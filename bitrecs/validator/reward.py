# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Bitrecs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import base64
import json
import hashlib
import traceback
import numpy as np
import bittensor as bt
import jsonschema
import json_repair
from typing import List, Tuple
from datetime import datetime, timezone
from bitrecs.commerce.user_action import UserAction
from bitrecs.protocol import BitrecsRequest, SignedResponse
from bitrecs.commerce.product import Product, ProductFactory
from bitrecs.utils import constants as CONST
from bitrecs.utils.reasoning import ReasonReport
from bitrecs.llms.prompt_factory import PromptFactory

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

BASE_REWARD = 0.80
CONSENSUS_BONUS_MULTIPLIER = 1.05
REASONING_BONUS_MULTIPLIER = 1.025
VERFIED_BONUS_MULTIPLIER = 1.15
SUSPECT_MINER_DECAY = 0.980
PERMITTED_CLOCK_DIFF_SECONDS = 300


class CatalogValidator:
    def __init__(self, store_catalog: List[Product]):
        self.sku_set = {product.sku.lower().strip() for product in store_catalog}
    
    def validate_sku(self, sku: str) -> bool:
        if not sku:
            return False
        return sku.lower().strip() in self.sku_set


def validate_result_schema(num_recs: int, results: list) -> bool:
    """
    Ensure results from Miner match the required schema
    """
    if num_recs < 1 or num_recs > CONST.MAX_RECS_PER_REQUEST:
        return False
    if len(results) != num_recs:
        bt.logging.error("Error validate_result_schema num_recs mismatch")
        return False
    
    schema = {
        "type": "object",
        "properties": {
            "sku": {"type": "string"},
            "name": {"type": "string"},
            "price": {"type": ["string", "number"]},
            "reason": {"type": "string"}
        },
        "required": ["sku", "name", "price", "reason"],
    }

    count = 0
    for item in results:
        try:
            thing = json_repair.loads(item)
            jsonschema.validate(thing, schema)           
            count += 1
        except json.decoder.JSONDecodeError as e:            
            bt.logging.trace(f"JSON JSONDecodeError ERROR: {e}")
            break
        except jsonschema.exceptions.ValidationError as e:            
            bt.logging.trace(f"JSON ValidationError ERROR: {e}")
            break
        except Exception as e:            
            bt.logging.trace(f"JSON Exception ERROR: {e}")
            break

    return count == len(results)


def verify_miner_signature(response: BitrecsRequest) -> bool:
    try:
        if not response.miner_signature:
            bt.logging.error("Response missing miner_signature")
            return False
        payload = {
            "name": response.name,
            "axon_hotkey": response.axon.hotkey,
            "dendrite_hotkey": response.dendrite.hotkey,
            "created_at": response.created_at,
            "num_results": response.num_results,
            "query": response.query,
            "site_key": response.site_key,
            "results": response.results,
            "models_used": response.models_used,
            "miner_uid": response.miner_uid,
            "miner_hotkey": response.miner_hotkey,
            "verified_proof": response.verified_proof
        }
        payload_str = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).digest()
        signature = bytes.fromhex(response.miner_signature)
        miner_hotkey = response.miner_hotkey
        return bt.Keypair(ss58_address=miner_hotkey).verify(payload_hash, signature)
    except Exception as e:
        bt.logging.error(f"Error verifying response signature: {e}")
        return False


def verify_time(response: BitrecsRequest) -> bool:
    response_time = datetime.fromisoformat(response.created_at)
    if response_time.tzinfo is None:
        response_time = response_time.replace(tzinfo=timezone.utc)
    utc_now = datetime.now(timezone.utc)
    age = (utc_now - response_time).total_seconds()
    if age <= 0 or age > PERMITTED_CLOCK_DIFF_SECONDS:
        bt.logging.error(f"Failed verify_time: {age} seconds for {response.axon.hotkey[:8]}")
        return False
    return True


def validate_proof_skus(valid_skus: set, signed_response: SignedResponse) -> bool:
    try:
        if len(valid_skus) == 0:
            bt.logging.error("No valid SKUs provided")
            return False
        response = signed_response.response
        skus = PromptFactory.extract_skus_from_response(response) or []
        extracted_skus = set(sku.upper().strip() for sku in skus)
        expected_skus = set(sku.upper().strip() for sku in valid_skus)
        if extracted_skus != expected_skus:
            bt.logging.error(f"SKU mismatch: extracted {extracted_skus}, expected {expected_skus}")
            return False
        return True
    except Exception as e:
        bt.logging.error(f"Error in validate_proof_skus: {e}")
        return False
    

def verify_proof_with_recs(
    recs: set,
    response: SignedResponse,
    public_key: Ed25519PublicKey
) -> bool:
    proof = response.proof
    signature_b64 = response.signature
    timestamp = response.timestamp
    ttl = response.ttl
    try:
        current_time = datetime.now(timezone.utc)
        proof_time = datetime.fromisoformat(timestamp)
        ttl_time = datetime.fromisoformat(ttl)
        time_diff = abs((current_time - proof_time).total_seconds())
        if time_diff > PERMITTED_CLOCK_DIFF_SECONDS:
            bt.logging.error(f"Timestamp too old or future: {time_diff} seconds")
            return False
        if current_time > ttl_time:
            bt.logging.error(f"Proof expired: TTL {ttl_time}, current {current_time}")
            return False
        validate_recs = validate_proof_skus(recs, response)
        if not validate_recs:
            bt.logging.error("SKU proof validation failed")
            return False

        signed_data = {
            "proof": proof,
            "timestamp": timestamp,
            "ttl": ttl
        }
        serialized_data = json.dumps(signed_data, sort_keys=True).encode()
        signature_bytes = base64.b64decode(signature_b64)
        public_key.verify(signature_bytes, serialized_data)
        return True
    except InvalidSignature:
        bt.logging.error("verify_proof_with_recs failed: Invalid signature")
        return False
    except Exception as e:
        bt.logging.error(f"verify_proof_with_recs failed: {e}")
        bt.logging.debug(f"Traceback: {traceback.format_exc()}")
        return False




def reward(
    validator_hotkey: str,
    ground_truth: BitrecsRequest,
    catalog_validator: CatalogValidator, 
    response: BitrecsRequest,
    reasoning_report: ReasonReport = None,
    actions: List[UserAction] = None,
    r_limit: float = 1.0,
    max_f_score: float = 1.0,
    verified_public_key: Ed25519PublicKey = None
) -> Tuple[float, str]:
    """
    Score the Miner's response to the BitrecsRequest   

    Returns:
    - Tuple[float, str]: A tuple containing the score and notes.  
    """
    
    try:
        score = 0.0
        if response.dendrite.hotkey != validator_hotkey:
            bt.logging.error(f"Response from different hotkey: {response.dendrite.hotkey} != {validator_hotkey}")
            bt.logging.error(f"Miner hotkey: {response.axon.hotkey[:8]}")
            return 0.0, "Invalid_Validator_Hotkey"
        if not response.dendrite.signature:
            bt.logging.error(f"Response missing signature: {response.axon.hotkey[:8]}")
            return 0.0, "Missing_Dendrite_Signature"
        if response.is_timeout:
            bt.logging.error(f"{response.axon.hotkey[:8]} is_timeout is True, status: {response.dendrite.status_code}")
            return 0.0, "Timeout"
        if response.is_failure:
            bt.logging.error(f"{response.axon.hotkey[:8]} is_failure is True, status: {response.dendrite.status_code}")
            return 0.0, "Failure"
        if not response.is_success:
            bt.logging.error(f"{response.axon.hotkey[:8]} is_success is False, status: {response.dendrite.status_code}")
            return 0.0, "Unsuccessful"
        if not verify_time(response):
            bt.logging.error(f"{response.axon.hotkey[:8]} response time verification failed")
            return 0.0, "Invalid_Timestamp"
        if not verify_miner_signature(response):
            bt.logging.error(f"{response.axon.hotkey[:8]} signature verification failed")
            return 0.0, "Invalid_Signature"
        if not response.miner_uid or not response.miner_hotkey:
            bt.logging.error(f"{response.axon.hotkey[:8]} is not reporting correctly (missing ids)")
            return 0.0, "Missing_Miner_UID"
        if response.miner_hotkey.lower() != response.axon.hotkey.lower():
            bt.logging.error(f"{response.miner_uid} hotkey mismatch: {response.miner_hotkey} != {response.axon.hotkey}")
            return 0.0, "Invalid_Miner_Hotkey"
        if len(response.results) != ground_truth.num_results:
            bt.logging.error(f"{response.miner_uid} num_recs mismatch, expected {ground_truth.num_results} but got {len(response.results)}")
            return 0.0, "Invalid_Num_Results"
        if len(response.models_used) != 1:
            bt.logging.error(f"{response.miner_uid} has invalid models used: {response.miner_hotkey[:8]}")
            return 0.0, "Invalid_Models_Used"
        if response.axon.process_time < r_limit:
            bt.logging.error(f"\033[33m WARNING Miner {response.miner_uid} axon time: {response.axon.process_time} < {r_limit}\033[0m")
            return 0.0, "Invalid_Axon_Time"
        if response.dendrite.process_time < r_limit:
            bt.logging.error(f"\033[33m WARNING Miner {response.miner_uid} dendrite time: {response.dendrite.process_time} < {r_limit}\033[0m")
            return 0.0, "Invalid_Dendrite_Time"
        if response.query != ground_truth.query:
            bt.logging.error(f"{response.miner_uid} query mismatch: {response.query} != {ground_truth.query}")
            return 0.0, "Invalid_Query"
        if response.context != "[]":
            bt.logging.error(f"{response.miner_uid} context is not empty: {response.context}")
            return 0.0, "Invalid_Context"
        if not validate_result_schema(ground_truth.num_results, response.results):
            bt.logging.error(f"{response.miner_uid} failed schema validation: {response.miner_hotkey[:8]}")
            return 0.0, "Invalid_Schema"
     
        valid_items = set()
        query_lower = response.query.lower().strip()
        for result in response.results:
            try:
                product = json_repair.loads(result)
                sku = product["sku"]
                if sku.lower() == query_lower:
                    bt.logging.error(f"{response.miner_uid} has query in results: {response.miner_hotkey[:8]}")
                    return 0.0, "Query_In_Results"
                if sku in valid_items:
                    bt.logging.error(f"{response.miner_uid} has duplicate results: {response.miner_hotkey[:8]}")
                    return 0.0, "Duplicate_Results"
                if not catalog_validator.validate_sku(sku):
                    bt.logging.error(f"{response.miner_uid} has skus not in the catalog: {response.miner_hotkey[:8]}")
                    return 0.0, "Invalid_SKU"
                
                valid_items.add(sku)
            except Exception as e:
                bt.logging.error(f"JSON ERROR: {e}, miner data: {response.miner_hotkey}")
                return 0.0, "Invalid_JSON"

        if len(valid_items) != ground_truth.num_results:
            bt.logging.error(f"{response.miner_uid} invalid number of valid_items: {response.miner_hotkey[:8]}")
            return 0.0, "Numrecs_Mismatch"
        
        score = BASE_REWARD
        score_notes = ["Base_Reward"]        
        if CONST.REASONING_SCORING_ENABLED:
            if not reasoning_report:
                score = BASE_REWARD / 4
                bt.logging.warning(f"\033[33m{response.miner_hotkey[:8]} no report score:{score}\033[0m")
                score_notes.append("No_Reasoning_Report")
            elif reasoning_report.f_score <= 0:
                score = BASE_REWARD / 2
                bt.logging.warning(f"\033[33m{response.miner_hotkey[:8]} no/low reasoning score:{score}\033[0m")
                score_notes.append("Low_Reasoning_Score")
            else:
                f_score = min(reasoning_report.f_score, max_f_score)
                score = BASE_REWARD + f_score
                score *= REASONING_BONUS_MULTIPLIER
                bt.logging.trace(f"\033[32m{response.miner_hotkey[:8]} score:{score:.6f} f_score: {f_score:.6f} rank: {reasoning_report.rank}\033[0m")
                score_notes.append("Reasoning_Bonus")

        if response.verified_proof and response.verified_proof != {} and verified_public_key:
            signed_response = SignedResponse(**response.verified_proof)            
            verified = verify_proof_with_recs(valid_items, signed_response, verified_public_key)
            if not verified:
                bt.logging.error(f"{response.axon.hotkey[:8]} Verified Inference Failed: {response.miner_uid}")
                return 0.0, "Invalid_Verified_Proof"
            else:
                score *= VERFIED_BONUS_MULTIPLIER
                bt.logging.trace(f"\033[32m{response.axon.hotkey[:8]} Verified Inference Success: {response.miner_uid}\033[0m")
                score_notes.append("Verified_Proof_Bonus")
        
        notes = " | ".join(score_notes)
        return score, notes
    except Exception as e:
        bt.logging.error(f"Error in rewards: {e}, miner data: {response}")
        return 0.0, "Exception_Error"



def get_rewards(
    validator_hotkey: str,
    ground_truth: BitrecsRequest,
    responses: List[BitrecsRequest],
    reasoning_reports: List[ReasonReport] = None,
    actions: List[UserAction] = None,    
    r_limit: float = 1.0,
    batch_size: int = 16,
    entity_threshold: float = 0.2,
    verified_public_key: Ed25519PublicKey = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns an array of rewards for the given query and responses.
    - validator_hotkey: The hotkey of the validator.
    - ground_truth: The BitrecsRequest object containing the ground truth query
    - responses: A list of BitrecsRequest objects containing the responses from miners.
    - actions: A list of UserAction objects containing the actions performed by users.
    - r_limit: The rlimit for responses.
    - batch_size: The number of responses in this batch.
    - entity_threshold: The threshold for considering nodes as entities.
    - verified_public_key: The public key used to verify proofs of inference.
    Returns:
    - Tuple[np.ndarray, List[str]]: A tuple containing an array of rewards and a list of reward notes.    
    
    """

    if ground_truth.num_results < CONST.MIN_RECS_PER_REQUEST or ground_truth.num_results > CONST.MAX_RECS_PER_REQUEST:
        bt.logging.error(f"Invalid number of recommendations: {ground_truth.num_results}")
        raise ValueError(f"Invalid number of recommendations: {ground_truth.num_results}")

    store_catalog: list[Product] = ProductFactory.try_parse_context_strict(ground_truth.context)
    if len(store_catalog) < CONST.MIN_CATALOG_SIZE or len(store_catalog) > CONST.MAX_CATALOG_SIZE:
        bt.logging.error(f"Invalid catalog size: {len(store_catalog)}")
        raise ValueError(f"Invalid catalog size: {len(store_catalog)}")
    catalog_validator = CatalogValidator(store_catalog)

    if not reasoning_reports or len(reasoning_reports) == 0:
        bt.logging.warning(f"\033[1;33m WARNING - no reasoning_reports found in get_rewards \033[0m")

    if not actions or len(actions) == 0:
        bt.logging.warning(f"\033[1;33m WARNING - no actions found in get_rewards \033[0m")

    axon_times = []
    for response in responses:
        axon_time = response.axon.process_time if response.axon and response.axon.process_time else None
        axon_times.append(axon_time)

    valid_times = [t for t in axon_times if t is not None and t > 0]
    if len(valid_times) > 1:
        min_time = min(valid_times)
        max_time = max(valid_times)
        avg_time = sum(valid_times) / len(valid_times)
        spread = max_time - min_time
        bt.logging.trace(f"Batch: min={min_time:.4f}s, max={max_time:.4f}s, avg={avg_time:.4f}s, spread={spread:.4f}s")
        if spread > 2.0:
            bt.logging.info(f"\033[33mWide Spread detected: {spread:.4f}s\033[0m")

    ip_counts = {}
    for response in responses:
        ip = response.axon.ip
        ip_counts[ip] = ip_counts.get(ip, 0) + 1

    max_ip_count = max(ip_counts.values()) if ip_counts else 0
    max_ip_percent = max_ip_count / len(responses) if responses else 0
    
    bt.logging.trace("----------------|----|----------")
    for ip, count in sorted(ip_counts.items(), key=lambda x: -x[1]):
        this_hk = [r.axon.hotkey for r in responses if r.axon.ip == ip][0]
        bt.logging.trace(f"{ip:<15} | {count}  | {this_hk[:8]}")

    entity_ips = set()
    if max_ip_percent >= entity_threshold:
        entity_ips = {ip for ip, count in ip_counts.items() if count / len(responses) >= entity_threshold}
        bt.logging.warning(f"Entity threshold > \033[33m{max_ip_percent:.2%}\033[0m from {entity_ips}")

    difficulty_decay = measure_request_difficulty(
        sku=ground_truth.query,
        catalog_size=len(store_catalog),
        num_recs=ground_truth.num_results,
        num_participants=len(responses),
        min_catalog_size=CONST.MIN_CATALOG_SIZE,
        max_catalog_size=CONST.MAX_CATALOG_SIZE,
        min_recs=CONST.MIN_RECS_PER_REQUEST,
        max_recs=CONST.MAX_RECS_PER_REQUEST,
        min_participants=1,
        max_participants=batch_size,
        base=1.0,
        min_decay=0.9,   # 10% penalty for easiest
        max_decay=1.0    # no penalty for hardest
    )

    difficulty_statement = get_difficulty_statement(difficulty_decay)
    bt.logging.trace(f"{difficulty_statement}")
    if not CONST.DIFFICULTY_SCORING_ENABLED:
        bt.logging.trace(f"\033[33mDifficulty adjustment skipped!\033[0m")
    
    if CONST.REASONING_SCORING_ENABLED:
        bt.logging.trace(f"\033[32mReasoning scoring is enabled\033[0m")

    rewards = []
    reward_notes = []
    for i, response in enumerate(responses):
        if response.axon.ip in entity_ips and not CONST.REWARD_ENTITIES:
            rewards.append(0.0)
            reward_notes.append("Entity_No_Reward")
            continue
        
        r_report = get_reasoning_report(response, reasoning_reports)
        max_f_score = max((r.f_score for r in reasoning_reports), default=1.0)
        miner_reward, reward_note = reward(validator_hotkey, 
                              ground_truth, 
                              catalog_validator, 
                              response, 
                              r_report, 
                              actions, 
                              r_limit, 
                              max_f_score, 
                              verified_public_key)
        reward_notes.append(reward_note)
        if miner_reward <= 0.0:
            rewards.append(0.0)
            continue
        if CONST.DIFFICULTY_SCORING_ENABLED:
            final_score = miner_reward * difficulty_decay
        else:
            final_score = miner_reward
        rewards.append(final_score)

    result = (np.array(rewards, dtype=float), reward_notes)
    return result    


def measure_request_difficulty(
    sku: str,
    catalog_size: int,
    num_recs: int,
    num_participants: int,
    min_catalog_size: int = 5,
    max_catalog_size: int = 1000,
    min_recs: int = CONST.MIN_RECS_PER_REQUEST,
    max_recs: int = CONST.MAX_RECS_PER_REQUEST,
    min_participants: int = 1,
    max_participants: int = 16,
    base: float = 1.0,
    min_decay: float = 0.9,   # 10% penalty for easiest
    max_decay: float = 1.0    # no penalty for hardest
) -> float:
    """
    Returns a decay factor in [min_decay, max_decay].
    - Easiest requests get min_decay (0.9).
    - Hardest requests get max_decay (1.0, no penalty).
    """
    catalog_weight = 0.4
    recs_weight = 0.1
    participants_weight = 0.2

    # Use catalog size instead of context length
    catalog_factor = (catalog_size - min_catalog_size) / (max_catalog_size - min_catalog_size)
    catalog_factor = max(0.0, min(catalog_factor, 1.0))
    recs_factor = (num_recs - min_recs) / (max_recs - min_recs)
    recs_factor = max(0.0, min(recs_factor, 1.0))
    part_factor = (num_participants - min_participants) / (max_participants - min_participants)
    part_factor = max(0.0, min(part_factor, 1.0))

    raw_difficulty = base * (1 + catalog_weight * catalog_factor) * (1 + recs_weight * recs_factor) * (1 + participants_weight * part_factor)
    max_difficulty = base * (1 + catalog_weight) * (1 + recs_weight) * (1 + participants_weight)

    # Map to decay factor in [min_decay, max_decay]
    decay = min_decay + (max_decay - min_decay) * (raw_difficulty - 1.0) / (max_difficulty - 1.0)
    decay = max(min_decay, min(decay, max_decay))
    return float(decay)


def get_difficulty_statement(difficulty: float) -> str:
    """
    Returns a human-readable statement about the request difficulty.
    """
    if difficulty <= 0.93:
        return f"Difficulty is very easy: \033[1;32m{difficulty:.3f}\033[0m"
    elif difficulty <= 0.97:
        return f"Difficulty is medium: \033[1;33m{difficulty:.3f}\033[0m"
    else:
        return f"Difficulty is hard: \033[1;31m{difficulty:.3f}\033[0m"
    
def get_reasoning_report(
    response: BitrecsRequest,
    reasoning_reports: List[ReasonReport] = None
) -> ReasonReport | None:
    if not reasoning_reports or len(reasoning_reports) == 0:
        return None
    reasoning_report = next(
        (r for r in reasoning_reports if r.miner_hotkey.lower().strip() == response.axon.hotkey.lower().strip()),
        None
    )
    return reasoning_report