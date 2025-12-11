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

import json
import os
import time
import bittensor as bt
import asyncio
import httpx
import numpy as np
import traceback
import requests
import signal
from random import SystemRandom
safe_random = SystemRandom()
from dotenv import load_dotenv
load_dotenv()
from datetime import timedelta
from bitrecs import __version__ as this_version
from bitrecs.base.validator import BaseValidatorNeuron
from bitrecs.utils.r2 import ValidatorUploadRequest
from bitrecs.utils.runtime import execute_periodically
from bitrecs.utils.version import LocalMetadata
from bitrecs.validator import forward
from bitrecs.protocol import BitrecsRequest
from bitrecs.utils import constants as CONST
from bitrecs.utils.epoch import get_current_epoch_info
from bitrecs.utils.r2 import put_r2_upload
from bitrecs.metrics.score_metrics import (
    check_score_health,
    run_complete_score_analysis
)
from bitrecs.utils.reasoning import ReasoningReport
from bitrecs.utils.rarity import RarityReport
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


validator_instance = None  # Global reference for signal handler

def handle_exit(signum, frame):
    bt.logging.info(f"Signal {signum} received, saving state and exiting...")
    if validator_instance is not None:
        validator_instance.save_state()
        if validator_instance.api_server:
            validator_instance.api_server.stop()
    exit(0)

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)


class Validator(BaseValidatorNeuron):
    """
    Bitrecs Validator Neuron

    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.load_state()
        self.total_request_in_interval = 0
        if not os.environ.get("BITRECS_PROXY_URL"):
            raise Exception("Please set the BITRECS_PROXY_URL environment variable.")        


    async def forward(self, request: BitrecsRequest = None):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Selecting a top candidate from the responses
        - Return top candidate to the client
        - Rewarding the miners
        - Updating the scores
        """                
        return await forward(self, request)
    
    
    @execute_periodically(timedelta(seconds=CONST.TEMPO_SYNC_INTERVAL))
    async def tempo_sync(self):
        """
        Periodically checks if a new epoch has started and resets miner batches.
        
        """

        def get_current_tempo(self):
            current_block = self.block
            netuid = self.config.netuid
            bt.logging.trace(f"Get epoch on block {current_block} and netuid {netuid}")
            current_tempo, _, _ = get_current_epoch_info(current_block, netuid)
            return current_tempo
        
        bt.logging.info(f"\033[35mTempo sync ran at {int(time.time())}\033[0m")
        if self.should_sync_metagraph():
            bt.logging.info(f"Sync metagraph tempo_sync size: {len(self.total_uids)} at block {self.block}")
            self.resync_metagraph()
            await self.update_total_uids()
            bt.logging.info(f"Metagraph resynced - final size: {len(self.total_uids)}")

        current_tempo = get_current_tempo(self)
        if self.last_tempo != current_tempo:
            self.last_tempo = current_tempo
            await self.start_new_tempo()
            bt.logging.info(f"\033[1;32mNew tempo started: {current_tempo} - Miner batches reset\033[0m")
    
     
    @execute_periodically(timedelta(seconds=CONST.VERSION_CHECK_INTERVAL))
    async def version_sync(self):
        bt.logging.info(f"\033[35mVersion sync ran at {int(time.time())}\033[0m")
        try:
            self.local_metadata = LocalMetadata.local_metadata()
            self.local_metadata.uid = self.uid
            self.local_metadata.hotkey = self.wallet.hotkey.ss58_address
            local_head = self.local_metadata.head
            remote_head = self.local_metadata.remote_head
            code_version = self.local_metadata.version
            bt.logging.info(f"Bitrecs Version:\033[32m {code_version}\033[0m")
            if local_head != remote_head:
                bt.logging.info(f"Head:\033[33m {local_head}\033[0m / Remote: \033[33m{remote_head}\033[0m")                
                bt.logging.warning(f"{self.neuron_type} version mismatch: Please update your code to the latest version.")
            else:
                 bt.logging.info(f"Head:\033[32m {local_head}\033[0m / Remote: \033[32m{remote_head}\033[0m")
        except Exception as e:
            bt.logging.error(f"Failed to get version with exception: {e}")
        return
       

    # @execute_periodically(timedelta(seconds=CONST.ACTION_SYNC_INTERVAL))
    # async def action_sync(self):
    #     """
    #     #TODO: This method is currently a placeholder and does not perform any actions.
                
    #     """
    #     bt.logging.trace(f"\033[35mAction sync ran at {int(time.time())}\033[0m")
    #     self.user_actions = []
    #     return
    
    
    @execute_periodically(timedelta(seconds=CONST.R2_SYNC_INTERVAL))
    async def r2_sync(self):
        """
        Periodically sync miner responses to R2
        """        
        bt.logging.info(f"\033[35mR2 sync ran at {int(time.time())}\033[0m")
        r2_enabled = self.config.r2.sync_on
        if not r2_enabled:
            bt.logging.trace(f"R2 Sync OFF at {int(time.time())}")        
            bt.logging.warning("R2 Sync is OFF set --r2.sync_on to enable")
            return

        start_time = time.perf_counter()
        #bt.logging.info(f"Starting R2 Sync at {int(time.time())}")
        if not self.wallet or not self.wallet.hotkey:
            bt.logging.error("Hotkey not found - skipping R2 sync")
            return
        try:
            keypair = self.wallet.hotkey
            bt.logging.trace(f"Using hotkey with address: {keypair.ss58_address}")
                
            update_request = ValidatorUploadRequest(
                hot_key=self.wallet.hotkey.ss58_address,
                val_uid=self.config.netuid,
                step=str(self.step),
                llm_provider="OPEN_ROUTER",
                llm_model="ignored"
            )
            bt.logging.trace(f"Sending response sync request: {update_request}")
            #sync_result = put_r2_upload(update_request, keypair)
            loop = asyncio.get_event_loop()
            sync_result = await loop.run_in_executor(
                None,
                lambda: put_r2_upload(update_request, keypair)
            )
            if sync_result:
                bt.logging.trace(f"\033[1;32m Success - R2 updated sync_result: {sync_result} \033[0m")
            else:
                bt.logging.error("\033[1;31m Failed to update R2 \033[0m")

        except Exception as e:
            bt.logging.error(f"Failed to update R2 with exception: {e}")
        finally:
            duration = time.perf_counter() - start_time
            bt.logging.info(f"R2 Sync complete in {duration:.2f} seconds")

    
    @execute_periodically(timedelta(seconds=CONST.SCORE_DISPLAY_INTERVAL))
    async def score_sync(self):
        """
        Enhanced score display with normalized weights and EMA insights
        """        
        bt.logging.info(f"\033[35mScore sync ran at {int(time.time())}\033[0m")
        try:           
            active_scores = {}
            for uid, score in enumerate(self.scores):
                if score > 0:
                    active_scores[uid] = score
            
            if not active_scores:
                bt.logging.info("No active scores to display")
                return            
            
            sorted_scores = sorted(active_scores.items(), key=lambda x: x[1], reverse=True)
            scores_array = np.array(list(active_scores.values()))
            stats = {
                'count': len(active_scores),
                'mean': np.mean(scores_array),
                'std': np.std(scores_array),
                'min': np.min(scores_array),
                'max': np.max(scores_array),
                'median': np.median(scores_array),
                'cv': np.std(scores_array) / np.mean(scores_array) if np.mean(scores_array) > 0 else 0
            }
            
            # Display summary
            bt.logging.info(f"\033[1;36m=== SCORE SUMMARY (Step {self.step}) ===\033[0m")
            bt.logging.info(f"Active miners: {stats['count']}")
            bt.logging.info(f"Mean: {stats['mean']:.6f} | Std: {stats['std']:.6f} | CV: {stats['cv']:.3f}")
            bt.logging.info(f"Min: {stats['min']:.6f} | Max: {stats['max']:.6f} | Median: {stats['median']:.6f}")
            
            # Calculate max/min ratio more safely
            min_threshold = 1e-6
            safe_min = max(stats['min'], min_threshold)
            max_min_ratio = stats['max'] / safe_min

            if stats['min'] < min_threshold:
                #bt.logging.warning(f"⚠️Very small minimum score detected: {stats['min']:.8f}")
                bt.logging.info(f"Max/Min ratio (safe): {max_min_ratio:.2f}")
            else:
                bt.logging.info(f"Max/Min ratio: {max_min_ratio:.2f}")
            
            # Display top performers
            bt.logging.info("\033[1;32m=== TOP PERFORMERS ===\033[0m")
            for i, (uid, score) in enumerate(sorted_scores[:10], 1):
                percentile = (len(sorted_scores) - i + 1) / len(sorted_scores) * 100
                bt.logging.info(f"{i:2d}. UID {uid:2d}: {score:.6f} ({percentile:.1f}%)")
            
            # Display score distribution
            bt.logging.info("\033[1;34m=== SCORE DISTRIBUTION ===\033[0m")
            quartiles = np.percentile(scores_array, [25, 50, 75])
            bt.logging.info(f"Q1: {quartiles[0]:.6f} | Q2: {quartiles[1]:.6f} | Q3: {quartiles[2]:.6f}")
            iqr = quartiles[2] - quartiles[0]
            bt.logging.info(f"IQR: {iqr:.6f}")
            
            # Track score changes over time
            if not hasattr(self, 'score_history'):
                self.score_history = []
            
            # Store current snapshot
            current_snapshot = {
                'step': self.step,
                'timestamp': time.time(),
                'stats': stats,
                'top_3': sorted_scores[:3],
                'active_uids': list(active_scores.keys())
            }
            
            self.score_history.append(current_snapshot)
            
            # Keep only last 20 snapshots
            if len(self.score_history) > 20:
                self.score_history = self.score_history[-20:]            
            
            run_complete_score_analysis(self)
            
            # Enhanced health checks
            check_score_health(self, stats, max_min_ratio)
            
            # Store alpha history for insights
            if hasattr(self, 'last_alpha_used'):
                if not hasattr(self, 'alpha_history'):
                    self.alpha_history = []
                self.alpha_history.append(self.last_alpha_used)
                if len(self.alpha_history) > 50:
                    self.alpha_history = self.alpha_history[-50:]
            
        except Exception as e:
            bt.logging.error(f"Error in enhanced score_sync: {e}")
            bt.logging.error(traceback.format_exc())
        

    @execute_periodically(timedelta(seconds=CONST.COOLDOWN_SYNC_INTERVAL))
    async def cooldown_sync(self):
        """
        Load cooldowns to decay lazy miners.

        """
        try:            
            bt.logging.info(f"\033[35mCooldown sync ran at {int(time.time())}\033[0m")
            proxy_url = os.environ.get("BITRECS_PROXY_URL").removesuffix("/")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('BITRECS_API_KEY')}",
                "User-Agent": f"Bitrecs-Node/{this_version}"
            }
            r = requests.get(f"{proxy_url}/cooldown", headers=headers, timeout=10)
            r.raise_for_status()
            cooldowns = r.json()
            self.banned_ips = cooldowns["banned_ips"] or []
            self.banned_coldkeys = cooldowns["banned_coldkeys"] or []
            self.banned_hotkeys = cooldowns["banned_hotkeys"] or []
            self.r_limit = cooldowns["r_limit"] or 1.0
            bt.logging.trace(f"Cooldowns updated: Limit: {self.r_limit},  {len(self.banned_ips)} IPs, {len(self.banned_coldkeys)} coldkeys, {len(self.banned_hotkeys)} hotkeys")
        except Exception as e:
            bt.logging.error(f"cooldown_sync Exception: {e}")


    @execute_periodically(timedelta(seconds=CONST.REASONING_SYNC_INTERVAL))
    async def reasoning_sync(self):
        """
        Load reasoning reports
        
        """
        try:
            if not CONST.REASONING_SCORING_ENABLED:
                return
            bt.logging.info(f"\033[35mReasoning sync ran at {int(time.time())}\033[0m")
            reports = ReasoningReport.get_reports()
            if not reports or len(reports) == 0:
                bt.logging.error("\033[31mNo reasoning reports found!\033[0m")
                self.reasoning_reports = []
                return
            self.reasoning_reports = reports
            bt.logging.info(f"Reasoning sync complete with \033[32m{len(self.reasoning_reports)}\033[0m reports")
            
            delta = ReasoningReport.get_delta_uids(reports, self.metagraph)
            self.missing_evals_uids = set(
                uid for uid in self.total_uids
                if uid in delta
            )
            if len(self.missing_evals_uids) > 0:
                bt.logging.warning(f"\033[33mReasoning evals delta: {list(self.missing_evals_uids)}\033[0m")
            else:
                bt.logging.info("\033[32mAll miners have reasoning evals\033[0m")
            
        except Exception as e:
            bt.logging.error(f"reasoning_sync Exception: {e}")
  
    
    @execute_periodically(timedelta(seconds=CONST.VERFIED_KEY_SYNC_INTERVAL))
    async def verified_sync(self):

        async def get_verified_public_key() -> Ed25519PublicKey:
            async with httpx.AsyncClient(timeout=30.0) as client:
                public_key_response = await client.get(f"{CONST.VERIFIED_INFERENCE_URL}/public_key")
                public_key_response.raise_for_status()
                public_key_string = json.loads(public_key_response.text)["public_key"]
                raw_bytes = bytes.fromhex(public_key_string)
                return Ed25519PublicKey.from_public_bytes(raw_bytes)
         
        bt.logging.info(f"\033[35mVerified sync ran at {int(time.time())}\033[0m")
        self.verified_public_key = await get_verified_public_key()
        if self.verified_public_key:
            bt.logging.info(f"\033[32mVerifier key loaded for network: {self.network}\033[0m")
            self.rarity_reports = RarityReport.get_reports()
            bt.logging.info(f"Verifier loaded \033[32m{len(self.rarity_reports)}\033[0m rarity reports")
        else:
            bt.logging.error(f"\033[31mFailed to load verifier key for network: {self.network}\033[0m")
            raise Exception("Failed to load verifier public key")
        
        
    @execute_periodically(timedelta(seconds=CONST.BACKUP_WEIGHT_SYNC_INTERVAL))
    async def weights_sync(self):
        """
        Weights are synced via the Validator inbound, this is a backup sync incase inbound fails for extended periods.
        
        """
        bt.logging.info(f"\033[35mWeights sync ran at {int(time.time())}\033[0m")
        if self.should_set_weights():
            self.set_weights()
            bt.logging.info("Weights synchronized via weights_sync.")

    

async def main():
    global validator_instance
    bt.logging.info(f"\033[32m Starting Bitrecs Validator\033[0m ... {int(time.time())}")    
    with Validator() as validator:
        validator_instance = validator
        start_time = time.time()
        await validator.cooldown_sync()
        await validator.update_total_uids()
        while True:
            tasks = [
                asyncio.create_task(validator.verified_sync()),
                asyncio.create_task(validator.tempo_sync()),
                asyncio.create_task(validator.version_sync()),
                asyncio.create_task(validator.r2_sync()),
                asyncio.create_task(validator.cooldown_sync()),
                asyncio.create_task(validator.reasoning_sync()),
                asyncio.create_task(validator.weights_sync())
            ]
            if validator.config.logging.trace and CONST.SCORE_DISPLAY_ENABLED:
                tasks.append(asyncio.create_task(validator.score_sync()))
                
            await asyncio.gather(*tasks, return_exceptions=True)
            
            bt.logging.info(f"Validator {validator.uid} running... {int(time.time())}")
            if time.time() - start_time > 300:
                bt.logging.info(
                    f"---Total request in last 5 minutes: {validator.total_request_in_interval}"
                )
                start_time = time.time()
                validator.total_request_in_interval = 0
            await asyncio.sleep(15)

if __name__ == "__main__": 
    asyncio.run(main())
