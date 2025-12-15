import socket
import bittensor as bt
import numpy as np
import random
from typing import List, Tuple


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_miner_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))

    bt.logging.trace(f"\033[32m get_random_uids - pre candidate_uids: {candidate_uids} from k {k} \033[0m")

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = np.array(random.sample(available_uids, k))
    return uids



def get_all_miner_uids(self,   
    banned_coldkeys: set = None, 
    banned_hotkeys: set = None,
    banned_ips: set = None) -> Tuple[list[int], list[int]]:

    """Fetch random miners that meet criteria."""    
    
    avail_uids = [] 
    suspect_uids = []  
    for uid in range(self.metagraph.n.item()):
        if uid == 0 or uid == self.uid:
            continue
        if not self.metagraph.axons[uid].is_serving:
            continue
        this_stake = float(self.metagraph.S[uid])
        stake_limit = float(self.config.neuron.vpermit_tao_limit)
        if self.metagraph.validator_permit[uid] and this_stake > stake_limit:
            continue
        hk = self.metagraph.axons[uid].hotkey
        if hk not in self.hotkeys:
            continue
        if banned_coldkeys and self.metagraph.axons[uid].coldkey in banned_coldkeys:
            suspect_uids.append(uid)
            continue
        if banned_hotkeys and self.metagraph.axons[uid].hotkey in banned_hotkeys:
            suspect_uids.append(uid)
            continue
        if banned_ips and self.metagraph.axons[uid].ip in banned_ips:
            suspect_uids.append(uid)
            continue
    
        avail_uids.append(uid)

    suspect_uids = list(set(suspect_uids))
    bt.logging.trace(f"\033[32mALL candidate_uids: {avail_uids}\033[0m")
    bt.logging.trace(f"\033[33mcooldown nodes: {suspect_uids}\033[0m")

    return list(set(avail_uids)), suspect_uids


def best_uid(metagraph: bt.Metagraph) -> int:
    """Returns the best performing UID in the metagraph."""
    return max(range(metagraph.n), key=lambda uid: metagraph.I[uid].item()) 


def ping_miner_uid(self, uid, port=8091, timeout=5) -> bool:
    """
    Connect to a miner UID to check their availability.
    Returns True if successful, false otherwise
    """  
    ip = self.metagraph.axons[uid].ip
    ignored = ["localhost", "127.0.0.1", "0.0.0.0"]
    if ip in ignored:
        bt.logging.trace("Ignoring localhost ping.")
        return False

    try:        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((ip, port))
            return True
    except ConnectionRefusedError:
        bt.logging.warning(f"Port {port} on for UID {uid} is not connected.")
        return False
    except socket.timeout:
        bt.logging.warning(f"Timeout on Port {port} for UID {uid}.")
        return False
    except Exception as e:
        bt.logging.error(f"An error occurred: {e}")
        return False