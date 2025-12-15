from typing import List
from bittensor.core.subtensor import Subtensor
from bittensor.core.chain_data import (
    NeuronInfo,
)

def get_all_neurons(
    netuid: int = 296,
    network: str = "test"    
) -> List[NeuronInfo]:    
    subtensor = Subtensor(network=network)
    metagraph = subtensor.metagraph(netuid=netuid)
    neurons = metagraph.neurons
    return neurons


def test_list_all_uids_and_meta():
    neurons = get_all_neurons()
    print(f"\nTotal neurons fetched: {len(neurons)}")    
    for n in neurons:
        # NeuronInfo has these attributes:
        print(f"UID: {n.uid}")
        print(f"  Hotkey: {n.hotkey}")
        print(f"  Coldkey: {n.coldkey}")
        print(f"  Active: {n.active}")
        print(f"  Stake: {n.stake}")
        print(f"  Rank: {n.rank}")
        print(f"  Trust: {n.trust}")
        print(f"  Consensus: {n.consensus}")
        print(f"  Incentive: {n.incentive}")
        print(f"  Dividends: {n.dividends}")
        print(f"  Emission: {n.emission}")
        print(f"  Validator Permit: {n.validator_permit}")
        print(f"  Axon (IP:Port): {n.axon_info.ip}:{n.axon_info.port}")
        print(f"  Is Serving: {n.axon_info.is_serving}")
        print("---")

    active_count = sum(1 for n in neurons if n.active)
    print(f"\nTotal active neurons: {active_count} out of {len(neurons)}")
    inactive_count = len(neurons) - active_count
    print(f"Total inactive neurons: {inactive_count} out of {len(neurons)}")