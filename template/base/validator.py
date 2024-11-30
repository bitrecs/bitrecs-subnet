# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 bitrecs.ai

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


import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt
import time
import traceback
import anyio.to_thread
import random


from typing import List, Union, Optional
from traceback import print_exception

from template.base.neuron import BaseNeuron
from template.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit, 
)  # TODO: Replace when bittensor switches to numpy
from template.mock import MockDendrite
from template.utils.config import add_validator_args

from template.api.api_server import ApiServer
from template.protocol import BitrecsRequest
from dataclasses import dataclass
from queue import SimpleQueue, Empty

from template.utils.uids import check_uid_availability, get_random_uids, clamp

api_queue = SimpleQueue() # Queue of SynapseEventPair

@dataclass
class SynapseWithEvent:
    """ Object that API server can send to main thread to be serviced. """
    input_synapse: BitrecsRequest
    event: threading.Event
    output_synapse: BitrecsRequest


async def api_forward(synapse: BitrecsRequest) -> BitrecsRequest:
    bt.logging.info(f"api_forward validator synapse: {synapse}")
    bt.logging.info(f"api_forward validator synapse type: {type(synapse)}")
    
    """ Forward function for API server. """
    synapse_with_event = SynapseWithEvent(
        input_synapse=synapse,
        event=threading.Event(),
        output_synapse=BitrecsRequest(
            name=synapse.name,                     
            created_at=synapse.created_at,
            user=synapse.user,
            num_results=synapse.num_results,
            query=synapse.query,
            context=synapse.context,
            site_key=synapse.site_key,
            results=synapse.results,
            models_used=synapse.models_used,
            miner_uid=synapse.miner_uid,
            miner_hotkey=synapse.miner_hotkey
        )
    )
    api_queue.put(synapse_with_event)
    # Wait until the main thread marks this synapse as processed.
    await anyio.to_thread.run_sync(synapse_with_event.event.wait)
    return synapse_with_event.output_synapse



class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        if self.config.api.enabled:
            # external requests
            api_server = ApiServer(
                axon_port=self.config.axon.port,
                forward_fn=api_forward,
                api_json=self.config.api_json,          
                ngrok_domain="bitrecs.ai"
            )
            api_server.start()            
            bt.logging.info(f"\033[1;32m 🐸 API Endpoint Started: {api_server.fast_server.config.host} on Axon: {api_server.fast_server.config.port} \033[0m")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config, port=self.config.axon.port)
            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    async def concurrent_forward2(self, pr: BitrecsRequest):
        coroutines = [
            self.forward(pr)
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        return await asyncio.gather(*coroutines)
    
    def select_top_result(self, original_request: BitrecsRequest, miner_results: List[BitrecsRequest]) -> BitrecsRequest:
        """Selects the top result from the list of results."""
        for r in miner_results:
            bt.logging.info(f"select_top_result Result: {r}")            
            if len(r.results) == original_request.num_results:
                return r            
        return None


    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        #bt.logging.info(f"Validator starting at block: {self.block}")
        bt.logging.info(
            f"\033[1;32m 🐸 Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}\033[0m")
        if hasattr(self, "axon"):
            f"Axon: {self.axon}"

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                try:

                    api_enabled = self.config.api.enabled
                    api_exclusive = self.config.api.exclusive

                    bt.logging.info(f"api_enabled: {api_enabled}")
                    bt.logging.info(f"api_exclusive: {api_exclusive}")

                    synapse_with_event: Optional[SynapseWithEvent] = None
                    try:
                        synapse_with_event = api_queue.get(timeout=5)
                        bt.logging.info(f"api_queue queue found a Request {synapse_with_event}")
                    except Empty:
                        # No synapse from API server.
                        pass

                    if synapse_with_event is not None and api_enabled: #API request
                        bt.logging.info("** Processing synapse from API server **")

                        available_uids = get_random_uids(self, k=self.config.neuron.sample_size)
                        # available_uids = [
                        #     uid
                        #     for uid in range(self.metagraph.n.item())
                        #     if check_uid_availability(
                        #         metagraph=self.metagraph,
                        #         uid=uid,
                        #         vpermit_tao_limit=0.1
                        #     )
                        # ]
                        bt.logging.trace(f"available_uids: {available_uids}")
                        chosen_uids = random.sample(
                            available_uids,
                            k=clamp(min=1, max=10, x=len(available_uids))
                        )
                        bt.logging.debug(f"len(chosen_uids): {len(chosen_uids)}")
                        bt.logging.debug(f"chosen_uids: {chosen_uids}")
                        chosen_axons = [
                            self.metagraph.axons[uid]
                            for uid in chosen_uids
                        ]
                        bt.logging.trace(f"chosen_axons: {chosen_axons}")

                        api_request = synapse_with_event.input_synapse

                        responses = self.dendrite.query(
                            # Send the query to all axons in the network.
                            chosen_axons,
                            # Construct a query.
                            api_request,
                            # All responses have the deserialize function called on them before returning.
                            deserialize=False,
                            timeout=10.0
                        )

                        bt.logging.debug(f"len(responses): {len(responses)}")

                        #TODO ranking and scoring
                        selected_response = self.select_top_result(api_request, responses)
                        if selected_response is None:
                            bt.logging.error("No valid result could be parsed ! skipping request")
                            continue

                        synapse_with_event.output_synapse = selected_response
                        # thing = self.loop.run_until_complete(self.concurrent_forward2(synapse_with_event.input_synapse))
                        # bt.logging.info(f"thing: {thing}")
                        synapse_with_event.event.set()

                    else:     
                        if not api_exclusive: #Regular validator loop                
                            bt.logging.info("Processing synthetic concurrent forward")
                            self.loop.run_until_complete(self.concurrent_forward())

                    if self.should_exit:
                        return

                    try:                        
                        self.sync()
                    except Exception as e:
                        bt.logging.error(traceback.format_exc())
                        bt.logging.error(f"Failed to sync with exception: {e}")

                    self.step += 1

                except Exception as e:
                    bt.logging.error(f"Failed to run forward with exception: {e}")
                    time.sleep(60)
                finally:                   
                    if api_enabled and api_exclusive:
                        bt.logging.info(f"forward finished, ready for next request")
                        #time.sleep(10)
                        pass
                    else:
                        bt.logging.info(f"forward finished, sleep for {10} seconds")
                        time.sleep(10)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(traceback.format_exc(err))
            # bt.logging.debug(
            #     str(print_exception(type(err), err, err.__traceback__))
            # )

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        bt.logging.info(f"set_weights on chain start")       
        bt.logging.info(f"Scores: {self.scores}")       

        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )
        
        if np.all(self.scores == 0):
            bt.logging.warning(
                f"Scores are all zero. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )
            return

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN
        
        bt.logging.debug("norm", norm)
        
        # Compute raw_weights safely
        raw_weights = self.scores / norm         
        
        # Printing type of arr object
        bt.logging.debug("Array is of type: ", type(raw_weights))
        # Printing array dimensions (axes)
        bt.logging.debug("No. of dimensions: ", raw_weights.ndim)
        # Printing shape of array
        bt.logging.debug("Shape of array: ", raw_weights.shape)
        # Printing size (total number of elements) of array
        bt.logging.debug("Size of array: ", raw_weights.size)
        # Printing type of elements in array
        bt.logging.debug("Array stores elements of type: ", raw_weights.dtype)        
        bt.logging.debug("uids", str(self.metagraph.uids.tolist()))
        bt.logging.debug("raw_weights", str(raw_weights))
        
        # Process the raw weights to final_weights via subtensor limitations.
        try:

            (
                processed_weight_uids,
                processed_weights,
            ) = process_weights_for_netuid(
                uids=self.metagraph.uids,
                weights=raw_weights,
                netuid=self.config.netuid,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )
        except Exception as e:
            bt.logging.error(f"process_weights_for_netuid function error: {e}")
            pass
            
        bt.logging.debug(f"processed_weight_uids {processed_weight_uids}")        
        bt.logging.debug(f"processed_weights {processed_weights}")

        # Convert to uint16 weights and uids.
        try:
            (
                uint_uids,
                uint_weights,
            ) = convert_weights_and_uids_for_emit(
                uids=processed_weight_uids, weights=processed_weights
            )
                        
            bt.logging.debug(f"uint_weights {uint_weights}")        
            bt.logging.debug(f"uint_uids {uint_uids}")

        except Exception as e:
            bt.logging.error(f"convert_weights_and_uids_for_emit function error: {e}")
            pass

         # Set the weights on chain via our subtensor connection.
        try:
            result, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=True,
                wait_for_inclusion=True,
                version_key=self.spec_version,
            )
            if result is True:
                #write_timestamp(time.time())
                print(f"updated timestamp to {time.time()}")
                bt.logging.info(f"set_weights on chain successfully! msg: {msg}")
            else:
                bt.logging.error(f"set_weights on chain failed {msg}")
        except Exception as e:
            bt.logging.error(f"set_weights failed with exception: {e}")


    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = (
            alpha * scattered_rewards + (1 - alpha) * self.scores
        )
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        # logger.info("Saving validator state start.")
        #
        # # Save the state of the validator to file.
        # np.savez(self.config.neuron.full_path + "/state.npz",
        #          step=self.step,
        #          scores=self.scores,
        #          hotkeys=self.hotkeys)
        # logger.info("Saving validator state end.")
        pass

    def load_state(self):
        """Loads the state of the validator from a file."""
        # logger.info("Loading validator state.")
        #
        # # Load the state of the validator from file.
        # state = np.load(self.config.neuron.full_path + "/state.npz")
        # self.step = state["step"]
        # self.scores = state["scores"]
        # self.hotkeys = state["hotkeys"]
        pass


