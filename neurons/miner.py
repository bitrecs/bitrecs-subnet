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


import sys
import time
import typing
import asyncio
import json
import json_repair
import bittensor as bt
import bitrecs.utils.constants as CONST
from typing import List
from datetime import datetime, timedelta, timezone
from bitrecs.base.miner import BaseMinerNeuron
from bitrecs.commerce.user_profile import UserProfile
from bitrecs.protocol import BitrecsRequest
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.llms.factory import LLM, LLMFactory
from bitrecs.utils.runtime import execute_periodically
from bitrecs.utils.uids import best_uid
from bitrecs.utils.version import LocalMetadata
from dotenv import load_dotenv
load_dotenv()


async def do_work(user_prompt: str,
                  context: str, 
                  num_recs: int,
                  server: LLM,
                  model: str,
                  system_prompt="You are a helpful assistant.", 
                  profile : UserProfile = None,
                  debug_prompts=False) -> List[str]:
    """
    Miner work is done here.
    This function is invoked by the API validator to generate recommendations.
    You can use any method you prefer to generate the data.
    The default setup will use OPEN_ROUTER.

    Args:
        user_prompt (str): The user query (generally the SKU they are browsing)
        context (str): The context of the user query - this is set of products to chose from (store catalog)
        num_recs (int): The number of recommendations to generate.
        server (LLM): The LLM server type to query.
        model (str): The LLM model to use.
        system_prompt (str): The system prompt for the LLM.
        profile (UserProfile): The user profile to use when generating recommendations.
        debug_prompts (bool): Whether to log debug information about the prompts.

    Returns:
        typing.List[str]: A list of product recommendations generated by the miner.

    """
    bt.logging.info(f"do_work Prompt: {user_prompt}")
    bt.logging.info(f"do_work LLM server: {server}")
    bt.logging.info(f"do_work LLM model: {model}")  
    bt.logging.trace(f"do_work profile: {profile}")

    factory = PromptFactory(sku=user_prompt,
                            context=context, 
                            num_recs=num_recs,                                                         
                            debug=debug_prompts,
                            profile=profile)
    prompt = factory.generate_prompt()
    try:
        llm_response = LLMFactory.query_llm(server=server, 
                                            model=model, 
                                            system_prompt=system_prompt, 
                                            temp=0.0, user_prompt=prompt)
        if not llm_response or len(llm_response) < 10:
            bt.logging.error("LLM response is empty.")
            return []
        
        parsed_recs = PromptFactory.tryparse_llm(llm_response)
        if debug_prompts:
            bt.logging.trace(f" {llm_response} ")
            bt.logging.trace(f"LLM response: {parsed_recs}")

        return parsed_recs
    except Exception as e:
        bt.logging.error(f"Error calling LLM: {e}")

    return []


class Miner(BaseMinerNeuron):
    """
    Main miner class which generates product recommendations based on incoming requests.
    You are encouraged to modify the do_work function to generate high quality recommendations using whatever method you prefer.

    Default: By default this miner uses OPEN_ROUTER and google/gemini-2.0-flash-lite-001 to generate recommendations.
    
    You can override this by setting the --llm.provider argument in the config.
    For example, --llm.provider OLLAMA_LOCAL will use the local ollama instance to generate recommendations.
    Additionally, --llm.model "model_name" can be used to override the default model.

    Note: check your .env file for the appropriate API key settings and urls for the LLM provider configured.

    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        bt.logging.info(f"\033[1;32m 🐸 Bitrecs Miner started uid: {self.uid}\033[0m")

        try:
            self.llm = self.config.llm.provider
            provider = LLMFactory.try_parse_llm(self.llm)
            bt.logging.info(f"\033[1;35m Miner LLM Provider: [{self.llm}]\033[0m")
            self.llm_provider = provider
            self.model = ""
        except ValueError as ve:
            bt.logging.error(f"Invalid LLM provider: {ve}")
            sys.exit()      

        if self.llm_provider == LLM.VLLM:
            bt.logging.info(f"\033[1;35m Please ensure vLLM Server is running\033[0m")
        elif self.llm_provider == LLM.OLLAMA_LOCAL:
            bt.logging.info(f"\033[1;35m Please ensure Ollama Server is running\033[0m")
        else:
            bt.logging.info(f"\033[1;35m Please ensure your API keys are set in the environment\033[0m")             

        bt.logging.info(f"\033[1;35m Miner is warming up\033[0m")
        warmup_result = self.warmup()
        if not warmup_result:
            bt.logging.error(f"\033[31mMiner warmup failed. Exiting.\033[0m")
            sys.exit()
        if not self.model:
            bt.logging.error(f"\033[31mMiner model not set. Exiting.\033[0m")
            sys.exit()

        best_performing_uid = best_uid(self.metagraph)        
        if self.uid == best_performing_uid:
            bt.logging.info(f"\033[1;32m 🐸 You are the BEST performing miner in the subnet, keep it up!\033[0m")

        self.total_request_in_interval = 0
        
        if(self.config.logging.trace):
            bt.logging.trace(f"TRACE ENABLED Miner {self.uid} - {self.llm_provider} - {self.model}")
    

    async def forward(
        self, synapse: BitrecsRequest
    ) -> BitrecsRequest:
        """
        Takes an API request and generates recs

        Args:
            synapse (bitrecs.protocol.BitrecsRequest): The synapse object containing the 'BitrecsRequest' data.

        Returns:
            bitrecs.protocol.BitrecsRequest: The synapse object with the recs - same object modified with updated fields.

        """
        bt.logging.info(f"MINER {self.uid} FORWARD PASS {synapse.query}")

        results = []
        query = synapse.query
        context = synapse.context
        num_recs = synapse.num_results
        model = self.model
        server = self.llm_provider        
        st = time.time()
        debug_prompts = self.config.logging.trace
        user_profile = UserProfile.tryparse_profile(synapse.user)

        try:
            results = await do_work(user_prompt=query,
                                    context=context, 
                                    num_recs=num_recs, 
                                    server=server, 
                                    model=model, 
                                    profile=user_profile,
                                    debug_prompts=debug_prompts)            
            bt.logging.info(f"LLM {self.model} - Results: count ({len(results)})")
        except Exception as e:
            bt.logging.error(f"\033[31mFATAL ERROR calling do_work: {e!r} \033[0m")
        finally:
            et = time.time()
            bt.logging.info(f"{self.model} Query - Elapsed Time: \033[1;32m {et-st} \033[0m")

        utc_now = datetime.now(timezone.utc)
        created_at = utc_now.strftime("%Y-%m-%dT%H:%M:%S")

        #Do some cleanup - schema is validated in the reward function
        final_results = []
        for item in results:
            try:
                item_str = str(item)
                try:
                    dictionary_item = json.loads(item_str)
                except json.JSONDecodeError:
                    repaired = json_repair.repair_json(item_str)
                    dictionary_item = json.loads(repaired)
                
                if "name" not in dictionary_item:
                    bt.logging.error(f"Item missing 'name' key: {dictionary_item}")
                    continue
                dictionary_item["name"] = CONST.RE_PRODUCT_NAME.sub("", str(dictionary_item["name"]))

                if "reason" in dictionary_item:
                    dictionary_item["reason"] = CONST.RE_REASON.sub("", str(dictionary_item["reason"]))
                
                recommendation = json.dumps(dictionary_item, separators=(',', ':'))
                final_results.append(recommendation)
            except Exception as e:
                bt.logging.error(f"Failed to parse LLM result: {item}, error: {e}")
                continue
        
        output_synapse=BitrecsRequest(
            name=synapse.name, 
            axon=synapse.axon,
            dendrite=synapse.dendrite,
            created_at=created_at,
            user="",
            num_results=num_recs,
            query=synapse.query,
            context="[]",
            site_key=synapse.site_key,
            results=final_results,
            models_used=[self.model],
            miner_uid=str(self.uid),
            miner_hotkey=self.wallet.hotkey.ss58_address
        )
        
        bt.logging.info(f"MINER {self.uid} FORWARD PASS RESULT -> {output_synapse}")
        self.total_request_in_interval += 1
        return output_synapse
        

    async def blacklist(
        self, synapse: BitrecsRequest
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (bitrecs.protocol.BitrecsRequest): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )

        bt.logging.debug(
            f"GOOD hotkey {synapse.dendrite.hotkey}"
        )

        return False, "Hotkey recognized!"

    async def priority(self, synapse: BitrecsRequest) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (bitrecs.protocol.BitrecsRequest): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.debug(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority
    
    
    def save_state(self):
        pass


    def warmup(self):
        """
        On startup, try querying the LLM to ensure it is working and loaded into memory.    
        You can override the base model with --llm.model "model_name"

        """
        match self.llm_provider:
            case LLM.OLLAMA_LOCAL:
                model = "mistral-nemo"                
            case LLM.OPEN_ROUTER:
                model = "google/gemini-2.0-flash-lite-001"
            case LLM.CHAT_GPT:
                model = "gpt-4o-mini"
            case LLM.VLLM:
                model = "NousResearch/Meta-Llama-3-8B-Instruct"                
            case LLM.GEMINI:                                
                model = "gemini-2.0-flash-001"
            case LLM.GROK:
                model = "grok-beta"
            case LLM.CLAUDE:
                model = "anthropic/claude-3.5-haiku"
            case _:
                bt.logging.error("Unknown LLM server")
                raise ValueError("Unknown LLM server")
                
        #If user specified model override it here
        if self.config.llm.model and len(self.config.llm.model) > 2:
            model = self.config.llm.model
             
        bt.logging.info(f"Miner Warmup: {self.llm} - Model: {model}")
        try:
            result = LLMFactory.query_llm(server=self.llm_provider, 
                                 model=model, 
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.1, user_prompt="Tell me a sarcastic joke")
            self.model = model
            bt.logging.info(f"Warmup SUCCESS: {self.model} - Result: {result}")
            return True
        except Exception as e:
            bt.logging.error(f"\033[31mFATAL ERROR calling warmup: {e!r} \033[0m")
        return False
    
    
    @execute_periodically(timedelta(seconds=CONST.VERSION_CHECK_INTERVAL))
    async def version_sync(self):
        bt.logging.trace(f"Version sync ran at {int(time.time())}")
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


        
async def main():
    with Miner() as miner:
        start_time = time.time()        
        while True:            
            version_sync_task = asyncio.create_task(miner.version_sync())
            await version_sync_task

            bt.logging.info(f"Miner {miner.uid} running, waiting for work ... {int(time.time())}")
            if time.time() - start_time > 300:
                bt.logging.info(
                    f"---Total request in last 5 minutes: {miner.total_request_in_interval}"
                )
                start_time = time.time()
                miner.total_request_in_interval = 0

            await asyncio.sleep(10)

if __name__ == "__main__":  
    asyncio.run(main())
