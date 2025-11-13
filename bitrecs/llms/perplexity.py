import time
import json
import requests
import bittensor as bt
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.verified_utils import sign_verified_request
from bitrecs.protocol import MinerResponse, SignedResponse
from bitrecs.utils import constants as CONST


class Perplexity:    
    def __init__(self, 
                 key,
                 model="sonar", 
                 system_prompt="You are a helpful assistant.", 
                 temp=0.0,
                 use_verified_inference: bool = False,
                 miner_wallet: "bt.Wallet" = None
        ):

        self.PERPLEXITY_API_KEY = key
        if not self.PERPLEXITY_API_KEY:
            raise ValueError("PERPLEXITY_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.use_verified_inference = use_verified_inference
        self.miner_wallet = miner_wallet
        self.provider = LLM.PERPLEXITY.name

    def call_perplexity(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError()

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bitrecs.ai",
            "X-Title": "bitrecs"
        }        

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }],
            "stream": False,
            "temperature": self.temp,
            "extra_body": {"chat_template_kwargs": {"thinking": False }}
        }
        
        timeout = (5, 30) #connect, read timeout
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            #print(data)
            return data['choices'][0]['message']['content']
        except requests.exceptions.ConnectTimeout:
            raise TimeoutError(f"OpenRouter connect timed out after {timeout[0]}s")
        except requests.exceptions.ReadTimeout:
            raise TimeoutError(f"OpenRouter read timed out after {timeout[1]}s")
        except requests.exceptions.RequestException as e:
            # bubble up other network / HTTP errors
            raise RuntimeError(f"OpenRouter request failed: {e}") from e
        

    def call_perplexity_verified(self, prompt) -> MinerResponse:       
        if not prompt or len(prompt) < 10:
                raise ValueError()
        if not self.use_verified_inference:
            raise ValueError("use_verified_inference is False")
        if not self.miner_wallet:
            raise ValueError("miner_wallet is not set for verified inference")
        
        headers = {
            "Authorization": f"Bearer {self.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bitrecs.ai",
            "x-title": "bitrecs",
            "x-hotkey": self.miner_wallet.hotkey.ss58_address,
            "x-provider": self.provider
        }
        url = f"{CONST.VERIFIED_INFERENCE_URL}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }],
            "stream": False,
            "temperature": self.temp,
            "extra_body": {"chat_template_kwargs": {"thinking": False }}
        }
        ts = str(int(time.time()))
        signature, nonce = sign_verified_request(self.miner_wallet, self.provider, payload, ts)
        headers["x-signature"] = signature
        headers["x-nonce"] = nonce
        headers["x-timestamp"] = ts
        
        timeout = (5, 30) #connect, read timeout
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()            
            response = data["response"]
            proof = data["proof"]
            signature = data["signature"]
            timestamp = data["timestamp"]
            ttl = data["ttl"]
            miner_response = MinerResponse(
                results=response['choices'][0]['message']['content'],
                signed_response=SignedResponse(
                    response=response,
                    proof=proof,
                    signature=signature,
                    timestamp=timestamp,
                    ttl=ttl
                )                    
            )
            return miner_response
            
        except requests.exceptions.ConnectTimeout:
            raise TimeoutError(f"OpenRouter connect timed out after {timeout[0]}s")
        except requests.exceptions.ReadTimeout:
            raise TimeoutError(f"OpenRouter read timed out after {timeout[1]}s")
        except requests.exceptions.RequestException as e:
            # bubble up other network / HTTP errors
            raise RuntimeError(f"OpenRouter request failed: {e}") from e       


    