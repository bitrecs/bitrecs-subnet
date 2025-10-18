import time
import requests
import bittensor as bt
from bitrecs.llms.llm_provider import LLM
from bitrecs.protocol import MinerResponse, SignedResponse
from bitrecs.llms.verified_utils import sign_verified_request
from bitrecs.utils import constants as CONST


class Chutes:
    def __init__(self, 
                 key, 
                 model="deepseek-ai/DeepSeek-V3", 
                 system_prompt="You are a helpful assistant.", 
                 temp=0.0,
                use_verified_inference: bool = False,
                miner_wallet: "bt.Wallet" = None):
        
        self.CHUTES_API_KEY = key
        if not self.CHUTES_API_KEY:
            raise ValueError("CHUTES_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.use_verified_inference = use_verified_inference
        self.miner_wallet = miner_wallet
        self.provider = LLM.CHUTES.name
                


    def call_chutes(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError()
        url = "https://llm.chutes.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.CHUTES_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "max_tokens": 2048,
            "temperature": self.temp
        }
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        #print(result)
        thing = result["choices"][0]["message"]["content"]
        return thing
    


    def call_chutes_verified(self, prompt) -> MinerResponse:
        """Verified Chutes Implementation"""
        if not prompt or len(prompt) < 10:
            raise ValueError()
        if not self.use_verified_inference:
            raise ValueError("use_verified_inference must be True for verified inference")
        if not self.miner_wallet:
            raise ValueError("miner_wallet is not set for verified inference")
        
        headers = {
            "Authorization": f"Bearer {self.CHUTES_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bitrecs.ai",
            "X-Title": "bitrecs",
            "x-hotkey": self.miner_wallet.hotkey.ss58_address,
            "x-provider": self.provider
        }
        url = f"{CONST.VERIFIED_INFERENCE_URL}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temp,
            "max_tokens": 2048,
            "stream": False
        }
        ts = str(int(time.time()))
        signature, nonce = sign_verified_request(self.miner_wallet, self.provider, payload, ts)
        headers["x-signature"] = signature
        headers["x-nonce"] = nonce
        headers["x-timestamp"] = ts
       
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        results = data["response"]['choices'][0]['message']['content']
        proof = data["proof"]
        signature = data["signature"]
        timestamp = data["timestamp"]
        ttl = data["ttl"]
        miner_response = MinerResponse(
            results=results,
            signed_response=SignedResponse(
                response=data["response"],
                proof=proof,
                signature=signature,
                timestamp=timestamp,
                ttl=ttl 
            )                    
        )
        return miner_response
    
    

