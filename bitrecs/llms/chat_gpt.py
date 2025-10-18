import time
import requests
import bittensor as bt
from openai import OpenAI
from openai.types.responses import Response
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.verified_utils import sign_verified_request
from bitrecs.protocol import MinerResponse, SignedResponse
from bitrecs.utils import constants as CONST

class ChatGPT:
    def __init__(self, 
                key,
                model="gpt-4o-mini", 
                system_prompt="You are a helpful assistant.", 
                temp=0.0,
                use_verified_inference: bool = False,
                miner_wallet: "bt.Wallet" = None):
        
        self.CHATGPT_API_KEY = key
        if not self.CHATGPT_API_KEY:
            raise ValueError("CHATGPT_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.use_verified_inference = use_verified_inference
        self.miner_wallet = miner_wallet
        self.provider = LLM.CHAT_GPT.name


    def call_chat_gpt(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError()        
        if "gpt-5" not in self.model.lower():
            return self.call_chat_gpt_legacy(prompt)

        client = OpenAI(api_key=self.CHATGPT_API_KEY)        
        chat_response : Response = client.responses.create(
            extra_headers={
                "HTTP-Referer": "https://bitrecs.ai",
                "X-Title": "bitrecs"
            },
            model=self.model,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
            instructions=self.system_prompt,
            input=prompt,
            #temperature=self.temp, #temp not supported in gpt5
            max_output_tokens=2048
        )
        thing = chat_response.output_text
        return thing
    

    def call_chat_gpt_legacy(self, prompt) -> str:
        """used for pre gpt5 models"""
        if not prompt or len(prompt) < 10:
            raise ValueError()
        client = OpenAI(api_key=self.CHATGPT_API_KEY)
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://bitrecs.ai",
                "X-Title": "bitrecs"
            }, 
            model=self.model,
            messages=[
            {
                "role": "user",
                "content": prompt,
            }],
            temperature=self.temp,
            max_tokens=2048
        )
        thing = completion.choices[0].message.content                
        return thing
    
    
    def call_chat_gpt_verified(self, prompt) -> MinerResponse:
        """Verified GPT5 Implementation"""
        if not prompt or len(prompt) < 10:
            raise ValueError()
        if not self.use_verified_inference:
            raise ValueError("use_verified_inference must be True for verified inference")
        if not self.miner_wallet:
            raise ValueError("miner_wallet is not set for verified inference")
         
        if "gpt-5" not in self.model.lower():
            return self.call_chat_gpt_verified_legacy(prompt)
        
        headers = {
            "Authorization": f"Bearer {self.CHATGPT_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bitrecs.ai",
            "X-Title": "bitrecs",
            "x-hotkey": self.miner_wallet.hotkey.ss58_address,
            "x-provider": self.provider
        }      
        url = f"{CONST.VERIFIED_INFERENCE_URL}/v1/chat/completions"
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "text": {
                "format": {
                    "type": "text"
                },
                "verbosity": "low"
            },
            "reasoning": {
                "effort": "low"
            }
        }
        ts = str(int(time.time()))
        signature, nonce = sign_verified_request(self.miner_wallet, self.provider, payload, ts)
        headers["x-signature"] = signature
        headers["x-nonce"] = nonce
        headers["x-timestamp"] = ts
       
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        response_output = data["response"]["output"]
        for output in response_output:
            if output.get("type") == "message":
                message_output = output
                break
        else:
            raise ValueError("No message output found in response")        
        results = message_output["content"][0]["text"]        
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
    
    
    def call_chat_gpt_verified_legacy(self, prompt) -> MinerResponse:
        """Verified GPT4 Implementation - legacy, use call_chat_gpt_verified for gpt5"""
        if not prompt or len(prompt) < 10:
            raise ValueError()
        if not self.use_verified_inference:
            raise ValueError("use_verified_inference must be True for verified inference")
        
        url = f"{CONST.VERIFIED_INFERENCE_URL}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temp,
            "max_tokens": 2048
        }
        headers = {
            "Authorization": f"Bearer {self.CHATGPT_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bitrecs.ai",
            "X-Title": "bitrecs",
            "x-hotkey": self.miner_wallet.hotkey.ss58_address,
            "x-provider": self.provider
        }
        ts = str(int(time.time()))
        signature, nonce = sign_verified_request(self.miner_wallet, self.provider, payload, ts)
        headers["x-signature"] = signature
        headers["x-nonce"] = nonce
        headers["x-timestamp"] = ts
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        results = data["response"]["choices"][0]["message"]["content"]
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

    