import requests
from openai import OpenAI
from openai.types.responses import Response
from bitrecs.protocol import MinerResponse, SignedResponse

class ChatGPT:
    def __init__(self, 
                key,
                model="gpt-4o-mini", 
                system_prompt="You are a helpful assistant.", 
                temp=0.0,
                use_verified_inference: bool = False,
                miner_hotkey: str = None):
        
        self.CHATGPT_API_KEY = key
        if not self.CHATGPT_API_KEY:
            raise ValueError("CHATGPT_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.use_verified_inference = use_verified_inference
        self.miner_hotkey = miner_hotkey

    def call_chat_gpt_legacy(self, prompt) -> MinerResponse:
        """used for pre gpt5 models"""
        if not prompt or len(prompt) < 10:
            raise ValueError()
        
        if self.use_verified_inference:
            # Use requests for custom verified endpoint (returns dict)
            url = "https://verified.bitrecs.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.CHATGPT_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://bitrecs.ai",
                "X-Title": "bitrecs",
                "x-hotkey": self.miner_hotkey,
                "x-provider": "CHAT_GPT"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temp,
                "max_tokens": 2048
            }
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            chat_response = data["response"]
            proof = data["proof"]
            signature = data["signature"]
            timestamp = data["timestamp"]
            ttl = data["ttl"]
            miner_response = MinerResponse(
                results=chat_response['choices'][0]['message']['content'],
                signed_response=SignedResponse(
                    response=chat_response,
                    proof=proof,
                    signature=signature,
                    timestamp=timestamp,
                    ttl=ttl
                )                    
            )
            return miner_response
        else:
            # Standard OpenAI: use client
            headers = {
                "HTTP-Referer": "https://bitrecs.ai",
                "X-Title": "bitrecs"
            }
            client = OpenAI(api_key=self.CHATGPT_API_KEY)
            data = client.chat.completions.create(
                extra_headers=headers,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temp,
                max_tokens=2048
            )
            # data is ChatCompletion object
            miner_response = MinerResponse(
                results=data.choices[0].message.content,
                signed_response=None
            )
            return miner_response
       

    def call_chat_gpt(self, prompt) -> MinerResponse:
        if not prompt or len(prompt) < 10:
            raise ValueError()
        
        if "gpt-5" not in self.model.lower():
            return self.call_chat_gpt_legacy(prompt)
        
        if self.use_verified_inference:             
            url = "https://verified.bitrecs.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.CHATGPT_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://bitrecs.ai",
                "X-Title": "bitrecs",
                "x-hotkey": self.miner_hotkey,
                "x-provider": "CHAT_GPT"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "text" : {
                    "format": {
                        "type": "text"
                    },
                    "verbosity": "low"
                },
                "reasoning" : {
                    "effort": "low"
                }                
            }
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            chat_response = data["response"]
            proof = data["proof"]
            signature = data["signature"]
            timestamp = data["timestamp"]
            ttl = data["ttl"]
            miner_response = MinerResponse(
                results=chat_response['choices'][0]['message']['content'],
                signed_response=SignedResponse(
                    response=chat_response,
                    proof=proof,
                    signature=signature,
                    timestamp=timestamp,
                    ttl=ttl
                )                    
            )
            return miner_response
            
        else:
            # Standard OpenAI: use client for GPT-5
            client = OpenAI(api_key=self.CHATGPT_API_KEY)        
            chat_response = client.responses.create(
                extra_headers={
                    "HTTP-Referer": "https://bitrecs.ai",
                    "X-Title": "bitrecs"
                },
                model=self.model,
                reasoning={"effort": "minimal"},
                text={"verbosity": "low"},
                instructions=self.system_prompt,
                input=prompt,
                max_output_tokens=2048
            )
            output_text = chat_response.output_text
            miner_response = MinerResponse(
                results=output_text,
                signed_response=None
            )
            return miner_response