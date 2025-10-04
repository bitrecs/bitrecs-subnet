import json
import requests
from openai import OpenAI
from bitrecs.protocol import MinerResponse, SignedResponse
from bitrecs.utils import constants as CONST

class OpenRouter:    
    def __init__(self, 
                 key,
                 model="google/gemini-flash-1.5-8b", 
                 system_prompt="You are a helpful assistant.", 
                 temp=0.0,
                 use_verified_inference: bool = False,
                 miner_hotkey: str = None,
        ):

        self.OPENROUTER_API_KEY = key
        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.use_verified_inference = use_verified_inference
        self.miner_hotkey = miner_hotkey

    def call_open_router(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError()

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bitrecs.ai",
            "X-Title": "bitrecs"
        }
        reasoning = {
            "enabled": False,
            "exclude": True,
            "effort": "minimal"
        }
        # Handle specific models that require different reasoning settings
        if "gpt-5" in self.model.lower():
            reasoning = {
                "exclude": True,
                "effort": "minimal"
            }

        payload = {
            "model": self.model,
            "messages": [
                #{"role": "system", "content": "/no_think"},
                {
                    "role": "user", 
                    "content": prompt
                }],
            "reasoning": reasoning,
            "stream": False,
            "temperature": self.temp
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
        

    def call_open_router_verified(self, prompt) -> MinerResponse:       
        if not prompt or len(prompt) < 10:
                raise ValueError()
        
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bitrecs.ai",
            "X-Title": "bitrecs"
        }
        url = f"{CONST.VERIFIED_INFERENCE_URL}/v1/chat/completions"
        headers["x-hotkey"] = self.miner_hotkey
        headers["x-provider"] = "OPEN_ROUTER"
      
        reasoning = {
            "enabled": False,
            "exclude": True,
            "effort": "minimal"
        }
        # Handle specific models that require different reasoning settings
        if "gpt-5" in self.model.lower():
            reasoning = {
                "exclude": True,
                "effort": "minimal"
            }

        payload = {
            "model": self.model,
            "messages": [
                #{"role": "system", "content": "/no_think"},
                {
                    "role": "user", 
                    "content": prompt
                }],
            "reasoning": reasoning,
            "stream": False,
            "temperature": self.temp,
            # "thinking": {
            #     "type": "disabled"
            # },
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


    