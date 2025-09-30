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
        """used for pre gpt5 models"""
        if not prompt or len(prompt) < 10:
            raise ValueError()
        if not self.use_verified_inference:
            raise ValueError("use_verified_inference must be True for verified inference")
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