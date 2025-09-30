from openai import OpenAI

class Cerebras:
    def __init__(self, 
                key, 
                model="llama-4-scout-17b-16e-instruct", 
                system_prompt="You are a helpful assistant.",
                temp=0.0,
                miner_hotkey: str = None,
                use_verified_inference: bool = False):
        
        self.CEREBRAS_API_KEY = key
        if not self.CEREBRAS_API_KEY:
            raise ValueError("CEREBRAS_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp        
        self.miner_hotkey = miner_hotkey
        self.use_verified_inference = use_verified_inference
        

    def call_cerebras(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError()

        client = OpenAI(api_key=self.CEREBRAS_API_KEY,
                        base_url="https://api.cerebras.ai/v1")

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://bitrecs.ai",
                "X-Title": "bitrecs"
            }, 
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temp,
            max_tokens=2048
        )
        thing = completion.choices[0].message.content                
        return thing