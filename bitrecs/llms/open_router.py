import json
import requests
from openai import OpenAI

class OpenRouter:    
    def __init__(self, 
                 key,
                 model="google/gemini-flash-1.5-8b", 
                 system_prompt="You are a helpful assistant.", 
                 temp=0.0
        ):

        self.OPENROUTER_API_KEY = key
        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp


    def call_open_router_legacy(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError()

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.OPENROUTER_API_KEY,
        )

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
            max_tokens=2048,
            reasoning_effort="low"            
        )

        thing = completion.choices[0].message.content                
        return thing    


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
        
        
    
