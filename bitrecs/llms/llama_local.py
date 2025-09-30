import os
import base64
import requests

class OllamaLocal():
    def __init__(self, 
                 ollama_url: str, 
                 model: str, 
                 system_prompt: str, 
                 temp=0.0):
        
        if not ollama_url:
            raise Exception
        self.ollama_url = ollama_url
        self.model = model
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
        self.system_prompt = system_prompt        
        if temp < 0 or temp > 1:
            raise Exception
        self.temp = temp
        self.keep_alive = 3600


    def file_to_base64(self, file_path) -> str:
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
        
    def ask_ollama(self, prompt) -> str:
        #return self.ask_ollama_long_ctx(prompt, 8000)
        data = {
            "model": self.model,
            "system": self.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temp                
            }
        }
        # print(data)
        return self.call_ollama(data)
    
        
    def ask_ollama_long_ctx(self, prompt, num_ctx: int = None) -> str:
        """Send a prompt to Ollama with optional longer context window.
        
        Args:
            prompt (str): The prompt to send to the model
            num_ctx (int, optional): Context window size. If None, uses environment variable
                
        Returns:
            str: The model's response
        """
        options = {
            "temperature": self.temp,
        }        
     
        if num_ctx is not None:
            options["num_ctx"] = max(int(num_ctx), 2048)        
        elif os.environ.get("num_ctx") is not None:
            env_ctx = os.environ.get("num_ctx")
            try:
                ctx_value = max(int(env_ctx), 2048)
                options["num_ctx"] = ctx_value
                print(f"Using context length from environment: {ctx_value}")
            except ValueError:
                print(f"Invalid context length in environment: {env_ctx}, using default 2048")
                options["num_ctx"] = 2048
        else:
            options["num_ctx"] = 2048

        data = {
            "model": self.model,       
            "system": self.system_prompt,    
            "messages": [              
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options
        }
        return self.call_ollama(data)
    

    def get_ollama_caption(self, file_path) -> str:        
        base64_image = self.file_to_base64(file_path)
        data = {
            "model": self.model,
            "system": self.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": "What is this?",
                    "stream": False,
                    "images": [base64_image]
                }
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
             "options": {
                "temperature": self.temp
            }
        }
        # print(data)
        return self.call_ollama(data)   


    def call_ollama(self, data) -> str:        
        response = requests.post(self.ollama_url, json=data)
        if response.status_code == 200:
            response_json = response.json()
            message = response_json["message"]
            content = message["content"]            
            return content
        else:
            print(response.text)
            return "Error: Unable to get caption from LLama server status {}".format(response.status_code)

