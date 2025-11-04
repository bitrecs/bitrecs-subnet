import os
import bittensor as bt
from bitrecs.llms.cerebras import Cerebras
from bitrecs.llms.claude import Claude
from bitrecs.llms.gemini import Gemini
from bitrecs.llms.groq import Groq
from bitrecs.llms.llama_local import OllamaLocal
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.open_router import OpenRouter
from bitrecs.llms.chat_gpt import ChatGPT
from bitrecs.llms.perplexity import Perplexity
from bitrecs.llms.vllm_router import vLLM
from bitrecs.llms.chutes import Chutes
from bitrecs.llms.grok import Grok
from bitrecs.llms.nvidia_inference import NvidiaInference
from bitrecs.protocol import MinerResponse


class LLMFactory:

    @staticmethod
    def query_llm(server: LLM, model: str, 
                  system_prompt="You are a helpful assistant", 
                  temp=0.0, user_prompt="") -> str:
        match server:
            case LLM.OLLAMA_LOCAL:
                return OllamaLocalInterface(model, system_prompt, temp).query(user_prompt)
            case LLM.OPEN_ROUTER:
                return OpenRouterInterface(model, system_prompt, temp).query(user_prompt)
            case LLM.CHAT_GPT:
                return ChatGPTInterface(model, system_prompt, temp).query(user_prompt)
            case LLM.VLLM:
                return VllmInterface(model, system_prompt, temp).query(user_prompt)
            case LLM.GEMINI:
                return GeminiInterface(model, system_prompt, temp).query(user_prompt)         
            case LLM.CHUTES:
                return ChutesInterface(model, system_prompt, temp).query(user_prompt)
            case LLM.GROK:
                return GrokInterface(model, system_prompt, temp).query(user_prompt)                
            case LLM.CLAUDE:
                return ClaudeInterface(model, system_prompt, temp).query(user_prompt)                
            case LLM.CEREBRAS:
                return CerebrasInterface(model, system_prompt, temp).query(user_prompt)
            case LLM.GROQ:
                return GroqInterface(model, system_prompt, temp).query(user_prompt)
            case LLM.NVIDIA:
                return NvidiaInterface(model, system_prompt, temp).query(user_prompt)
            case LLM.PERPLEXITY:
                return PerplexityInterface(model, system_prompt, temp).query(user_prompt)
            case _:
                raise ValueError("Unknown LLM server")
            
    @staticmethod
    def query_llmv(server: LLM, model: str, 
                  system_prompt="You are a helpful assistant", 
                  temp=0.0, user_prompt="", 
                  miner_wallet: "bt.Wallet" = None, 
                  use_verified_inference=False) -> MinerResponse:
        """Verified inference"""
        match server:
            case LLM.OLLAMA_LOCAL:
                raise NotImplementedError("Ollama Local does not support verified inference")
            case LLM.VLLM:
                raise NotImplementedError("VLLM does not support verified inference")
            case LLM.OPEN_ROUTER:
                return OpenRouterInterface(model, system_prompt, temp, miner_wallet, use_verified_inference).query_verified(user_prompt)
            case LLM.CHAT_GPT:
                return ChatGPTInterface(model, system_prompt, temp, miner_wallet, use_verified_inference).query_verified(user_prompt)          
            case LLM.GEMINI:
                return GeminiInterface(model, system_prompt, temp, miner_wallet, use_verified_inference).query_verified(user_prompt)         
            case LLM.CHUTES:                
                return ChutesInterface(model, system_prompt, temp, miner_wallet, use_verified_inference).query_verified(user_prompt)
            case LLM.GROK:
                return GrokInterface(model, system_prompt, temp, miner_wallet, use_verified_inference).query_verified(user_prompt)
            case LLM.CLAUDE:
                return ClaudeInterface(model, system_prompt, temp, miner_wallet, use_verified_inference).query_verified(user_prompt)                
            case LLM.CEREBRAS:
                raise NotImplementedError("Cerebras is not implemented yet")
            case LLM.GROQ:
                raise NotImplementedError("Groq is not implemented yet")
            case LLM.NVIDIA:
                return NvidiaInterface(model, system_prompt, temp, miner_wallet, use_verified_inference).query_verified(user_prompt)
            case LLM.PERPLEXITY:
                return PerplexityInterface(model, system_prompt, temp, miner_wallet, use_verified_inference).query_verified(user_prompt)
            case _:
                raise ValueError("Unknown LLM server")
            
   
            
    @staticmethod
    def try_parse_llm(value: str) -> LLM:
        match value.strip().upper():
            case "OLLAMA_LOCAL":
                return LLM.OLLAMA_LOCAL
            case "OPEN_ROUTER":
                return LLM.OPEN_ROUTER
            case "CHAT_GPT":
                return LLM.CHAT_GPT
            case "VLLM":
                return LLM.VLLM
            case "GEMINI":
                return LLM.GEMINI
            case "GROK":
                return LLM.GROK
            case "CLAUDE":
                return LLM.CLAUDE
            case "CHUTES":
                return LLM.CHUTES
            case "CEREBRAS":
                return LLM.CEREBRAS
            case "GROQ":
                return LLM.GROQ
            case "NVIDIA":
                return LLM.NVIDIA
            case "PERPLEXITY":
                return LLM.PERPLEXITY
            case _:
                raise ValueError("Unknown LLM server")
        
        
class OllamaLocalInterface:
    def __init__(self, model, system_prompt, temp):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp        
        self.OLLAMA_LOCAL_URL = os.environ.get("OLLAMA_LOCAL_URL").removesuffix("/")
        if not self.OLLAMA_LOCAL_URL:
             bt.logging.error("OLLAMA_LOCAL_URL not set.")        
    
    def query(self, user_prompt) -> str:
        llm = OllamaLocal(ollama_url=self.OLLAMA_LOCAL_URL, model=self.model, 
                          system_prompt=self.system_prompt, temp=self.temp)
        return llm.ask_ollama(user_prompt)
    
    
class OpenRouterInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference
        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set")
    
    def query(self, user_prompt) -> str:
        router = OpenRouter(self.OPENROUTER_API_KEY, model=self.model, 
                            system_prompt=self.system_prompt, temp=self.temp)
        return router.call_open_router(user_prompt)

    def query_verified(self, user_prompt) -> MinerResponse:
        router = OpenRouter(self.OPENROUTER_API_KEY, model=self.model,
                            system_prompt=self.system_prompt, temp=self.temp, miner_wallet=self.miner_wallet,
                            use_verified_inference=self.use_verified_inference)
        return router.call_open_router_verified(user_prompt)  
    
    
class ChatGPTInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.CHATGPT_API_KEY = os.environ.get("CHATGPT_API_KEY")
        if not self.CHATGPT_API_KEY:            
            raise ValueError("CHATGPT_API_KEY is not set")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference

    def query(self, user_prompt) -> str:
        router = ChatGPT(self.CHATGPT_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp)
        return router.call_chat_gpt(user_prompt)
        
    def query_verified(self, user_prompt) -> MinerResponse:
        router = ChatGPT(self.CHATGPT_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp, miner_wallet=self.miner_wallet, 
                         use_verified_inference=self.use_verified_inference)
        return router.call_chat_gpt_verified(user_prompt)
    
    
class VllmInterface:
    def __init__(self, model, system_prompt, temp):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.VLLM_API_KEY = os.environ.get("VLLM_API_KEY")
        if not self.VLLM_API_KEY:            
            raise ValueError("VLLM_API_KEY is not set")
        self.VLLM_LOCAL_URL = os.environ.get("VLLM_LOCAL_URL").removesuffix("/")
        if not self.VLLM_LOCAL_URL:
            self.VLLM_LOCAL_URL = "http://localhost:8000/v1"          
    
    def query(self, user_prompt) -> str:
        router = vLLM(key=self.VLLM_API_KEY, model=self.model, 
                      system_prompt=self.system_prompt, 
                      temp=self.temp, base_url=self.VLLM_LOCAL_URL)
        return router.call_vllm(user_prompt)
    
    
class GeminiInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:            
            raise ValueError("GEMINI_API_KEY is not set")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference
        
    def query(self, user_prompt) -> str:
        router = Gemini(self.GEMINI_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp)
        return router.call_gemini(user_prompt)
    
    def query_verified(self, user_prompt) -> MinerResponse:
        router = Gemini(self.GEMINI_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp, miner_wallet=self.miner_wallet, 
                         use_verified_inference=self.use_verified_inference)
        return router.call_gemini_verified(user_prompt)
    
    
class GrokInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.GROK_API_KEY = os.environ.get("GROK_API_KEY")
        if not self.GROK_API_KEY:            
            raise ValueError("GROK_API_KEY is not set")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference
        
    def query(self, user_prompt) -> str:
        router = Grok(self.GROK_API_KEY, model=self.model, 
                        system_prompt=self.system_prompt, temp=self.temp)
        return router.call_grok(user_prompt)
    
    def query_verified(self, user_prompt) -> MinerResponse:
        router = Grok(self.GROK_API_KEY, model=self.model,
                    system_prompt=self.system_prompt, temp=self.temp, 
                    miner_wallet=self.miner_wallet, 
                    use_verified_inference=self.use_verified_inference)
        return router.call_grok_verified(user_prompt)
    

class ClaudeInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
        if not self.CLAUDE_API_KEY:            
            raise ValueError("CLAUDE_API_KEY is not set")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference
        
    def query(self, user_prompt) -> str:
        router = Claude(self.CLAUDE_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp, 
                         miner_wallet=self.miner_wallet, 
                         use_verified_inference=self.use_verified_inference)
        return router.call_claude(user_prompt)
    
    def query_verified(self, user_prompt) -> MinerResponse:
        router = Claude(self.CLAUDE_API_KEY, model=self.model,
                    system_prompt=self.system_prompt, temp=self.temp, 
                    miner_wallet=self.miner_wallet, 
                    use_verified_inference=self.use_verified_inference)
        return router.call_claude_verified(user_prompt)


class ChutesInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY")
        if not self.CHUTES_API_KEY:            
            raise ValueError("CHUTES_API_KEY is not set")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference
        
    def query(self, user_prompt) -> str:
        router = Chutes(self.CHUTES_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp)        
        return router.call_chutes(user_prompt)

    def query_verified(self, user_prompt) -> MinerResponse:
        router = Chutes(self.CHUTES_API_KEY, model=self.model,
                    system_prompt=self.system_prompt, temp=self.temp, 
                    miner_wallet=self.miner_wallet, 
                    use_verified_inference=self.use_verified_inference)
        return router.call_chutes_verified(user_prompt)
    

class CerebrasInterface:
    def __init__(self, model, system_prompt, temp, hotkey = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
        if not self.CEREBRAS_API_KEY:            
            raise ValueError("CEREBRAS_API_KEY is not set")
        self.miner_hotkey = hotkey
        self.use_verified_inference = use_verified_inference
        
    def query(self, user_prompt) -> str:
        router = Cerebras(self.CEREBRAS_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp, 
                         miner_hotkey=self.miner_hotkey, 
                         use_verified_inference=self.use_verified_inference)
        return router.call_cerebras(user_prompt)
    

class GroqInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not self.GROQ_API_KEY:            
            raise ValueError("GROQ_API_KEY is not set")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference
        
    def query(self, user_prompt) -> str:
        router = Groq(self.GROQ_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp, 
                         miner_wallet=self.miner_wallet, 
                         use_verified_inference=self.use_verified_inference)
        return router.call_groq(user_prompt)
    

class NvidiaInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
        if not self.NVIDIA_API_KEY:            
            raise ValueError("NVIDIA_API_KEY is not set")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference
        
    def query(self, user_prompt) -> str:
        router = NvidiaInference(self.NVIDIA_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp, 
                         miner_wallet=self.miner_wallet, 
                         use_verified_inference=self.use_verified_inference)
        return router.call_nvidia(user_prompt)
    
    def query_verified(self, user_prompt) -> MinerResponse:
        router = NvidiaInference(self.NVIDIA_API_KEY, model=self.model,
                    system_prompt=self.system_prompt, temp=self.temp, 
                    miner_wallet=self.miner_wallet, 
                    use_verified_inference=self.use_verified_inference)
        return router.call_nvidia_verified(user_prompt)


class PerplexityInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
        if not self.PERPLEXITY_API_KEY:            
            raise ValueError("PERPLEXITY_API_KEY is not set")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference
        
    def query(self, user_prompt) -> str:
        router = Perplexity(self.PERPLEXITY_API_KEY, model=self.model, 
                         system_prompt=self.system_prompt, temp=self.temp, 
                         miner_wallet=self.miner_wallet, 
                         use_verified_inference=self.use_verified_inference)
        return router.call_perplexity(user_prompt)
    
    def query_verified(self, user_prompt) -> MinerResponse:
        router = Perplexity(self.PERPLEXITY_API_KEY, model=self.model,
                    system_prompt=self.system_prompt, temp=self.temp, 
                    miner_wallet=self.miner_wallet, 
                    use_verified_inference=self.use_verified_inference)
        return router.call_perplexity_verified(user_prompt)
