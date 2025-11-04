from enum import Enum


class LLM(Enum):
    OLLAMA_LOCAL = 1
    OPEN_ROUTER = 2
    CHAT_GPT = 3
    VLLM = 4
    GEMINI = 5
    GROK = 6
    CLAUDE = 7
    CHUTES = 8
    CEREBRAS = 9
    GROQ = 10
    NVIDIA = 11
    PERPLEXITY = 12

