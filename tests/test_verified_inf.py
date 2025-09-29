import os
os.environ["NEST_ASYNCIO"] = "0"
import json
import time
import pytest    
import sqlite3
import concurrent.futures
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict
from random import SystemRandom
safe_random = SystemRandom()
from typing import Counter
from bitrecs.commerce.product import CatalogProvider, ProductFactory
from bitrecs.llms.factory import LLM, LLMFactory
from bitrecs.llms.prompt_factory import PromptFactory
from dotenv import load_dotenv
load_dotenv()



LOCAL_OLLAMA_URL = "http://10.0.0.40:11434/api/chat"
OLLAMA_MODEL = "mistral-nemo"

map = [
    {"provider": LLM.OLLAMA_LOCAL, "model": "mistral-nemo"},
    {"provider": LLM.VLLM, "model": "NousResearch/Meta-Llama-3-8B-Instruct"},
    {"provider": LLM.CHAT_GPT, "model": "gpt-5-nano-2025-08-07"},

    #{"provider": LLM.OPEN_ROUTER, "model": "nvidia/llama-3.1-nemotron-70b-instruct"},
    #{"provider": LLM.OPEN_ROUTER, "model": "nousresearch/deephermes-3-llama-3-8b-preview:free"},

    {"provider": LLM.OPEN_ROUTER, "model": "amazon/nova-lite-v1"},
    {"provider": LLM.OPEN_ROUTER, "model": "google/gemini-2.5-flash-lite"},
    {"provider": LLM.OPEN_ROUTER, "model": "meta-llama/llama-4-scout"},
    {"provider": LLM.OPEN_ROUTER, "model": "openai/gpt-4.1-nano"},
    
    {"provider": LLM.GROK, "model": "grok-2-latest"},
    {"provider": LLM.GEMINI, "model": "gemini-2.0-flash-001"},
    {"provider": LLM.CLAUDE, "model": "anthropic/claude-3.5-haiku"}
]

# CLOUD_BATTERY = ["amazon/nova-lite-v1", "google/gemini-flash-1.5-8b", "google/gemini-2.0-flash-001",
#                  "x-ai/grok-2-1212", "qwen/qwen-turbo", "openai/gpt-4o-mini"]

#CLOUD_PROVIDERS = [LLM.OPEN_ROUTER, LLM.GEMINI, LLM.CHAT_GPT, LLM.GROK, LLM.CLAUDE]
CLOUD_PROVIDERS = [LLM.OPEN_ROUTER, LLM.GEMINI, LLM.CHAT_GPT]


#LOCAL_PROVIDERS = [LLM.OLLAMA_LOCAL, LLM.VLLM]
LOCAL_PROVIDERS = [LLM.OLLAMA_LOCAL]


MASTER_SKU = "B08XYRDKDV" 
#HP Envy 6455e Wireless Color All-in-One Printer with 6 Months Free Ink (223R1A) (Renewed Premium)

# 1 failed, 8 passed, 1 skipped, 4 warnings in 147.16s (0:02:27
# 7 passed, 1 skipped, 4 warnings in 35.79s
#7 passed, 4 warnings in 42.26s
#7 passed, 4 warnings in 60.06s (0:01:00)
#2 failed, 6 passed, 4 warnings in 200.12s (0:03:20)

def product_woo():
    woo_catalog = "./tests/data/woocommerce/product_catalog.csv" #2038 records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, woo_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)
    return products

def product_shopify():
    shopify_catalog = "./tests/data/shopify/electronics/shopify_products.csv"
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.SHOPIFY, shopify_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.SHOPIFY)
    return products

def product_1k():
    with open("./tests/data/amazon/office/amazon_office_sample_1000.json", "r") as f:
        data = f.read()
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    return products

def product_5k():
    with open("./tests/data/amazon/office/amazon_office_sample_5000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    return products

def product_20k():    
    with open("./tests/data/amazon/office/amazon_office_sample_20000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    return products

def get_local_answer(provider: LLM, prompt: str, model: str, num_recs: int) -> list:
    local_providers = [LLM.OLLAMA_LOCAL, LLM.VLLM]
    if provider not in local_providers:
        raise ValueError("Invalid provider for local call")
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model, 
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    return parsed_recs


def test_print_setup():
    print(f"\nMASTER_SKU: {MASTER_SKU}")
    print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
        
    print(f"\nLOCAL: {LOCAL_PROVIDERS}")
    print(f"CLOUD: {CLOUD_PROVIDERS}")



def test_latest_openrouter_model():
    raw_products = product_woo()      
    products = ProductFactory.dedupe(raw_products)    
    rp = safe_random.choice(products)
    user_prompt = rp.sku    
    num_recs = safe_random.choice([3, 4, 5])
    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(match)
    print(f"\033[32mSelected product: {match.sku} - {match.name} \033[0m")

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()    
    print(f"PROMPT SIZE: {len(prompt)}") 
    wc = PromptFactory.get_word_count(prompt)
    print(f"word count: {wc}")
    tc = PromptFactory.get_token_count(prompt)
    print(f"token count: {tc}")
    
    #model = "x-ai/grok-code-fast-1"
    #model = "ai21/jamba-mini-1.7"    
    #model = "qwen/qwen3-next-80b-a3b-instruct"
    # model = "x-ai/grok-4-fast:free"
    # provider = LLM.OPEN_ROUTER

    #model = "gpt-4.1-nano"
    model = "gpt-5-nano"
    provider = LLM.CHAT_GPT

    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()

    hotkey = "ASLJALSJASLKFJALSKJDFK"
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt, 
                                 miner_hotkey=hotkey,
                                 use_verified_inference=True)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response.results)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus