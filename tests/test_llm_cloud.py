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
from bitrecs.llms.factory import LLMFactory
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.prompt_factory import PromptFactory
from dotenv import load_dotenv
load_dotenv()



LOCAL_OLLAMA_URL = "http://10.0.0.40:11434/api/chat"
OLLAMA_MODEL = "mistral-nemo"

map = [
    {"provider": LLM.OLLAMA_LOCAL, "model": "mistral-nemo"},
    {"provider": LLM.VLLM, "model": "NousResearch/Meta-Llama-3-8B-Instruct"},
    {"provider": LLM.CHAT_GPT, "model": "gpt-5-nano"},

    #{"provider": LLM.OPEN_ROUTER, "model": "nvidia/llama-3.1-nemotron-70b-instruct"},
    #{"provider": LLM.OPEN_ROUTER, "model": "nousresearch/deephermes-3-llama-3-8b-preview:free"},

    {"provider": LLM.OPEN_ROUTER, "model": "amazon/nova-lite-v1"},
    {"provider": LLM.OPEN_ROUTER, "model": "google/gemini-2.5-flash-preview-09-2025"},
    {"provider": LLM.OPEN_ROUTER, "model": "x-ai/grok-4.1-fast"},
    {"provider": LLM.OPEN_ROUTER, "model": "openai/gpt-5-nano"},
    
    {"provider": LLM.GROK, "model": "grok-4-1-fast-non-reasoning"},
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


def test_warmup():
    prompt = "Tell me a joke"
    model = OLLAMA_MODEL
    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model, 
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)
    print(llm_response)
    assert llm_response is not None


def test_all_sets_matryoshka():
    list1 = product_1k()
    list2 = product_5k()
    list3 = product_20k()
    
    set1 = set(item.sku for item in list1)
    set2 = set(item.sku for item in list2)
    set3 = set(item.sku for item in list3)

    assert set1.issubset(set2)
    assert set2.issubset(set3)
    assert (set1 & set2).issubset(set3)


def test_product_dupes():
    list1 = product_1k()
    print(f"loaded {len(list1)} records")
    assert len(list1) == 1000
    d1 = ProductFactory.get_dupe_count(list1)
    print(f"dupe count: {d1}")
    assert d1 == 36
    dd1 = ProductFactory.dedupe(list1)
    print(f"after de-dupe: {len(dd1)} records") 
    assert len(dd1) == (len(list1) - d1)

    list2 = product_5k()
    print(f"loaded {len(list2)} records")
    assert len(list2) == 5000
    d2 = ProductFactory.get_dupe_count(list2)
    print(f"dupe count: {d2}")
    assert d2 == 568
    dd2 = ProductFactory.dedupe(list2)
    print(f"after de-dupe: {len(dd2)} records") 
    assert len(dd2) == (len(list2) - d2)

    list3 = product_20k()
    print(f"loaded {len(list3)} records")
    assert len(list3) == 19_999
    d3 = ProductFactory.get_dupe_count(list3)
    print(f"dupe count: {d3}")
    assert d3 == 4500
    dd3 = ProductFactory.dedupe(list3)
    print(f"after de-dupe: {len(dd3)} records") 
    assert len(dd3) == (len(list3) - d3)


def test_call_local_llm_with_1k_for_baseline():
    products = product_1k() 
    products = ProductFactory.dedupe(products)
    
    user_prompt = MASTER_SKU
    num_recs = 3
    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(match)

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    #print(prompt)    
    model = OLLAMA_MODEL
    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model, 
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)
    parsed_recs = PromptFactory.tryparse_llm(llm_response)   
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    
    assert len(parsed_recs) == num_recs
    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1   
  


def test_call_all_cloud_providers_warmup():    
    #prompt = "Don't be alarmed, we're going to talk about Product Recommendations"
    prompt = f"Tell me a joke today is {safe_random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])}"
    
    count = 0
    for provider in CLOUD_PROVIDERS:

        # if count > 2:
        #     break

        print(f"provider: {provider}")        
        model = safe_random.choice([m for m in map if m["provider"] == provider])["model"]
        #model = [m for m in map if m["provider"] == provider][0]["model"]
        print(f"provider: {provider}")
        try:
            print(f"asked: {prompt}")
            llm_response = LLMFactory.query_llm(server=provider,
                                model=model,
                                system_prompt="You are a helpful assistant", 
                                temp=0.0, 
                                user_prompt=prompt)                        
            print(f"response: {llm_response}")                      
            assert llm_response is not None 
            assert len(llm_response) > 0
            print(f"provider: \033[32m {provider} PASSED \033[0m with: {model}")
            count += 1

        except Exception:
            print(f"provider: {provider} \033[31m FAILED \033[0m using: {model}")            
            continue            
                     
    assert count == len(CLOUD_PROVIDERS)


#@pytest.mark.skip(reason="skipped")
def test_call_all_cloud_providers_1k_woo_products():    
    raw_products = product_woo()
    products = ProductFactory.dedupe(raw_products)
    print(f"after de-dupe: {len(products)} records")
  
    rp = safe_random.choice(products)
    user_prompt = rp.sku
    #num_recs = 3
    num_recs = safe_random.choice([1, 3, 5, 6, 8])
    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(match)    
    print(f"num_recs: {num_recs}")

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    #print(prompt)
    print(f"prompt length: {len(prompt)}")

    print("********** LOOPING PROVIDERS ")
    success_count = 0
    for provider in CLOUD_PROVIDERS:
        print(f"provider: {provider}")
        #model = [m for m in map if m["provider"] == provider][0]["model"]
        model = safe_random.choice([m for m in map if m["provider"] == provider])["model"]
        try:   
            st = time.time()         
            llm_response = LLMFactory.query_llm(server=provider,
                                model=model,
                                system_prompt="You are a helpful assistant", 
                                temp=0.0,
                                user_prompt=prompt)
            parsed_recs = PromptFactory.tryparse_llm(llm_response)
            print(f"parsed {len(parsed_recs)} records")
            print(parsed_recs)
            et = time.time()
            diff = et-st
            print(f"provider: \033[32m {provider} run \033[0m {model} : {diff:.2f} seconds")
            assert len(parsed_recs) == num_recs
            
            skus = [item['sku'] for item in parsed_recs]
            counter = Counter(skus)
            for sku, count in counter.items():
                print(f"{sku}: {count}")
                assert count == 1

            assert user_prompt not in skus

            success_count += 1
           
            print(f"provider: \033[32m {provider} PASSED woocommerce catalog \033[0m with: {model} in {diff:.2f} seconds")                 
        except Exception:
            print(f"provider: {provider} \033[31m FAILED woocommerce catalog \033[0m using: {model}")
            continue

    assert len(CLOUD_PROVIDERS) == success_count


#@pytest.mark.skip(reason="skipped - stalled")
def test_call_multiple_open_router_1k_amazon_random():
    raw_products = product_1k()    
    products = ProductFactory.dedupe(raw_products)
    print(f"after de-dupe: {len(products)} records")

    time.sleep(1)
    rp = safe_random.choice(products)
    user_prompt = rp.sku
    #num_recs = 3
    num_recs = safe_random.choice([1, 5, 9, 10, 11, 16, 20])

    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(match)    
    print(f"num_recs: {num_recs}")

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    #print(prompt)
    print(f"prompt length: {len(prompt)}")

    print("********** LOOPING PROVIDERS ")
    
    providers = [p for p in map if p["provider"] == LLM.OPEN_ROUTER]
    attempt_count = 0
    success_count = 0
    for provider in providers:
        print(f"provider: {provider}")
        attempt_count += 1

        model = provider["model"]
        this_provider = provider["provider"]

        try:            
            llm_response = LLMFactory.query_llm(server=this_provider,
                                model=model,
                                system_prompt="You are a helpful assistant", 
                                temp=0.0, 
                                user_prompt=prompt)
            parsed_recs = PromptFactory.tryparse_llm(llm_response)
            print(f"parsed {len(parsed_recs)} records")
            print(parsed_recs)

            assert len(parsed_recs) == num_recs

            skus = [item['sku'] for item in parsed_recs]
            counter = Counter(skus)
            for sku, count in counter.items():
                print(f"{sku}: {count}")
                assert count == 1

            print("asserting user_prompt not in sku")
            assert user_prompt not in skus
            
            success_count += 1
            print(f"provider: \033[32m {this_provider} PASSED amazon \033[0m with: {model}")
        except Exception:
            print(f"provider: {this_provider} \033[31m FAILED amazon \033[0m using: {model}")            
            continue

    provider_length = len(providers)
    assert attempt_count == provider_length
    print("PARTIAL PASS")

    assert attempt_count == success_count
    print("FULL PASS")


def test_call_multiple_open_router_amazon_5k_random():
    raw_products = product_5k()    
    products = ProductFactory.dedupe(raw_products)
    print(f"after de-dupe: {len(products)} records")

    time.sleep(1)
    rp = safe_random.choice(products)
    user_prompt = rp.sku
    #num_recs = 3
    #num_recs = safe_random.choice([1, 5, 9, 10, 11, 16, 20])

    num_recs = safe_random.choice([3, 4, 5])

    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(match)    
    print(f"num_recs: {num_recs}")

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    #print(prompt)
    print(f"PROMPT SIZE: {len(prompt)}")
 
    wc = PromptFactory.get_word_count(prompt)
    print(f"word count: {wc}")

    tc = PromptFactory.get_token_count(prompt)
    print(f"token count: {tc}")    


    print("********** LOOPING PROVIDERS ")    
    providers = [p for p in map if p["provider"] == LLM.OPEN_ROUTER]
    attempt_count = 0
    success_count = 0
    for provider in providers:
        print(f"provider: {provider}")
        attempt_count += 1

        model = provider["model"]
        this_provider = provider["provider"]
        try:
            llm_response = LLMFactory.query_llm(server=this_provider,
                                model=model,
                                system_prompt="You are a helpful assistant", 
                                temp=0.0, 
                                user_prompt=prompt)
            parsed_recs = PromptFactory.tryparse_llm(llm_response)
            print(f"parsed {len(parsed_recs)} records")
            print(parsed_recs)

            assert len(parsed_recs) == num_recs

            skus = [item['sku'] for item in parsed_recs]
            counter = Counter(skus)
            for sku, count in counter.items():
                print(f"{sku}: {count}")
                assert count == 1

            print("asserting user_prompt not in sku")
            assert user_prompt not in skus
            
            success_count += 1
            print(f"provider: \033[32m {this_provider} PASSED amazon \033[0m with: {model}")
        except Exception:
            print(f"provider: {this_provider} \033[31m FAILED amazon \033[0m using: {model}")            
            continue

    provider_length = len(providers)
    assert attempt_count == provider_length
    print("PARTIAL PASS")
    
    assert attempt_count == success_count
    print("FULL PASS")


@pytest.mark.skip(reason="skipped - chutes missing provider")
def test_call_chutes():
    #raw_products = product_5k() 
    #raw_products = product_1k()
    raw_products = product_woo()
      
    products = ProductFactory.dedupe(raw_products)
    #print(f"after de-dupe: {len(products)} records")    
    rp = safe_random.choice(products)
    user_prompt = rp.sku    
    num_recs = safe_random.choice([3, 4, 5])

    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(match)    
    # print(f"num_recs: {num_recs}")

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    #print(prompt)
    print(f"PROMPT SIZE: {len(prompt)}")
 
    wc = PromptFactory.get_word_count(prompt)
    print(f"word count: {wc}")

    tc = PromptFactory.get_token_count(prompt)
    print(f"token count: {tc}")    
    
    llm_response = LLMFactory.query_llm(server=LLM.CHUTES,
                                 model="deepseek-ai/DeepSeek-V3",
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    #print(parsed_recs)
    assert len(parsed_recs) == num_recs
    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus


@pytest.mark.skip(reason="skipped")
def test_call_nousresearch_deephermes_3_mistral_24b_preview():
    #nousresearch/deephermes-3-mistral-24b-preview
    raw_products = product_woo()
      
    products = ProductFactory.dedupe(raw_products)
    #print(f"after de-dupe: {len(products)} records")    
    rp = safe_random.choice(products)
    user_prompt = rp.sku    
    num_recs = safe_random.choice([3, 4, 5])

    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(match)    
    # print(f"num_recs: {num_recs}")

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    #print(prompt)
    print(f"PROMPT SIZE: {len(prompt)}")
 
    wc = PromptFactory.get_word_count(prompt)
    print(f"word count: {wc}")

    tc = PromptFactory.get_token_count(prompt)
    print(f"token count: {tc}")    
    
    llm_response = LLMFactory.query_llm(server=LLM.OPEN_ROUTER,
                                 model="nousresearch/deephermes-3-mistral-24b-preview",
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    #print(parsed_recs)
    assert len(parsed_recs) == num_recs
    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus
    


@pytest.mark.skip(reason="skipped")
def test_call_horizon_alpha():    
    raw_products = product_woo()      
    products = ProductFactory.dedupe(raw_products)    
    rp = safe_random.choice(products)
    user_prompt = rp.sku    
    num_recs = safe_random.choice([3, 4, 5])
    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(match)
    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    #print(prompt)
    print(f"PROMPT SIZE: {len(prompt)}")
 
    wc = PromptFactory.get_word_count(prompt)
    print(f"word count: {wc}")

    tc = PromptFactory.get_token_count(prompt)
    print(f"token count: {tc}")        
    
    #model = "openrouter/horizon-alpha"
    model = "openrouter/horizon-beta"
    st = time.time()
    llm_response = LLMFactory.query_llm(server=LLM.OPEN_ROUTER,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs
    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus


@pytest.mark.skip(reason="skipped")
def test_call_openai_route_gpt4_and_gpt5_ok():
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
    
    # Test Legacy
    model = "gpt-4.1-nano"
    print(f"\033[32mTesting OpenAI with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=LLM.CHAT_GPT,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus

    # Test GPT5    
    model = "gpt-5-nano-2025-08-07"
    #model = "gpt-5-mini-2025-08-07"
    print(f"\033[32mTesting OpenAI with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=LLM.CHAT_GPT,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus



#@pytest.mark.skip(reason="skipped")
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
    #model = "x-ai/grok-4-fast:free"
    #model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    #model = "deepseek/deepseek-v3.2-exp"
    #model = "alibaba/tongyi-deepresearch-30b-a3b:free"
    #model = "nvidia/nemotron-nano-9b-v2:free"
    #model = "openai/gpt-5-nano"
    #model = "z-ai/glm-4.5-air"
    #model = "moonshotai/kimi-k2:free"
    #model = "openrouter/polaris-alpha"
    #model = "openrouter/sherlock-dash-alpha"
    #model = "openai/gpt-5.1-chat"
    model = "x-ai/grok-4.1-fast"
    #model = "openrouter/bert-nebulon-alpha"
        
    provider = LLM.OPEN_ROUTER
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus



@pytest.mark.skip(reason="skipped")
def test_cerebras():
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
    
    #model = "llama-4-scout-17b-16e-instruct"    
    model = "llama-4-maverick-17b-128e-instruct"
    provider = LLM.CEREBRAS
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus


@pytest.mark.skip(reason="skipped")
def test_grok():
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
    
    model = "grok-4-fast-non-reasoning"
    provider = LLM.GROK
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus




@pytest.mark.skip(reason="skipped")
def test_claude():
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
    
    #model = "claude-3-haiku-20240307"
    model = "claude-3-5-haiku-20241022"
    provider = LLM.CLAUDE
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus



@pytest.mark.skip(reason="skipped")
def test_groq():
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
    
    #model = "llama-3.1-8b-instant"
    model = "groq/compound"
    #model = "groq/compound-mini"
    provider = LLM.GROQ
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus



#@pytest.mark.skip(reason="skipped")
def test_nvidia_inference():
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
        
    #model = "deepseek-ai/deepseek-v3.1-terminus"        
    #model = "meta/llama-3.3-70b-instruct"
    model = "qwen/qwen3-next-80b-a3b-instruct"
    provider = LLM.NVIDIA
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus


#@pytest.mark.skip(reason="skipped")
def test_perplexity_inference():
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
        
    #model = "deepseek-ai/deepseek-v3.1-terminus"        
    #model = "meta/llama-3.3-70b-instruct"
    model = "sonar"
    provider = LLM.PERPLEXITY
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus



@pytest.mark.skip(reason="skipped")
def test_gemini_inference():
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
    
    
    model = "gemini-3-pro-preview"
    provider = LLM.GEMINI
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()
    llm_response = LLMFactory.query_llm(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")
    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    assert len(parsed_recs) == num_recs    
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    assert user_prompt not in skus



def query_llm_with_timeout(server, model, system_prompt, temp, user_prompt, timeout=30):   
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            LLMFactory.query_llm,
            server,
            model,
            system_prompt,
            temp,
            user_prompt
        )
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            fut.cancel()
            raise TimeoutError(f"LLM query timed out after {timeout} seconds") from e

@pytest.mark.skip(reason="skipped - long running test")
def test_cycle_models_and_store_results():   
    db_path = "./tests/llm_results.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT,
            model TEXT,
            prompt TEXT,
            raw_response TEXT,
            parsed_json TEXT,
            parsed_count INTEGER,
            requested_num_recs INTEGER,
            duration REAL,
            timestamp TEXT,
            success INTEGER,
            error TEXT
        )
        """
    )
    conn.commit()
    
    #raw_products = product_1k()
    raw_products = product_shopify()
    #raw_products = product_woo()
    products = ProductFactory.dedupe(raw_products)
    rp = safe_random.choice(products)
    user_prompt = rp.sku
    num_recs = 3

    context = json.dumps([asdict(p) for p in products], separators=(',', ':'))
    factory = PromptFactory(sku=user_prompt, context=context, num_recs=num_recs, debug=True)
    prompt = factory.generate_prompt()

    providers = [

        {"provider": LLM.OPEN_ROUTER, "model": "x-ai/grok-code-fast-1"},
        {"provider": LLM.OPEN_ROUTER, "model": "meta-llama/llama-4-maverick:free"},
        {"provider": LLM.OPEN_ROUTER, "model": "amazon/nova-lite-v1"},
        #{"provider": LLM.OPEN_ROUTER, "model": "nousresearch/hermes-4-70b"},
        {"provider": LLM.OPEN_ROUTER, "model": "inception/mercury-coder"},
        {"provider": LLM.OPEN_ROUTER, "model": "mistralai/mistral-small-3.2-24b-instruct"},

        #{"provider": LLM.OPEN_ROUTER, "model": "z-ai/glm-4.5-air:free"},
        #{"provider": LLM.OPEN_ROUTER, "model": "google/gemini-2.5-flash-lite-preview-06-17"},
        {"provider": LLM.OPEN_ROUTER, "model": "google/gemini-2.5-flash-lite"},
        {"provider": LLM.OPEN_ROUTER, "model": "meta-llama/llama-4-scout"},
        {"provider": LLM.OPEN_ROUTER, "model": "openai/gpt-4.1-nano"},
        {"provider": LLM.OPEN_ROUTER, "model": "openai/gpt-5-nano"},

        {"provider": LLM.OPEN_ROUTER, "model": "moonshotai/kimi-k2-0905"},
        {"provider": LLM.OPEN_ROUTER, "model": "qwen/qwen3-30b-a3b-instruct-2507"},
        {"provider": LLM.OPEN_ROUTER, "model": "ai21/jamba-mini-1.7"},

        #{"provider": LLM.OPEN_ROUTER, "model": "openrouter/sonoma-dusk-alpha"},
        #{"provider": LLM.OPEN_ROUTER, "model": "openrouter/sonoma-sky-alpha"},
        {"provider": LLM.OPEN_ROUTER, "model": "qwen/qwen3-next-80b-a3b-instruct"},
        {"provider": LLM.OPEN_ROUTER, "model": "x-ai/grok-4-fast"},
        {"provider": LLM.OPEN_ROUTER, "model":"google/gemini-2.5-flash-lite-preview-09-2025"}

    ]
    
    safe_random.shuffle(providers)
    #providers = providers[:3]

    success_count = 0
    for entry in providers:
        provider = entry["provider"]
        model = entry["model"]
        start = time.time()
        raw_resp = None
        parsed_recs = None
        parsed_count = 0
        success = 0
        error_msg = None
        try:
            print(f"Querying {provider} with model {model}...")          
            raw_resp = query_llm_with_timeout(
                server=provider,
                model=model,
                system_prompt="You are a helpful assistant",
                temp=0.0,
                user_prompt=prompt,
                timeout=30
            )
            parsed_recs = PromptFactory.tryparse_llm(raw_resp)
            parsed_count = len(parsed_recs) if parsed_recs else 0
            success = 1 if parsed_count == num_recs else 0
            success_count += success
                 
        except Exception as e:
            print(f"Error querying {provider} with model {model}: {e}")
            error_msg = repr(e)
            success = 0
        finally:
            duration = time.time() - start
            timestamp = datetime.now(timezone.utc).isoformat()
            cur.execute(
                """
                INSERT INTO llm_results
                (provider, model, prompt, raw_response, parsed_json, parsed_count, requested_num_recs, duration, timestamp, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(provider),
                    str(model),
                    prompt,
                    raw_resp,
                    json.dumps(parsed_recs, default=str) if parsed_recs is not None else None,
                    parsed_count,
                    num_recs,
                    duration,
                    timestamp,
                    success,
                    error_msg,
                ),
            )
            conn.commit()

    conn.close()
    
    assert success_count == len(providers), f"Not all provider/model combinations succeeded: {success_count} out of {len(providers)}"


@pytest.mark.skip(reason="skipped")
def test_print_summary_table():
    db_path = "./tests/llm_results.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT provider, model, COUNT(*) as attempts, SUM(success) as successes, AVG(duration) as avg_duration
        FROM llm_results
        GROUP BY provider, model
        ORDER BY successes DESC, avg_duration ASC
        """
    )
    rows = cur.fetchall()
    print("\nLLM Providers Summary:")
    print(f"{'Provider':<15} {'Model':<50} {'Attempts':>10} {'Successes':>10} {'Avg Duration (s)':>20}")
    print("-" * 105)    
    for row in rows:
        provider, model, attempts, successes, avg_duration = row
        # Truncate model name if longer than 49 chars to fit column
        model_display = model[:49] if len(model) > 49 else model
        print(f"{provider:<15} {model_display:<50} {attempts:>10} {successes:>10} {avg_duration:>20.2f}")    
    conn.close()
    assert 1==1
