import os
os.environ["NEST_ASYNCIO"] = "0"
import json
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

OLLAMA_MODEL = "mistral-nemo" #8/8 8 passed, 5 skipped, 4 warnings in 61.01s (0:01:01)  | 8 passed, 5 skipped, 3 warnings in 62.52s (0:01:02) |  7 passed, 5 skipped, 3 warnings in 50.73s | 8 passed, 5 skipped, 4 warnings in 59.86s 
#6 passed, 5 skipped, 3 warnings in 52.57s | 6 passed, 5 skipped, 3 warnings in 57.67s | 1 failed, 5 passed, 5 skipped, 3 warnings in 67.16s (0:01:07) |  1 failed, 5 passed, 5 skipped, 3 warnings in 62.38s (0:01:02) |1 failed, 6 passed, 5 skipped, 3 warnings in 57.55s 
#2 failed, 6 passed, 5 skipped, 4 warnings in 80.10s (0:01:20) |  1 failed, 7 passed, 5 skipped, 4 warnings in 69.80s (0:01:09) | 1 failed, 7 passed, 5 skipped, 4 warnings in 79.59s | 1 failed, 7 passed, 5 skipped, 3 warnings in 68.21s (0:01:08)

#OLLAMA_MODEL = "nemotron:70b-instruct-q4_K_M" #6/6 6 passed, 5 skipped, 3 warnings in 159.35s (0:02:39) 
#OLLAMA_MODEL = "llama3.1:70b" #6/6 6 passed, 5 skipped, 3 warnings in 133.20s (0:02:13)
#OLLAMA_MODEL = "llama3.1:70b-instruct-q4_0" #6 passed, 5 skipped, 3 warnings in 132.29s (0:02:12)
#OLLAMA_MODEL = "qwen2.5:32b" #6/6  6 passed, 5 skipped, 3 warnings in 119.75s (0:01:59) 
#OLLAMA_MODEL = "qwen2.5:32b-instruct" #6/6

#OLLAMA_MODEL = "mistral-nemo:12b" #5/6 1 failed, 6 passed, 5 skipped, 3 warnings in 58.90s
#OLLAMA_MODEL = "mistral-nemo:12b-instruct-2407-q8_0" #5/6  1 failed, 5 passed, 5 skipped, 3 warnings in 76.60s (0:01:16) | 1 failed, 6 passed, 5 skipped, 3 warnings in 87.11s (0:01:27) | 1 failed, 6 passed, 5 skipped, 3 warnings in 75.76s (0:01:15)
#OLLAMA_MODEL = "nemotron" #5/6 1 failed, 5 passed, 5 skipped, 3 warnings in 226.01s (0:03:46)

#OLLAMA_MODEL = "qwen2.5-coder:32b" #5/6
#OLLAMA_MODEL = "llama3.2-vision:90b" #5/6 1 failed, 5 passed, 5 skipped, 3 warnings in 163.90s (0:02:43)
#OLLAMA_MODEL = "falcon3:10b-instruct-fp16" # 1 failed, 5 passed, 5 skipped, 3 warnings in 123.59s (0:02:03)


#OLLAMA_MODEL = "llama3.1" #3/6
#OLLAMA_MODEL = "llama3.3" #3/5 context length errors in logs
#OLLAMA_MODEL = "llama3.3:70b-instruct-q2_K" #2/5
#OLLAMA_MODEL = "qwq" #4/6
#OLLAMA_MODEL = "gemma2:27b-instruct-fp16" # 4 failed, 2 passed, 5 skipped, 3 warnings in 119.31s (0:01:59)
#OLLAMA_MODEL = "deepseek-coder-v2:latest" # 4 failed, 2 passed, 5 skipped, 3 warnings in 71.11s (0:01:11)
#OLLAMA_MODEL = "llama3.2-vision:90b-instruct-q4_K_M" #2 failed, 4 passed, 5 skipped, 3 warnings in 216.71s (0:03:36)
#OLLAMA_MODEL = "qwen2.5:72b-instruct-q4_0" dnf
#OLLAMA_MODEL = "mistral-nemo:12b-instruct-2407-fp16"

#OLLAMA_MODEL = "phi4" # 1 failed, 7 passed, 5 skipped, 3 warnings in 124.66s (0:02:04) coldstart |  2 failed, 6 passed, 5 skipped, 3 warnings in 202.52s (0:03:22) 
#OLLAMA_MODEL = "phi4:14b-fp16" # 4 failed, 4 passed, 5 skipped, 3 warnings in 129.96s (0:02:09)
#OLLAMA_MODEL = "phi4:14b-q8_0" #1 failed, 7 passed, 5 skipped, 3 warnings in 90.35s (0:01:30) 
#OLLAMA_MODEL = "gemma3" #3 failed, 5 passed, 5 skipped, 2 warnings in 34.93s =============

MASTER_SKU = "B08XYRDKDV" #HP Envy 6455e Wireless Color All-in-One Printer with 6 Months Free Ink (223R1A) (Renewed Premium)
#MASTER_SKU = "B004KKX6IO" #Seville Classics Modern Ergonomic Pneumatic Height Adjustable 360-Degree Swivel Stool Chair, for Drafting, Office, Home, Garage, Work Desk

print(f"MASTER_SKU: {MASTER_SKU}\n")
print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")


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


def test_call_local_llm_with_1k():
    products = product_1k() 
    products = ProductFactory.dedupe(products)
    
    user_prompt = MASTER_SKU
    num_recs = 5    
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
    #print(llm_response)    
    parsed_recs = PromptFactory.tryparse_llm(llm_response)   
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    
    assert len(parsed_recs) == num_recs

    #check uniques
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1
    
  
def test_call_local_llm_with_5k():
    products = product_5k()
    products = ProductFactory.dedupe(products)
    print(f"after de-dupe: {len(products)} records")    
    
    user_prompt = MASTER_SKU
    num_recs = 5
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
    model = OLLAMA_MODEL
    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model, 
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)
    #print(llm_response)    
    parsed_recs = PromptFactory.tryparse_llm(llm_response)   
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    
    assert len(parsed_recs) == num_recs

    #check uniques
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1    


def test_call_local_llm_with_20k():
    products = product_20k()  
    products = ProductFactory.dedupe(products)    
    print(f"after de-dupe: {len(products)} records")
    
    user_prompt = MASTER_SKU
    #num_recs = 5
    num_recs = safe_random.choice([5, 6, 7, 8, 9, 10, 11, 12, 16, 20])
    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(match)    

    #context = json.dumps([asdict(products) for products in products])
    context = json.dumps([asdict(products) for products in products], separators=(',', ':'))
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    tc = PromptFactory.get_token_count(prompt)
    print(f"prompt length: {len(prompt)}")
    print(f"prompt token count: {tc}")

    #print(prompt)
    model = OLLAMA_MODEL

    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)
    #print(llm_response)    
    parsed_recs = PromptFactory.tryparse_llm(llm_response)   
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
  
    assert len(parsed_recs) == num_recs

    #check uniques
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1


def test_call_local_llm_with_20k_random_logic():
    raw_products = product_20k()
    products = ProductFactory.dedupe(raw_products)    
    print(f"after de-dupe: {len(products)} records")
   
    rp = safe_random.choice(products)
    user_prompt = rp.sku    
    num_recs = safe_random.choice([5, 6, 7, 8, 9, 10, 11, 12, 16, 20])

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
       
    model = OLLAMA_MODEL
    
    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model,
                                 system_prompt="You are a helpful assistant\n /no_think", 
                                 temp=0.0, user_prompt=prompt)
    #print(llm_response)    
    parsed_recs = PromptFactory.tryparse_llm(llm_response)   
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
  
    assert len(parsed_recs) == num_recs  

    #check uniques
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1

    assert user_prompt not in skus



def test_call_local_llm_with_shopify_1k_random_logic():
    raw_products = product_shopify()
    products = ProductFactory.dedupe(raw_products)    
    print(f"after de-dupe: {len(products)} records")
   
    rp = safe_random.choice(products)
    user_prompt = rp.sku        
    num_recs = safe_random.choice([5, 6, 7, 8, 9, 10])

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
       
    model = OLLAMA_MODEL
    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)
    #print(llm_response)    
    parsed_recs = PromptFactory.tryparse_llm(llm_response)   
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
  
    assert len(parsed_recs) == num_recs  

    #check uniques
    skus = [item['sku'] for item in parsed_recs]
    counter = Counter(skus)
    for sku, count in counter.items():
        print(f"{sku}: {count}")
        assert count == 1

    assert user_prompt not in skus