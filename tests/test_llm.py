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
from tests.utils import write_prompt_to_file
from dotenv import load_dotenv
load_dotenv()


LOCAL_OLLAMA_URL = "http://10.0.0.40:11434/api/chat"

OLLAMA_MODEL = "mistral-nemo" #7/7 7 passed, 3 skipped, 4 warnings in 48.95s | 7 passed, 3 skipped, 4 warnings in 48.15s
#OLLAMA_MODEL = "llama3.1" #6/7
#OLLAMA_MODEL = "nemotron:70b-instruct-q4_K_M" #7/7
#OLLAMA_MODEL = "nemotron" #5/6
#OLLAMA_MODEL = "llama3.1:70b" #6/7
#OLLAMA_MODEL = "llama3.3" #3/5
#OLLAMA_MODEL = "llama3.3:70b-instruct-q2_K" #4/5
#OLLAMA_MODEL = "qwen2.5:32b-instruct" #0/5
#OLLAMA_MODEL = "gemma3" #3 failed, 4 passed, 2 skipped, 2 warnings in 30.60
#OLLAMA_MODEL = "gemma3:27b" #7 passed, 2 skipped, 2 warnings in 74.68s (0:01:14)


MASTER_SKU = "B07BG1CZ8X"   #iJuqi Mom Gifts from Daughter Son - 3PCS Stainless Steel Expendable Motivational 
                            # #Charm Bangle Bracelets Set for Mother's Day, Birthday Gifts for Mom, Mother Jewelry for Christmas (Silver)

print(f"MASTER_SKU: {MASTER_SKU}\n")
print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")

NUM_RECS = safe_random.choice([5, 6, 7, 8, 9, 10, 16, 20])

def product_woo():
    woo_catalog = "./tests/data/woocommerce/product_catalog.csv" #2038 records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, woo_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)
    return products

def product_1k():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    return products

def product_5k():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_5000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    return products

def product_20k():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_20000.json", "r") as f:
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



def test_call_local_llm_with_woo_catalog():
    products = product_woo()
    print(f"loaded {len(products)} records")
    assert len(products) == 2038
    #print(products)
    dd = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dd}")
    assert dd == 0
    
    #24-WB02 =Compete Track Tote
    #The Compete Track Tote holds a host of exercise supplies with ease. Stash your towel, jacket and street shoes inside. 
    # Tuck water bottles in easy-access external spaces. 
    # Perfect for trips to gym or yoga studio, with dual top handles for convenience to and from.
    
    user_prompt = "24-WB02"
    num_recs = 7
    debug_prompts = False

    match = [products for products in products if products.sku == user_prompt][0]
    print(f"Getting rec for: {match}")

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




def test_call_local_llm_with_1k():
    products = product_1k()
    print(f"loaded {len(products)} records")
    assert len(products) == 907
    
    dd = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dd}")
    assert dd == 61
    
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
    print(f"loaded {len(products)} records")
    assert len(products) == 4544

    dd = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dd}")
    assert dd == 416

    products = ProductFactory.dedupe(products)
    print(f"after de-dupe: {len(products)} records")
    
    user_prompt = MASTER_SKU
    num_recs = safe_random.choice([5, 6, 7])
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
    raw_products = product_20k()
    print(f"loaded: {len(raw_products)} records")
    assert len(raw_products) == 18_088

    dd = ProductFactory.get_dupe_count(raw_products)
    print(f"dupe count: {dd}")
    #assert dd == 2106    

    products = ProductFactory.dedupe(raw_products)    
    print(f"after de-dupe: {len(products)} records")
    
    user_prompt = MASTER_SKU
    num_recs = 6
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
    if 1==2:
        write_prompt_to_file(prompt)
    
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
    print(f"loaded: {len(raw_products)} records")
    assert len(raw_products) == 18_088

    dd = ProductFactory.get_dupe_count(raw_products)
    print(f"dupe count: {dd}")
    #assert dd == 2106    

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
