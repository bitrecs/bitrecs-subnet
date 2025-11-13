import os
os.environ["NEST_ASYNCIO"] = "0"
import json
import secrets
from random import SystemRandom
safe_random = SystemRandom()
from datetime import datetime
from bitrecs.protocol import BitrecsRequest
from dataclasses import asdict
from typing import List, Optional
from bitrecs.commerce.product import CatalogProvider, Product, ProductFactory
from bitrecs.llms.factory import LLMFactory
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.utils.misc import ttl_cache
from dotenv import load_dotenv
load_dotenv()

LOCAL_OLLAMA_URL = "http://10.0.0.40:11434/api/chat"
WARMUP_OLLAMA_MODEL = "mistral-nemo" #mistral-small3.1:latest

MODEL_BATTERY = [ "mistral-nemo", "phi4", "gemma3:27b", "qwen2.5:32b"]

#MODEL_BATTERY = ["llama3.1:70b-instruct-q4_0", "qwen2.5:32b", "gemma3:27b", "nemotron:latest"]

#MODEL_BATTERY = ["qwen2.5:32b", "gemma3:27b", "nemotron:latest", "phi4"]

# CLOUD_BATTERY = ["deepseek/deepseek-chat-v3-0324",
#                  "amazon/nova-lite-v1", "google/gemini-flash-1.5-8b",
#                  "x-ai/grok-2-1212", "openai/chatgpt-4o-latest", "anthropic/claude-2.1",
#                  "google/gemini-2.0-flash-001", "anthropic/claude-3.7-sonnet",
#                  "openai/gpt-4o-mini-search-preview", "openai/gpt-4o-mini", "qwen/qwen-turbo"]

CLOUD_BATTERY = ["amazon/nova-lite-v1", "google/gemini-2.0-flash-001", "deepseek/deepseek-chat-v3-0324:free"]


_FIRST_GET_REC = False
_FIRST_GET_MOCK_REC = False
DEBUG_ALL_PROMPTS = False

class TestConfig:
    similarity_threshold: float = 0.33
    top_n: int = 2
    num_recs: int = 6
    real_set_count: int = len(MODEL_BATTERY)
    fake_set_count: int = 9


@ttl_cache(ttl=900)
def product_1k() -> List[Product]:
    asos_catalog = "./tests/data/asos/sample_1k.csv" 
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, asos_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)    
    return products

@ttl_cache(ttl=900)
def product_5k() -> List[Product]:
    asos_catalog = "./tests/data/asos/sample_5k.csv" 
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, asos_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)    
    return products

@ttl_cache(ttl=900)
def product_10k_a() -> List[Product]:
    asos_catalog = "./tests/data/asos/sample_10k.csv" 
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, asos_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)    
    return products

@ttl_cache(ttl=900)
def product_20k() -> List[Product]:
    asos_catalog = "./tests/data/asos/asos_30k_trimmed.csv" 
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, asos_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)    
    return products

@ttl_cache(ttl=900)
def product_1k_a() -> List[Product]:
    walmart_catalog = "./tests/data/walmart/wallmart_1k_kaggle_trimmed.csv" 
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WALMART, walmart_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WALMART)    
    return products

@ttl_cache(ttl=900)
def product_5k_a() -> List[Product]:
    walmart_catalog = "./tests/data/walmart/wallmart_5k_kaggle_trimmed.csv"
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WALMART, walmart_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WALMART)
    return products

@ttl_cache(ttl=900)
def product_20k_a() -> List[Product]:   
    walmart_catalog = "./tests/data/walmart/wallmart_30k_kaggle_trimmed.csv" #30k records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WALMART, walmart_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WALMART)    
    return products    


def get_rec(products, sku, model=None, num_recs=5) -> List:
    global _FIRST_GET_REC
    if not sku or not products:
        raise ValueError("sku and products are required")
    products = ProductFactory.dedupe(products)
    user_prompt = sku
    debug_prompts = DEBUG_ALL_PROMPTS
    match = [products for products in products if products.sku == user_prompt][0]
    print(match)
    
    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    if not model:
        model = safe_random.choice(MODEL_BATTERY)
    print(f"Local Model:\033[32m {model} \033[0m")
    if not _FIRST_GET_REC:
        print("\nFirst prompt template:")
        print(prompt)
        _FIRST_GET_REC = True

    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model,
                                 system_prompt="You are a helpful assistant",
                                 temp=0.0, user_prompt=prompt)
    
    print(f"LLM response: {llm_response}")

    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    assert len(parsed_recs) == num_recs
    return parsed_recs


def get_rec_fake(sku, num_recs=5) -> List:
    if not sku:
        raise ValueError("sku is required")
    products = product_1k()
    products = ProductFactory.dedupe(products)
    result = safe_random.sample(products, num_recs)
    final = [thing.to_dict() for thing in result]    
    return final


def mock_br_request(products: List[Product], 
                    group_id: str, 
                    sku: str, 
                    model: str, 
                    num_recs: int) -> Optional[BitrecsRequest]:
    
    global _FIRST_GET_MOCK_REC
    assert num_recs > 0 and num_recs <= 20   
    products = ProductFactory.dedupe(products)
    user_prompt = sku    
    debug_prompts = DEBUG_ALL_PROMPTS
    match = [products for products in products if products.sku == user_prompt][0]
    print(match)    
    
    context = json.dumps([asdict(products) for products in products])    
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()    
    if not model:
        model = safe_random.choice(MODEL_BATTERY)
    print(f"Model:\033[32m {model} \033[0m")

    if not _FIRST_GET_MOCK_REC:
        print("\nFirst prompt template:")
        print(prompt)
        _FIRST_GET_MOCK_REC = True

    tc = PromptFactory.get_token_count(prompt)
    print(f"Token count: {tc}")

    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model, 
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)    
    parsed_recs = PromptFactory.tryparse_llm(llm_response) 
    assert len(parsed_recs) == num_recs

    m = BitrecsRequest(
        created_at=datetime.now().isoformat(),
        user="test_user",
        num_results=num_recs,
        query=sku,
        context="[]",
        site_key=group_id,
        results=parsed_recs,
        models_used=[model],
        miner_uid=str(safe_random.randint(10, 1000)),
        miner_hotkey=secrets.token_hex(16)
    )    
    return m


def mock_br_request_cloud(products: List[Product], 
                    group_id: str, 
                    sku: str, 
                    model: str, 
                    num_recs: int) -> Optional[BitrecsRequest]:
    assert num_recs > 0 and num_recs <= 20   
    products = ProductFactory.dedupe(products)
    user_prompt = sku    
    debug_prompts = DEBUG_ALL_PROMPTS
    match = [products for products in products if products.sku == user_prompt][0]
    print(match)
  
    context = json.dumps([asdict(products) for products in products])    
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()
    if not model:
        model = safe_random.choice(CLOUD_BATTERY)
    print(f"Cloud Model:\033[32m {model} \033[0m")

    tc = PromptFactory.get_token_count(prompt)
    print(f"Token count: {tc}")

    llm_response = LLMFactory.query_llm(server=LLM.OPEN_ROUTER,
                                 model=model, 
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)    
    parsed_recs = PromptFactory.tryparse_llm(llm_response) 
    assert len(parsed_recs) == num_recs
    
    m = BitrecsRequest(
        created_at=datetime.now().isoformat(),
        user="test_user",
        num_results=num_recs,
        query=sku,
        context="[]",
        site_key=group_id,
        results=parsed_recs,
        models_used=[model],
        miner_uid=str(safe_random.randint(10, 1000)),
        miner_hotkey=secrets.token_hex(16)
    )    
    return m


def product_name_by_sku_trimmed(sku: str, take_length: int = 50, products = None) -> str:
    try:
        if not products:
            products = product_20k()        
        selected_product = [p for p in products if p.sku == sku][0]
        name = selected_product.name
        if len(name) > take_length:
            name = name[:take_length] + "..."
        return name        
    except Exception as e:
        print(e)
        return f"Error loading sku {sku}"    


def test_results_have_reasoning_tags_local():
    num_recs = 5
    products = product_1k()
    products = ProductFactory.dedupe(products)
    product = safe_random.choice(products)
    product_name = product_name_by_sku_trimmed(product.sku, 500)
    
    print(f"Product name: {product_name}")
    print(f"Product sku: {product.sku}")

    user_prompt = product.sku
    debug_prompts = DEBUG_ALL_PROMPTS
    match = [products for products in products if products.sku == user_prompt][0]
    print(match)
    
    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    
    prompt = factory.generate_prompt()   
    model = safe_random.choice(MODEL_BATTERY)    
    print(prompt)    
    print(f"Local Model:\033[32m {model} \033[0m")

    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model,
                                 system_prompt="You are a helpful assistant",
                                 temp=0.0, user_prompt=prompt)
    
    print(f"LLM response: {llm_response}")

    parsed_recs = PromptFactory.tryparse_llm(llm_response)
    assert len(parsed_recs) == num_recs
    
    print(f"\033[32m For product: {product_name} \033[0m")
    
    for rec in parsed_recs: 
        assert rec.get("sku") is not None, f"Recommendation {rec} does not have sku"                
        assert rec.get("reason") is not None, f"Recommendation {rec} does not have reasoning tags"        
        print(f"Product name: {product_name_by_sku_trimmed(rec['sku'], 500)}")
        print(f"Recommendation: {rec['sku']}")
        print(f"Reasoning: {rec['reason']}")        

    print(f"\033[32;1m Recs provided by  {model} \033[0m")
        
        


def test_results_have_reasoning_tags_cloud():
    num_recs = 5
    products = product_1k()
    products = ProductFactory.dedupe(products)
    product = safe_random.choice(products)
    product_name = product_name_by_sku_trimmed(product.sku, 500)
    
    print(f"Product name: {product_name}")
    print(f"Product sku: {product.sku}")  

    user_prompt = product.sku
    debug_prompts = DEBUG_ALL_PROMPTS
    match = [products for products in products if products.sku == user_prompt][0]
    print(match)
    
    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt, 
                            context=context, 
                            num_recs=num_recs,
                            debug=debug_prompts)
    prompt = factory.generate_prompt()
    print(prompt)

    group_id = secrets.token_hex(16)    
    rec_requests : List[BitrecsRequest] = []
    models_used = []
    
    print("\nGenerating cloud recommendations...")
    battery = CLOUD_BATTERY    
    safe_random.shuffle(battery)
    for model in battery:        
        try:                        
            req = mock_br_request_cloud(products, group_id, user_prompt, model, num_recs)
            rec_requests.append(req)
            print(f"Set {model}: {[r['sku'] for r in req.results]}")
            models_used.append(model)
        except Exception as e:
            print(f"SKIPPED - Error with model {model}: {e}")
            continue   
    
    print(f"\033[32m For product: {product_name} \033[0m")
    
    for i, req in enumerate(rec_requests):
        print(f"\n=== Recommendations for {models_used[i]} ===")
        print(f"Model: {models_used[i]}")
        
        for rec in req.results:
            assert rec.get("sku") is not None, f"Recommendation {rec} does not have sku"
            assert rec.get("reason") is not None, f"Recommendation {rec} does not have reasoning tags"

            print(f"REC ------ {i}")
            print(f"Product name: {product_name_by_sku_trimmed(rec['sku'], 500)}")
            print(f"Recommendation: {rec['sku']}")
            print(f"Reasoning: {rec['reason']}")




def test_results_have_reasoning_tags_hybrid():
    group_id = secrets.token_hex(16)
    products = product_1k()    
    products = ProductFactory.dedupe(products)
    selected_product = safe_random.choice(products)
    sku = selected_product.sku
    product_name = product_name_by_sku_trimmed(sku, 500, products)
    
    rec_requests : List[BitrecsRequest] = []
    models_used = []   

    config = TestConfig()    
    
    #MIX = ['RANDOM']    
    MIX = ['RANDOM', 'CLOUD', 'LOCAL']
    RANDOM_COUNT = 5
    CLOUD_COUNT = 2
    LOCAL_COUNT = 3
        
    print("\n=== Hybrid Reason Tags Test ===")
    print(f"This test is using {len(products)} products ")
    print("Original Product:")
    print(f"SKU: \033[32m {sku} \033[0m")
    print(f"Name: \033[32m {selected_product.name} \033[0m")
    print(f"Price: ${selected_product.price}")
    
    if "RANDOM" in MIX:
        print("\nGenerating random recommendations...")
        for i in range(RANDOM_COUNT):
            fake_recs = get_rec_fake(sku, config.num_recs)
            fake_model = f"random-{i}"
            req = BitrecsRequest(
                created_at=datetime.now().isoformat(),
                user="test_user",
                num_results=config.num_recs,
                query=sku,
                context="[]",
                site_key=group_id,
                results=[{"sku": r['sku']} for r in fake_recs],
                models_used=[fake_model],
                miner_uid=str(safe_random.randint(10, 100)),
                miner_hotkey=secrets.token_hex(16)
            )
            rec_requests.append(req)
            models_used.append(fake_model)  

    if "CLOUD" in MIX:
        print("\nGenerating cloud recommendations...")
        battery = CLOUD_BATTERY[:CLOUD_COUNT]
        #battery = CLOUD_BATTERY
        safe_random.shuffle(battery)
        for model in battery:        
            try:                        
                req = mock_br_request_cloud(products, group_id, sku, model, config.num_recs)
                rec_requests.append(req)
                print(f"Set {model}: {[r['sku'] for r in req.results]}")
                models_used.append(model)
            except Exception as e:
                print(f"SKIPPED - Error with model {model}: {e}")
                continue

    if "LOCAL" in MIX:
        print("\nGenerating Local recommendations...")
        local_battery = MODEL_BATTERY[:LOCAL_COUNT]
        #local_battery = MODEL_BATTERY
        safe_random.shuffle(local_battery)  
        for model in local_battery:
            try:
                mock_req = mock_br_request(products, group_id, sku, model, config.num_recs)
                rec_requests.append(mock_req)
                print(f"Set {model}: {[r['sku'] for r in mock_req.results]}")
                models_used.append(model)
            except Exception as e:
                print(f"SKIPPED - Error with model {model}: {e}")
                continue

  
    print(f"\033[32m For product: {product_name} \033[0m")
    
    for i, req in enumerate(rec_requests):
        print(f"\n=== Recommendations for {models_used[i]} ===")
        print(f"Model: {models_used[i]}")

        if RANDOM_COUNT > 0 and i < RANDOM_COUNT:
            print(f"Fake Model: {models_used[i]}")
            print("skipped - fake recs")
            continue
        else:
            for rec in req.results:            
                assert rec.get("sku") is not None, f"Recommendation {rec} does not have sku"
                assert rec.get("reason") is not None, f"Recommendation {rec} does not have reasoning tags"

                print(f"REC ------ {i}")
                print(f"Product name: {product_name_by_sku_trimmed(rec['sku'], 500)}")
                print(f"Recommendation: {rec['sku']}")
                print(f"Reasoning: {rec['reason']}")
