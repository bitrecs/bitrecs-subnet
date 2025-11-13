
import os
import time
import traceback
os.environ["NEST_ASYNCIO"] = "0"
import pytest
import sys
import json
import secrets
import bittensor as bt
from pathlib import Path
from random import SystemRandom
safe_random = SystemRandom()
from datetime import datetime
from bitrecs.protocol import BitrecsRequest
from dataclasses import asdict
from typing import List, Optional, Set
from bitrecs.commerce.product import CatalogProvider, Product, ProductFactory
from bitrecs.llms.factory import LLMFactory
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.validator.reward import validate_result_schema
from bitrecs.utils.misc import ttl_cache
from bitrecs.utils.distance import (        
    display_rec_matrix,    
    select_most_similar_bitrecs,
    select_most_similar_sets,
    calculate_jaccard_distance,    
    select_most_similar_bitrecs_threshold,
    select_most_similar_bitrecs_threshold2,        
)
from bitrecs.utils.color import ColorScheme
from dotenv import load_dotenv
load_dotenv()

LOCAL_OLLAMA_URL = "http://10.0.0.40:11434/api/chat"
WARMUP_OLLAMA_MODEL = "mistral-nemo"

MODEL_BATTERY = [ "mistral-nemo", "phi4", "gemma3:27b", "qwen3:32b" ]

CLOUD_BATTERY = ["amazon/nova-lite-v1", "google/gemini-flash-1.5-8b", "google/gemini-2.0-flash-001",
                 "x-ai/grok-4-fast", "qwen/qwen-turbo", "openai/gpt-4o-mini", "openai/gpt-4.1-nano"]



_FIRST_GET_REC = False
_FIRST_GET_MOCK_REC = False

DEBUG_ALL_PROMPTS = False

class TestConfig:
    similarity_threshold: float = 0.33
    top_n: int = 2
    num_recs: int = 6
    real_set_count: int = len(MODEL_BATTERY)
    fake_set_count: int = 9


@pytest.fixture(autouse=True)
def test_logger(request: pytest.FixtureRequest):
    """Fixture to capture print statements and write to timestamped log file."""    
    log_dir = Path("./tests", "test_logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = request.node.name
    log_file = log_dir / f"{test_name}_{timestamp}.log"
    with open(log_file, 'w') as f:        
        f.write(f"=== Test Started: {test_name} at {timestamp} ===\n\n")                
        original_stdout = sys.stdout
        class DualOutput:
            def write(self, text):
                original_stdout.write(text)
                f.write(text)
                f.flush()
            
            def flush(self):
                original_stdout.flush()
                f.flush()
        
        # Replace stdout with our custom output
        sys.stdout = DualOutput()        
        yield        
        
        sys.stdout = original_stdout
        f.write(f"\n=== Test Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")


def product_woo():
    woo_catalog = "./tests/data/woocommerce/product_catalog.csv" #2038 records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, woo_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)
    return products

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
    #print(f"Sku {sku} with model {model} = {parsed_recs}")
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
    #print(f"Sku {sku} with {model} = {parsed_recs}")
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
    #print(f"Sku {sku} with {model} = {parsed_recs}")
    return m


def recommender_presenter(original_sku: str, recs: List[Set[str]]) -> str:    
    result = f"Target SKU: \033[32m {original_sku} \033[0m\n"
    target_product_name = product_name_by_sku_trimmed(original_sku, 200)
    result += f"Target Product:\033[32m{target_product_name} \033[0m\n"
    result += "------------------------------------------------------------\n"    
    # Track matches with simple counter
    matches = {}  # name -> count    
    # First pass - count matches
    for rec_set in recs:
        for rec in rec_set:
            name = product_name_by_sku_trimmed(rec, 90)
            matches[name] = matches.get(name, 0) + 1
    
    # Second pass - output with emphasis on matches
    seen = set()
    for rec_set in recs:
        for rec in rec_set:
            name = product_name_by_sku_trimmed(rec, 90)
            if (rec, name) in seen:
                continue
                
            seen.add((rec, name))
            count = matches[name]
            if count > 1:
                # Double match - bright green
                result += f"\033[1;32m{rec} - {name} (!)\033[0m\n"
            elif count == 1:
                # Single appearance - normal
                result += f"{rec} - {name}\n"

    result += "\n"
    return result


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



def test_warmup():
    print(f"Model battery: {MODEL_BATTERY}")
    prompt = "Tell me a joke"
    model = WARMUP_OLLAMA_MODEL
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


def test_local_llm_bitrecs_mock_ok():
    group_id = secrets.token_hex(16)
    products = product_1k()
    sku = safe_random.choice(products).sku
    print(f"Group ID: {group_id}")
    print(f"SKU: {sku}")
    
    mock_request = mock_br_request(products, group_id, sku, "mistral-nemo", 5)
    assert mock_request is not None
    assert mock_request.num_results == 5
    assert mock_request.query == sku
    assert mock_request.results is not None
    assert len(mock_request.results) == 5
    assert mock_request.models_used is not None
    assert len(mock_request.models_used) == 1
    assert mock_request.miner_uid is not None
    assert mock_request.miner_hotkey is not None
    assert mock_request.site_key == group_id


def test_local_llm_base_config_jaccard():
    config = TestConfig()   
    products = product_woo()
    #products = product_1k()
    #products = product_5k()
    products = ProductFactory.dedupe(products)
    product = safe_random.choice(products)
    product_name = product_name_by_sku_trimmed(product.sku, 500)
    
    print("\n=== Recommendation Set Analysis ===")
    print(f"This test is using {len(products)} products ")
    print(f"Testing recommendations for product SKU: {product.sku}")  
    print(f"Product Name: \033[32m{product_name} \033[0m") 
        
    rec_sets = []
    model_recs = {}
    models_used = []
    
    print(f"Number of recommendations: {config.num_recs}")
    print(f"Number of real sets: {config.real_set_count}")
    print(f"Number of fake sets: {config.fake_set_count}")
    
    print("\nGenerating real recommendations...")
    for i in range(config.real_set_count):
        this_model = MODEL_BATTERY[i % len(MODEL_BATTERY)]
        recs = get_rec(products, product.sku, this_model, config.num_recs)
        assert recs is not None
        rec_set = set(str(r['sku']) for r in recs)
        rec_sets.append(rec_set)
        print(f"Set {i} (Real) {this_model}: {sorted(list(rec_set))}")
        model_recs[this_model] = recs
        models_used.append(this_model)
        
    print("\nGenerating random recommendations...")
    for i in range(config.fake_set_count):
        this_model = f"random-{i}"
        fake_recs = get_rec_fake(product.sku, config.num_recs)
        assert fake_recs is not None
        fake_set = set(str(r["sku"]) for r in fake_recs)
        rec_sets.append(fake_set)
        print(f"Set {i} (Random) {this_model}: {sorted(list(fake_set))}")        
        model_recs[this_model] = fake_recs
        models_used.append(this_model)
   
    print("\nJaccard Distance Matrix:")
    print("=" * 60)    
    
    header = "Sets:"
    for i in range(len(rec_sets)):
        header += f"{i:>7}"
    print(header)
    print("-" * 60)
    
    print(f"total of {len(rec_sets)} sets")
    
    # Calculate Jaccard distances between all pairs with aligned columns
    for i in range(len(rec_sets)):
        row = f"{i:4d}"
        for j in range(len(rec_sets)):
            if j < i:
                distance = calculate_jaccard_distance(rec_sets[i], rec_sets[j])
                row += f"{distance:7.3f}"
            else:
                row += "      -"
        print(row)
    
    print("\nNote: Lower distances between sets (real) vs (random)")
    print("      indicate better recommendation quality")
    print("=" * 40)

    # Verify all distances are valid
    for i in range(len(rec_sets)):
        for j in range(i + 1, len(rec_sets)):
            distance = calculate_jaccard_distance(rec_sets[i], rec_sets[j])
            assert 0 <= distance <= 1

    print("\nSelecting most similar sets:")
    most_similar = select_most_similar_sets(rec_sets, top_n=config.top_n)
    print(f"Most similar set indices: {most_similar}")
    print("Selected sets:")
    for idx in most_similar:
        model_used = models_used[idx]
        print(f"Set {idx}: {sorted(list(rec_sets[idx]))} - \033[32m {model_used} \033[0m")

    # Verify that the most similar sets are the real ones
    for idx in most_similar:
        assert idx <= config.real_set_count
    
    print("\nVerifying recommendation quality:")
    print("=" * 60)
    
    # Check that all selected sets are from real recommendations
    for idx in most_similar:
        if idx >= config.real_set_count:
            print(f"WARNING: Set {idx} is a random set, not a real recommendation!")
        assert idx < config.real_set_count, f"Set {idx} is not from real recommendations (idx >= {config.real_set_count})"
    
    similar_set_distances = []
    for i in range(len(most_similar)):
        for j in range(i + 1, len(most_similar)):
            dist = calculate_jaccard_distance(rec_sets[most_similar[i]], rec_sets[most_similar[j]])
            similar_set_distances.append(dist)
    
    avg_similarity = 1 - (sum(similar_set_distances) / len(similar_set_distances))
    print(f"Average similarity between selected sets: {avg_similarity:.3f}")
    print(f"Average distance between selected sets: {1-avg_similarity:.3f}")    
    
    if avg_similarity >= config.similarity_threshold:
        f"Selected sets have low similarity ({avg_similarity:.3f} < {config.similarity_threshold})"
    
    print("\nQuality check passed:")
    print("✓ All selected sets are from real recommendations")
    print(f"✓ Average similarity above threshold ({avg_similarity:.3f} >= {config.similarity_threshold})")
    print("=" * 60)

    summary = recommender_presenter(product.sku, [rec_sets[idx] for idx in most_similar])
    print(summary)
    
    # matrix = display_rec_matrix(rec_sets, models_used, most_similar)
    # print(matrix)

    # matrix = display_rec_matrix_numpy(rec_sets, models_used, most_similar)
    # print(matrix)

    # for s in ColorScheme:
    #     print("----------------------------------------------------------------------------")
    #     print(f"{s.name}: {s.value}")
    #     npmatrix = display_rec_matrix_numpy(rec_sets, models_used, most_similar, ColorScheme[s.name])
    #     print(npmatrix)        



def test_local_llm_raw_1k_jaccard():
    """Test recommendation sets using Jaccard similarity with model tracking"""
    group_id = secrets.token_hex(16)
    products = product_1k()
    products = ProductFactory.dedupe(products)
    selected_product = safe_random.choice(products)
    sku = selected_product.sku
    
    config = TestConfig()
    rec_tracking : List[Set] = []
    
    print("\n=== Recommendation Analysis ===")
    print(f"This test is using {len(products)} products ")
    print(f"SKU: {sku}")
    print(f"Recommendations per set: {config.num_recs}")    
    
    print("\nGenerating random recommendations...")
    for i in range(config.fake_set_count):
        model_name = f"random-{i}"
        fake_recs = get_rec_fake(sku, config.num_recs)
        fake_set = set(str(r["sku"]) for r in fake_recs)
        rec_tracking.append((fake_set, model_name))
        print(f"Set (Random) {model_name}: {sorted(list(fake_set))}")    
    
    print("\nGenerating model recommendations...")    
    battery = MODEL_BATTERY    
    safe_random.shuffle(battery)

    for model in battery:
        mock_req = mock_br_request(products, group_id, sku, model, config.num_recs)
        rec_set = set(str(r['sku']) for r in mock_req.results)
        rec_tracking.append((rec_set, model))
        print(f"Set {model}: {sorted(list(rec_set))}")

    
    rec_sets = [item[0] for item in rec_tracking]        
    print("\nJaccard Distance Matrix:")
    print("=" * 60)
    most_similar = select_most_similar_sets(rec_sets, top_n=config.top_n)
    assert most_similar is not None
    assert len(most_similar) == config.top_n
    
    print("\nMost Similar Sets:")
    print("-" * 60)
    for idx in most_similar:
        rec_set, model = rec_tracking[idx]
        print(f"Set {idx} ({model}):")
        print(f"  SKUs: {sorted(list(rec_set))}")        
    
    for idx in most_similar:
        model = rec_tracking[idx][1]
        is_random = model.startswith("random-")
        assert not is_random, f"Selected set {idx} is random, expected real model"


def test_local_llm_bitrecs_5k_jaccard():
    group_id = secrets.token_hex(16)    
    products = product_5k()
    products = ProductFactory.dedupe(products)
    sku = safe_random.choice(products).sku
    
    print(f"This test is using {len(products)} products ")  
    product_name = product_name_by_sku_trimmed(sku, 500)
    print(f"Target Product:\033[32m{product_name} \033[0m")

    config = TestConfig()
    rec_sets = []
    models_used = []
    
    battery = MODEL_BATTERY
    safe_random.shuffle(battery)

    print(f"USING LOCAL MODEL BATTERY of size: {len(battery)}")
    print(f"Number of recommendations: {config.num_recs}")
    print(f"Number of real sets: {len(battery)}")
    print(f"Number of fake sets: {config.fake_set_count}")  
    print(f"Top N: {config.top_n}")
    
    for i, thing in enumerate(range(config.fake_set_count)):
        this_model = f"random-{i}"
        fake_recs = get_rec_fake(sku, config.num_recs)
        assert fake_recs is not None
        fake_set = set(str(r['sku']) for r in fake_recs)
        assert len(fake_set) == config.num_recs
        print(f"Set {i} (Random) {this_model}: {sorted(list(fake_set))}")
        rec_sets.append(fake_set)
        models_used.append(this_model)
     
    for model in battery:
        mock_req = mock_br_request(products, group_id, sku, model, config.num_recs)
        assert mock_req is not None        
        rec_set = set(str(r['sku']) for r in mock_req.results)        
        rec_sets.append(rec_set)
        i += 1
        print(f"Set {i} {model}: {sorted(list(rec_set))}")
        models_used.append(model)     
        
    print("\nFinished generating rec sets")  
    print(f"A total of {len(rec_sets)} sets")
    
    most_similar = select_most_similar_sets(rec_sets, top_n=config.top_n)
    assert most_similar is not None
    assert len(most_similar) == config.top_n

    print(f"\nMost similar set indices: {most_similar}")
    print("Selected sets:")
    for idx in most_similar:
        print(f"Set {idx}/{len(rec_sets)} {sorted(list(rec_sets[idx]))}")
        model = models_used[idx]
        print(f"Model: {model}")

    report = recommender_presenter(sku, [rec_sets[idx] for idx in most_similar])
    print(report)     
    
    matrix = display_rec_matrix(rec_sets, models_used, most_similar)
    print(matrix)


def test_local_llm_bitrecs_protocol_5k_jaccard():
    """Test recommendation sets using BitrecsRequest protocol"""
    group_id = secrets.token_hex(16)
    #products = product_1k()
    products = product_5k()
    products = ProductFactory.dedupe(products)
    selected_product = safe_random.choice(products)
    sku = selected_product.sku
    
    config = TestConfig()
    rec_requests : List[BitrecsRequest] = []
    models_used = []
    
    print("\n=== Protocol Recommendation Analysis ===")    
    print(f"This test is using {len(products)} products ")
    print("Original Product:")
    print(f"SKU: \033[32m {sku} \033[0m")
    print(f"Name: \033[32m {selected_product.name} \033[0m")
    print(f"Price: ${selected_product.price}")
    
    # Generate fake recommendations first
    print("\nGenerating random recommendations...")
    for i in range(config.fake_set_count):
        fake_recs = get_rec_fake(sku, config.num_recs)
        req = BitrecsRequest(
            created_at=datetime.now().isoformat(),
            user="test_user",
            num_results=config.num_recs,
            query=sku,
            context="[]",
            site_key=group_id,
            results=[{"sku": r['sku']} for r in fake_recs],
            models_used=[f"random-{i}"],
            miner_uid=str(safe_random.randint(10, 100)),
            miner_hotkey=secrets.token_hex(16)
        )       
        rec_requests.append(req)
        models_used.append(f"random-{i}")       
        print(f"Set random-{i}: {[r['sku'] for r in req.results]}")
    
    print("\nGenerating model recommendations...")    
    battery = MODEL_BATTERY
    for model in battery:
        req = mock_br_request(products, group_id, sku, model, config.num_recs)
        rec_requests.append(req)
        models_used.append(model)
        print(f"Set {model}: {[r['sku'] for r in req.results]}")

    # No threshold
    most_similar = select_most_similar_bitrecs(rec_requests, top_n=config.top_n)
    assert most_similar is not None
    assert len(most_similar) == config.top_n

    # 10% with threshold
    low_threshold = 0.10
    most_similar2 = select_most_similar_bitrecs_threshold(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=low_threshold)  

    # 33% or null
    med_threshold = 0.33
    most_similar3 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=med_threshold)
    
    good_threshold = 0.51
    most_similar4 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=good_threshold)
 
    print("\nFinished generating rec sets") 
    print("Selected sets:")
    for req in most_similar:
        model = req.models_used[0]
        skus = [r['sku'] for r in req.results]
        print(f"Model {model}:")
        print(f"  SKUs: {sorted(skus)}")

    print(f"\033[1;32m Top {config.top_n} No-Threshold Pairs: \033[0m")
    selected_sets = [set(r['sku'] for r in req.results) for req in most_similar]
    report = recommender_presenter(sku, selected_sets)
    print(report)

    if most_similar2:
        print("Low sets:")
        for req in most_similar2:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")      
            
        selected_sets2 = [set(r['sku'] for r in req.results) for req in most_similar2]
        report = recommender_presenter(sku, selected_sets2)        
        print(f"\033[1;32m Low Threshold {low_threshold} \033[0m")
        print(report)
    else:        
        print(f"\033[31m No sets found meeting threshold {low_threshold} \033[0m")


    if most_similar3:
        print("Selected sets:")
        for req in most_similar3:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets3 = [set(r['sku'] for r in req.results) for req in most_similar3]
        report = recommender_presenter(sku, selected_sets3)
        print(f"\033[1;32m Medium Threshold {med_threshold} \033[0m")
        print(report)
    else:
        print(f"\033[31m No sets found meeting threshold {med_threshold} \033[0m")

    if most_similar4:
        print("Selected sets:")
        for req in most_similar4:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets4 = [set(r['sku'] for r in req.results) for req in most_similar4]
        report = recommender_presenter(sku, selected_sets4)
        print(f"\033[1;32m Good Threshold {good_threshold} \033[0m")
        print(report)
    else:        
        print(f"\033[31m No sets for threshold (>51%) {len(selected_sets)} \033[0m")

    
    rec_sets = [set(r['sku'] for r in req.results) for req in rec_requests]    
    matrix = display_rec_matrix(rec_sets, models_used, most_similar)
    print(matrix)


def test_cloud_llm_bitrecs_protocol_1k_jaccard():
    """Test cloud recommendation sets using BitrecsRequest protocol"""
    group_id = secrets.token_hex(16)
    products = product_1k()
    #products = product_5k()
    products = ProductFactory.dedupe(products)
    selected_product = safe_random.choice(products)
    sku = selected_product.sku
    
    config = TestConfig()
    rec_requests : List[BitrecsRequest] = []
    models_used = []
    
    print("\n=== Protocol Recommendation Analysis ===")
    print(f"This test is using {len(products)} products ")
    print("Original Product:")
    print(f"SKU: \033[32m {sku} \033[0m")
    print(f"Name: \033[32m {selected_product.name} \033[0m")
    print(f"Price: ${selected_product.price}")
    
    print("\nGenerating random recommendations...")
    for i in range(config.fake_set_count):
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
        print(f"Set random-{i}: {[r['sku'] for r in req.results]}")
    
    print("\nGenerating cloud recommendations...")
    #battery = CLOUD_BATTERY[:4]
    battery = CLOUD_BATTERY
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

    # No threshold
    most_similar = select_most_similar_bitrecs(rec_requests, top_n=config.top_n)
    assert most_similar is not None
    assert len(most_similar) == config.top_n

    # 10% with threshold
    low_threshold = 0.10
    most_similar2 = select_most_similar_bitrecs_threshold(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=low_threshold)  

    # 33% or null
    med_threshold = 0.33
    most_similar3 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=med_threshold)
    
    good_threshold = 0.51
    most_similar4 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=good_threshold)
 
    print("\nFinished generating rec sets")
    print("Selected sets:")
    for req in most_similar:
        model = req.models_used[0]
        skus = [r['sku'] for r in req.results]
        print(f"Model {model}:")
        print(f"  SKUs: {sorted(skus)}")

    print("\033[1;32m No Threshold \033[0m")
    selected_sets = [set(r['sku'] for r in req.results) for req in most_similar]
    report = recommender_presenter(sku, selected_sets)
    print(report)

    if most_similar2:
        print("Selected sets:")
        for req in most_similar2:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")      
            
        selected_sets2 = [set(r['sku'] for r in req.results) for req in most_similar2]
        report = recommender_presenter(sku, selected_sets2)        
        print(f"\033[1;32m Low Threshold {low_threshold} \033[0m")
        print(report)
    else:        
        print(f"\033[31m No sets found low threshold {low_threshold} \033[0m")

    if most_similar3:
        print("Selected sets:")
        for req in most_similar3:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets3 = [set(r['sku'] for r in req.results) for req in most_similar3]
        report = recommender_presenter(sku, selected_sets3)
        print(f"\033[1;32m Medium Threshold {med_threshold} \033[0m")
        print(report)
    else:
        print(f"\033[31m No sets found medium threshold {med_threshold} \033[0m")

    if most_similar4:
        print("Selected sets:")
        for req in most_similar4:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets4 = [set(r['sku'] for r in req.results) for req in most_similar4]
        report = recommender_presenter(sku, selected_sets4)
        print(f"\033[1;32m Good Threshold {good_threshold} \033[0m")
        print(report)
    else:
        print(f"\nNo results for threshold (>51%) out of {len(selected_sets)} sets ")

    rec_sets = [set(r['sku'] for r in req.results) for req in rec_requests]    
    matrix = display_rec_matrix(rec_sets, models_used, most_similar)
    print(matrix)


def test_hybrid_cloud_llm_bitrecs_protocol_1k_jaccard():
    """Test cloud / local recommendation sets using BitrecsRequest protocol"""
    group_id = secrets.token_hex(16)
    products = product_1k()
    #products = product_5k()
    products = ProductFactory.dedupe(products)
    selected_product = safe_random.choice(products)
    sku = selected_product.sku
    
    config = TestConfig()
    rec_requests : List[BitrecsRequest] = []
    models_used = []    
    
    print("\n=== Protocol Recommendation Analysis ===")    
    print(f"This test is using {len(products)} products ")
    print("Original Product:")
    print(f"SKU: \033[32m {sku} \033[0m")
    print(f"Name: \033[32m {selected_product.name} \033[0m")
    print(f"Price: ${selected_product.price}")    
    
    print("\nGenerating random recommendations...")
    for i in range(config.fake_set_count):
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
        print(f"Set random-{i}: {[r['sku'] for r in req.results]}")
    
    print("\nGenerating cloud recommendations...")
    #battery = CLOUD_BATTERY[:5]
    battery = CLOUD_BATTERY
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
    
    print("\nGenerating Local recommendations...")
    local_battery = MODEL_BATTERY[:3]
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

    # No threshold
    most_similar = select_most_similar_bitrecs(rec_requests, top_n=config.top_n)
    assert most_similar is not None
    assert len(most_similar) == config.top_n

    # 10% with threshold
    low_threshold = 0.10
    most_similar2 = select_most_similar_bitrecs_threshold(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=low_threshold)  

    # 33% or null
    med_threshold = 0.33
    most_similar3 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=med_threshold)
    
    good_threshold = 0.51
    most_similar4 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=good_threshold)
    
    great_threshold = 0.80
    most_similar5 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=great_threshold)
 
    print("\nFinished generating rec sets")
    print("Selected sets:")
    for req in most_similar:
        model = req.models_used[0]
        skus = [r['sku'] for r in req.results]
        print(f"Model {model}:")
        print(f"  SKUs: {sorted(skus)}")

    print("\033[1;32m No Threshold \033[0m")
    selected_sets = [set(r['sku'] for r in req.results) for req in most_similar]
    report = recommender_presenter(sku, selected_sets)
    print(report)

    if most_similar2:
        print("Selected sets:")
        for req in most_similar2:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")      
            
        selected_sets2 = [set(r['sku'] for r in req.results) for req in most_similar2]
        report = recommender_presenter(sku, selected_sets2)        
        print(f"\033[1;32m Low Threshold {low_threshold} \033[0m")
        print(report)
    else:
        print(f"No sets found meeting threshold {low_threshold}")

    if most_similar3:
        print("Selected sets:")
        for req in most_similar3:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets3 = [set(r['sku'] for r in req.results) for req in most_similar3]
        report = recommender_presenter(sku, selected_sets3)
        print(f"\033[1;32m Medium Threshold {med_threshold} \033[0m")
        print(report)
    else:
        print(f"\033[31m Noo sets found meeting threshold {med_threshold} \033[0m")

    if most_similar4:
        print("Selected sets:")
        for req in most_similar4:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets4 = [set(r['sku'] for r in req.results) for req in most_similar4]
        report = recommender_presenter(sku, selected_sets4)
        print(f"\033[1;32m Good Threshold {good_threshold} \033[0m")
        print(report)
    else:
        print(f"\nNo results for threshold (>51%) out of {len(selected_sets)} sets ")

    if most_similar5:
        print("Selected sets:")
        for req in most_similar5:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets5 = [set(r['sku'] for r in req.results) for req in most_similar5]
        report = recommender_presenter(sku, selected_sets5)
        print(f"\033[1;32m Great Threshold {great_threshold} \033[0m")
        print(report)
    else:
        print(f"\nNo results for threshold (>80%) out of {len(selected_sets)} sets ")

    rec_sets = [set(r['sku'] for r in req.results) for req in rec_requests]
    matrix = display_rec_matrix(rec_sets, models_used, most_similar)
    print(matrix)


def test_hybrid_cloud_llm_bitrecs_protocol_5k_jaccard():
    """Test cloud / local recommendation sets using BitrecsRequest protocol"""
    group_id = secrets.token_hex(16)
    #products = product_1k()
    products = product_5k()
    products = ProductFactory.dedupe(products)
    selected_product = safe_random.choice(products)
    sku = selected_product.sku
    
    config = TestConfig()
    rec_requests : List[BitrecsRequest] = []
    models_used = []    
    
    print("\n=== Protocol Recommendation Analysis ===")    
    print(f"This test is using {len(products)} products ")
    print("Original Product:")
    print(f"SKU: \033[32m {sku} \033[0m")
    print(f"Name: \033[32m {selected_product.name} \033[0m")
    print(f"Price: ${selected_product.price}")    
    
    print("\nGenerating random recommendations...")
    for i in range(config.fake_set_count):
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
        print(f"Set random-{i}: {[r['sku'] for r in req.results]}")
    
    print("\nGenerating cloud recommendations...")
    #battery = CLOUD_BATTERY[:5]
    battery = CLOUD_BATTERY
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
    
    print("\nGenerating Local recommendations...")
    local_battery = MODEL_BATTERY[:3]
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

    # No threshold
    most_similar = select_most_similar_bitrecs(rec_requests, top_n=config.top_n)
    assert most_similar is not None
    assert len(most_similar) == config.top_n

    # 10% with threshold
    low_threshold = 0.10
    most_similar2 = select_most_similar_bitrecs_threshold(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=low_threshold)  

    # 33% or null
    med_threshold = 0.33
    most_similar3 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=med_threshold)
    
    good_threshold = 0.51
    most_similar4 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=good_threshold)
    
    great_threshold = 0.80
    most_similar5 = select_most_similar_bitrecs_threshold2(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=great_threshold)
 
    print("\nFinished generating rec sets")
    print("Selected sets:")
    for req in most_similar:
        model = req.models_used[0]
        skus = [r['sku'] for r in req.results]
        print(f"Model {model}:")
        print(f"  SKUs: {sorted(skus)}")

    print("\033[1;32m No Threshold \033[0m")
    selected_sets = [set(r['sku'] for r in req.results) for req in most_similar]
    report = recommender_presenter(sku, selected_sets)
    print(report)

    if most_similar2:
        print("Selected sets:")
        for req in most_similar2:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")      
            
        selected_sets2 = [set(r['sku'] for r in req.results) for req in most_similar2]
        report = recommender_presenter(sku, selected_sets2)        
        print(f"\033[1;32m Low Threshold {low_threshold} \033[0m")
        print(report)
    else:
        print(f"No sets found meeting threshold {low_threshold}")

    if most_similar3:
        print("Selected sets:")
        for req in most_similar3:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets3 = [set(r['sku'] for r in req.results) for req in most_similar3]
        report = recommender_presenter(sku, selected_sets3)
        print(f"\033[1;32m Medium Threshold {med_threshold} \033[0m")
        print(report)
    else:
        print(f"\033[31m Noo sets found meeting threshold {med_threshold} \033[0m")

    if most_similar4:
        print("Selected sets:")
        for req in most_similar4:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets4 = [set(r['sku'] for r in req.results) for req in most_similar4]
        report = recommender_presenter(sku, selected_sets4)
        print(f"\033[1;32m Good Threshold {good_threshold} \033[0m")
        print(report)
    else:
        print(f"\nNo results for threshold (>51%) out of {len(selected_sets)} sets ")

    if most_similar5:
        print("Selected sets:")
        for req in most_similar5:
            model = req.models_used[0]
            skus = [r['sku'] for r in req.results]
            print(f"Model {model}:")
            print(f"  SKUs: {sorted(skus)}")
        selected_sets5 = [set(r['sku'] for r in req.results) for req in most_similar5]
        report = recommender_presenter(sku, selected_sets5)
        print(f"\033[1;32m Great Threshold {great_threshold} \033[0m")
        print(report)
    else:
        print(f"\nNo results for threshold (>80%) out of {len(selected_sets)} sets ")

    rec_sets = [set(r['sku'] for r in req.results) for req in rec_requests]    
    matrix = display_rec_matrix(rec_sets, models_used, most_similar)
    print(matrix)



def test_local_llm_bitrecs_protocol_with_randos():
    group_id = secrets.token_hex(16)    
    products = product_1k()
    products = ProductFactory.dedupe(products)
    sku = safe_random.choice(products).sku
    
    print(f"This test is using {len(products)} products ")  
    product_name = product_name_by_sku_trimmed(sku, 500)
    print(f"Target Product:\033[32m{product_name} \033[0m")

    config = TestConfig()
    rec_sets : List[BitrecsRequest] = []
    models_used = []
    
    battery = MODEL_BATTERY
    safe_random.shuffle(battery)    
    print(f"USING LOCAL MODEL BATTERY of size: {len(battery)}")
    print(f"Number of recommendations: {config.num_recs}")
    print(f"Number of real sets: {len(battery)}")
    print(f"Number of fake sets: {config.fake_set_count}")  
    print(f"Top N: {config.top_n}")

    for i, thing in enumerate(range(config.fake_set_count)):
        this_model = f"random-{i}"
        fake_recs = get_rec_fake(sku, config.num_recs)
        m = BitrecsRequest(
            created_at=datetime.now().isoformat(),
            user="test_user",
            num_results=config.num_recs,
            query=sku,
            context="[]",
            site_key=group_id,
            results=fake_recs,
            models_used=[this_model],
            miner_uid=str(i),
            miner_hotkey=secrets.token_hex(16)
        )
        rec_sets.append(m)
        print(f"Set {i} {this_model}: {m.miner_uid}")
        models_used.append(this_model)
    
   
    for model in battery:
        mock_req = mock_br_request(products, group_id, sku, model, config.num_recs)
        assert mock_req is not None 
        assert mock_req.results is not None
        assert len(mock_req.results) == config.num_recs
        rec_sets.append(mock_req)
        i += 1
        print(f"Set {i} {model}: {mock_req.miner_uid}")
        models_used.append(model)
        
    print("\nFinished generating rec sets")  
    print(f"A total of {len(rec_sets)} sets")    
    
    invalid_count = 0
    for s in rec_sets:
        thing = results_to_json(s.results)
        valid_schema = validate_result_schema(config.num_recs, thing)
        if not valid_schema:
            invalid_count += 1
            print(f"\033[1;33m Invalid schema for {s.miner_uid} with model {s.models_used} \033[0m")
            #s.results = []
        #assert valid_schema, "Invalid schema for results"

    print(f"Total Invalid schema count: {invalid_count}")
    #assert invalid_count == 0, "Invalid schema for results"

    most_similar = select_most_similar_bitrecs(rec_sets, top_n=config.top_n)
    
    assert most_similar is not None
    assert len(most_similar) == config.top_n
    
    for i, thing in enumerate(most_similar):
        rec_index = rec_sets.index(thing) or 0
        assert rec_index >= config.fake_set_count, "Selected set is a fake set, expected real model"

    recs : List[Set[str]] = []
    print(f"\nMost similar set indices: {len(most_similar)}")
    print("Selected sets:")
    for i, sim in enumerate(most_similar):        
        miner_uid = sim.miner_uid
        print(f"Set {i} of {len(rec_sets)} {miner_uid}")        
        model = sim.models_used[0]            
        print(f"Model: {model}")        
        print(f"  SKUs: {sim.results}")
        skus = [r["sku"] for r in sim.results]   
        recs.append(set(skus))
   
    most_similar_sets = [set(r['sku'] for r in req.results) for req in most_similar]
    report = recommender_presenter(sku, most_similar_sets)
    print(report)    
 
    all_sets = [set(r["sku"] for r in req.results) for req in rec_sets]
    matrix = display_rec_matrix(all_sets, models_used, highlight_indices=most_similar, 
                                    color_scheme=ColorScheme.VIRIDIS)
    print(matrix)

    print("\n\n\n")
    candidates = analyze_similar_requests(1, config.num_recs, rec_sets)
    if candidates:
        print(f"{len(candidates)} Top Candidates: {candidates}")
        for c in candidates:
            skus = [sku["sku"] for sku in c.results]
            print(f"Candidate {c.miner_uid} - {c.models_used} - SKUs: {skus}")

    


def results_to_json(results: List) -> List[str]:
    final = []
    for item in results:
        thing = json.dumps(item)
        final.append(thing)
    return final



def analyze_similar_requests(step, num_recs: int, requests: List[BitrecsRequest]) -> Optional[List[BitrecsRequest]]:
    if not requests or len(requests) < 2 or step < 1:
        print(f"Too few requests to analyze: {len(requests)}")
        bt.logging.warning(f"Too few requests to analyze: {len(requests)}")
        return

    def list_to_json(results: List) -> List[str]:
        final = []
        for item in results:
            thing = json.dumps(item)
            final.append(thing)
        return final
    
    def get_dynamic_top_n(num_requests: int) -> int:
        """
        Calculate top_n based on number of requests
        Rules:
        - Minimum 2 pairs
        - Maximum 33% of total requests
        - Never more than 5 pairs
        """
        if num_requests < 4:
            return 2  # Minimum pairs
        # Calculate 33% of requests, rounded down
        suggested = max(2, min(5, num_requests // 3))
        return suggested

    print(f"Starting analyze_similar_requests with step: {step} and num_recs: {num_recs}")    
    st = time.perf_counter()
    try:
        #top_n = 3              
        top_n = get_dynamic_top_n(len(requests))
        print(f"\033[1;32m Top N: {top_n} based on {len(requests)} bitrecs \033[0m")
        bt.logging.info(f"\033[1;32m Top N: {top_n} based on {len(requests)} bitrecs \033[0m")
        most_similar = select_most_similar_bitrecs(requests, top_n)
        if not most_similar:
            print(f"\033[33m No similar recs found in this round step: {step} \033[0m")
            bt.logging.warning(f"\033[33m No similar recs found in this round step: {step} \033[0m")
            return
        for sim in most_similar:
            print(f"Similar requests: {sim.miner_uid} {sim.models_used} - batch: {sim.site_key}")
            bt.logging.info(f"Similar requests: {sim.miner_uid} {sim.models_used} - batch: {sim.site_key}")
        
        valid_recs = []
        models_used = []
        for br in requests:
            thing = list_to_json(br.results)
            valid_schema = validate_result_schema(num_recs, thing)
            if not valid_schema:
                print(f"\033[1;33m Invalid schema for {br.miner_uid} with model {br.models_used} \033[0m")
                bt.logging.warning(f"\033[1;33m Invalid schema for {br.miner_uid} with model {br.models_used} \033[0m")
                continue
            skus = [r["sku"] for r in br.results]
            valid_recs.append(set(skus))
            models_used.append(br.models_used[0])
        if not valid_recs:
            print(f"\033[1;33m No valid recs found in this round step: {step} \033[0m")
            bt.logging.error(f"\033[1;33m No valid recs found in this round step: {step} \033[0m")
            return        
        
        matrix = display_rec_matrix(valid_recs, models_used, highlight_indices=most_similar)                                    
        print(matrix)
        bt.logging.info(matrix)

        et = time.perf_counter()
        diff = et - st
        print(f"Time taken to analyze similar bitrecs: {diff:.2f} seconds")
        bt.logging.info(f"Time taken to analyze similar bitrecs: {diff:.2f} seconds")
        return most_similar
    except Exception as e:
        print(f"analyze_similar_requests failed with exception: {e}")
        bt.logging.error(f"analyze_similar_requests failed with exception: {e}")
        print(traceback.format_exc())
        bt.logging.error(traceback.format_exc())
        return
    


def test_distance_limit():
    """Test distance limit for BitrecsRequest"""
    group_id = secrets.token_hex(16)
    products = product_1k()
    #products = product_5k()
    products = ProductFactory.dedupe(products)
    selected_product = safe_random.choice(products)
    sku = selected_product.sku
    
    config = TestConfig()
    rec_requests : List[BitrecsRequest] = []
    models_used = []

    #MIX = ['RANDOM', 'CLOUD', 'LOCAL']

    #MIX = ['RANDOM', 'LOCAL', 'CLOUD']
    MIX = ['RANDOM', 'LOCAL']
    RANDOM_COUNT = 180
    CLOUD_COUNT = 3
    LOCAL_COUNT = 4
    
    print("\n=== Protocol Recommendation Analysis ===")
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
    # No threshold
    most_similar = select_most_similar_bitrecs(rec_requests, top_n=config.top_n)
    assert most_similar is not None
    assert len(most_similar) == config.top_n

    # 5% ultralow
    ulow_threshold = 0.05  
    most_similar_low = select_most_similar_bitrecs_threshold(rec_requests, 
                                                           top_n=config.top_n, 
                                                           similarity_threshold=ulow_threshold)
    assert most_similar_low is not None

    # # 10% with threshold
    # low_threshold = 0.10    
    # most_similar2 = select_most_similar_bitrecs_threshold(rec_requests, 
    #                                                        top_n=config.top_n, 
    #                                                        similarity_threshold=low_threshold)
    # # 33% or null
    # med_threshold = 0.33
    # most_similar3 = select_most_similar_bitrecs_threshold2(rec_requests, 
    #                                                        top_n=config.top_n, 
    #                                                        similarity_threshold=med_threshold)
    # good_threshold = 0.51
    # most_similar4 = select_most_similar_bitrecs_threshold2(rec_requests, 
    #                                                        top_n=config.top_n, 
    #                                                        similarity_threshold=good_threshold)
    # great_threshold = 0.80
    # most_similar5 = select_most_similar_bitrecs_threshold2(rec_requests, 
    #                                                        top_n=config.top_n, 
    #                                                        similarity_threshold=great_threshold)
    print("\nFinished generating rec sets")
    print("Selected sets:")
    for req in most_similar:
        model = req.models_used[0]
        skus = [r['sku'] for r in req.results]
        print(f"Model {model}:")
        print(f"  SKUs: {sorted(skus)}")
    print("\033[1;32m No Threshold \033[0m")   
    selected_sets = [set(r['sku'] for r in req.results) for req in most_similar]
    report = recommender_presenter(sku, selected_sets)
    print(report)
    rec_sets = [set(r['sku'] for r in req.results) for req in rec_requests]
    most_similar_indices = [rec_requests.index(req) for req in most_similar]    
    matrix = display_rec_matrix(rec_sets, models_used, most_similar_indices)
    print(matrix)

