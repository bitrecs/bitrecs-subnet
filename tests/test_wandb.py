import json
import time
import wandb
import secrets
import bittensor as bt
from typing import Dict, Any, Optional, List, Set
from bitrecs.commerce.product import CatalogProvider, Product, ProductFactory
from bitrecs.llms.factory import LLMFactory
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.protocol import BitrecsRequest
from bitrecs.utils.distance import (    
    calculate_jaccard_distance, 
    display_rec_matrix, 
    display_rec_matrix_html, 
    select_most_similar_bitrecs
)
from bitrecs.utils.color import ColorScheme
from bitrecs.utils.misc import ttl_cache
from dataclasses import asdict
from datetime import datetime
from random import SystemRandom
safe_random = SystemRandom()
from dotenv import load_dotenv
load_dotenv()


DEBUG_ALL_PROMPTS = False
_FIRST_GET_MOCK_REC = True
_FIRST_GET_REC = True

LOCAL_OLLAMA_URL = "http://10.0.0.40:11434/api/chat"
WARMUP_OLLAMA_MODEL = "mistral-nemo"

MODEL_BATTERY = [ "mistral-nemo", "phi4", "gemma3:12b", "qwen2.5:14b", "llama3.1" ]

#MODEL_BATTERY = ["llama3.1:70b-instruct-q4_0", "qwen2.5:32b", "gemma3:27b", "nemotron:latest"]

#MODEL_BATTERY = ["qwen2.5:32b", "gemma3:27b", "nemotron:latest", "phi4"]

# CLOUD_BATTERY = ["deepseek/deepseek-chat-v3-0324",
#                  "amazon/nova-lite-v1", "google/gemini-flash-1.5-8b",
#                  "x-ai/grok-2-1212", "openai/chatgpt-4o-latest", "anthropic/claude-2.1",
#                  "google/gemini-2.0-flash-001", "anthropic/claude-3.7-sonnet",
#                  "openai/gpt-4o-mini-search-preview", "openai/gpt-4o-mini", "qwen/qwen-turbo"]

CLOUD_BATTERY = ["amazon/nova-lite-v1", "google/gemini-flash-1.5-8b", "google/gemini-2.0-flash-001",
                 "x-ai/grok-2-1212", "qwen/qwen-turbo", "openai/gpt-4o-mini"]


class WandbHelper:
    def __init__(
        self,
        project_name: str,
        entity: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None
    ):
        """
        Initialize WandB tracking
        """      
        self.default_config = {
            "network": "testnet",
            "neuron_type": "validator",
            "sample_size": 5,
            "num_concurrent_forwards": 1,
            "vpermit_tao_limit": 1024,
            "run_name": f"validator_{wandb.util.generate_id()}"
        }
        
        if config:
            self.default_config.update(config)

        try:
            self.run = wandb.init(
                project=project_name,
                entity=entity,
                config=self.default_config,
                tags=tags,
                reinit=True
            )
        except Exception as e:
            bt.logging.error(f"Error initializing wandb: {e}")
            self.run = None
        
    def log_weights(self, step: int, weights: Dict[str, float], prefix: str = "weights"):
        """
        Log weight updates to wandb
        """
        try:
            metrics = {f"{prefix}/{k}": v for k, v in weights.items()}
            metrics["step"] = step
            wandb.log(metrics)
        except Exception as e:
            bt.logging.error(f"Error logging weights to wandb: {e}")

    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log arbitrary metrics to wandb
        """
        try:
            wandb.log(metrics)
        except Exception as e:
            bt.logging.error(f"Error logging metrics to wandb: {e}")
    
    def finish(self):
        """
        Close wandb run and clear run reference
        """
        if self.run:
            self.run.finish()
            self.run = None  # Clear the run reference after finishing

    def log_cluster_metrics(
        self,
        step: int,
        most_similar: Optional[List[BitrecsRequest]],
        valid_recs: List[BitrecsRequest],
        models_used: List[str], 
        matrix: str,
        analysis_time: float
    ):
        """Log cluster metrics including similarity matrix to WandB"""
        if not self.run:
            bt.logging.warning("WandB not initialized, skipping logging")
            return

        try:
            # Basic metrics
            metrics = {
                "step": step,
                "analysis_time": analysis_time,
                "num_valid_requests": len(valid_recs),
                "num_similar_clusters": len(most_similar) if most_similar else 0,
            }
            self.log_metrics(metrics)

            # Log HTML matrix display
            wandb.log({
                "matrix_display": wandb.Html(f"{matrix}"),
            }, step=step)

            if not most_similar:
                return

            # Convert BitrecsRequests to sets of SKUs for distance calculation
            valid_sets = [set(r['sku'] for r in req.results) for req in valid_recs]
            n = len(valid_sets)  # Get actual size

            # Create complete n x n matrix
            matrix_data = []
            for i in range(n):
                row = []
                for j in range(n):
                    if j >= i:  # Upper triangle and diagonal
                        row.append("-")
                    else:  # Lower triangle
                        distance = calculate_jaccard_distance(valid_sets[i], valid_sets[j])
                        row.append(f"{distance:.3f}")
                matrix_data.append(row)

            # Log the distance matrix
            columns = [f"Set_{i}" for i in range(n)]  # Make n columns
            table = wandb.Table(
                columns=columns,
                data=matrix_data
            )
            wandb.log({"similarity_matrix": table}, step=step)

            # Log cluster pairs
            cluster_table = wandb.Table(
                columns=["miner_uid", "model_used", "site_key", "distance"]
            )
            for i in range(len(most_similar)-1):
                for j in range(i+1, len(most_similar)):
                    # Get indices in valid_sets
                    idx1 = valid_recs.index(most_similar[i])
                    idx2 = valid_recs.index(most_similar[j])
                    # Calculate distance
                    distance = calculate_jaccard_distance(valid_sets[idx1], valid_sets[idx2])
                    
                    # Log both miners in the pair
                    cluster_table.add_data(
                        most_similar[i].miner_uid,
                        most_similar[i].models_used[0] if most_similar[i].models_used else "unknown",
                        most_similar[i].site_key,
                        f"{distance:.3f}"
                    )
                    cluster_table.add_data(
                        most_similar[j].miner_uid,
                        most_similar[j].models_used[0] if most_similar[j].models_used else "unknown", 
                        most_similar[j].site_key,
                        f"{distance:.3f}"
                    )
            wandb.log({"clusters": cluster_table}, step=step)

            # Log model distribution
            model_counts = {}
            for model in models_used:
                model_counts[model] = model_counts.get(model, 0) + 1
            wandb.log({
                "model_distribution": wandb.Histogram(
                    list(model_counts.values()),
                    num_bins=len(model_counts)
                )
            }, step=step)

        except Exception as e:
            bt.logging.error(f"Error logging cluster metrics to WandB: {e}")



@ttl_cache(ttl=900)
def product_1k() -> List[Product]:
    walmart_catalog = "./tests/data/walmart/wallmart_1k_kaggle_trimmed.csv" 
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WALMART, walmart_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WALMART)    
    return products

@ttl_cache(ttl=900)
def product_5k() -> List[Product]:
    walmart_catalog = "./tests/data/walmart/wallmart_5k_kaggle_trimmed.csv"
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WALMART, walmart_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WALMART)
    return products

@ttl_cache(ttl=900)
def product_20k() -> List[Product]:
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


def recommender_presenter(catalog: List[Product], original_sku: str, recs: List[Set[str]]) -> str:    
    result = f"Target SKU: \033[32m {original_sku} \033[0m\n"
    target_product_name = product_name_by_sku_trimmed(original_sku, 200, catalog)
    result += f"Target Product:\033[32m{target_product_name} \033[0m\n"
    result += "------------------------------------------------------------\n"    
    # Track matches with simple counter
    matches = {}  # name -> count    
    # First pass - count matches
    for rec_set in recs:
        for rec in rec_set:
            name = product_name_by_sku_trimmed(rec, 200, catalog)
            matches[name] = matches.get(name, 0) + 1
    
    # Second pass - output with emphasis on matches
    seen = set()
    for rec_set in recs:
        for rec in rec_set:
            name = product_name_by_sku_trimmed(rec, 200, catalog)
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



class TestConfig:
    similarity_threshold: float = 0.33
    top_n: int = 2
    num_recs: int = 6
    real_set_count: int = len(MODEL_BATTERY)
    fake_set_count: int = 9



def test_wandb_cluster_logging():
    """
    Test WandB cluster logging functionality.
    """

    group_id = secrets.token_hex(16)
    products = product_1k()
    #products = product_5k()
    products = ProductFactory.dedupe(products)
    selected_product = safe_random.choice(products)
    sku = selected_product.sku
    config = TestConfig()
    rec_requests : List[BitrecsRequest] = []
    models_used = []
   
    COLORS = [ColorScheme.VIRIDIS, ColorScheme.MAKOTO, ColorScheme.ROCKET, ColorScheme.SPECTRAL]
    
    #MIX = ['RANDOM']    
    MIX = ['RANDOM', 'CLOUD', 'LOCAL']
    RANDOM_COUNT = 5
    CLOUD_COUNT = 3
    LOCAL_COUNT = 3
        
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

    st = time.perf_counter()
    valid_recs = [set([r['sku'] for r in rec.results]) for rec in rec_requests]
    most_similar = select_most_similar_bitrecs(rec_requests, top_n=config.top_n)
    if not most_similar:
        print("No similar clusters found.")
        return    
    for sim in most_similar:
        print(f"Miner UID: {sim.miner_uid}, Site Key: {sim.site_key}, Models Used: {sim.models_used}")
        

    #Matrix display
    most_similar_indices = [rec_requests.index(req) for req in most_similar]    
    scheme = safe_random.choice(COLORS)
    matrix = display_rec_matrix(valid_recs, models_used, 
                                highlight_indices=most_similar_indices, color_scheme=scheme)
    print("Similarity Matrix:")
    print(matrix)

    html_matrix = display_rec_matrix_html(valid_recs, 
                                          models_used,
                                        highlight_indices=most_similar_indices)

    selected_sets = [set(r['sku'] for r in req.results) for req in most_similar]
    report = recommender_presenter(products, sku, selected_sets)
    print(report)

    et = time.perf_counter()    
    analysis_time = et - st
    print(f"Analysis time: {analysis_time} seconds")


    for sim in most_similar:
        for m in sim.models_used:
            assert "random" not in m, "Random model found in similar clusters"

    wandb_project = "bitrecs_localnet"
    wandb_entity = "bitrecs"  
    wandb_config = {
        "network": "testnet",
        "neuron_type": "validator_test",
        "sample_size": str(len(rec_requests)),
        "num_concurrent_forwards": 1,
        "vpermit_tao_limit": 1024,
        "run_name": f"validator_{wandb.util.generate_id()}"
    }
    
    wandb_helper = WandbHelper(
        project_name=wandb_project,
        entity=wandb_entity,
        config=wandb_config,
        tags=["test"]
    )
    assert wandb_helper.run is not None, "WandB run initialization failed"
    assert wandb_helper.run.project == wandb_project, "Project name mismatch"  
    assert wandb_helper.run.entity == wandb_entity, "Entity name mismatch"
    
    wandb_helper.log_cluster_metrics(
        step=1,
        most_similar=most_similar,
        valid_recs=rec_requests,
        models_used=models_used,
        matrix=html_matrix,
        analysis_time=analysis_time
    )
    
    wandb_helper.finish()
    assert wandb_helper.run is None, "WandB run not finished properly"

  