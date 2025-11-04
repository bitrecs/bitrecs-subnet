import os
import httpx
os.environ["NEST_ASYNCIO"] = "0"
import json
import time
import pytest    
import bittensor as bt
from bitrecs.protocol import SignedResponse
from dataclasses import asdict
from random import SystemRandom
safe_random = SystemRandom()
from typing import Counter
from bitrecs.commerce.product import CatalogProvider, ProductFactory
from bitrecs.llms.factory import LLMFactory
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.prompt_factory import PromptFactory
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from dotenv import load_dotenv
load_dotenv()
from bitrecs.utils import constants as CONST
from bitrecs.validator.reward import validate_proof_skus, verify_proof_with_recs

VERIFIED_URL = CONST.VERIFIED_INFERENCE_URL
TEST_WALLET = bt.Wallet(name=os.getenv("TEST_WALLET_NAME", "test_wallet_1"))
 

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



# @pytest.mark.asyncio
# async def test_openrouter_verified_inf():
#     raw_products = product_woo()    
#     #raw_products = product_shopify()
#     products = ProductFactory.dedupe(raw_products)    
#     rp = safe_random.choice(products)
#     user_prompt = rp.sku    
#     num_recs = safe_random.choice([3, 4, 5])    
#     debug_prompts = False

#     match = [products for products in products if products.sku == user_prompt][0]
#     print(match)
#     print(f"\033[32mSelected product: {match.sku} - {match.name} \033[0m")

#     context = json.dumps([asdict(products) for products in products])
#     factory = PromptFactory(sku=user_prompt, 
#                             context=context, 
#                             num_recs=num_recs,
#                             debug=debug_prompts)
    
#     prompt = factory.generate_prompt()    
#     print(f"PROMPT SIZE: {len(prompt)}") 
#     wc = PromptFactory.get_word_count(prompt)
#     print(f"word count: {wc}")
#     tc = PromptFactory.get_token_count(prompt)
#     print(f"token count: {tc}")    
    
#     #model = "ai21/jamba-mini-1.7"    
#     #model = "qwen/qwen3-next-80b-a3b-instruct"
#     #model = "x-ai/grok-4-fast:free"
#     #model = "x-ai/grok-4-fast"
#     #model = "google/gemini-2.5-flash-lite-preview-09-2025"
#     #model = "anthropic/claude-3-haiku"  #claude-3-haiku-20240307
#     #model = "deepseek/deepseek-v3.2-exp" timeout
#     #model = "meituan/longcat-flash-chat"
#     #model = "nvidia/nemotron-nano-9b-v2"
#     #model = "meta-llama/llama-4-maverick"
#     #model = "meta-llama/llama-4-scout"
#     #model = "qwen/qwen-turbo"
#     #model = "amazon/nova-lite-v1"
#     #model = "z-ai/glm-4.6"
#     #model = "moonshotai/kimi-k2-0905"
#     #model = "inclusionai/ling-1t" #timeout
#     #model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
#     #model = "qwen/qwen3-vl-8b-instruct"
#     #provider = LLM.OPEN_ROUTER

#     #model = "gpt-4.1-nano"
#     #model = "gpt-5-nano"
#     #provider = LLM.CHAT_GPT

#     model = "gemini-2.5-flash-lite-preview-09-2025"
#     provider = LLM.GEMINI

#     #model = "grok-4-fast-non-reasoning"
#     #provider = LLM.GROK
    
#     #model = "claude-haiku-4-5-20251001"
#     #provider = LLM.CLAUDE

#     #model = "zai-org/GLM-4.5-Air"
#     #model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
#     #provider = LLM.CHUTES
    
#     print(f"\033[32mTesting {provider} with model: {model} \033[0m")
#     st = time.time()
    
#     llm_response = LLMFactory.query_llmv(server=provider,
#                                 model=model,
#                                 system_prompt="You are a helpful assistant", 
#                                 temp=0.0, 
#                                 user_prompt=prompt,
#                                 miner_wallet=TEST_WALLET,
#                                 use_verified_inference=True)                                 
                                 
#     et = time.time()
#     diff = et - st  
#     print(f"LLM response time: {diff:.2f} seconds")

#     # Local Verification
#     public_key = await get_public_key()
#     response = llm_response.signed_response
#     assert verify_proof(response, public_key), "Signature verification failed"
#     print(f"\033[32mLOCAL Signature verification succeeded \033[0m")

#     # Remote verification
#     assert remote_verify_proof(response), "Remote signature verification failed"
#     print(f"\033[32mREMOTE signature verification succeeded \033[0m")

#     parsed_recs = PromptFactory.tryparse_llm(llm_response.results)
#     print(f"parsed {len(parsed_recs)} records")
#     print(parsed_recs)
#     assert len(parsed_recs) == num_recs    
#     skus = [item['sku'] for item in parsed_recs]
#     counter = Counter(skus)
#     for sku, count in counter.items():
#         print(f"{sku}: {count}")
#         assert count == 1
#     assert user_prompt not in skus   
    




@pytest.mark.asyncio
async def test_verify_with_rec_match():
    raw_products = product_woo()    
    #raw_products = product_shopify()
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

    model = "gemini-2.5-flash-lite-preview-09-2025"
    provider = LLM.GEMINI   
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()
    
    llm_response = LLMFactory.query_llmv(server=provider,
                                model=model,
                                system_prompt="You are a helpful assistant", 
                                temp=0.0, 
                                user_prompt=prompt,
                                miner_wallet=TEST_WALLET,
                                use_verified_inference=True)                                 
                                 
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")        
    public_key = await get_public_key()
    response = llm_response.signed_response

    # # Remote verification
    assert remote_verify_proof(response), "Remote signature verification failed"
    print(f"\033[32mREMOTE signature verification succeeded \033[0m")

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

    skus2 = set([item['sku'] for item in parsed_recs])
    assert validate_proof_skus(skus2, response), "verify_proof_skus verification failed"
    print(f"\033[32mSKU set verification succeeded \033[0m")    

    # Local verification
    assert verify_proof_with_recs(skus2, response, public_key), "verify_proof_with_recs verification failed"
    print(f"\033[32mFull recommendation verification succeeded \033[0m")




async def get_public_key() -> Ed25519PublicKey:
    """Get public key from proxy server."""    
    async with httpx.AsyncClient(timeout=30.0) as client:
        public_key_response = await client.get(f"{VERIFIED_URL}/public_key")
        public_key_response.raise_for_status()
        public_key_string = json.loads(public_key_response.text)["public_key"]
        raw_bytes = bytes.fromhex(public_key_string)
        return Ed25519PublicKey.from_public_bytes(raw_bytes)
    

def remote_verify_proof(signed_response: SignedResponse) -> bool:
    """Remote verify the signature of the response."""
    print(f"RAW signed response: {signed_response}")
    payload = signed_response.model_dump()
    try:
        r = httpx.post(f"{VERIFIED_URL}/verify", json=payload, timeout=30.0)
        r.raise_for_status()
        result = r.json()
        print(f"Remote verification result: {result}")
        return result.get("valid", False)
    except Exception as e:
        print(f"Remote verification failed: {e}")
        return False