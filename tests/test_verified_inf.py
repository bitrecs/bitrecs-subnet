import os
import httpx
os.environ["NEST_ASYNCIO"] = "0"
import base64
import json
import time
import pytest    
from bitrecs.protocol import SignedResponse
from dataclasses import asdict
from random import SystemRandom
safe_random = SystemRandom()
from typing import Counter
from bitrecs.commerce.product import CatalogProvider, ProductFactory
from bitrecs.llms.factory import LLM, LLMFactory
from bitrecs.llms.prompt_factory import PromptFactory
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from dotenv import load_dotenv
load_dotenv()

VERIFIED_URL = "https://verified.bitrecs.ai"

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

@pytest.mark.asyncio
async def test_openrouter_verified_inf():
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
    
    #model = "ai21/jamba-mini-1.7"    
    #model = "qwen/qwen3-next-80b-a3b-instruct"
    model = "x-ai/grok-4-fast:free"
    provider = LLM.OPEN_ROUTER

    #model = "gpt-4.1-nano"
    #model = "gpt-5-nano"
    #provider = LLM.CHAT_GPT
    
    print(f"\033[32mTesting {provider} with model: {model} \033[0m")
    st = time.time()

    hotkey = "ASLJALSJASLKFJALSKJDFK"
    llm_response = LLMFactory.query_llmv(server=provider,
                                 model=model,
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, 
                                 user_prompt=prompt, 
                                 miner_hotkey=hotkey,
                                 use_verified_inference=True)
    et = time.time()
    diff = et - st  
    print(f"LLM response time: {diff:.2f} seconds")

    public_key = await get_public_key()
    response = llm_response.signed_response
    assert verify_signature(response, public_key), "Signature verification failed"
    print(f"\033[32mSignature verification succeeded \033[0m")

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


async def get_public_key() -> Ed25519PublicKey:
    """Get public key from proxy server."""    
    async with httpx.AsyncClient(timeout=30.0) as client:
        public_key_response = await client.get(f"{VERIFIED_URL}/public_key")
        public_key_response.raise_for_status()
        public_key_string = json.loads(public_key_response.text)["public_key"]
        raw_bytes = bytes.fromhex(public_key_string)
        return Ed25519PublicKey.from_public_bytes(raw_bytes)    


def verify_signature(
    response:  SignedResponse, 
    public_key: Ed25519PublicKey
) -> bool:
    """Verify the signature of the response."""
    proof = response.proof
    signature_b64 = response.signature

    print(f"Proof: {proof}")
    print(f"Signature (base64): {signature_b64}")

    signature_bytes = base64.b64decode(signature_b64)
    serialized_proof = json.dumps(proof, sort_keys=True).encode()
    try:
        public_key.verify(signature_bytes, serialized_proof)
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False