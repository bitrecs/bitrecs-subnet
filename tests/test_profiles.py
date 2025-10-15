import json
import random
from dataclasses import asdict
from bitrecs.llms.factory import LLMFactory
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.utils.misc import ttl_cache
from bitrecs.commerce.user_profile import UserProfile
from bitrecs.commerce.product import CatalogProvider, ProductFactory
from dotenv import load_dotenv
load_dotenv()

TEST_PROFILE = """{"id":"123","created_at":"2025-05-31T18:45:13Z",
"cart":[{"sku":"24-WB02","name":"Compete Track Tote","price":"32"},{"sku":"24-WG087","name":"Sprite Yoga Strap 10 foot","price":"21"}],
"orders":[
{"order_id":"3354","total":"59.89","status":"checkout-draft","created_at":"2025-05-31T18:34:02Z","items":[{"sku":"24-WB02","name":"Compete Track Tote","price":"32"},{"sku":"24-WG087","name":"Sprite Yoga Strap 10 foot","price":"21"}]},
{"order_id":"3265","total":"48.59","status":"processing","created_at":"2025-03-23T18:12:15Z","items":[{"sku":"24-WB03","name":"Driven Backpack","price":"36"},{"sku":"24-UG06","name":"Affirm Water Bottle","price":"7"}]},
{"order_id":"3263","total":"275.72","status":"processing","created_at":"2025-03-19T14:23:21Z","items":[{"sku":"24-UG03","name":"Harmony Lumaflex™ Strength Band Kit","price":"22"},{"sku":"24-MB03","name":"Crown Summit Backpack","price":"38"},{"sku":"24-WG02","name":"Didi Sport Watch","price":"92"}]},
{"order_id":"3019","total":"134.47","status":"processing","created_at":"2024-12-31T02:46:19Z","items":[{"sku":"WS07-S-Black","name":"Juliana Short-Sleeve Tee - S, Black","price":"42"},{"sku":"MP01-33-Black","name":"Caesar Warm-Up Pant - 33, Black","price":"35"}]},
{"order_id":"3010","total":"101.70","status":"processing","created_at":"2024-12-30T21:12:15Z","items":[{"sku":"WP07-28-Black","name":"Aeon Capri - 28, Black","price":"48"},{"sku":"WH05-XS-Orange","name":"Selene Yoga Hoodie - XS, Orange","price":"42"}]},
{"order_id":"3006","total":"36.16","status":"processing","created_at":"2024-12-30T20:56:01Z","items":[{"sku":"WJ09-S-Blue","name":"Jade Yoga Jacket - S, Blue","price":"32"}]},
{"order_id":"3004","total":"83.62","status":"processing","created_at":"2024-12-30T20:48:19Z","items":[{"sku":"MH13-XS-Blue","name":"Marco Lightweight Active Hoodie - XS, Blue","price":"74"}]},
{"order_id":"2998","total":"77.97","status":"processing","created_at":"2024-12-30T15:32:04Z","items":[{"sku":"MH09-XS-Blue","name":"Abominable Hoodie - XS, Blue","price":"69"}]},
{"order_id":"2994","total":"77.97","status":"processing","created_at":"2024-12-30T15:00:07Z","items":[{"sku":"MH09-S-Blue","name":"Abominable Hoodie - S, Blue","price":"69"}]},
{"order_id":"2992","total":"47.46","status":"processing","created_at":"2024-12-30T02:47:12Z","items":[{"sku":"WH05-XS-Orange","name":"Selene Yoga Hoodie - XS, Orange","price":"42"}]}
],"site_config":{"profile":"ecommerce_retail_store_manager"}}"""


@ttl_cache(ttl=900)
def product_woo():
    woo_catalog = "./tests/data/woocommerce/product_catalog.csv" #2038 records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, woo_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)
    return products


def test_parse_profile_str():
    #profile_str = '{"id": "123", "created_at": "2025-05-01T12:00:00Z", "cart": [], "orders": [], "site_config": {"profile": "ecommerce_retail_store_manager"} }'
    profile_str = TEST_PROFILE
    profile = UserProfile.tryparse_profile(profile_str)
    assert isinstance(profile, UserProfile)
    assert profile.id == "123"
    assert profile.created_at == "2025-05-31T18:45:13Z"
    assert len(profile.cart) == 2
    assert len(profile.orders) == 10
    assert profile.site_config == {"profile": "ecommerce_retail_store_manager"}


def test_parse_profile_dict():
    profile_dict = {
        "id": "456",
        "created_at": "2025-05-01T12:00:00Z",
        "cart": [],
        "orders": [],
        "site_config": {"profile": "ecommerce_retail_store_manager"}
    }
    profile = UserProfile.tryparse_profile(profile_dict)    
    assert isinstance(profile, UserProfile)
    assert profile.id == "456"
    assert profile.created_at == "2025-05-01T12:00:00Z"
    assert profile.cart == []
    assert profile.orders == []
    assert profile.site_config == {"profile": "ecommerce_retail_store_manager"}


def test_parse_profile_with_cart():
    cart_count = 5
    cart = product_woo()[:cart_count]
    cart_dicts = [product.to_dict() for product in cart]

    profile_dict = {
        "id": "123", 
        "created_at": "2025-05-01T12:00:00Z", 
        "cart": cart_dicts, 
        "orders": [], 
        "site_config": {"profile": "default"}
    }

    profile_str = json.dumps(profile_dict)    
    profile = UserProfile.tryparse_profile(profile_str)
    
    assert isinstance(profile, UserProfile)
    assert profile.id == "123"
    assert profile.created_at == "2025-05-01T12:00:00Z"
    assert len(profile.cart) == cart_count
    assert profile.orders == []
    assert profile.site_config == {"profile": "default"}


def test_parse_profile_invalid():
    invalid_profile = "This is not a valid profile"
    profile = UserProfile.tryparse_profile(invalid_profile)    
    assert profile is None
    invalid_dict = {"invalid_key": "value"}
    profile = UserProfile.tryparse_profile(invalid_dict)    
    assert profile is None


def test_persona_parse_default():
    thing = PromptFactory.PERSONAS["ecommerce_retail_store_manager"]    
    assert thing["description"] is not None
    assert thing["tone"] is not None
    assert thing["response_style"] is not None
    assert thing["priorities"] is not None


def test_profile_load_cart_orders_in_prompt_factory():
    profile = UserProfile.tryparse_profile(TEST_PROFILE)
    products = product_woo()   
    context = json.dumps([asdict(products) for products in products], separators=(',', ':'))

    viewing_product = random.choice(products)
    user_prompt = viewing_product.sku

    factory = PromptFactory(sku=user_prompt,
                            context=context, 
                            num_recs=5,
                            profile=profile,
                            debug=False)   

    prompt = factory.generate_prompt()
    print(f"Prompt: {prompt}")

    tc = factory.get_token_count(prompt)
    print(f"Token count: {tc}")    
    
    assert factory.profile is not None    
    assert factory.profile.id == "123"
    assert factory.profile.created_at == "2025-05-31T18:45:13Z"
    assert factory.profile.site_config == {"profile": "ecommerce_retail_store_manager"}    
    assert factory.num_recs == 5
    assert len(factory.profile.cart) == 2
    assert len(factory.profile.orders) == 10   
    

def test_profile_call_local_llm_cart_not_in_recs():
    profile = UserProfile.tryparse_profile(TEST_PROFILE)
    products = product_woo()   
    context = json.dumps([asdict(products) for products in products], separators=(',', ':'))

    viewing_product = random.choice(products)
    user_prompt = viewing_product.sku
    num_recs = 5
    factory = PromptFactory(sku=user_prompt,
                            context=context, 
                            num_recs=num_recs,
                            profile=profile,
                            debug=False)
    

    assert factory.profile is not None    
    assert factory.profile.id == "123"
    assert factory.profile.created_at == "2025-05-31T18:45:13Z"
    assert factory.profile.site_config == {"profile": "ecommerce_retail_store_manager"}    
    assert factory.num_recs == 5
    assert len(factory.profile.cart) == 2
    assert len(factory.profile.orders) == 10   

    prompt = factory.generate_prompt()
    #print(f"Prompt: {prompt}")

    tc = factory.get_token_count(prompt)
    print(f"Token count: {tc}")    

    model = "mistral-nemo"
    llm_response = LLMFactory.query_llm(server=LLM.OLLAMA_LOCAL,
                                 model=model, 
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)
    #print(llm_response)    
    parsed_recs = PromptFactory.tryparse_llm(llm_response)   
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)
    
    assert len(parsed_recs) == num_recs
    
    for rec in parsed_recs:
        sku = rec['sku']        
        assert sku not in [product['sku'] for product in profile.cart], f"SKU {sku} should not be in cart"
        
    


