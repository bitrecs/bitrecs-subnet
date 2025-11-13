import os
os.environ["NEST_ASYNCIO"] = "0"
import json
import json_repair
import jsonschema
from dataclasses import asdict
from random import SystemRandom
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.commerce.product import CatalogProvider, Product, ProductFactory
from bitrecs.validator.reward import CatalogValidator, validate_result_schema
from bitrecs.utils.constants import RE_MODEL_NAME
safe_random = SystemRandom()


def test_basic_parsing():
    single_rec = '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}'
    single_rec2 = '{"sku": "24-UG02", "name": "Pursuit Lumaflex&trade; Tone Band", "price": "16.04"}'
    single_rec3 = '{"sku": "24-MG05", "name": "Cruise Dual Analog Watch", "price": "55.90"}'
    multi_rec = [single_rec, single_rec2, single_rec3]
    final_recs = [json.loads(idx.replace("'", '"')) for idx in multi_rec]
    print(final_recs)    
    assert len(multi_rec) == len(final_recs)    


def test_basic_parsing2():
    results =  ['{"sku": "24-WG088", "name": "Sprite Foam Roller"}',
                '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick"}',
                '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band"}',
                '{"sku": "24-UG05", "name": "Go-Get\'r Pushup Grips"}',
                '{"sku": "24-UG02", "name": "Pursuit Lumaflex&trade; Tone Band"}',
                '{"sku": "24-UG07", "name": "Dual Handle Cardio Ball"}']

    final = []
    for idx in results:        
        fixed1 = json_repair.repair_json(idx, logging=False)  
        print(f"fixed: {fixed1}")
        product = json_repair.loads(fixed1)        
        final.append(product)    
    print("FINAL RESULTS")
    print(final)
    assert len(results) == len(final)


def test_schema_validation():
    broken_json =  ['{"sku": "24-WG088", "name": "Sprite Foam Roller"}',
                    '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick"}',
                    '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band"}',
                    '{"sku": "24-UG05", "name": "Go-Get\'r Pushup Grips"}',
                    '{"sku": "24-UG02", "name": "Pursuit Lumaflex&trade; Tone Band"}',
                    '{"sku": "24-UG07", "name": "Dual Handle Cardio Ball"}']
    
    partial_json =  ['{"sku": "24-WG088", "name": "Sprite Foam Roller"}',
                     '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": 5.00}',
                     '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band"}',
                     '{"sku": "24-UG05", "name": "Go-Get\'r Pushup Grips"}',
                     '{"sku": "24-UG02", "name": "Pursuit Lumaflex&trade; Tone Band"}',
                     '{"sku": "24-UG07", "name": "Dual Handle Cardio Ball", "price": "19"}']
    
    good_json =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}', 
                  '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                  '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32.0"}', 
                  '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19.11"}', 
                  '{"sku": "24-UG05", "name": "Go-Get\'r Pushup Grips", "price": "19.00"}', 
                  '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']
    
    schema = {
        "type": "object",
        "properties": {
            "sku": {"type": "string"},
            "name": {"type": "string"},
            "price": {"type": "string"}
        },
        "required": ["sku", "name", "price"]
    }

    broken_count = 0
    for item in broken_json:
        try:
            thing = json_repair.loads(item)
            jsonschema.validate(thing, schema)
            broken_count += 1
        except json.decoder.JSONDecodeError:            
            continue
        except jsonschema.exceptions.ValidationError:
            continue
    #print(broken_count)
    assert broken_count == 0

    partial_count = 0
    for item in partial_json:
        try:
            thing = json_repair.loads(item)
            jsonschema.validate(thing, schema)
            partial_count += 1
        except json.decoder.JSONDecodeError:            
            continue
        except jsonschema.exceptions.ValidationError:
            continue
    
    assert partial_count == 1


    good_count = 0
    for item in good_json:
        try:
            thing = json_repair.loads(item)
            jsonschema.validate(thing, schema)
            good_count += 1
        except json.decoder.JSONDecodeError:
            continue
        except jsonschema.exceptions.ValidationError:            
            continue

    assert good_count == len(good_json)


def test_load_1k_raw():
    rows = []
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()
        rows = json_repair.loads(data)    
    print(f"loaded {len(rows)} records")
    assert len(rows) == 1000


def test_load_5k_raw():
    rows = []
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_5000.json", "r") as f:
        data = f.read()
        rows = json_repair.loads(data)    
    print(f"loaded {len(rows)} records")
    assert len(rows) == 5000


def test_parse_1k_into_products():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.try_parse_context(data)
    print(f"loaded {len(products)} records")    
    assert len(products) == 1000


def test_parse_5k_into_products():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_5000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.try_parse_context(data)
    print(f"loaded {len(products)} records")    
    assert len(products) == 5000


def test_parse_20k_into_products():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_20000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.try_parse_context(data)
    print(f"loaded {len(products)} records")    
    assert len(products) == 20000


def test_parse_1k_products_have_missing_fields():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.try_parse_context(data)
    print(f"loaded {len(products)} records")       
    assert len(products) == 1000

    broken = False
    for product in products:
        if not hasattr(product, "sku"):
            broken = True
            break
        if not hasattr(product, "name"):
            broken = True
            break
        if not hasattr(product, "price"):
            broken = True
            break

    assert broken # should be broken

            
def test_convert_1k_amazon_to_bitrecs():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    print(f"converted {len(products)} records")       
    assert len(products) == 907

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 61

    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False


def test_convert_5k_amazon_to_bitrecs():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_5000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    print(f"converted {len(products)} records")       
    assert len(products) == 4544

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 416

    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False


def test_convert_20k_amazon_to_bitrecs():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_20000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    print(f"converted {len(products)} records")       
    assert len(products) == 18088

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 3324

    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False


def test_convert_1k_woocommerce_to_bitrecs():
    woo_catalog = "./tests/data/woocommerce/product_catalog.csv" #2038 records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, woo_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)
    print(f"converted {len(products)} records")       
    assert len(products) == 2038

    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False


def test_convert_1k_shopify_to_bitrecs():
    shopify_catalog = "./tests/data/shopify/electronics/shopify_products.csv" #824 records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.SHOPIFY, shopify_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.SHOPIFY)
    print(f"converted {len(products)} records")
    assert len(products) == 359
    
    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 9

    products = ProductFactory.dedupe(products)
    assert len(products) == 350
  

def test_convert_30k_walmart_to_bitrecs():
    walmart_catalog = "./tests/data/walmart/wallmart_30k_kaggle_trimmed.csv" #30k records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WALMART, walmart_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WALMART)
    print(f"converted {len(products)} records")
    assert len(products) == 30000
    
    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 0

    products = ProductFactory.dedupe(products)
    assert len(products) == 30000

    sample = safe_random.sample(products, 10)
    for p in sample:
        print(f"{p.sku} - {p.name} - {p.price}")    
    


def test_product_factory_parse_all():
    products =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']
    context = json.dumps(products)
    result = ProductFactory.try_parse_context(context)
    assert len(result) == 5


def test_product_factory_parse_all_dataclass():
    products =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"skuere": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']
    context = json.dumps(products)
    print(f"context: {context}")  

    #regular json loads
    result : list[Product] = ProductFactory.try_parse_context(context)    
    assert len(result) == 5   

    #strict schmea  json loads
    result : list[Product] = ProductFactory.try_parse_context_strict(context)
    assert len(result) == 0 #sku not present in last record, entire context is rejected


def test_product_factory_parse_all_dataclass_from_dict():
    products =  [{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"},
                 {"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"},
                 {"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"},
                 {"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"},
                 {"skuere": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}]
    context = json.dumps(products)
    print(f"context: {context}")

    #regular json loads
    result : list[Product] = ProductFactory.try_parse_context(context)    
    assert len(result) == 5 

    #strict schmea  json loads
    result : list[Product] = ProductFactory.try_parse_context_strict(context)
    assert len(result) == 4 #sku not present in last record



def test_products_must_all_have_sku():
    products =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']

    sku_check = ProductFactory.check_all_have_sku(products)
    print(f"sku check: {sku_check}")
    assert sku_check is True


def test_products_must_all_have_sku_case_sensitive():
    products =  ['{"SkU": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']

    sku_check = ProductFactory.check_all_have_sku(products)
    print(f"sku check: {sku_check}")
    assert sku_check is False
    
    
def test_products_must_all_have_sku_no_upper_allowed():
    products =  ['{"SKU": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']

    sku_check = ProductFactory.check_all_have_sku(products)
    print(f"sku check: {sku_check}")
    assert sku_check is False   
    

def test_products_missing_sku_error():
    products =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']

    sku_check = ProductFactory.check_all_have_sku(products)
    print(f"sku check: {sku_check}")
    assert sku_check is False



def test_schema_validation_broken_testnet_json_03_03_2025():
    broken_json = ['{\'sku\': \'8772908155104\', \'name\': \'10" Table Top Selfie LED Lamp\', \'price\': \'46.74\', \'reason\': \'test\'}', 
    "{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29', 'reason': 'test'}", 
    "{'sku': '8772908450016', 'name': 'Galaxy Starry Sky Projector Rotating', 'price': '90.34', 'reason': 'test'}", 
    "{'sku': '8761138839776', 'name': 'beFree Sound Color LED Dual Gaming Speakers', 'price': '84.42', 'reason': 'test'}", 
    "{'sku': '8772908384480', 'name': 'Universal Wireless Charging Stand for Iphone Apple Watch Airpods', 'price': '40.33', 'reason': 'test'}", 
    '{\'sku\': \'8761139331296\', \'name\': \'Impress 16" Oscillating Stand Fan (black) IM-725B\', \'price\': \'56.91\', \'reason\': \'test\'}']

    is_valid = validate_result_schema(6, broken_json)
    assert is_valid is True
 

def test_schema_validation_broken_testnet_json_03_03_2025_2():
    broken_json = ['{\'sku\': \'8772908155104\', \'name\': \'10" Table Top Selfie LED Lamp\', \'price\': \'46.74\'}', 
    "{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29'}", 
    "{'sku': '8772908450016', 'name': 'Galaxy Starry Sky Projector Rotating', 'price': '90.34'}", 
    "{'sku': '8761138839776', 'name': 'beFree Sound Color LED Dual Gaming Speakers', 'price': '84.42'}", 
    "{'sku': '8772908384480', 'name': 'Universal Wireless Charging Stand for Iphone Apple Watch Airpods', 'price': '40.33'}", 
    '{\'sku\': \'8761139331296\', \'name\': \'Impress 16" Oscillating Stand Fan (black) IM-725B\', \'price\': \'56.91\'}']

    context = json.dumps(broken_json)
    products = ProductFactory.try_parse_context_strict(context)
    print(products)
    assert len(products) == 0
    

def test_schema_validation_broken_testnet_json_03_03_2025_4():
    broken_json = ['{\'sku\': \'8761139331296\', \'name\': \'Impress 16" Oscillating Stand Fan (black) IM-725B\', \'price\': \'56.91\'}', 
                   "{'sku': '8772909105376', 'name': 'Wireless Magnetic Charger And Power Bank For iPhone 12', 'price': '56.42'}", 
                   "{'sku': '8761139921120', 'name': 'HD 1080P Camera 360° Panoramic PTZ Wireless Wifi Camera', 'price': '57.33'}", 
                   "{'sku': '8772908712160', 'name': 'Watermelon iPhone Case', 'price': '24.17'}", 
                   "{'sku': '8761139101920', 'name': 'beFree Sound 2.0 Computer Gaming Speakers with LED RGB Lights', 'price': '87.01'}", 
                   "{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29'}"]

    context = json.dumps(broken_json)
    products = ProductFactory.try_parse_context_strict(context)
    print(products)
    assert len(products) == 0
    


def test_strict_parser_rejects_malformed_json_quotes():
    problematic_json = ['{\'sku\': \'8772908155104\', \'name\': \'10" Table Top Selfie LED Lamp\', \'price\': \'46.74\'}', 
    "{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29'}", 
    "{'sku': '8772908450016', 'name': 'Galaxy Starry Sky Projector Rotating', 'price': '90.34'}", 
    "{'sku': '8761138839776', 'name': 'beFree Sound Color LED Dual Gaming Speakers', 'price': '84.42'}", 
    "{'sku': '8772908384480', 'name': 'Universal Wireless Charging Stand for Iphone Apple Watch Airpods', 'price': '40.33'}", 
    '{\'sku\': \'8761139331296\', \'name\': \'Impress 16" Oscillating Stand Fan (black) IM-725B\', \'price\': \'56.91\'}']
    
    context = json.dumps(problematic_json)
    print(context)

    products = ProductFactory.try_parse_context_strict(context)
    
    # Verify specific rejections
    assert len(products) < len(problematic_json)
    
    # Verify surviving products have proper formatting
    for product in products:
        assert '"' not in product.sku  # No quotes in actual data
        assert "'" not in product.sku


def test_schema_validation_missing_reasoning():
    broken_json = ["{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29', 'reason': 'test'}", 
    "{'sku': '8772908450016', 'name': 'Galaxy Starry Sky Projector Rotating', 'price': '90.34', 'reason': 'test'}", 
    "{'sku': '8761138839776', 'name': 'beFree Sound Color LED Dual Gaming Speakers', 'price': '84.42', 'reason': 'test'}", 
    "{'sku': '8772908384480', 'name': 'Universal Wireless Charging Stand for Iphone Apple Watch Airpods', 'price': '40.33'}"]
    is_valid = validate_result_schema(4, broken_json)
    assert is_valid is False


def test_compact_product_json_1k():   
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)    
    
    assert len(products) == 907
    context = json.dumps([asdict(products) for products in products])    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 40300 == tc

    context = json.dumps([asdict(products) for products in products], separators=(',', ':'))    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 34859 == tc

    p = json.loads(context)
    assert len(p) == 907


def test_compact_product_json_20k():   
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_20000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)    
    
    assert len(products) == 18088
    context = json.dumps([asdict(products) for products in products])    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 803791 == tc

    context = json.dumps([asdict(products) for products in products], separators=(',', ':'))    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 695267 == tc

    p = json.loads(context)
    assert len(p) == 18088


def test_catalog_validator():
    #"PILOT Dr. Grip Refillable & Retractable Gel Ink Rolling Ball Pen, Fine Point, Blue Barrel, Black Ink, Single Pen (36260)", 
    # "images": [], "asin": "B00006IEBU", "parent_asin": "B08CH1V4DG", "
    with open("./tests/data/amazon/office/amazon_office_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    catalog_validator = CatalogValidator(products)

    sku = "B00006IEBU"
    is_valid = catalog_validator.validate_sku(sku)
    assert True is is_valid

    sku = "B00006IEBUe"
    is_valid = catalog_validator.validate_sku(sku)
    assert False is is_valid
    

def test_find_name_by_sku():
    with open("./tests/data/amazon/office/amazon_office_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)    
    products = ProductFactory.dedupe(products)
    prodcut_json = json.dumps([asdict(product) for product in products])

    sku = "B01MTP6Q01"
    match = ProductFactory.find_sku_name(sku, prodcut_json)
    print(f"match: {match}")
    assert match == "Giant Print Check Register Set of 2"


def test_model_name_re():
    valid_models = [        
        'somemodel',
        'openai/gpt-4o-mini-search-preview',
        'tencent/hunyuan-a13b-instruct:free',
        'model_custom',
        'gemini2.1',
        'chat-gpt',
        'openai/gpt-4.1',
        'openai/gpt-4.1:latest',
        'qwen2.5-coder:latest',
        'openai/gpt-4o-2024-05-13',
        'openai/gpt-3.5-turbo-1106',
        'anthropic/claude-3.7-sonnet:thinking',
        'gemini-2.5-flash-lite-preview-06-17',
        'deepseek/deepseek-chat-v3-0324',
        'NousResearch/DeepHermes-3-Mistral-24B-Preview',
        'openai/gpt-4ABC',
        'meta-llama/llama-4-scout',
        'meta-llama/Llama-4-Scout-17B-16E',
        'fast_mode',
        'SmartEngine+Templates'
    ]
    for model in valid_models:
        claned = RE_MODEL_NAME.sub("", model)
        print(f"model: {model} - cleaned: {claned}")
        assert claned == model, f"Model '{model}' did not match the regex correctly"


def test_responses_parse_to_skus():
    responses = [
        {
            "id": "gen-1762174546-KTPAqRYyKnHjfyGjaXs8",
            "provider": "Google AI Studio",
            "model": "google/gemini-2.0-flash-001",
            "object": "chat.completion",
            "created": 1762174546,
            "choices": [
                {
                    "logprobs": None,
                    "finish_reason": "stop",
                    "native_finish_reason": "STOP",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "```json\n[\n  {\n    \"sku\": \"WS02\",\n    \"name\": \"Gabrielle Micro Sleeve Top - Clothing|New Luma Yoga Collection|Tees\",\n    \"price\": \"28\",\n    \"reason\": \"This top complements the parachute pants as part of the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"WH10\",\n    \"name\": \"Helena Hooded Fleece - Clothing|Hoodies amp Sweatshirts|New Luma Yoga Collection\",\n    \"price\": \"55\",\n    \"reason\": \"This hoodie provides warmth and style and is part of the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"WJ11\",\n    \"name\": \"Neve Studio Dance Jacket - Clothing|Jackets|New Luma Yoga Collection\",\n    \"price\": \"69\",\n    \"reason\": \"This jacket is a stylish layering piece from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP03\",\n    \"name\": \"Geo Insulated Jogging Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"51\",\n    \"reason\": \"These jogging pants offer warmth and comfort and are part of the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP05\",\n    \"name\": \"Kratos Gym Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"57\",\n    \"reason\": \"These gym pants are a comfortable and stylish option from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP07\",\n    \"name\": \"Thorpe Track Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"68\",\n    \"reason\": \"These track pants are a versatile option from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP08\",\n    \"name\": \"Zeppelin Yoga Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"82\",\n    \"reason\": \"These yoga pants are a comfortable and stylish option from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"WSH03\",\n    \"name\": \"Gwen Drawstring Bike Short - Clothing|New Luma Yoga Collection|Performance Fabrics|Shorts\",\n    \"price\": \"50\",\n    \"reason\": \"These bike shorts are a comfortable and stylish option from the New Luma Yoga Collection\"\n  }\n]\n```",
                        "refusal": None
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 6350,
                "completion_tokens": 569,
                "total_tokens": 6919,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 0}
            }
        },
        {
            "id": "chatcmpl-CXoO4WZJeKccWAyTy97AgExo6cBLw",
            "object": "chat.completion",
            "created": 1762174548,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I assist you today?",
                        "refusal": None,
                        "annotations": []
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 9,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}
            },
            "service_tier": "default",
            "system_fingerprint": "fp_560af6e559"
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "```json\n[\n  {\n    \"sku\": \"MP03\",\n    \"name\": \"Geo Insulated Jogging Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"51\",\n    \"reason\": \"These jogging pants complement the shorts as part of the New Luma Yoga Collection providing a full outfit\"\n  },\n  {\n    \"sku\": \"MS09\",\n    \"name\": \"Ryker LumaTechtrade Tee Crew-neck - Tees\",\n    \"price\": \"32\",\n    \"reason\": \"This tee pairs well with the shorts for a complete athletic outfit\"\n  },\n  {\n    \"sku\": \"MT05\",\n    \"name\": \"Rocco Gym Tank - Tanks\",\n    \"price\": \"24\",\n    \"reason\": \"This tank top is a great option for workouts and complements the active shorts\"\n  },\n  {\n    \"sku\": \"24-MG03\",\n    \"name\": \"Summit Watch - New Luma Yoga Collection|Watches\",\n    \"price\": \"54\",\n    \"reason\": \"This watch from the New Luma Yoga Collection complements the shorts and enhances the athletic look\"\n  },\n  {\n    \"sku\": \"24-UG05\",\n    \"name\": \"Go-Getr Pushup Grips - Fitness Equipment|New Luma Yoga Collection\",\n    \"price\": \"19\",\n    \"reason\": \"These pushup grips enhance the workout experience when paired with the active shorts\"\n  },\n  {\n    \"sku\": \"MSH06\",\n    \"name\": \"Lono Yoga Short - Clothing|New Luma Yoga Collection|Shorts\",\n    \"price\": \"32\",\n    \"reason\": \"This yoga short is another option from the New Luma Yoga Collection providing a similar style\"\n  },\n  {\n    \"sku\": \"MP05\",\n    \"name\": \"Kratos Gym Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"57\",\n    \"reason\": \"These gym pants from the New Luma Yoga Collection provide an alternative to shorts for cooler weather\"\n  },\n  {\n    \"sku\": \"24-MB05\",\n    \"name\": \"Wayfarer Messenger Bag - Bags|New Luma Yoga Collection\",\n    \"price\": \"45\",\n    \"reason\": \"This messenger bag from the New Luma Yoga Collection is great for carrying workout gear\"\n  }\n]\n```",
                        "role": "assistant"
                    }
                }
            ],
            "created": 1762174634,
            "id": "paYIac_XLdqr-8YP96e_2Ao",
            "model": "gemini-2.0-flash-001",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 558,
                "prompt_tokens": 6353,
                "total_tokens": 6911
            }
        }
    ]
    
    total_skus = 0
    for response in responses:
        skus = PromptFactory.extract_skus_from_response(response)
        print(f"skus: {skus}")
        total_skus += len(skus)

    assert total_skus == 16


    
def test_responses_parse_to_model_list():
    responses = [
        {
            "id": "gen-1762174546-KTPAqRYyKnHjfyGjaXs8",
            "provider": "Google AI Studio",
            "model": "google/gemini-2.0-flash-001",
            "object": "chat.completion",
            "created": 1762174546,
            "choices": [
                {
                    "logprobs": None,
                    "finish_reason": "stop",
                    "native_finish_reason": "STOP",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "```json\n[\n  {\n    \"sku\": \"WS02\",\n    \"name\": \"Gabrielle Micro Sleeve Top - Clothing|New Luma Yoga Collection|Tees\",\n    \"price\": \"28\",\n    \"reason\": \"This top complements the parachute pants as part of the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"WH10\",\n    \"name\": \"Helena Hooded Fleece - Clothing|Hoodies amp Sweatshirts|New Luma Yoga Collection\",\n    \"price\": \"55\",\n    \"reason\": \"This hoodie provides warmth and style and is part of the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"WJ11\",\n    \"name\": \"Neve Studio Dance Jacket - Clothing|Jackets|New Luma Yoga Collection\",\n    \"price\": \"69\",\n    \"reason\": \"This jacket is a stylish layering piece from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP03\",\n    \"name\": \"Geo Insulated Jogging Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"51\",\n    \"reason\": \"These jogging pants offer warmth and comfort and are part of the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP05\",\n    \"name\": \"Kratos Gym Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"57\",\n    \"reason\": \"These gym pants are a comfortable and stylish option from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP07\",\n    \"name\": \"Thorpe Track Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"68\",\n    \"reason\": \"These track pants are a versatile option from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP08\",\n    \"name\": \"Zeppelin Yoga Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"82\",\n    \"reason\": \"These yoga pants are a comfortable and stylish option from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"WSH03\",\n    \"name\": \"Gwen Drawstring Bike Short - Clothing|New Luma Yoga Collection|Performance Fabrics|Shorts\",\n    \"price\": \"50\",\n    \"reason\": \"These bike shorts are a comfortable and stylish option from the New Luma Yoga Collection\"\n  }\n]\n```",
                        "refusal": None
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 6350,
                "completion_tokens": 569,
                "total_tokens": 6919,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 0}
            }
        },
        {
            "id": "chatcmpl-CXoO4WZJeKccWAyTy97AgExo6cBLw",
            "object": "chat.completion",
            "created": 1762174548,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I assist you today?",
                        "refusal": None,
                        "annotations": []
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 9,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}
            },
            "service_tier": "default",
            "system_fingerprint": "fp_560af6e559"
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "```json\n[\n  {\n    \"sku\": \"MP03\",\n    \"name\": \"Geo Insulated Jogging Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"51\",\n    \"reason\": \"These jogging pants complement the shorts as part of the New Luma Yoga Collection providing a full outfit\"\n  },\n  {\n    \"sku\": \"MS09\",\n    \"name\": \"Ryker LumaTechtrade Tee Crew-neck - Tees\",\n    \"price\": \"32\",\n    \"reason\": \"This tee pairs well with the shorts for a complete athletic outfit\"\n  },\n  {\n    \"sku\": \"MT05\",\n    \"name\": \"Rocco Gym Tank - Tanks\",\n    \"price\": \"24\",\n    \"reason\": \"This tank top is a great option for workouts and complements the active shorts\"\n  },\n  {\n    \"sku\": \"24-MG03\",\n    \"name\": \"Summit Watch - New Luma Yoga Collection|Watches\",\n    \"price\": \"54\",\n    \"reason\": \"This watch from the New Luma Yoga Collection complements the shorts and enhances the athletic look\"\n  },\n  {\n    \"sku\": \"24-UG05\",\n    \"name\": \"Go-Getr Pushup Grips - Fitness Equipment|New Luma Yoga Collection\",\n    \"price\": \"19\",\n    \"reason\": \"These pushup grips enhance the workout experience when paired with the active shorts\"\n  },\n  {\n    \"sku\": \"MSH06\",\n    \"name\": \"Lono Yoga Short - Clothing|New Luma Yoga Collection|Shorts\",\n    \"price\": \"32\",\n    \"reason\": \"This yoga short is another option from the New Luma Yoga Collection providing a similar style\"\n  },\n  {\n    \"sku\": \"MP05\",\n    \"name\": \"Kratos Gym Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"57\",\n    \"reason\": \"These gym pants from the New Luma Yoga Collection provide an alternative to shorts for cooler weather\"\n  },\n  {\n    \"sku\": \"24-MB05\",\n    \"name\": \"Wayfarer Messenger Bag - Bags|New Luma Yoga Collection\",\n    \"price\": \"45\",\n    \"reason\": \"This messenger bag from the New Luma Yoga Collection is great for carrying workout gear\"\n  }\n]\n```",
                        "role": "assistant"
                    }
                }
            ],
            "created": 1762174634,
            "id": "paYIac_XLdqr-8YP96e_2Ao",
            "model": "gemini-2.0-flash-001",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 558,
                "prompt_tokens": 6353,
                "total_tokens": 6911
            }
        }
    ]

    models = set()
    for response in responses:
        model_name = response["model"]
        model_name = model_name.split('/')[-1] if '/' in model_name else model_name
        print(f"model_name: {model_name}")
        models.add(model_name)
    
    assert len(models) == 2
    assert "gemini-2.0-flash-001" in models
    assert "gpt-4o-mini-2024-07-18" in models

    