import json
from dataclasses import asdict
from random import SystemRandom
safe_random = SystemRandom()
from bitrecs.commerce.product import CatalogProvider, ProductFactory
from bitrecs.llms.prompt_factory import PromptFactory
from dotenv import load_dotenv
load_dotenv()


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


copy_pastas = ["ok I need PRICE TO GO UP. I cant take this anymore. every day I am checking price and its dipping. every day, check price, bad price. I cant take this anymore, I have over invested, by a lot. it is what it is. but I need price to GO UP ALREADY. can devs DO SOMETHING",
               "You think it's funny to take screenshots of people's NFTS, huh? Property theft is a joke to you? l'll have you know that the blockchain doesn't lie. I own it. Even if you save it, it's my property. You are mad that you don't own the art that I own. Delete that screenshot.",
               "Some things are better left unsaid. Recommend no more news like these, for the sake of the people, our industry (and your business)",
               "Awesome pics. Great size. Look thick. Solid. Tight. Keep us all posted on your continued progress with any new progress pics or vid clips. Show us what you got man. Wanna see how freakin' huge, solid, thick and tight you can get. Thanks for the motivation."]


def test_get_word_count():    
    for i, pasta in enumerate(copy_pastas):
        #print(f"pasta: {pasta}")
        wc = PromptFactory.get_word_count(pasta)
        print(wc)
        match i:
            case 0: 
                assert wc == 56
            case 1: 
                assert wc == 53
            case 2: 
                assert wc == 23
            case 3: 
                assert wc == 46


def test_get_token_count():    
    for i, pasta in enumerate(copy_pastas):
        #print(f"pasta: {pasta}")
        tc = PromptFactory.get_token_count(pasta)
        print(tc)
        match i:
            case 0: 
                assert tc == 71
            case 1: 
                assert tc == 64
            case 2: 
                assert tc == 29
            case 3: 
                assert tc == 59


def test_get_token_count_random1k_prompt():
    raw_products = product_1k()
    products = ProductFactory.dedupe(raw_products)
    print(f"after de-dupe: {len(products)} records")

    rp = safe_random.choice(products)
    user_prompt = rp.sku
    num_recs = safe_random.choice([1, 5, 9, 10, 11, 16, 20])    
    debug_prompts = False

    match = [p for p in products if p.sku == user_prompt][0]
    print(match)

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt,
                            context=context,
                            num_recs=num_recs,
                            debug=debug_prompts)    


    prompt = factory.generate_prompt()
    #print(prompt)

    wc = PromptFactory.get_word_count(prompt)
    print(f"word count: {wc}")

    tc = PromptFactory.get_token_count(prompt)
    print(f"token count o200k_base: {tc}")

    tc2 = PromptFactory.get_token_count(prompt, encoding_name="cl100k_base")
    print(f"token count cl100k_base: {tc2}")
    
    assert wc > 20_00
    assert tc > 48_000
    assert tc2 > 48_000


def test_get_token_count_random5k_prompt():
    raw_products = product_5k()
    products = ProductFactory.dedupe(raw_products)
    print(f"after de-dupe: {len(products)} records")

    rp = safe_random.choice(products)
    user_prompt = rp.sku
    num_recs = safe_random.choice([1, 5, 9, 10, 11, 16, 20])    
    debug_prompts = False

    match = [p for p in products if p.sku == user_prompt][0]
    print(match)

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt,
                            context=context,
                            num_recs=num_recs,
                            debug=debug_prompts)


    prompt = factory.generate_prompt()
    #print(prompt)

    wc = PromptFactory.get_word_count(prompt)
    print(f"word count: {wc}")

    tc = PromptFactory.get_token_count(prompt)
    print(f"token count o200k_base: {tc}")

    tc2 = PromptFactory.get_token_count(prompt, encoding_name="cl100k_base")
    print(f"token count cl100k_base: {tc2}")
    
    assert wc > 90_000
    assert tc > 200_000
    assert tc2 > 200_000



def test_get_token_count_random20k_prompt():
    raw_products = product_20k()
    products = ProductFactory.dedupe(raw_products)
    print(f"after de-dupe: {len(products)} records")

    rp = safe_random.choice(products)
    user_prompt = rp.sku
    num_recs = safe_random.choice([1, 5, 9, 10, 11, 16, 20])    
    debug_prompts = False

    match = [p for p in products if p.sku == user_prompt][0]
    print(match)

    context = json.dumps([asdict(products) for products in products])
    factory = PromptFactory(sku=user_prompt,
                            context=context,
                            num_recs=num_recs,
                            debug=debug_prompts)


    prompt = factory.generate_prompt()
    #print(prompt)

    wc = PromptFactory.get_word_count(prompt)
    print(f"word count: {wc}")

    tc = PromptFactory.get_token_count(prompt)
    print(f"token count o200k_base: {tc}")

    tc2 = PromptFactory.get_token_count(prompt, encoding_name="cl100k_base")
    print(f"token count cl100k_base: {tc2}")
    
    assert wc > 300_000
    assert tc > 790_000
    assert tc2 > 800_000

