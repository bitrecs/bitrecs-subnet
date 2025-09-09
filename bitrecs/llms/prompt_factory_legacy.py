import re
import json
import tiktoken
import bittensor as bt
import bitrecs.utils.constants as CONST
from functools import lru_cache
from typing import List, Optional
from datetime import datetime
from bitrecs.commerce.user_profile import UserProfile
from bitrecs.commerce.product import ProductFactory

class PromptFactory:

    SEASON = "spring/summer"

    ENGINE_MODE = "complimentary"  #similar, sequential
    
    PERSONAS = {
        "luxury_concierge": {
            "description": "an elite American Express-style luxury concierge with impeccable taste and a deep understanding of high-end products across all categories. You cater to discerning clients seeking exclusivity, quality, and prestige",
            "tone": "sophisticated, polished, confident",
            "response_style": "Recommend only the finest, most luxurious products with detailed descriptions of their premium features, craftsmanship, and exclusivity. Emphasize brand prestige and lifestyle enhancement",
            "priorities": ["quality", "exclusivity", "brand prestige"]
        },
        "general_recommender": {
            "description": "a friendly and practical product expert who helps customers find the best items for their needs, balancing seasonality, value, and personal preferences across a wide range of categories",
            "tone": "warm, approachable, knowledgeable",
            "response_style": "Suggest well-rounded products that offer great value, considering seasonal relevance and customer needs. Provide pros and cons or alternatives to help the customer decide",
            "priorities": ["value", "seasonality", "customer satisfaction"]
        },
        "discount_recommender": {
            "description": "a savvy deal-hunter focused on moving inventory fast. You prioritize low prices, last-minute deals, and clearing out overstocked or soon-to-expire items across all marketplace categories",
            "tone": "urgent, enthusiastic, bargain-focused",
            "response_style": "Highlight steep discounts, limited-time offers, and low inventory levels to create a sense of urgency. Focus on price savings and practicality over luxury or long-term value",
            "priorities": ["price", "inventory levels", "deal urgency"]
        },
        "ecommerce_retail_store_manager": {
            "description": "an experienced e-commerce retail store manager with a strategic focus on optimizing sales, customer satisfaction, and inventory turnover across a diverse marketplace",
            "tone": "professional, practical, results-driven",
            "response_style": "Provide balanced recommendations that align with business goals, customer preferences, and current market trends. Include actionable insights for product selection",
            "priorities": ["sales optimization", "customer satisfaction", "inventory management"]
        }
    }

    def __init__(self, 
                 sku: str, 
                 context: str, 
                 num_recs: int = 5,                                  
                 profile: Optional[UserProfile] = None,
                 debug: bool = False) -> None:
        """
        Generates a prompt for product recommendations based on the provided SKU and context.
        :param sku: The SKU of the product being viewed.
        :param context: The context string containing available products.
        :param num_recs: The number of recommendations to generate (default is 5).
        :param profile: Optional UserProfile object containing user-specific data.
        :param debug: If True, enables debug logging."""

        if len(sku) < CONST.MIN_QUERY_LENGTH or len(sku) > CONST.MAX_QUERY_LENGTH:
            raise ValueError(f"SKU must be between {CONST.MIN_QUERY_LENGTH} and {CONST.MAX_QUERY_LENGTH} characters long")
        if num_recs < 1 or num_recs > CONST.MAX_RECS_PER_REQUEST:
            raise ValueError(f"num_recs must be between 1 and {CONST.MAX_RECS_PER_REQUEST}")

        self.sku = sku
        self.context = context
        self.num_recs = num_recs
        self.debug = debug
        self.catalog = []
        self.cart = []
        self.cart_json = "[]"
        self.orders = []
        self.order_json = "[]"
        self.season =  PromptFactory.SEASON       
        self.engine_mode = PromptFactory.ENGINE_MODE 
        if not profile:
            self.persona = "ecommerce_retail_store_manager"
        else:
            self.profile = profile
            self.persona = profile.site_config.get("profile", "ecommerce_retail_store_manager")
            if not self.persona or self.persona not in PromptFactory.PERSONAS:
                bt.logging.error(f"Invalid persona: {self.persona}. Must be one of {list(PromptFactory.PERSONAS.keys())}")
                self.persona = "ecommerce_retail_store_manager"
            self.cart = self._sort_cart_keys(profile.cart)
            self.cart_json = json.dumps(self.cart, separators=(',', ':'))
            self.orders = profile.orders
            # self.order_json = json.dumps(self.orders, separators=(',', ':'))
        
        self.sku_info = ProductFactory.find_sku_name(self.sku, self.context)    


    def _sort_cart_keys(self, cart: List[dict]) -> List[str]:
        ordered_cart = []
        for item in cart:
            ordered_item = {
                'sku': item.get('sku', ''),
                'name': item.get('name', ''),
                'price': item.get('price', '')
            }
            ordered_cart.append(ordered_item)
        return ordered_cart
    
    
    def generate_prompt(self) -> str:
        """Generates a text prompt for product recommendations with persona details."""
        bt.logging.info("PROMPT generating prompt: {}".format(self.sku))

        today = datetime.now().strftime("%Y-%m-%d")
        season = self.season
        persona_data = self.PERSONAS[self.persona]

        prompt = f"""# SCENARIO
    A shopper is viewing a product with SKU <sku>{self.sku}</sku> named <sku_info>{self.sku_info}</sku_info> on your e-commerce store.
    They are looking for {self.engine_mode} products to add to their cart.
    You will build a recommendation set with no duplicates based on the provided context and your persona qualities.
        
    # YOUR PERSONA
    <persona>{self.persona}</persona>

    <core_attributes>
    You embody: {persona_data['description']}
    Your mindset: {persona_data['tone']}
    Your expertise: {persona_data['response_style']}
    Core values: {', '.join(persona_data['priorities'])}
    </core_attributes>

    # YOUR ROLE:
    - Recommend **{self.num_recs}** {self.engine_mode} products (A -> X,Y,Z)
    - Increase average order value and conversion rate
    - Use deep product catalog knowledge
    - Understand product attributes and revenue impact
    - Avoid variant duplicates (same product in different colors/sizes)
    - Consider seasonal relevance

    Current season: <season>{season}</season>
    Today's date: {today} 

    # TASK
    Given a product SKU <sku>{self.sku}</sku> select **{self.num_recs}** complementary unique products from the context.
    Use your persona qualities to THINK about which products to select, but return ONLY a JSON array.
    Evaluate each product name and price fields before making your recommendations.
    The name field is the most important attribute followed by price.
    The product name will often contain important information like which category it belongs to, sometimes denoted by | characters indicating the category hierarchy.    
    Leverage the complete information ecosystem - product catalog, user context, seasonal trends, and your role expertise as a {self.persona} - to deliver {self.engine_mode} recommendations.
    Apply comprehensive analysis using all available inputs: product attributes from the context, user cart history, seasonal relevance, pricing considerations and your persona's core values to create a cohesive recommendation set.
    Utilize your core_attributes to make the best recommendations.
    Do **not** recommend products that are already in the cart.

    # INPUT
    Query SKU: <sku>{self.sku}</sku><sku_info>{self.sku_info}</sku_info>

    Current cart:
    <cart>
    {self.cart_json}
    </cart>

    Available products:
    <context>
    {self.context}
    </context>   

    # OUTPUT REQUIREMENTS
    - Return ONLY a JSON array.
    - NO Python dictionary syntax (no single quotes).
    - Each item must be valid JSON with: "sku": "...", "name": "...", "price": "...", "reason": "..."
    - Each item must have: sku, name, price and reason.
    - If the Query SKU product is gendered consider recommending products that match the gender of the Query SKU.
    - If the Query SKU is gender neutral recommend more gender neutral products.
    - Never mix gendered products in the recommendation set for example if the user is looking at womans shoes, do not recommend mens shoes.
    - Do not conflate pet products with baby products, they are different categories.
    - Must return exactly {self.num_recs} items.
    - Return items MUST exist in context.
    - Return items must NOT exist in the cart.
    - No duplicates. *Very important* The final result MUST be a unique set of products from the context.
    - Product matching Query SKU must not be included in the set of recommendations.
    - Return items should be ordered by relevance/profitability, the first being your top recommendation.
    - Each item must have a reason explaining why the product is a good recommendation for the related Query SKU.
    - The reason should be a single succinct sentence consisting of plain words without punctuation, or line breaks.
    - You will be graded on your reason so make sure to provide a good reason for each recommendation which is relevant to the Query SKU.    
    - No explanations or text outside the JSON array.

    Example format:
    
    [{{"sku": "XYZ", "name": "Hunter Original Play Boot Chelsea", "price": "115", "reason": "User is viewing rainboots, we recommend this alternative pair of rainboots which is our best seller"}},
        {{"sku": "ABC", "name": "Men's Lightweight Hooded Rain Jacket", "price": "149", "reason": "Since the user is looking at mens rainboots, given the season a mens raincoat should be a good fit"}},
        {{"sku": "DEF", "name": "Davek Elite Umbrella", "price": "159", "reason": "An Umbrella would go nicely with ABC Lightweight Hooded Rain Jacket and is often paired with it"}}]"""

        prompt_length = len(prompt)
        bt.logging.info(f"LLM QUERY Prompt length: {prompt_length}")
        
        if self.debug:
            token_count = PromptFactory.get_token_count(prompt)
            bt.logging.info(f"LLM QUERY Prompt Token count: {token_count}")
            bt.logging.debug(f"Persona: {self.persona}")
            bt.logging.debug(f"Season {season}")
            bt.logging.debug(f"Values: {', '.join(persona_data['priorities'])}")
            bt.logging.debug(f"Prompt: {prompt}")
            #print(prompt)

        return prompt
    
    
    @staticmethod
    def get_token_count(prompt: str, encoding_name: str="o200k_base") -> int:
        encoding = PromptFactory._get_cached_encoding(encoding_name)
        tokens = encoding.encode(prompt)
        return len(tokens)
    
    
    @staticmethod
    @lru_cache(maxsize=4)
    def _get_cached_encoding(encoding_name: str):
        return tiktoken.get_encoding(encoding_name)
    
    
    @staticmethod
    def get_word_count(prompt: str) -> int:
        return len(prompt.split())
    

    @staticmethod
    def tryparse_llm(input_str: str) -> list:
        """
        Take raw LLM output and parse to an array 

        """
        try:
            if not input_str:
                bt.logging.error("Empty input string tryparse_llm")   
                return []
            input_str = input_str.replace("```json", "").replace("```", "").strip()
            pattern = r'\[.*?\]'
            regex = re.compile(pattern, re.DOTALL)
            match = regex.findall(input_str)        
            for array in match:
                try:
                    llm_result = array.strip()
                    return json.loads(llm_result)
                except json.JSONDecodeError:                    
                    bt.logging.error(f"Invalid JSON in prompt factory: {array}")
            return []
        except Exception as e:
            bt.logging.error(str(e))
            return []
    