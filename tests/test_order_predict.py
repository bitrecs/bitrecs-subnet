import json
import sqlite3
import time
from typing import List
from random import SystemRandom
safe_random = SystemRandom()
from dataclasses import asdict
from bitrecs.llms.factory import LLMFactory
from bitrecs.llms.llm_provider import LLM
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.commerce.product import Product, ProductFactory
from bitrecs.commerce.user_profile import UserProfile
from dotenv import load_dotenv
load_dotenv()


DB_PATH = "./tests/data/testdb/store.sqlite"
MIN_ORDER_CLIP = 25.00


class OrderPrediction:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def _validate_inputs(self, sku: str, rec_skus: list[str]) -> tuple[str, list[str]]:
        """Validate and clean inputs"""
        if not sku or not isinstance(sku, str):
            raise ValueError("SKU must be a non-empty string")        
        if not rec_skus or not isinstance(rec_skus, list):
            raise ValueError("rec_skus must be a non-empty list")
        cleaned_rec_skus = []
        for rec_sku in rec_skus:
            if rec_sku and isinstance(rec_sku, str):
                cleaned = rec_sku.strip()
                if cleaned and cleaned != sku:  # Don't include the same SKU
                    cleaned_rec_skus.append(cleaned)
        
        if not cleaned_rec_skus:
            raise ValueError("No valid recommendation SKUs provided")
            
        return sku.strip(), cleaned_rec_skus
    
    def find_similar_orders(self, sku: str, rec_skus: list[str]) -> dict:
        """Find orders that contain both the given SKU and one or more recommendation SKUs."""
        try:
            sku, rec_skus = self._validate_inputs(sku, rec_skus)
        except ValueError as e:
            return {'orders': [], 'co_occurrence_stats': {}, 'total_orders': 0, 'error': str(e)}
        
        rec_sku_placeholders = ','.join(['?' for _ in rec_skus])        
        query = f"""
        SELECT DISTINCT o.order_id,
               o.grand_total,
               o.status,
               o.subtotal,
               o.total_item_count,
               o.total_paid,
               o.total_qty_ordered,
               o.updated_at,
               o.group_id,
               GROUP_CONCAT(DISTINCT oi_rec.sku) as matching_rec_skus,
               GROUP_CONCAT(DISTINCT oi_rec.name) as matching_rec_names,
               COUNT(DISTINCT oi_rec.sku) as rec_sku_count,
               SUM(oi_rec.row_total) as rec_items_total
        FROM music_orders o
        JOIN music_order_items oi_main ON o.order_id = oi_main.order_id
        JOIN music_order_items oi_rec ON o.order_id = oi_rec.order_id
        WHERE oi_main.sku = ?
        AND oi_rec.sku IN ({rec_sku_placeholders})
        AND oi_main.sku != oi_rec.sku
        --AND o.status = 'complete'  -- Add status filter
        GROUP BY o.order_id
        ORDER BY rec_sku_count DESC, o.grand_total DESC
        """        
        
        try:
            params = [sku] + rec_skus
            cursor = self.db.execute(query, params)
            orders = cursor.fetchall()        
            co_occurrence_stats = self._get_co_occurrence_stats(sku, rec_skus)        
            return {
                'orders': [dict(order) for order in orders],
                'co_occurrence_stats': co_occurrence_stats,
                'total_orders': len(orders)
            }
        except sqlite3.Error as e:
            return {'orders': [], 'co_occurrence_stats': {}, 'total_orders': 0, 'error': f"Database error: {e}"}
    
    def _get_co_occurrence_stats(self, sku: str, rec_skus: list[str]) -> dict:
        """Get count of orders for each recommendation SKU with the given SKU"""
        stats = {}
        
        for rec_sku in rec_skus:
            query = """
            SELECT COUNT(DISTINCT o.order_id) as order_count
            FROM music_orders o
            WHERE o.order_id IN (
                SELECT order_id FROM music_order_items WHERE sku = ?
            )
            AND o.order_id IN (
                SELECT order_id FROM music_order_items WHERE sku = ?
            )
            """
            cursor = self.db.execute(query, [sku, rec_sku])
            result = cursor.fetchone()
            stats[rec_sku] = result[0] if result else 0 
            
        return stats
    
    def get_order_details_with_items(self, sku: str, rec_skus: list[str]) -> list:       
        if not sku or not rec_skus:
            return []
        
        rec_sku_placeholders = ','.join(['?' for _ in rec_skus])        
        query = f"""
        SELECT 
            o.order_id,
            o.grand_total,
            o.status,
            o.updated_at,
            oi.sku,
            oi.name,
            oi.price,
            oi.qty,
            oi.row_total,
            CASE 
                WHEN oi.sku = ? THEN 'target_sku'
                WHEN oi.sku IN ({rec_sku_placeholders}) THEN 'recommendation_sku'
                ELSE 'other_item'
            END as item_type
        FROM music_orders o
        JOIN music_order_items oi ON o.order_id = oi.order_id
        WHERE o.order_id IN (
            SELECT DISTINCT o2.order_id
            FROM music_orders o2
            JOIN music_order_items oi_main ON o2.order_id = oi_main.order_id
            JOIN music_order_items oi_rec ON o2.order_id = oi_rec.order_id
            WHERE oi_main.sku = ?
            AND oi_rec.sku IN ({rec_sku_placeholders})
            AND oi_main.sku != oi_rec.sku
        )
        ORDER BY o.order_id, oi.item_id
        """
        
        # Parameters: sku for CASE, rec_skus for CASE, sku for WHERE, rec_skus for WHERE
        params = [sku] + rec_skus + [sku] + rec_skus
        cursor = self.db.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recommendation_strength(self, sku: str, rec_skus: list[str]) -> dict:
        """
        Calculate recommendation strength based on co-occurrence frequency.
        """
        # Get total orders containing the main SKU
        total_sku_orders_query = """
        SELECT COUNT(DISTINCT order_id) as total_orders
        FROM music_order_items 
        WHERE sku = ?
        """
        cursor = self.db.execute(total_sku_orders_query, [sku])
        result = cursor.fetchone()
        total_sku_orders = result[0] if result else 0
        
        if total_sku_orders == 0:
            return {'recommendations': [], 'base_sku_orders': 0}
        
        # Use existing co-occurrence method
        co_occurrence_stats = self._get_co_occurrence_stats(sku, rec_skus)
        
        # Get additional details for each recommendation
        recommendations = []
        for rec_sku, co_count in co_occurrence_stats.items():
            if co_count > 0:
                strength = (co_count / total_sku_orders) * 100
                
                # Get revenue details for this rec_sku in co-occurring orders
                revenue_query = """
                SELECT 
                    AVG(o.grand_total) as avg_order_value,
                    SUM(oi.row_total) as total_rec_revenue,
                    AVG(oi.price) as avg_rec_price,
                    COUNT(oi.item_id) as total_rec_items_sold
                FROM music_order_items oi
                JOIN music_orders o ON oi.order_id = o.order_id
                WHERE oi.sku = ?
                AND oi.order_id IN (
                    SELECT order_id FROM music_order_items WHERE sku = ?
                )
                """
                cursor = self.db.execute(revenue_query, [rec_sku, sku])
                revenue_result = cursor.fetchone()
                
                recommendations.append({
                    'sku': rec_sku,
                    'co_occurrence_count': co_count,
                    'strength_percentage': round(strength, 2),
                    'confidence': 'high' if strength > 10 else 'medium' if strength > 5 else 'low',
                    'avg_order_value': round(revenue_result[0], 2) if revenue_result and revenue_result[0] else 0,
                    'total_revenue': round(revenue_result[1], 2) if revenue_result and revenue_result[1] else 0,
                    'avg_item_price': round(revenue_result[2], 2) if revenue_result and revenue_result[2] else 0,
                    'total_items_sold': revenue_result[3] if revenue_result and revenue_result[3] else 0
                })
        
        recommendations.sort(key=lambda x: x['strength_percentage'], reverse=True)        
        return {
            'recommendations': recommendations,
            'base_sku_orders': total_sku_orders
        }
    
    def get_customer_purchase_patterns(self, sku: str, rec_skus: list[str]) -> dict:
        """
        Analyze purchase patterns using group_id as customer identifier
        """
        if not sku or not rec_skus:
            return {'patterns': [], 'total_customers': 0}
        
        rec_sku_placeholders = ','.join(['?' for _ in rec_skus])
        
        # FIXED: Use subqueries to find orders with both SKUs
        query = f"""
        SELECT 
            o.group_id as customer_id,
            COUNT(DISTINCT o.order_id) as total_orders,
            MIN(o.updated_at) as first_purchase_date,
            MAX(o.updated_at) as last_purchase_date,
            SUM(o.grand_total) as total_spent,
            GROUP_CONCAT(DISTINCT rec_items.sku) as purchased_rec_skus,
            COUNT(DISTINCT rec_items.sku) as unique_rec_skus_bought
        FROM music_orders o
        JOIN music_order_items main_items ON o.order_id = main_items.order_id
        JOIN music_order_items rec_items ON o.order_id = rec_items.order_id
        WHERE main_items.sku = ?
        AND rec_items.sku IN ({rec_sku_placeholders})
        AND main_items.sku != rec_items.sku
        AND o.group_id IS NOT NULL
        --AND o.status = 'complete'
        GROUP BY o.group_id
        ORDER BY unique_rec_skus_bought DESC, total_spent DESC
        """
    
        params = [sku] + rec_skus
        cursor = self.db.execute(query, params)
        patterns = cursor.fetchall()
        
        return {
            'patterns': [dict(pattern) for pattern in patterns],
            'total_customers': len(patterns)
        }

    def find_sequential_orders(self, sku: str, rec_skus: list[str]) -> dict:
        """Find orders where customers bought the SKU first, then later bought rec_skus."""
        try:
            sku, rec_skus = self._validate_inputs(sku, rec_skus)
        except ValueError as e:
            return self._empty_sequential_result(error=str(e))
        
        rec_sku_placeholders = ','.join(['?' for _ in rec_skus])
        
        query = f"""
        SELECT 
            first_order.group_id as customer_id,
            first_order.order_id as first_order_id,
            first_order.updated_at as first_order_date,
            first_order.grand_total as first_order_total,
            second_order.order_id as second_order_id,
            second_order.updated_at as second_order_date,
            second_order.grand_total as second_order_total,
            oi_rec.sku as purchased_rec_sku,
            oi_rec.name as purchased_rec_name,
            oi_rec.price as purchased_rec_price,
            oi_rec.qty as purchased_rec_qty,
            oi_rec.row_total as purchased_rec_total,
            JULIANDAY(second_order.updated_at) - JULIANDAY(first_order.updated_at) as days_between
        FROM music_orders first_order
        JOIN music_order_items oi_first ON first_order.order_id = oi_first.order_id
        JOIN music_orders second_order ON first_order.group_id = second_order.group_id
        JOIN music_order_items oi_rec ON second_order.order_id = oi_rec.order_id
        WHERE oi_first.sku = ?
        AND oi_rec.sku IN ({rec_sku_placeholders})
        AND first_order.updated_at < second_order.updated_at  -- FIXED: Re-enabled
        AND first_order.group_id IS NOT NULL
        --AND first_order.status = 'complete'  
        --AND second_order.status = 'complete' 
        AND first_order.order_id != second_order.order_id  -- FIXED: Ensure different orders
        ORDER BY first_order.group_id, first_order.updated_at, second_order.updated_at
        """
        
        try:
            params = [sku] + rec_skus
            cursor = self.db.execute(query, params)
            results = cursor.fetchall()            
            if not results:
                return self._empty_sequential_result()
            
            return self._process_sequential_results(results)
            
        except sqlite3.Error as e:
            return self._empty_sequential_result(error=f"Database error: {e}")
    
    def _empty_sequential_result(self, error=None):
        """Return empty sequential result structure"""
        result = {
            'sequential_patterns': [], 
            'customers': [], 
            'summary_stats': {
                'total_customers': 0,
                'total_sequential_orders': 0,
                'avg_days_between_purchases': 0,
                'total_rec_revenue': 0,
                'rec_sku_frequency': {},
                'rec_sku_unique_customers': {},
                'conversion_rate_by_sku': {},
                'avg_purchases_per_customer_by_sku': {}
            }
        }
        if error:
            result['error'] = error
        return result
    
    def _process_sequential_results(self, results):
        """Process sequential query results into structured data"""
        sequential_patterns = [dict(row) for row in results]
        
        # Group by customer for customer-level analysis
        customers = {}
        for pattern in sequential_patterns:
            customer_id = pattern['customer_id']
            if customer_id not in customers:
                customers[customer_id] = {
                    'customer_id': customer_id,
                    'first_purchase_date': pattern['first_order_date'],
                    'first_order_id': pattern['first_order_id'],
                    'first_order_total': pattern['first_order_total'],
                    'subsequent_purchases': [],
                    'total_rec_spending': 0,
                    'unique_rec_skus': set(),
                    'avg_days_between_purchases': 0
                }
            
            customers[customer_id]['subsequent_purchases'].append({
                'order_id': pattern['second_order_id'],
                'order_date': pattern['second_order_date'],
                'order_total': pattern['second_order_total'],
                'sku': pattern['purchased_rec_sku'],
                'name': pattern['purchased_rec_name'],
                'price': pattern['purchased_rec_price'],
                'qty': pattern['purchased_rec_qty'],
                'item_total': pattern['purchased_rec_total'] or 0,  # FIXED: Handle None
                'days_after_first': pattern['days_between']
            })
            
            customers[customer_id]['total_rec_spending'] += pattern['purchased_rec_total'] or 0
            customers[customer_id]['unique_rec_skus'].add(pattern['purchased_rec_sku'])
        
        # Calculate summary statistics with better error handling
        customer_list = list(customers.values())
        for customer in customer_list:
            customer['unique_rec_skus'] = list(customer['unique_rec_skus'])
            if customer['subsequent_purchases']:
                valid_days = [
                    p['days_after_first'] for p in customer['subsequent_purchases'] 
                    if p['days_after_first'] is not None and p['days_after_first'] >= 0
                ]
                customer['avg_days_between_purchases'] = (
                    sum(valid_days) / len(valid_days) if valid_days else 0
                )

        # Overall summary stats
        total_customers = len(customer_list)
        total_sequential_orders = len(sequential_patterns)

        valid_overall_days = [
            p['days_between'] for p in sequential_patterns 
            if p['days_between'] is not None and p['days_between'] >= 0
        ]
        avg_days_between = sum(valid_overall_days) / len(valid_overall_days) if valid_overall_days else 0
        total_rec_revenue = sum(c['total_rec_spending'] for c in customer_list)
        
        # Count frequency and unique customers per SKU
        rec_sku_frequency = {}
        rec_sku_unique_customers = {}
        
        for pattern in sequential_patterns:
            sku_bought = pattern['purchased_rec_sku']
            customer_id = pattern['customer_id']
            
            rec_sku_frequency[sku_bought] = rec_sku_frequency.get(sku_bought, 0) + 1
            
            if sku_bought not in rec_sku_unique_customers:
                rec_sku_unique_customers[sku_bought] = set()
            rec_sku_unique_customers[sku_bought].add(customer_id)
        
        return {
            'sequential_patterns': sequential_patterns,
            'customers': customer_list,
            'summary_stats': {
                'total_customers': total_customers,
                'total_sequential_orders': total_sequential_orders,
                'avg_days_between_purchases': round(avg_days_between, 1),
                'total_rec_revenue': round(total_rec_revenue, 2),
                'rec_sku_frequency': rec_sku_frequency,
                'rec_sku_unique_customers': {
                    sku: len(customers_set) 
                    for sku, customers_set in rec_sku_unique_customers.items()
                },
                'conversion_rate_by_sku': {
                    sku: round((len(rec_sku_unique_customers[sku]) / total_customers) * 100, 2) 
                    if total_customers > 0 else 0
                    for sku in rec_sku_frequency.keys()
                },
                'avg_purchases_per_customer_by_sku': {
                    sku: round(rec_sku_frequency[sku] / len(rec_sku_unique_customers[sku]), 2)
                    for sku in rec_sku_frequency.keys()
                }
            }
        }
    
   
    

def products_music(truncate_names: bool = True):    
    catalog = load_products_from_db(DB_PATH, truncate_names=truncate_names)   
    return catalog


def random_product() -> Product:
    return safe_random.choice(products_music())


def load_products_from_db(sql_lite_path: str, truncate_names: bool = True) -> List[Product]:  
    products = []
    try:
        conn = sqlite3.connect(sql_lite_path)
        cursor = conn.cursor()        
        sql = """SELECT 
            sku,
            CASE 
                WHEN INSTR(name, '-') > 0 
                THEN SUBSTR(name, 1, INSTR(name, '-') - 1)
                ELSE name
            END AS name,
            price
        FROM music_products;"""
        if not truncate_names:
            sql = """SELECT sku, name, price FROM music_products"""
            print(" \033[1;31m  Warning Loading full product names, check token limits! \033[0m")
        else:
            print(" \033[1;33m  Product Names Truncated \033[0m")
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            product = Product(
                sku=str(row[0]), 
                name=str(row[1]),
                price=str(row[2])
            )
            products.append(product)            
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()    
    return ProductFactory.dedupe(products)


def get_sample_user_profile() -> UserProfile:
    sql = f"""
        select group_id, count(1) as count from music_orders
        where status == 'complete' and total_paid > {MIN_ORDER_CLIP}
        group by group_id
        having count(1) > 1"""
    profiles = []
    profile_orders = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            raise ValueError("No user profiles found in the database.")        
        profiles = [{"group_id": row[0], "count": row[1]} for row in rows]
        assert len(profiles) == 5606
        r = safe_random.choice(profiles)        
        sql = f"""select o.*, i.qty, i.sku, i.name, i.price from music_orders o 
                left join music_order_items i on o.order_id = i.order_id
                where group_id = '{r['group_id']}' and o.status = 'complete' and o.total_paid > 0"""
        cursor.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            raise ValueError("No orders found for the selected user profile.")
        orders = []        
        for row in rows:
            order = {
                "order_id": row[0],
                "grand_total": str(row[1]),
                "status": str(row[2]),
                "subtotal": str(row[3]),
                "subtotal_inc_tax": str(row[4]),
                "subtotal_invoiced": str(row[5]),
                "total_item_count": str(row[6]),
                "total_paid": str(row[7]),
                "total_qty_ordered": str(row[8]),
                "updated_at": str(row[9]),
                "group_id": str(row[10]),
                "qty": str(row[11]),
                "sku": str(row[12]),
                "name": str(row[13]),
                "price": str(row[14])
            }
            orders.append(order)
        print(f"Found {len(orders)} orders for user profile {r['group_id']}.")
        for order in orders:
            order_id = order["order_id"]
            if order_id not in profile_orders:
                profile_orders[order_id] = {
                    "order_id": order_id,
                    "total": str(order["grand_total"]),
                    "status": str(order["status"]),
                    "created_at": str(order["updated_at"]),
                    "items": []
                }
            profile_orders[order_id]["items"].append({
                "sku": str(order["sku"]),
                "name": str(order["name"]),
                "price": str(order["price"]),
                "quantity": str(order["qty"])
            })
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
    if not profiles:    
        raise ValueError("No user profiles found in the database.")
    
    print(f"Found {len(profiles)} distinct user profiles in the database.")
    user_profile : UserProfile = UserProfile(
        id=str(r['group_id']),
        created_at="2025-05-31T18:45:13Z",  
        site_config={"profile": "ecommerce_retail_store_manager"},
        cart=[],  
        orders=list(profile_orders.values())
    )
    print(f"Selected random profile: \033[1;32m  {user_profile} \033[0m")
    return user_profile   


def get_simple_sku_stats(sku: str) -> dict:  
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    cursor = db.cursor()    
    query = """
    SELECT 
        sku, 
        COUNT(DISTINCT order_id) as total_orders, 
        SUM(row_total) as total_revenue, 
        SUM(qty) as total_items_sold
    FROM music_order_items
    WHERE sku = ?
    GROUP BY sku
    """    
    cursor.execute(query, (sku,))
    row = cursor.fetchone()    
    if not row:
        return {'sku': sku, 'total_orders': 0, 'total_revenue': 0.0, 'total_items_sold': 0}
        
    stats = {
        'sku': row['sku'],
        'total_orders': row['total_orders'],
        'total_revenue': round(row['total_revenue'], 2),
        'total_items_sold': row['total_items_sold']
    }
    
    db.close()
    return stats


def test_load_products_from_db():
    products = load_products_from_db(DB_PATH)
    print(f"Loaded {len(products)} products from the database.")
    assert len(products) == 22664, "Expected 22680 products to be loaded from the database."
    for product in products:
        assert isinstance(product, Product), "Loaded item is not a Product instance."
        assert product.sku is not None, "Product SKU should not be None."
        assert product.name is not None, "Product name should not be None."
        assert product.price is not None, "Product price should not be None."


def test_sample_user_profile():
    profile = get_sample_user_profile()
    assert isinstance(profile, UserProfile), "Expected UserProfile instance."
    assert profile.id is not None, "UserProfile ID should not be None."
    assert profile.created_at is not None, "UserProfile created_at should not be None."  
    assert profile.site_config is not None, "UserProfile site_config should not be None."    
    assert len(profile.orders) > 0, "UserProfile should have at least one order."
    
    
#@pytest.skip("Skipping test_sample_profile_get_similar_orders, requires database setup")
def test_sample_profile_get_similar_orders():
    num_recs = 5
    profile = get_sample_user_profile()    
    #$products = products_music()[:5000]
    products = products_music()
    print(f"Loaded {len(products)} products from the database.")#
    print(f"User profile: {profile.id} with {len(profile.orders)} orders and {len(profile.cart)} items in cart.")

    context = json.dumps([asdict(products) for products in products], separators=(',', ':'))

    #first_order = profile.orders[0]
    #first_sku = first_order['items'][0]['sku']
    #first_sku = "FASDSLUSGSBLK"
    first_sku = "FSEFXOTSPCOMPRE"

    viewing_product = next((p for p in products if p.sku == first_sku), None)
    assert viewing_product is not None, f"Product with SKU {first_sku} not found in products list."
       
    user_prompt = viewing_product.sku
    print(f"Viewing product: {viewing_product.name} \033[1;32m (SKU: {viewing_product.sku}) \033[0m")
    stats = get_simple_sku_stats(viewing_product.sku)
    print(f"SKU Stats: {stats}")    

    factory = PromptFactory(sku=user_prompt,
                            context=context, 
                            num_recs=num_recs,
                            profile=profile,
                            debug=True)
    prompt = factory.generate_prompt()    
    tc = factory.get_token_count(prompt)
    print(f"Token count: {tc}")

    #model = "deepseek-r1:70b"
    #model = "mistral-large:latest"
    #model = "gemma3:27b"
    #model = "mistral-nemo"    
    #model = "qwen3:32b"
    #model = "llama3.3:70b"
    #model = "google/gemini-2.5-flash-lite-preview-06-17"
    #model = "google/gemini-2.0-flash-lite-001"
    #model = "google/gemini-2.0-flash-001"
    #model = "amazon/nova-lite-v1"
    #model = "openai/gpt-4.1-nano"
    #model = "openai/gpt-4.1-mini"
    #model = "meta-llama/llama-4-maverick"
    #model = "qwen/qwen-turbo"    
    #model = "openai/gpt-4.1"
    #model = "ai21/jamba-mini-1.7"
    #model = "openrouter/sonoma-dusk-alpha"
    #model = "openrouter/sonoma-sky-alpha"
    #model = "google/gemini-2.5-flash-lite"
    #model = "google/gemini-2.5-flash"        
    #model = "qwen3:30b-a3b-instruct-2507-q4_K_M"
    #model = "qwen/qwen3-next-80b-a3b-instruct"   
    #model = "x-ai/grok-4-fast"
    model = "google/gemini-2.5-flash-lite-preview-09-2025"

    #server = LLM.OLLAMA_LOCAL
    server = LLM.OPEN_ROUTER

    print(f"Using model:  \033[1;32m {model} \033[0m")
    st = time.perf_counter()
    llm_response = LLMFactory.query_llm(server=server,
                                 model=model, 
                                 system_prompt="You are a helpful assistant", 
                                 temp=0.0, user_prompt=prompt)
    et = time.perf_counter()
    print(f"LLM response time: {et - st:0.2f} seconds")    
    #print(llm_response)
    parsed_recs = PromptFactory.tryparse_llm(llm_response)   
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)    
    assert len(parsed_recs) == num_recs    
    for rec in parsed_recs:
        sku = rec['sku']
        assert sku not in [product['sku'] for product in profile.cart], f"SKU {sku} should not be in cart"
        this_product = next((p for p in products if p.sku == sku), None)
        if this_product:
            #assert this_product is not None, f"Product with SKU {first_sku} not found in products list."
            #assert sku != viewing_product.sku, f"Recommended SKU {sku} should not be the same as viewing SKU {viewing_product.sku}"
            print(f"Recommended: {this_product.name} \033[1;32m (SKU: {rec['sku']}) \033[0m - Reason: {rec['reason']}")
    
    analyze_recommendations_with_sequential(viewing_product.sku, [rec['sku'] for rec in parsed_recs])
    analyze_recommendations(viewing_product.sku, [rec['sku'] for rec in parsed_recs])



def analyze_recommendations_with_sequential(sku: str, rec_skus: List[str] = None):
    """Analyze recommendations including sequential purchase patterns"""
    if not sku:
        raise ValueError("SKU must be provided for analysis")
    if rec_skus is None:
        raise ValueError("rec_skus must be provided for analysis")    
    
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    predictor = OrderPrediction(db)
    print(f"Analyzing SKU: {sku}")
    print(f"Recommendation SKUs: {rec_skus}")
    
    # Get sequential patterns
    sequential = predictor.find_sequential_orders(sku, rec_skus)
    print("\nSequential Patterns Found:")
    print(f"  Customers: {sequential['summary_stats']['total_customers']}")
    print(f"  Sequential orders: {sequential['summary_stats']['total_sequential_orders']}")
    
    if sequential['summary_stats']['total_customers'] > 0:
        print(f"  Average days between purchases: {sequential['summary_stats']['avg_days_between_purchases']}")
        print(f"  Total revenue: ${sequential['summary_stats']['total_rec_revenue']}")
        
        print("\nRecommendation Performance:")
        for rec_sku in sequential['summary_stats']['rec_sku_frequency'].keys():
            frequency = sequential['summary_stats']['rec_sku_frequency'][rec_sku]
            unique_customers = sequential['summary_stats']['rec_sku_unique_customers'][rec_sku]
            conversion = sequential['summary_stats']['conversion_rate_by_sku'][rec_sku]
            avg_purchases = sequential['summary_stats']['avg_purchases_per_customer_by_sku'][rec_sku]
            
            print(f"  {rec_sku}: {frequency} total purchases by {unique_customers} customers "
                  f"({conversion}% conversion, {avg_purchases} purchases/customer)")
    
    # Get same-order patterns (fix the variable names)
    same_order_patterns = predictor.find_similar_orders(sku, rec_skus)
    same_order_count = same_order_patterns['total_orders']
    sequential_count = sequential['summary_stats']['total_sequential_orders']
    
    print("\nComprehensive Analysis:")
    print(f"  Same-order patterns: {same_order_count}")
    print(f"  Sequential patterns: {sequential_count}")
    
    # Show co-occurrence stats if any
    if same_order_count > 0:
        print("\nSame-Order Co-occurrence Stats:")
        for rec_sku, count in same_order_patterns['co_occurrence_stats'].items():
            print(f"  {rec_sku}: appears with {sku} in {count} orders")
    
    db.close()

def analyze_recommendations(sku: str, rec_skus: List[str] = None):
    if not sku:
        raise ValueError("SKU must be provided for analysis")
    if rec_skus is None:
        raise ValueError("rec_skus must be provided for analysis")    
   
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    predictor = OrderPrediction(db) 
    similar_orders = predictor.find_similar_orders(sku, rec_skus)
    print(f"Found {similar_orders['total_orders']} orders with co-occurrences")
    for order in similar_orders['orders']:
        print(f"Order ID: {order['order_id']}, Total: {order['grand_total']}, "
              f"Status: {order['status']}, Rec SKUs: {order['matching_rec_skus']}")
    print("Co-occurrence stats:", similar_orders['co_occurrence_stats'])    
    
    strength_analysis = predictor.get_recommendation_strength(sku, rec_skus)
    print(f"\nBase SKU appears in {strength_analysis['base_sku_orders']} orders")
    print("Recommendation Strength Analysis:")
    for rec in strength_analysis['recommendations']:
        print(f"  {rec['sku']}: {rec['strength_percentage']}% strength, "
              f"${rec['total_revenue']} revenue, {rec['total_items_sold']} items sold")
    
    # Get customer patterns
    patterns = predictor.get_customer_purchase_patterns(sku, rec_skus)
    print(f"\nFound {patterns['total_customers']} customers with purchase patterns")
    
    db.close()




