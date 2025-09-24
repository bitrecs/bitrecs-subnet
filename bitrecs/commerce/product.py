import os
import re
import json
import bittensor as bt
import pandas as pd
import bitrecs.utils.constants as CONST
from abc import abstractmethod
from enum import Enum
from typing import Any, Counter, Dict, Set
from pydantic import BaseModel
from dataclasses import asdict, dataclass


class CatalogProvider(Enum):
    BITRECS = 0
    SHOPIFY = 1
    AMAZON = 2
    WOOCOMMERCE = 3
    BIGCOMMERCE = 4
    WALMART = 5


@dataclass
class Product:
    sku: str
    name: str
    price: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(',', ':'))


class ProductFactory:

    @staticmethod
    def tryload_catalog(file_path: str, max_rows=100_000) -> list:
        """
        Try to load a woo catalog into a normalized list

        :param file_path: Path to the WooCommerce CSV file
        :param max_rows: Maximum number of rows to process
        :return: List of dictionaries with 'sku', 'name', 'price'
        """
        try:
            if not os.path.exists(file_path):   
                bt.logging.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            #WooCommerce Format
            columns = ["ID", "Type", "SKU", "Name", "Published", "Description", "In stock?", "Stock", "Regular price", "Categories"]
            df = df[[c for c in columns if c in df.columns]]            
            
            if 'Description' in df.columns:
                df['Description'] = df['Description'].str.replace(r'<[^<>]*>', '', regex=True)
            
            #Only take simple and variable products
            #product_types = ["simple", "variable"]
            #df = df[df['Type'].isin(product_types)]

            #Final renaming of columns
            df = df.rename(columns={'SKU': 'sku', 'Name': 'name', 'Regular price': 'price', 'In stock?': 'InStock', 'Stock': 'OnHand'})            
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype(object)
            df.fillna('', inplace=True)
            
            # df = df.sort_values(by=['sku', 'name', 'price'], 
            #                   ascending=[True, True, True],
            #                   na_position='last')

            df = df.sort_values(by=['name', 'price'], 
                            ascending=[True, True],
                            na_position='last')

            df = df.head(max_rows)
            df = df.to_dict(orient='records')
            return df
        except Exception as e:
            bt.logging.error(str(e))
            return []
        
        
        
    @staticmethod
    def tryload_catalog_to_json(provider: CatalogProvider, file_path: str, max_rows=100_000) -> str:
        """
        Convert a product export .csv into JSON

        """
        if not os.path.exists(file_path):   
            bt.logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        match provider:
            case CatalogProvider.WOOCOMMERCE:
                thing = ProductFactory.tryload_catalog(file_path, max_rows)
                return json.dumps(thing, indent=2)
            case CatalogProvider.SHOPIFY:
                thing = ShopifyConverter.tryload_catalog_shopify(file_path, max_rows)                
                return json.dumps(thing, indent=2)
            case CatalogProvider.WALMART:
                thing = WalmartConverter.tryload_catalog(file_path, max_rows)        
                return json.dumps(thing, indent=2)
            case _:
               raise ValueError(f"Invalid provider: {provider}")
 

    @staticmethod
    def try_parse_context(context: str) -> list[Product]:
        """
        Default converter expects a json array of products with sku/name/price fields

        """
        try:
            store_catalog: list[Product] = json.loads(context)
            return store_catalog
        except Exception as e:
            bt.logging.error(f"try_parse_context Exception: {e}")
            return []
        
        
    @staticmethod
    def try_parse_context_strict(context: str) -> list[Product]:
        """
        Strict converter expects a json array of products with sku/name/price fields

        """ 
        result: list[Product] = []        
        try:
            products_data = json.loads(context)

            for product in products_data:
                sku = product.get("sku")
                name = product.get("name")
                price = product.get("price", "0")
                if not (sku and name and price):
                    continue
                
                sku = str(sku)
                name = str(name)
                price = str(price)
                name = CONST.RE_PRODUCT_NAME.sub("", name).strip()
                if not name or not sku:
                    continue
                    
                result.append(Product(sku=sku, name=name, price=price))
        except Exception as e:
            bt.logging.error(f"try_parse_context_strict Exception: {e}")
            return []
        
        #result.sort(key=operator.attrgetter('name'))
        result = sorted(result, key=lambda x: (x.sku.lower(), x.name.lower()))
        return result

   
    @staticmethod
    def get_dupe_count(products: list[Product]) -> int:       
        try:
            if not products or len(products) == 0:
                return 0
            
            sku_counts = Counter(
                product.sku if isinstance(product, Product) else product.get('sku')
                for product in products
            )
            return sum(count - 1 for count in sku_counts.values() if count > 1)
        except AttributeError as a:
            bt.logging.error(f"WARNING - get_dupe_count failed: {a}")
            return -1
        except Exception as e:
            bt.logging.error(f"ERROR - get_dupe_count encountered an unexpected error: {e}")
            return -1    


    @staticmethod
    def dedupe(products: list[Product]) -> Set[Product]:
        """Dedupe and sort"""
        seen = set()
        deduped = []
        for product in products:
            if product.sku not in seen:
                seen.add(product.sku)
                deduped.append(product)
        deduped = sorted(deduped, key=lambda x: (x.name.lower(), x.price))
        return deduped
    
               
    @staticmethod
    def check_all_have_sku(product_list: list) -> bool:
        try:
            product_dicts = []
            for product in product_list:
                try:
                    product_dict = json.loads(product.replace("'", '"'))
                    if isinstance(product_dict, dict):
                        product_dicts.append(product_dict)
                    else:
                        bt.logging.error(f"Product is not a dictionary: {product}")
                except json.JSONDecodeError as e:
                    bt.logging.error(f"JSON parsing error: {e} for product: {product}")
                    continue

            all_have_sku = all('sku' in product for product in product_dicts)
            return all_have_sku
        except Exception as e:
            bt.logging.error(f"Unexpected error in check_all_have_sku: {e}")
            return False


    @staticmethod
    def find_sku_name(target_sku: str, catalog_json: str) -> str:
        """
        Case-insensitive regex search for SKU name in JSON catalog.
        Returns the name field for the matching SKU.
        """
        # Case-insensitive pattern with flexible spacing
        pattern = rf'"sku"\s*:\s*"{re.escape(target_sku)}"[^}}]*"name"\s*:\s*"([^"]+)"'        
        match = re.search(pattern, catalog_json, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""
    

    @staticmethod
    def find_sku_name_slow(target_sku: str, catalog_json: str) -> str:
        """
        Slower but more robust method to find SKU name in JSON catalog.
        Parses the JSON and searches for the SKU.
        """
        try:
            products = ProductFactory.try_parse_context_strict(catalog_json)
            match = next((p for p in products if p.sku.lower() == target_sku.lower()), None)
            return match.name if match else ""
        except Exception as e:
            bt.logging.error(f"Error in find_sku_name_slow: {e}")
            return ""


        
    @staticmethod
    def convert(context: str, provider: CatalogProvider) -> list[Product]:
        """
            Convert a raw store catalog json into Products

        """
        match provider:
            case CatalogProvider.SHOPIFY:
                return ShopifyConverter().convert(context)                
            case CatalogProvider.AMAZON:
                return AmazonConverter().convert(context)
            case CatalogProvider.WOOCOMMERCE:
                return WoocommerceConverter().convert(context)
            case CatalogProvider.BIGCOMMERCE:
                return BigcommerceConverter().convert(context)
            case CatalogProvider.WALMART:
                return WalmartConverter().convert(context)
            case _:
                raise NotImplementedError("invalid provider")


class BaseConverter(BaseModel):
    
    @abstractmethod
    def convert(self, context: str) -> list[Product]:
        raise NotImplementedError("BaseConverter not implemented")
    
    def clean(self, raw_value: str) -> str:       
        result = CONST.RE_PRODUCT_NAME.sub("", raw_value)
        return result.strip()
    

class WoocommerceConverter(BaseConverter):    
  
    def convert(self, context: str) -> list[Product]:
        """
        converts from product_catalog.csv to Products

        args:
            context: str - woocommerce product export converted to json            

        """
        result : list[Product] = []
        for p in json.loads(context):
            try:
                sku = p.get("sku")
                name = p.get("name")
                price = p.get("price", "0.00")             
                if not sku or not name:
                    continue
                if price is None or price == 'None':
                    price = "0.00"
                sku = str(sku)
                price = str(price)
                name = self.clean(name)
                result.append(Product(sku=sku, name=name, price=price))
            except Exception as e:
                bt.logging.error(f"WoocommerceConverter.convert Exception: {e}")
                continue
        return result
        

    
class AmazonConverter(BaseConverter):
    
    def convert(self, context: str) -> list[Product]:
        """
        converts from amazon_fashion_sample_1000.json format

        """
        result : list[Product] = []
        for p in json.loads(context):
            try:
                sku = p.get("asin")
                if p["metadata"]:
                    name = p["metadata"].get("title", "metadata not found")
                    price = p["metadata"].get("price", "0.00")
                if not sku or not name:
                    continue
                if "metadata not found" in name:
                    continue
                if price is None or price == 'None':                    
                    price = "0.00"
                price = str(price)
                name = self.clean(name)
                result.append(Product(sku=sku, name=name, price=price))
            except Exception as e:
                bt.logging.error(f"AmazonConverter.convert Exception: {e}")
                continue
        return result
    

class ShopifyConverter(BaseConverter):
    
    def convert(self, context: str) -> list[Product]:
        """
        converts from shopify export .csv format

        """
        result : list[Product] = []
        for p in json.loads(context):
            try:
                sku = p.get("sku")
                name = p.get("name")
                price = p.get("price", "0.00")             
                if not sku or not name:
                    continue
                if price is None or price == 'None':
                    price = "0.00"
                price = str(price)
                name = self.clean(name)
                result.append(Product(sku=sku, name=name, price=price))
            except Exception as e:
                bt.logging.error(f"ShopifyConverter.convert Exception: {e}")
                continue
        return result

    @staticmethod
    def tryload_catalog_shopify(file_path: str, max_rows=100_000) -> list:
        """
        Try to load a Shopify catalog into a normalized list
        *this will squash variants down 
        
        :param file_path: Path to the Shopify CSV file
        :param max_rows: Maximum number of rows to process
        :return: List of dictionaries with 'sku', 'name', 'price'
        """
        try:
            if not os.path.exists(file_path):   
                bt.logging.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            # Select relevant columns
            columns = [
                "Handle", "Title", "Variant SKU", "Variant Price", 
                "Option1 Name", "Option1 Value", 
                "Option2 Name", "Option2 Value", 
                "Option3 Name", "Option3 Value", 
                "Status"
            ]
            df = df[[c for c in columns if c in df.columns]]

            # Rename columns for clarity
            df = df.rename(columns={
                'Handle': 'handle',
                'Title': 'name',
                'Variant SKU': 'sku',
                'Variant Price': 'price',
                'Option1 Name': 'option1_name',
                'Option1 Value': 'option1_value',
                'Option2 Name': 'option2_name',
                'Option2 Value': 'option2_value',
                'Option3 Name': 'option3_name',
                'Option3 Value': 'option3_value'
            })

            # Clean and preprocess data
            df['name'] = df['name'].fillna('').str.replace(r'<[^<>]*>', '', regex=True)
            df['sku'] = df['sku'].astype(str).str.lstrip("'").replace('nan', '')  # Remove leading ' and invalid 'nan' values
            
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype(object)
            df.fillna('', inplace=True)

            # Fill empty names with the parent name grouped by 'handle'
            parent_names = df.groupby('handle')['name'].first().to_dict()
            df['name'] = df.apply(lambda row: parent_names.get(row['handle'], '') if row['name'] == '' else row['name'], axis=1)

            # Remove rows without a SKU
            df = df[df['sku'] != '']

            # Limit rows for processing
            df = df.head(max_rows)

            # Convert to list of dictionaries
            products = []
            for _, row in df.iterrows():
                product = {
                    'handle': row['handle'],
                    'name': row['name'],
                    'sku': row['sku'],
                    'price': row['price'],
                    'variants': []
                }

                # Add variant details if available
                for i in range(1, 4):
                    option_name = row.get(f'option{i}_name', '').strip()
                    option_value = row.get(f'option{i}_value', '').strip()
                    if option_name and option_value:
                        product['variants'].append({option_name: option_value})

                products.append(product)

            return products
        except Exception as e:
            print(f"Error loading catalog: {e}")
            return []    


class BitrecsConverter(BaseConverter):
    
    def convert(self, context: str) -> list[Product]:
        """
        converts from generic json format

        """
        result : list[Product] = []
        for p in json.loads(context):
            try:
                sku = p.get("sku")
                name = p.get("name")
                price = p.get("price", "0.00")
                if not sku or not name:
                    continue
                if price is None or price == 'None':
                    price = "0.00"
                price = str(price)
                name = self.clean(name)
                result.append(Product(sku=sku, name=name, price=price))
            except Exception as e:
                bt.logging.error(f"GenericConverter.convert Exception: {e}")
                continue
        return result

    
class BigcommerceConverter(BaseConverter):
    
    def convert(self, context: str) -> list[Product]:
        raise NotImplementedError("Bigcommerce not implemented")
    
    
class WalmartConverter(BaseConverter):    
  
    def convert(self, context: str) -> list[Product]:
        """
        converts from wallmart_30k_kaggle_trimmed.csv to Products

        args:
            context: str - wallmart_30k_kaggle_trimmed product export converted to json            

        """
        result : list[Product] = []
        for p in json.loads(context):
            try:
                sku = p.get("sku")
                name = p.get("name")
                price = p.get("price", "0.00")             
                if not sku or not name:
                    continue
                if price is None or price == 'None':
                    price = "0.00"
                sku = str(sku)
                price = str(price)
                name = self.clean(name)
                brand = p.get("brand", "")
                if brand:
                    brand = self.clean(brand)
                    name = f"{name} - {brand}"
                
                result.append(Product(sku=sku, name=name, price=price))
            except Exception as e:
                bt.logging.error(f"WalmartConverter.convert Exception: {e}")
                continue
        return result
    

    @staticmethod
    def tryload_catalog(file_path: str, max_rows=100_000) -> list:
        """
        Try to load a walmart catalog into a normalized list

        :param file_path: Path to the Walmart CSV file
        :param max_rows: Maximum number of rows to process
        :return: List of dictionaries with 'sku', 'name', 'price'
        """
        try:
            if not os.path.exists(file_path):   
                bt.logging.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)            
            columns = ["UNIQUE_ID", "PRODUCT_NAME", "LIST_PRICE", "SALE_PRICE", "BRAND", "ITEM_NUMBER", "GTIN", "CATEGORY", "IN_STOCK"]            
            df = df[[c for c in columns if c in df.columns]]            
            df['PRODUCT_NAME'] = df['PRODUCT_NAME'].str.replace(r'<[^<>]*>', '', regex=True)
            df['BRAND'] = df['BRAND'].str.replace(r'<[^<>]*>', '', regex=True)
            df['CATEGORY'] = df['CATEGORY'].str.replace(r'<[^<>]*>', '', regex=True)            
            
            #Final renaming of columns
            df = df.rename(columns={'GTIN': 'sku', 'PRODUCT_NAME': 'name', 'LIST_PRICE': 'price', 'IN_STOCK': 'InStock', 'BRAND' : 'brand'})
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype(object)
            df.fillna('', inplace=True)

            # df = df.sort_values(by=['sku', 'name', 'price'], 
            #                   ascending=[True, True, True],
            #                   na_position='last')

            df = df.sort_values(by=['name', 'price'], 
                              ascending=[True, True],
                              na_position='last')

            df = df.head(max_rows)
            df = df.to_dict(orient='records')
            return df
        except Exception as e:
            bt.logging.error(str(e))
            return []