import json
import os
import re
import bittensor as bt
import pandas as pd
from typing import Counter
from pydantic import BaseModel
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum


class CatalogProvider(Enum):
    SHOPIFY = 1
    AMAZON = 2
    WOOCOMMERCE = 3
    BIGCOMMERCE = 4


@dataclass
class Product:
    sku: str
    name: str
    price: str
    

    @staticmethod
    def tryload_catalog(file_path: str, max_rows=100_000) -> list:
        """
        Try to load a woo catalog into a normalized list

        :param file_path: Path to the WooCommerce CSV file
        :param max_rows: Maximum number of rows to process
        :return: List of dictionaries with 'sku', 'name', 'price'
        """
        try:
            df = pd.read_csv(file_path)
            #WooCommerce Format
            columns = ["ID", "Type", "SKU", "Name", "Published", "Description", "In stock?", "Stock", "Regular price", "Categories"]            
            df = df[[c for c in columns if c in df.columns]]            
            df['Description'] = df['Description'].str.replace(r'<[^<>]*>', '', regex=True)
            
            #Only take simple and variable products
            #product_types = ["simple", "variable"]
            #df = df[df['Type'].isin(product_types)]

            #Final renaming of columns
            df = df.rename(columns={'SKU': 'sku', 'Name': 'name', 'Regular price': 'price', 'In stock?': 'InStock', 'Stock': 'OnHand'})            
            df.fillna(' ', inplace=True)
            df = df.head(max_rows)
            df = df.to_dict(orient='records')
            return df
        except Exception as e:
            bt.logging.error(str(e))
            return []
        
        
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
                thing = Product.tryload_catalog(file_path, max_rows)
                return json.dumps(thing, indent=2)
            case CatalogProvider.SHOPIFY:
                thing = Product.tryload_catalog_shopify(file_path, max_rows)
                return json.dumps(thing, indent=2)
            case _:
               raise ValueError(f"Invalid provider: {provider}")
            

    @staticmethod
    def try_parse_context(context: str) -> list["Product"]:
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
    def get_dupe_count(products: list["Product"]) -> int:
        try:
            if not products or len(products) == 0:
                return 0
            sku_counts = Counter(product.sku for product in products)
            return sum(count - 1 for count in sku_counts.values() if count > 1)
        except AttributeError:
            return 0
        
    
    @staticmethod
    def dedupe(products: list["Product"]) -> list["Product"]:
        unique_products = {}
        for product in products:
            if product.sku not in unique_products:
                unique_products[product.sku] = product
        return list(unique_products.values())
        
        
    @staticmethod
    def convert(context: str, provider: CatalogProvider) -> list["Product"]:
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
            case _:
                raise NotImplementedError("invalid provider")


class BaseConverter(BaseModel):
    
    @abstractmethod
    def convert(self, context: str) -> list["Product"]:
        raise NotImplementedError("BaseConverter not implemented")
    
    def clean(self, raw_value: str) -> str:        
        result = re.sub(r"[^A-Za-z0-9 ]", "", raw_value)
        return result.strip()
    

class WoocommerceConverter(BaseConverter):    
  
    def convert(self, context: str) -> list["Product"]:
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
                price = str(price)
                name = self.clean(name)
                result.append(Product(sku=sku, name=name, price=price))
            except Exception as e:
                bt.logging.error(f"WoocommerceConverter.convert Exception: {e}")
                continue
        return result
        

    
class AmazonConverter(BaseConverter):
    
    def convert(self, context: str) -> list["Product"]:
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
    
    def convert(self, context: str) -> list["Product"]:
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
     
    

    
class BigcommerceConverter(BaseConverter):
    
    def convert(self, context: str) -> list["Product"]:
        raise NotImplementedError("Bigcommerce not implemented")