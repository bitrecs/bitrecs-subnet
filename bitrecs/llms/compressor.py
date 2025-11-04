import csv
import io
import json
import bittensor as bt
from typing import List, Dict, Any


def compress_catalog(data: str) -> str:
    """
    Compress a JSON string representation of a catalog into a compact CSV-like string.
    Assumes the JSON string represents a list of dictionaries with consistent keys.
    Output: Header row followed by data rows, separated by newlines.
    """
    try:        
        json_data = json.loads(data)
        return compress_json(json_data)
    except Exception as e:
        bt.logging.error(f"Error compressing JSON string: {e}")
        return data

def compress_json(data: List[Dict[str, Any]]) -> str:
    """
    Compress a list of dictionaries into a compact CSV-like string.
    Assumes all dicts have the same keys (in consistent order).
    Output: Header row followed by data rows, separated by newlines.
    """
    if not data:
        return ""
    
    # Get keys from the first dict (assumes consistent keys)
    keys = list(data[0].keys())
    
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(keys)
    for item in data:
        if set(item.keys()) != set(keys):
            raise ValueError("Inconsistent keys in data")
        row = [item.get(key, "") for key in keys]
        writer.writerow(row)
    
    return output.getvalue().rstrip('\n')



# def decompress_json(compressed: str) -> List[Dict[str, Any]]:
#     """
#     Decompress a CSV-like string back into a list of dictionaries.
#     Assumes the first row is headers, followed by data rows.
#     """
#     if not compressed.strip():
#         return []
    
#     # Use StringIO to read CSV
#     input_io = io.StringIO(compressed)
#     reader = csv.reader(input_io)
    
#     rows = list(reader)
#     if len(rows) < 2:
#         return []  # No data rows
    
#     headers = rows[0]
#     data = []
#     for row in rows[1:]:
#         if len(row) == len(headers):
#             item = {headers[i]: row[i] for i in range(len(headers))}
#             data.append(item)
    
#     return data