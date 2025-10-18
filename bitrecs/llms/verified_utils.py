import json
import hashlib
import secrets
import bittensor as bt
from typing import Tuple

def sign_verified_request(miner_wallet: "bt.Wallet", provider: str, payload: dict, ts: str) -> Tuple[str, str]:
    nonce = secrets.token_hex(16)
    payload_str = json.dumps({
        "hotkey": miner_wallet.hotkey.ss58_address,
        "provider": provider,
        "nonce": nonce,
        "payload": payload,
        "timestamp": ts
    }, separators=(',', ':'), sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode('utf-8')).digest()
    signature_bytes = miner_wallet.hotkey.sign(payload_hash)
    signature_hex = signature_bytes.hex()
    return signature_hex, nonce