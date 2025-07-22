import os
import json
import time
import hmac
import hashlib
import threading
import traceback
import bittensor as bt
from dotenv import load_dotenv
load_dotenv()
from dataclasses import asdict
from typing import Callable
from functools import partial
from fastapi import FastAPI, HTTPException, Request, APIRouter, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.utils import constants as CONST, epoch
from bitrecs.commerce.product import ProductFactory
from bitrecs.protocol import BitrecsRequest
from bitrecs.api.api_core import filter_allowed_ips, limiter
from bitrecs.api.utils import (
    api_key_validator, get_proxy_public_key, 
    json_only_middleware, parse_ip_whitelist
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from uvicorn.config import Config
from uvicorn.server import Server

ForwardFn = Callable[[BitrecsRequest], BitrecsRequest]

SECRET_KEY_LOCALNET = "change-me"


class ApiServer:
    app: FastAPI
    router: APIRouter
    forward_fn: ForwardFn    

    def __init__(self, validator, api_port: int, forward_fn: ForwardFn):
        self.validator = validator
        self.forward_fn = forward_fn
        self.allowed_ips = ["127.0.0.1"]
        self.bypass_whitelist: bool = True
        self.app = FastAPI()
        self.app.state.limiter = limiter
        self.network = os.environ.get("NETWORK").strip().lower() #localnet / testnet / mainnet
        self.hot_key = validator.wallet.hotkey.ss58_address

        # if self.network != "mainnet":
        #     bt.logging.warning(f"\033[1;33m WARNING - API Server is running in {self.network} mode \033[0m")
        #     raise ValueError(f"API Server is not supported in {self.network} mode, please use mainnet")
     
        self.proxy_url = os.environ.get("BITRECS_PROXY_URL").removesuffix("/")
        if not self.proxy_url:
            bt.logging.error(f"\033[1;31m ERROR - MISSING BITRECS_PROXY_URL \033[0m")
            raise Exception("Missing BITRECS_PROXY_URL")
        
        self.bitrecs_api_key = os.environ.get("BITRECS_API_KEY")
        if not self.bitrecs_api_key:
            bt.logging.error(f"\033[1;31m ERROR - MISSING BITRECS_API_KEY \033[0m")
            raise Exception("Missing BITRECS_API_KEY")
            
        
        async def general_exception_handler(request: Request, exc: Exception):
            bt.logging.error(f"Unhandled exception: {request.url} - {str(exc)}")
            return JSONResponse(
                status_code=500,
                content={
                    "status_code": 500,
                    "message": "Internal server error - General",
                    "detail" : "General",
                    "data": None
                }
            )        

        self.app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=5)
        self.app.middleware("http")(partial(json_only_middleware, self))
        self.app.middleware('http')(partial(api_key_validator, self))
        self.app.middleware("http")(partial(filter_allowed_ips, self))
        
        self.app.add_exception_handler(Exception, general_exception_handler)
        
        self.config = Config(
            app=self.app,
            host="0.0.0.0",
            port=api_port,
            log_level="trace" if bt.logging.__trace_on__ else "critical"
        )
        self.server = Server(config=self.config)
        self._server_thread = None

        self.router = APIRouter()
        self.router.add_api_route("/ping", self.ping, methods=["GET"])
        self.router.add_api_route("/version", self.version, methods=["GET"])
        if self.network == "localnet":
            self.router.add_api_route("/rec", self.generate_product_rec_localnet, methods=["POST"]) 
        elif self.network == "testnet":
            self.router.add_api_route("/rec", self.generate_product_rec_testnet, methods=["POST"])
        elif self.network == "mainnet":
            self.router.add_api_route("/rec", self.generate_product_rec_mainnet, methods=["POST"])
            self.bypass_whitelist = False
            self.allowed_ips = parse_ip_whitelist(os.environ.get("VALIDATOR_API_WHITELIST", ""))
            if len(self.allowed_ips) == 0:
                raise ValueError("No allowed IPs configured for mainnet API")
            bt.logging.info(f"\033[1;32m API Server has {len(self.allowed_ips)} IP whitelist entries \033[0m")
        else:
            raise ValueError(f"Unsupported network: {self.network}")
        self.app.include_router(self.router)
     
        try:
            bt.logging.trace(f"\033[1;33mAPI warmup, please standby ...\033[0m")
            self.proxy_key : bytes = get_proxy_public_key(self.proxy_url)
            self.public_key = Ed25519PublicKey.from_public_bytes(self.proxy_key)
        except Exception as e:
            bt.logging.error(f"\033[1;31mERROR API could not get proxy public key:  {e} \033[0m")
            bt.logging.warning(f"\033[1;33mWARNING - your validator is in limp mode, please restart\033[0m")
            raise Exception("Could not get proxy public key")
        
        bt.logging.info(f"\033[1;32m API Server initialized on {self.network} \033[0m")

    
    async def verify_request_localnet(self, request: BitrecsRequest, x_signature: str, x_timestamp: str):
        timestamp = int(x_timestamp)
        current_time = int(time.time())
        if current_time - timestamp > 300:
            raise HTTPException(status_code=401, detail="Request expired")
        
        d = {
            'created_at': request.created_at,
            'user': request.user,
            'num_results': request.num_results,
            'query': request.query,
            'context': request.context,
            'site_key': request.site_key,
            'results': request.results,
            'models_used': request.models_used,
            'miner_uid': request.miner_uid,
            'miner_hotkey': request.miner_hotkey
        }

        body_str = json.dumps(d, sort_keys=True)
        string_to_sign = f"{x_timestamp}.{body_str}"
        expected_signature = hmac.new(
            SECRET_KEY_LOCALNET.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(x_signature, expected_signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
                
        bt.logging.info(f"\033[1;32m New Request Signature Verified\033[0m")


    async def verify_request_signature(self, request: BitrecsRequest, x_signature: str, x_timestamp: str): 
        timestamp = int(x_timestamp)
        current_time = int(time.time())
        if current_time - timestamp > 300:
            bt.logging.error(f"\033[1;31m Expired Request!\033[0m")
            raise HTTPException(status_code=401, detail="Request expired")

        d = {
            'created_at': request.created_at,
            'user': request.user,
            'num_results': request.num_results,
            'query': request.query,
            'context': request.context,
            'site_key': request.site_key,
            'results': request.results,
            'models_used': request.models_used,
            'miner_uid': request.miner_uid,
            'miner_hotkey': request.miner_hotkey,
            'miner_signature': request.miner_signature
        }

        body_str = json.dumps(d, sort_keys=True)
        message = f"{x_timestamp}.{body_str}".encode('utf-8')
        signature = bytes.fromhex(x_signature)        
        try:
            self.public_key.verify(signature, message)
        except InvalidSignature:
            bt.logging.error(f"\033[1;31m Invalid signature!\033[0m")
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        bt.logging.info(f"\033[1;32m New Request - Signature Verified\033[0m")
    
    
    async def ping(self, request: Request):
        bt.logging.info(f"\033[1;32m API Server ping \033[0m")
        ts = int(time.time())        
        return JSONResponse(status_code=200, content={"detail": "pong", "ts": ts})
    
    
    async def version(self, request: Request):
        bt.logging.info(f"\033[1;32m API Server version \033[0m")
        ts = int(time.time())
        if not self.validator.local_metadata:
            bt.logging.error(f"\033[1;31m API Server version - No metadata \033[0m")
            return JSONResponse(status_code=500, content={"detail": "version", "meta_data": {}, "ts": ts, "status": "meta_data error", "metrics": {}})
        v = self.validator.local_metadata.to_dict()

        total_uids = self.validator.total_uids or []        
        suspect_miners = self.validator.suspect_miners or []
        batch_seen_uids = self.validator.batch_seen_uids or []
        batch_orphan_uids = self.validator.batch_orphan_uids or []
        last_tempo = self.validator.last_tempo or 0
        tempo_batch_index = self.validator.tempo_batch_index or 0
        batches_completed = self.validator.batches_completed or 0
        bad_set_count = self.validator.bad_set_count or 0
        coverage = (len(batch_seen_uids) / len(total_uids)) * 100 if len(total_uids) > 0 else 0        

        current_block = self.validator.block
        netuid = self.validator.config.netuid
        current_epoch, blocks_until_next_epoch, epoch_start_block = epoch.get_current_epoch_info(current_block, netuid)
        minutes_to_next_block = blocks_until_next_epoch * 12 / 60

        metrics = {
            "tempo": CONST.EPOCH_TEMPO,
            "batch_size": CONST.QUERY_BATCH_SIZE,
            "total_uids": sorted(total_uids),
            "suspect_miners": sorted(suspect_miners),
            "batch_seen_uids": sorted(batch_seen_uids),
            "batch_orphan_uids": sorted(batch_orphan_uids),
            "last_tempo": last_tempo,
            "tempo_batch_index": tempo_batch_index,
            "batches_completed": batches_completed,
            "bad_set_count": bad_set_count,
            "coverage": round(coverage, 2),
            "current_block": current_block,
            "current_epoch": current_epoch,
            "blocks_until_next_epoch": blocks_until_next_epoch,
            "epoch_start_block": epoch_start_block,
            "minutes_to_next_epoch": round(minutes_to_next_block, 2)
        }

        return JSONResponse(status_code=200, content={"detail": "version", "meta_data": v, "ts": ts, "status": "OK", "metrics": metrics})
    
    
    async def generate_product_rec_localnet(
            self, 
            request: BitrecsRequest,
            x_signature: str = Header(...),
            x_timestamp: str = Header(...)
    ):  
        """
            Main Bitrecs Handler - localnet

            Generate n recommendations for a given query and context.
            Query is sent to random miners to generate a valid response in a reasonable time.            

        """

        try:
          
            await self.verify_request_localnet(request, x_signature, x_timestamp)

            store_catalog = ProductFactory.try_parse_context(request.context)
            catalog_size = len(store_catalog)
            bt.logging.trace(f"REQUEST CATALOG SIZE: {catalog_size}")
            if catalog_size < CONST.MIN_CATALOG_SIZE or catalog_size > CONST.MAX_CATALOG_SIZE:
                bt.logging.error(f"API invalid catalog size")                
                return JSONResponse(status_code=400,
                                    content={"detail": "error - invalid catalog", "status_code": 400})            
            
            dupes = ProductFactory.get_dupe_count(store_catalog)
            if dupes > catalog_size * CONST.CATALOG_DUPE_THRESHOLD:
                bt.logging.error(f"API Too many duplicates in catalog: {dupes}")                
                return JSONResponse(status_code=400,
                                    content={"detail": "error - dupe threshold reached", "status_code": 400})

            st = time.perf_counter()
            response = await self.forward_fn(request)
            total_time = time.perf_counter() - st

            if len(response.results) == 0:
                bt.logging.error(f"API forward_fn response has no results")                
                return JSONResponse(status_code=500,
                                    content={"detail": "error - forward", "status_code": 500})

            final_recs = [json.loads(idx.replace("'", '"')) for idx in response.results]            
            response_text = "Bitrecs Took {:.2f} seconds to process this request".format(total_time)

            response = {
                "user": "", 
                "original_query": response.query,
                "status_code": "200", #front end widgets expects this do not change
                "status_text": "OK", #front end widgets expects this do not change
                "response_text": response_text,
                "created_at": response.created_at,
                "results": final_recs,
                "models_used": response.models_used,
                "catalog_size": str(catalog_size),
                "miner_uid": response.miner_uid,
                "miner_hotkey": response.miner_hotkey,
                "reasoning": f"Bitrecs AI - {self.network}"
            }
            
            return JSONResponse(status_code=200, content=response)
        
        except HTTPException as h:
            bt.logging.error(f"\033[31m HTTP ERROR API generate_product_rec_localnet:\033[0m {h}")            
            return JSONResponse(status_code=h.status_code,
                                content={"detail": "error", "status_code": h.status_code})

        except Exception as e:
            bt.logging.error(f"\033[31m ERROR API generate_product_rec_localnet:\033[0m {e}")            
            return JSONResponse(status_code=500,
                                content={"detail": "error", "status_code": 500})
        

    async def generate_product_rec_testnet(
            self, 
            request: BitrecsRequest,
            x_signature: str = Header(...),
            x_timestamp: str = Header(...)
    ):  
        """
            Main Bitrecs Handler - testnet

            Generate n recommendations for a given query and context.
            Query is sent to random miners to generate a valid response in a reasonable time.            

        """

        try:
            st_a = int(time.time())

            await self.verify_request_signature(request, x_signature, x_timestamp)

            if len(request.context) > 100_000:
                tc = PromptFactory.get_token_count(request.context)
                if tc > CONST.MAX_CONTEXT_TOKEN_COUNT:
                    bt.logging.error(f"API context too large: {tc} tokens")
                    return JSONResponse(status_code=400,
                                        content={"detail": "error - context too large", "status_code": 400})

            store_catalog = ProductFactory.try_parse_context_strict(request.context)
            catalog_size = len(store_catalog)
            bt.logging.trace(f"REQUEST CATALOG SIZE: {catalog_size}")
            if catalog_size < CONST.MIN_CATALOG_SIZE or catalog_size > CONST.MAX_CATALOG_SIZE:
                bt.logging.error(f"API invalid catalog size")
                return JSONResponse(status_code=400,
                                    content={"detail": "error - invalid catalog - size", "status_code": 400})
            
            request.context = json.dumps([asdict(store_catalog) for store_catalog in store_catalog], separators=(',', ':'))
            sn_t = time.perf_counter()
            response = await self.forward_fn(request)
            subnet_time = time.perf_counter() - sn_t
            response_text = "Bitrecs Subnet {} Took {:.2f} seconds to process this request".format(self.network, subnet_time)
            bt.logging.trace(response_text)

            # if len(response.results) == 0:
            #     bt.logging.error(f"API forward_fn response has no results")
            #     return JSONResponse(status_code=500,
            #                         content={"detail": "error - forward", "status_code": 500})
            
            # if not CONST.MIN_NUM_RESULTS <= len(response.results) <= CONST.MAX_NUM_RESULTS:
            #     bt.logging.error(f"API forward_fn response has num_recs out of bounds: {len(response.results)}")
            #     return JSONResponse(status_code=500,
            #                         content={"detail": "error - forward", "status_code": 500})

            if len(response.results) != request.num_results:
                bt.logging.error(f"API forward_fn response has a num_recs mismatch")
                return JSONResponse(status_code=500,
                                    content={"detail": "error - forward", "status_code": 500})

            # if 1==2:
            #     #TODO: reward is not 100% strict as we tolerate llms returning good enough json
            #     #however our standard to return to clients must be strict and fail gracefully
            #     final_recs = [None] * len(response.results)  # Pre-allocate list with same length
            #     for i, idx in enumerate(response.results):
            #         try:
            #             repaired = repair_json(idx)
            #             rec = json.loads(repaired)
            #             standardized = json.dumps(rec)
            #             final_recs[i] = json.loads(standardized)
            #         except Exception as e:
            #             bt.logging.error(f"Failed to standardize result at index {i}: {idx}, error: {e}")
            #             final_recs[i] = None  # Mark failed entries as None

            #     # Remove any None entries while preserving order
            #     final_recs = [r for r in final_recs if r is not None]

            #final_recs = [json.loads(idx.replace("'", '"')) for idx in response.results]
            
            final_recs = [json.loads(idx) for idx in response.results]
            response = {
                "user": "", 
                "original_query": response.query,
                "status_code": "200", #front end widgets expects this do not change
                "status_text": "OK", #front end widgets expects this do not change
                "response_text": response_text,
                "created_at": response.created_at,
                "results": final_recs,
                "models_used": response.models_used,
                "catalog_size": str(catalog_size),
                "miner_uid": response.miner_uid,
                "miner_hotkey": response.miner_hotkey,
                "reasoning": f"Bitrecs AI - {self.network}"
            }
            et_a = int(time.time())
            total_duration = et_a - st_a
            bt.logging.info("\033[1;32mValidator - Processed request in {:.2f} seconds \033[0m".format(total_duration))
            return JSONResponse(status_code=200, content=response)
        
        except HTTPException as h:
            bt.logging.error(f"\033[31m HTTP ERROR API generate_product_rec_testnet:\033[0m {h}")            
            return JSONResponse(status_code=h.status_code,
                                content={"detail": "error", "status_code": h.status_code})

        except Exception as e:
            bt.logging.error(f"\033[31m ERROR API generate_product_rec_testnet:\033[0m {e}")
            bt.logging.error(f"\033[31m TRACEBACK:\033[0m\n{traceback.format_exc()}")
            bt.logging.trace(response)
            return JSONResponse(status_code=500,
                                content={"detail": "error", "status_code": 500})
        

    async def generate_product_rec_mainnet(
            self, 
            request: BitrecsRequest,
            x_signature: str = Header(...),
            x_timestamp: str = Header(...)
    ):  
        """
            Main Bitrecs Handler - mainnet

            Generate n recommendations for a given query and context.
            Query is sent to random miners to generate a valid response in a reasonable time.            

        """

        try:
            st_a = int(time.time())

            await self.verify_request_signature(request, x_signature, x_timestamp)

            if len(request.context) > 100_000:
                tc = PromptFactory.get_token_count(request.context)
                if tc > CONST.MAX_CONTEXT_TOKEN_COUNT:
                    bt.logging.error(f"API context too large: {tc} tokens")
                    return JSONResponse(status_code=400,
                                        content={"detail": "error - context too large", "status_code": 400})

            store_catalog = ProductFactory.try_parse_context_strict(request.context)
            catalog_size = len(store_catalog)
            bt.logging.trace(f"REQUEST CATALOG SIZE: {catalog_size}")
            if catalog_size < CONST.MIN_CATALOG_SIZE or catalog_size > CONST.MAX_CATALOG_SIZE:
                bt.logging.error(f"API invalid catalog size: {catalog_size} skus")
                return JSONResponse(status_code=400,
                                    content={"detail": "error - invalid catalog - size", "status_code": 400})           
            
            
            request.context = json.dumps([asdict(store_catalog) for store_catalog in store_catalog], separators=(',', ':'))
            sn_t = time.perf_counter()
            response : BitrecsRequest = await self.forward_fn(request)
            subnet_time = time.perf_counter() - sn_t
            response_text = "Bitrecs Subnet {} Took {:.2f} seconds to process this request".format(self.network, subnet_time)
            bt.logging.trace(response_text)

            # if not CONST.MIN_NUM_RESULTS <= len(response.results) <= CONST.MAX_NUM_RESULTS:
            #     bt.logging.error(f"API forward_fn response has num_recs out of bounds: {len(response.results)}")
            #     return JSONResponse(status_code=500,
            #                         content={"detail": "error - forward", "status_code": 500})

            if len(response.results) != request.num_results:
                bt.logging.error(f"API forward_fn response has a num_recs mismatch")
                return JSONResponse(status_code=500,
                                    content={"detail": "error - forward", "status_code": 500})            
            
            final_recs = [json.loads(idx) for idx in response.results]
            response = {
                "user": "",
                "original_query": response.query,
                "status_code": "200", #front end widgets expects this do not change
                "status_text": "OK", #front end widgets expects this do not change
                "response_text": response_text,
                "created_at": response.created_at,
                "results": final_recs,
                "models_used": response.models_used,
                "catalog_size": str(catalog_size),
                "miner_uid": response.miner_uid,
                "miner_hotkey": response.miner_hotkey,
                "reasoning": f"Bitrecs AI - {self.network}"
            }
            et_a = int(time.time())
            total_duration = et_a - st_a
            bt.logging.info("\033[1;32mValidator - Processed request in {:.2f} seconds \033[0m".format(total_duration))
            return JSONResponse(status_code=200, content=response)
        
        except HTTPException as h:
            bt.logging.error(f"\033[31m HTTP ERROR API generate_product_rec_mainnet:\033[0m {h}")            
            return JSONResponse(status_code=h.status_code,
                                content={"detail": "error", "status_code": h.status_code})

        except Exception as e:            
            bt.logging.error(f"\033[31m ERROR API generate_product_rec_mainnet:\033[0m {e}")
            bt.logging.error(f"\033[31m TRACEBACK:\033[0m\n{traceback.format_exc()}")
            #bt.logging.trace(response)
            return JSONResponse(status_code=500,
                                content={"detail": "error", "status_code": 500})


    def start(self):
        """Start the API server in a dedicated thread"""
        if self._server_thread is not None:
            bt.logging.warning("API server is already running")
            return

        def run_server():
            self.server.run()

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        bt.logging.info(f"API server started at {self.config.host}:{self.config.port}")


    def stop(self):
        """Stop the API server and cleanup"""
        if self._server_thread is None:
            bt.logging.warning("API server is not running")
            return
        
        self.server.should_exit = True
        self._server_thread.join(timeout=5)
        if self._server_thread.is_alive():
            bt.logging.warning("API server thread did not stop gracefully")
        self._server_thread = None
        bt.logging.info("API server stopped")


    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()