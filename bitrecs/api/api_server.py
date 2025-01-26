
import os
import json
import time
import uvicorn
import bittensor as bt
import hmac
import hashlib
from typing import Callable
from functools import partial
from bittensor.core.axon import FastAPIThreadedServer
from fastapi import FastAPI, HTTPException, Request, APIRouter, Response, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded
from bitrecs.utils import constants as CONST
from bitrecs.commerce.product import ProductFactory
from bitrecs.protocol import BitrecsRequest
from bitrecs.api.api_counter import APICounter
from bitrecs.api.api_core import OnlyJSONMiddleware, filter_allowed_ips, limiter
from bitrecs.api.utils import api_key_validator, get_proxy_public_key
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from dotenv import load_dotenv
load_dotenv()

ForwardFn = Callable[[BitrecsRequest], BitrecsRequest]

SECRET_KEY = "change-me"
PROXY_URL = os.environ.get("BITRECS_PROXY_URL").removesuffix("/")


class ApiServer:
    app: FastAPI
    fast_server: FastAPIThreadedServer
    router: APIRouter
    forward_fn: ForwardFn    

    def __init__(self, validator, axon_port: int, forward_fn: ForwardFn):
        self.validator = validator
        self.forward_fn = forward_fn
        self.allowed_ips = ["127.0.0.1", "10.0.0.1"]
        self.bypass_whitelist = True

        self.app = FastAPI()
        self.app.state.limiter = limiter
        self.bitrecs_api_key = os.environ.get("BITRECS_API_KEY")
        if not self.bitrecs_api_key:
            bt.logging.error(f"\033[1;31m ERROR - MISSING BITRECS_API_KEY \033[0m")
            raise Exception("Missing BITRECS_API_KEY")
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            bt.logging.error(f"Unhandled exception: {request.url} - {str(exc)}")
            return JSONResponse(
                status_code=500,
                content={
                    "status_code": 500,
                    "message": "Internal server error - General",
                    "data": None
                }
            )
        
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
            bt.logging.error(f"{request}: {exc_str}")
            content = {'status_code': 10422, 'message': exc_str, 'data': None}
            return JSONResponse(content=content, status_code=422)
        
        
        @self.app.exception_handler(RateLimitExceeded)
        async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
            bt.logging.warning(f"Rate limit exceeded for {request.client.host}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "status_code": 429,
                    "retry_after": exc.retry_after if hasattr(exc, 'retry_after') else 60
                },
                headers={"Retry-After": str(exc.retry_after if hasattr(exc, 'retry_after') else 60)}
            )
        
        self.app.middleware("http")(partial(filter_allowed_ips, self))
        self.app.middleware('http')(partial(api_key_validator, self))
        self.app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=5)
        self.app.add_middleware(OnlyJSONMiddleware)
      
        self.hot_key = validator.wallet.hotkey.ss58_address
        self.proxy_public_key : bytes = None
        self.network = os.environ.get("NETWORK").strip().lower() #localnet / testnet / mainnet

        self.fast_server = FastAPIThreadedServer(config=uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=axon_port,
            log_level="trace" if bt.logging.__trace_on__ else "critical",         
        ))

        self.router = APIRouter()
        self.router.add_api_route(
            "/ping", 
            self.ping,
            methods=["GET"]            
        )
        self.router.add_api_route(
            "/version", 
            self.version,
            methods=["GET"]            
        )

        if self.network == "localnet":
            self.router.add_api_route(
                "/rec",
                self.generate_product_rec_localnet,
                methods=["POST"]                
            ) 
        elif self.network == "testnet":
             self.router.add_api_route(
                "/rec",
                self.generate_product_rec_testnet,
                methods=["POST"]                
            )
        else:
            raise not NotImplementedError("Mainnet API not implemented")

        self.app.include_router(self.router)
     
        try:
            bt.logging.trace(f"\033[1;33mAPI warmup, please standby ...\033[0m")
            self.proxy_public_key = get_proxy_public_key(PROXY_URL)
        except Exception as e:
            bt.logging.error(f"\033[1;31mERROR API could not get proxy public key:  {e} \033[0m")
            bt.logging.warning(f"\033[1;33mWARNING - your validator is in limp mode, please restart\033[0m")
        
        self.api_counter = APICounter(os.path.join(self.app.root_path, "api_counter.json"))
        bt.logging.info(f"\033[1;32m API Counter set {self.api_counter.save_path} \033[0m")
        
        bt.logging.info(f"\033[1;32m API Server initialized on {self.network} \033[0m")

    
    async def verify_request(self, request: BitrecsRequest, x_signature: str, x_timestamp: str):     
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
            SECRET_KEY.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(x_signature, expected_signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        timestamp = int(x_timestamp)
        current_time = int(time.time())
        if current_time - timestamp > 300:  # 5 minutes
            raise HTTPException(status_code=401, detail="Request expired")
        
        bt.logging.info(f"\033[1;32m New Request Signature Verified\033[0m")


    async def verify_request2(self, request: BitrecsRequest, x_signature: str, x_timestamp: str): 
        timestamp = int(x_timestamp)
        current_time = int(time.time())
        if current_time - timestamp > 300:  # 5 minutes
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
            'miner_hotkey': request.miner_hotkey
        }

        body_str = json.dumps(d, sort_keys=True)
        message = f"{x_timestamp}.{body_str}".encode('utf-8')
        signature = bytes.fromhex(x_signature)
        public_key = Ed25519PublicKey.from_public_bytes(self.proxy_public_key)
        try:
            public_key.verify(signature, message)
        except InvalidSignature:
            bt.logging.error(f"\033[1;31m Invalid signature!\033[0m")
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        bt.logging.info(f"\033[1;32m New Request - Signature Verified\033[0m")
    
    
    async def ping(self, request: Request):
        bt.logging.info(f"\033[1;32m API Server ping \033[0m")
        st = int(time.time())
        return JSONResponse(status_code=200, content={"detail": "pong", "st": st})
    
    
    async def version(self, request: Request):
        bt.logging.info(f"\033[1;32m API Server version \033[0m")
        st = int(time.time())
        if not self.validator.local_metadata:
            bt.logging.error(f"\033[1;31m API Server version - No metadata \033[0m")
            return JSONResponse(status_code=200, content={"detail": "version", "meta_data": {}, "st": st})
        v = self.validator.local_metadata.to_dict()
        return JSONResponse(status_code=200, content={"detail": "version", "meta_data": v, "st": st})
    
    
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
          
            await self.verify_request(request, x_signature, x_timestamp)

            store_catalog = ProductFactory.try_parse_context(request.context)
            catalog_size = len(store_catalog)
            bt.logging.trace(f"REQUEST CATALOG SIZE: {catalog_size}")
            if catalog_size < CONST.MIN_CATALOG_SIZE or catalog_size > CONST.MAX_CATALOG_SIZE:
                bt.logging.error(f"API invalid catalog size")
                await self.log_counter(False)
                return JSONResponse(status_code=400,
                                    content={"detail": "error - invalid catalog", "status_code": 400})            
            
            dupes = ProductFactory.get_dupe_count(store_catalog)
            if dupes > catalog_size * CONST.CATALOG_DUPE_THRESHOLD:
                bt.logging.error(f"API Too many duplicates in catalog: {dupes}")
                await self.log_counter(False)
                return JSONResponse(status_code=400,
                                    content={"detail": "error - dupe threshold reached", "status_code": 400})

            st = time.time()
            response = await self.forward_fn(request)
            total_time = time.time() - st

            if len(response.results) == 0:
                bt.logging.error(f"API forward_fn response has no results")
                await self.log_counter(False)
                return JSONResponse(status_code=500,
                                    content={"detail": "error - forward", "status_code": 500})

            final_recs = [json.loads(idx.replace("'", '"')) for idx in response.results]            
            response_text = "Bitrecs Took {:.2f} seconds to process this request".format(total_time)

            response = {
                "user": response.user, 
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

            await self.log_counter(True)            
            return JSONResponse(status_code=200, content=response)
        
        except HTTPException as h:
            bt.logging.error(f"\033[31m HTTP ERROR API generate_product_rec_localnet:\033[0m {h}")
            await self.log_counter(False)
            return JSONResponse(status_code=h.status_code,
                                content={"detail": "error", "status_code": h.status_code})

        except Exception as e:
            bt.logging.error(f"\033[31m ERROR API generate_product_rec_localnet:\033[0m {e}")
            await self.log_counter(False)
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
          
            await self.verify_request2(request, x_signature, x_timestamp)

            store_catalog = ProductFactory.try_parse_context(request.context)
            catalog_size = len(store_catalog)
            bt.logging.trace(f"REQUEST CATALOG SIZE: {catalog_size}")
            if catalog_size < CONST.MIN_CATALOG_SIZE or catalog_size > CONST.MAX_CATALOG_SIZE:
                bt.logging.error(f"API invalid catalog size")
                await self.log_counter(False)
                return JSONResponse(status_code=400,
                                    content={"detail": "error - invalid catalog", "status_code": 400})            
            
            dupes = ProductFactory.get_dupe_count(store_catalog)
            if dupes > catalog_size * CONST.CATALOG_DUPE_THRESHOLD:
                bt.logging.error(f"API Too many duplicates in catalog: {dupes}")
                await self.log_counter(False)
                return JSONResponse(status_code=400,
                                    content={"detail": "error - dupe threshold reached", "status_code": 400})

            st = time.time()
            response = await self.forward_fn(request)
            total_time = time.time() - st

            if len(response.results) == 0:
                bt.logging.error(f"API forward_fn response has no results")
                await self.log_counter(False)
                return JSONResponse(status_code=500,
                                    content={"detail": "error - forward", "status_code": 500})

            final_recs = [json.loads(idx.replace("'", '"')) for idx in response.results]            
            response_text = "Bitrecs Took {:.2f} seconds to process this request".format(total_time)

            response = {
                "user": response.user, 
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

            await self.log_counter(True)            
            return JSONResponse(status_code=200, content=response)
        
        except HTTPException as h:
            bt.logging.error(f"\033[31m HTTP ERROR API generate_product_rec_testnet:\033[0m {h}")
            await self.log_counter(False)
            return JSONResponse(status_code=h.status_code,
                                content={"detail": "error", "status_code": h.status_code})

        except Exception as e:
            bt.logging.error(f"\033[31m ERROR API generate_product_rec_testnet:\033[0m {e}")
            await self.log_counter(False)
            return JSONResponse(status_code=500,
                                content={"detail": "error", "status_code": 500})
        

    async def generate_product_rec_mainnet(
            self, 
            request: BitrecsRequest,
            x_signature: str = Header(...),
            x_timestamp: str = Header(...)
    ):  
        raise NotImplementedError("Mainnet API not implemented")


    def start(self):
        self.fast_server.start()
        bt.logging.info(f"API server started at {self.fast_server.config.host}:{self.fast_server.config.port}")


    def stop(self):
        self.fast_server.stop()
        bt.logging.info("API server stopped")


    async def log_counter(self, success: bool) -> None:
        try: 
            self.api_counter.update(is_success=success)
            self.api_counter.save()
        except Exception as e:
            bt.logging.error(f"ERROR API could not update counter log:  {e}")
            pass            
   
