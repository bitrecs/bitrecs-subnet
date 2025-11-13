import ipaddress
import bittensor as bt
from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse


def get_client_ip(request: Request) -> str:
    """
    Gets the client IP address, handling proxies and potential spoofing.
    Prioritizes x-real-ip, then x-forwarded-for (last IP), then falls back to request.client.host.
    """
    if "x-real-ip" in request.headers:
        return request.headers["x-real-ip"].strip()

    if "x-forwarded-for" in request.headers:
        forwarded_for = request.headers["x-forwarded-for"].strip()
        ips = [ip.strip() for ip in forwarded_for.split(",")]
        if ips:
            # Get the first IP in the list (the original client IP)
            return ips[0]

    if request.client:
        return str(request.client.host)

    return get_remote_address(request)  # Fallback

limiter = Limiter(key_func=get_client_ip)

@limiter.limit("120/minute")
async def filter_allowed_ips(self, request: Request, call_next) -> Response:
    """
    Filters requests based on allowed IPs, handling bypass and rate limiting.
    """
    try:
        if self.bypass_whitelist:
            response = await call_next(request)
            return response

        client_ip = get_client_ip(request)
        bt.logging.trace(f"Resolved client IP: {client_ip}")

        # Validate IP address
        try:
            ipaddress.ip_address(client_ip)
        except ValueError:
            bt.logging.warning(f"Invalid IP address: {client_ip}")
            return Response(
                content="Invalid IP address",
                status_code=400,
            )

        if self.allowed_ips and client_ip not in self.allowed_ips:
            bt.logging.error(f"Blocked IP: {client_ip}")
            return Response(
                content="You do not have permission to access this resource",
                status_code=403,
            )

        bt.logging.trace(f"Allowed IP {client_ip}")
        response = await call_next(request)
        return response

    except RateLimitExceeded as re:
        bt.logging.error(f"Rate limit exceeded for {client_ip}")
        bt.logging.error(str(re))
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "status_code": 429,
                "retry_after": 60
            },
            headers={"Retry-After": "60"}
        )

