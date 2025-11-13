import json
import os
os.environ["NEST_ASYNCIO"] = "0"
import time
import warnings
import requests
import pandas as pd
from bitrecs.utils.version import LocalMetadata
from bitrecs.validator.forward import get_bitrecs_dummy_request
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import socket_ip
from dotenv import load_dotenv
load_dotenv()


TEST_VALIDATOR_IP = os.getenv("TEST_VALIDATOR_IP")
if not TEST_VALIDATOR_IP:
    TEST_VALIDATOR_IP = "127.0.0.1"
VALIDATOR_PORT = 7779
VALIDATOR_IS_SSL = False
print(f"Running tests on validator IP: {TEST_VALIDATOR_IP} with port: {VALIDATOR_PORT}")

BITRECS_API_KEY = os.getenv("BITRECS_API_KEY")
if BITRECS_API_KEY is None:
    raise ValueError("BITRECS_API_KEY")

INVALID_SIG = "EAD3BC14FB4773877809033E110FC23A48E74741E0155E2B1E14FADC68F74CFF23AE34A83C1DE3A92F507F9EBD70A1914D927A2AA7C986E904E01C9854BDAD09"
NUM_REQUESTS = 100
NUM_THREADS = 4


def make_get_request(url, headers):
    start_time = time.time()  
    try:
        response = requests.get(url, headers=headers)
        print(f"Request to {url} - {response.status_code}")
        end_time = time.time()
        if response.status_code == 200:
            return {"status": "OK", "time": end_time - start_time}
        else:
            return {"status": f"Failed {response.status_code}", "time": end_time - start_time}
    except requests.RequestException as e:
        end_time = time.time()
        return {"status": f"Exception {str(e)}", "time": end_time - start_time}    


def make_endpoint_request(url, headers, num_requests, num_threads) -> pd.DataFrame:
    df = pd.DataFrame(columns=['Request_Number', 'Status', 'Response_Time'])
    start_time = time.time()
    if headers is None:
        headers = {
            "Authorization": f"Bearer {BITRECS_API_KEY}"                          
        }        

    with warnings.catch_warnings():

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_request = {executor.submit(make_get_request, url, headers): i for i in range(num_requests)}
            for future in as_completed(future_to_request):
                request_num = future_to_request[future]
                result = future.result()

                warnings.filterwarnings("ignore", category=FutureWarning)
                df = pd.concat([df, pd.DataFrame({
                    'Request_Number': [request_num],
                    'Status': [result['status']],
                    'Response_Time': [result['time']]
                })], ignore_index=True)

    end_time = time.time()
    total_time = end_time - start_time    
    total_requests = len(df)
    ok_requests = (df['Status'] == 'OK').sum()
    failed_requests = total_requests - ok_requests    
    summary = pd.DataFrame({
        'Metric': ['Total Requests', 'OK Requests', 'Failed Requests', 'Total Time (sec)', 'Avg Response Time (sec)'],
        'Value': [total_requests, ok_requests, failed_requests, total_time, df['Response_Time'].mean()]
    })    
    print("\nDetailed Results:")
    print(df.sort_values(by='Request_Number').to_string(index=False))

    print("\n\033[32mSummary of Performance Test: \033[0m")
    print("url_{} ".format(url))
    print(summary.to_string(index=False))
    return summary


def test_can_reach_validator():    
    success = socket_ip(TEST_VALIDATOR_IP, VALIDATOR_PORT)    
    assert success is True

def test_no_auth_error_validator_root():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/"    
    response = requests.get(url)
    print(response.text)
    assert response.status_code == 400

def test_no_auth_error_validator_ping():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/ping"   
    response = requests.get(url)
    print(response.text)
    assert response.status_code == 400

def test_no_auth_error_validator_version():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/version"
    response = requests.get(url)
    print(response.text)
    assert response.status_code == 400

def test_no_auth_error_validator_rec():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/rec"
    response = requests.get(url)
    print(response.text)
    assert response.status_code == 400

def test_wrong_auth_error_validator():    
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/ping"
    headers = {       
        "Authorization": "Bearer wrong",
        "Content-Type": "application/json"
    }            
    response = requests.get(url, headers=headers)
    print(response.text)
    assert response.status_code == 401

def test_good_auth_root_validator():    
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/"
    headers = {        
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    print(response.text)
    if response.status_code == 429:
        print("Rate limit hit")
        return
    
    assert response.status_code == 404

def test_good_auth_ping_validator():    
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/ping"
    headers = {        
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    print(response.text)
    if response.status_code == 429:
        print("Rate limit hit")
        return
    
    assert response.status_code == 200


def test_good_server_time_validator():    
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/ping"
    headers = {        
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    print(response.text)
    if response.status_code == 429:
        print("Rate limit hit")
        return
    
    assert response.status_code == 200
    st = response.json()["ts"]
    assert st > 0
    current_time = int(time.time())
    over_5_minutes = current_time - st > 300
    assert over_5_minutes is False


def test_version_ok_validator():    
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/version"
    headers = {        
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    print(response.text)    
    if response.status_code == 429: 
        print("Rate limit hit")
        return
    
    assert response.status_code == 200
    meta_json = response.json()["meta_data"]
    md = LocalMetadata(**meta_json)
    #print(md)
    assert md.head != "head error"


def test_rec_no_sig_is_rejected_ok():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/rec"
    headers = {        
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "x-timestamp": str(int(time.time())),
        "Content-Type": "application/json"
    }
    br = get_bitrecs_dummy_request(5)
    data = br.model_dump()
    response = requests.post(url, headers=headers, json=data)
    print(response.text)
    if response.status_code == 429: 
        print("Rate limit hit")
        return
    assert response.status_code == 422 #missing headers


def test_rec_invalid_sig_rejected_ok():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/rec"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "x-signature": "wrong",
        "x-timestamp": str(int(time.time()))
    }
    br = get_bitrecs_dummy_request(5)
    data = br.model_dump()
    response = requests.post(url, headers=headers, json=data)
    print(response.text)
    if response.status_code == 429: 
        print("Rate limit hit")
        return
    
    assert response.status_code == 500


def test_rec_wrong_sig_rejected_ok():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/rec"    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "x-signature": INVALID_SIG,
        "x-timestamp": str(int(time.time()))
    }
    br = get_bitrecs_dummy_request(5)
    data = br.model_dump()
    response = requests.post(url, headers=headers, json=data)
    print(response.text)
    if response.status_code == 429: 
        print("Rate limit hit")
        return
    
    assert response.status_code == 401


def test_rate_limit_hit_root_ok():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/"
    headers = {
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "Content-Type": "application/json"             
    }

    num_requests = NUM_REQUESTS
    num_threads = NUM_THREADS

    print(f"\033[33m{num_requests} requests using {num_threads} threads to {url} \033[0m")
    results = make_endpoint_request(url, headers, num_requests, num_threads)
    print(results.head())
    total_requests = results['Value'][0]
    ok_requests = results['Value'][1]
    failed_requests = results['Value'][2]
    assert total_requests == num_requests
    assert ok_requests == num_requests - failed_requests

    assert failed_requests == total_requests


def test_rate_limit_hit_ping_ok():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/ping"
    headers = {
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "Content-Type": "application/json"        
    }

    num_requests = NUM_REQUESTS
    num_threads = NUM_THREADS

    print(f"\033[33m{num_requests} requests using {num_threads} threads to {url} \033[0m")
    results = make_endpoint_request(url, headers, num_requests, num_threads)
    print(results.head())
    total_requests = results['Value'][0]
    ok_requests = results['Value'][1]
    failed_requests = results['Value'][2]

    assert total_requests == num_requests
    assert ok_requests == num_requests - failed_requests

    if failed_requests > 0:
        assert failed_requests > total_requests * 0.3

   
def test_rate_limit_hit_version_ok():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/version"
    headers = {
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "Content-Type": "application/json"
    }

    num_requests = NUM_REQUESTS
    num_threads = NUM_THREADS

    print(f"\033[33m{num_requests} requests using {num_threads} threads to {url} \033[0m")
    results = make_endpoint_request(url, headers, num_requests, num_threads)
    print(results.head())
    total_requests = results['Value'][0]
    ok_requests = results['Value'][1]
    failed_requests = results['Value'][2]

    assert total_requests == num_requests
    assert ok_requests == num_requests - failed_requests
    if failed_requests > 0:
        assert failed_requests > total_requests * 0.3


def test_rate_limit_hit_rec_ok():

    def do_rec():
        url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/rec"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BITRECS_API_KEY}",
            "x-signature": INVALID_SIG,
            "x-timestamp": str(int(time.time()))
        }
        br = get_bitrecs_dummy_request(5)
        data = br.model_dump(mode="json")
        response = requests.post(url, headers=headers, json=data)
        return response
    
    num_requests = NUM_REQUESTS
    codes = []
    for i in range(num_requests):
        response = do_rec()
        print(response.text)
        print(response.status_code)
        codes.append(response.status_code)
        assert response.status_code > 400
        #time.sleep(0.1)
    
    rejected = codes.count(401)    
    rate_limited = codes.count(429)
    rate_limited_threshold = num_requests * 0.3

    assert len(codes) == num_requests
    print(f"Rejected: {rejected}, Rate Limited: {rate_limited}")
    if rate_limited > 0:
        assert rate_limited > rate_limited_threshold



def test_non_json_post_rejected():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/rec"
    headers = {
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "x-timestamp": str(int(time.time()))
    }
    files = {
        'file': ('data.txt', 'This is not JSON data', 'text/plain')
    }
    response = requests.post(url, headers=headers, files=files)
    print(response.text)
    assert response.status_code == 415


def test_post_node_info():
    from start_validator import post_node_report
    payload = {"bitrecs_test": "foo"}
    result = post_node_report(payload=payload)
    assert True is result


def test_trigger_update():
    thing = True
    assert True is thing


def test_api_has_metrics():
    url = f"http://{TEST_VALIDATOR_IP}:{VALIDATOR_PORT}/version"
    headers = {
        "Authorization": f"Bearer {BITRECS_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    print(response.text)
    thing = json.loads(response.text)
    print(json.dumps(thing, indent=2))
    assert "metrics" in thing

