#!/usr/bin/env python3
# =============================================================
# test_api.py - Component 4 API Test Suite
# A.A.N. Dulsara | IT23344006 | J-26-CS-314
# =============================================================

import requests
import json
import time

BASE_URL = "http://localhost:8080"

def print_result(test_name, response):
    print("=" * 50)
    print(f"TEST: {test_name}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_health():
    r = requests.get(f"{BASE_URL}/api/v1/health")
    print_result("Health Check", r)
    assert r.status_code == 200
    print("PASSED")

def test_ddos_block():
    payload = {"device_id": "cardiac", "attack_type": "DDoS-SYN", "severity": "Critical", "confidence": 0.95, "critical_protection_flag": False}
    r = requests.post(f"{BASE_URL}/api/v1/alert", json=payload)
    print_result("DDoS Critical - Block", r)
    assert r.status_code == 200
    print("PASSED")

def test_dos_ratelimit():
    payload = {"device_id": "bp_mon", "attack_type": "DoS", "severity": "High", "confidence": 0.85, "critical_protection_flag": False}
    r = requests.post(f"{BASE_URL}/api/v1/alert", json=payload)
    print_result("DoS High - Rate Limit", r)
    assert r.status_code == 200
    print("PASSED")

def test_probe_monitor():
    payload = {"device_id": "admsrv", "attack_type": "Probe", "severity": "Medium", "confidence": 0.75, "critical_protection_flag": False}
    r = requests.post(f"{BASE_URL}/api/v1/alert", json=payload)
    print_result("Probe Medium - Monitor", r)
    assert r.status_code == 200
    print("PASSED")

def test_critical_device():
    payload = {"device_id": "venti", "attack_type": "DDoS-SYN", "severity": "Critical", "confidence": 0.90, "critical_protection_flag": True}
    r = requests.post(f"{BASE_URL}/api/v1/alert", json=payload)
    print_result("Critical Device Ventilator Protection", r)
    assert r.status_code == 200
    print("PASSED")

def test_get_incidents():
    r = requests.get(f"{BASE_URL}/api/v1/incidents")
    print_result("Get All Incidents", r)
    assert r.status_code == 200
    print(f"Total incidents: {r.json()['total']}")
    print("PASSED")

def test_get_status():
    r = requests.get(f"{BASE_URL}/api/v1/status")
    print_result("Controller Status", r)
    assert r.status_code == 200
    print("PASSED")

if __name__ == "__main__":
    print("Component 4 - API Test Suite")
    print("A.A.N. Dulsara | IT23344006")
    tests = [test_health, test_ddos_block, test_dos_ratelimit, test_probe_monitor, test_critical_device, test_get_incidents, test_get_status]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
        time.sleep(0.5)
    print(f"RESULTS: {passed} passed, {failed} failed")
