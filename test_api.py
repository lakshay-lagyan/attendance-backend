import requests
import json

BASE_URL = "http://localhost:10000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_login():
    """Test login endpoint"""
    print("\n=== Testing Login Endpoint ===")
    data = {
        "email": "admin@admin.com",
        "password": "password123"
    }
    response = requests.post(f"{BASE_URL}/api/auth/login", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_api_info():
    """Test API info endpoint"""
    print("\n=== Testing API Info Endpoint ===")
    response = requests.get(f"{BASE_URL}/api")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Smart Attendance API")
    print("=" * 50)
    
    results = {
        "health": test_health(),
        "api_info": test_api_info(),
        "login": test_login()
    }
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20} {status}")
    
    all_passed = all(results.values())
    print("=" * 50)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 50)
