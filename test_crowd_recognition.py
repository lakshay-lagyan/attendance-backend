"""
Test crowd face recognition
"""
import requests
import json

BASE_URL = "http://localhost:10000"

def test_single_face_recognition():
    """Test single face recognition"""
    print("\n=== Testing Single Face Recognition ===")
    
    # Replace with your test image path
    test_image = "path/to/test_image.jpg"
    
    with open(test_image, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/api/face/recognize", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_crowd_recognition():
    """Test crowd face recognition"""
    print("\n=== Testing Crowd Recognition ===")
    
    # Replace with your test image with multiple faces
    crowd_image = "path/to/crowd_image.jpg"
    
    with open(crowd_image, 'rb') as f:
        files = {'image': f}
        data = {
            'enable_tracking': 'false',
            'cooldown': '5'
        }
        response = requests.post(
            f"{BASE_URL}/api/face/recognize/crowd", 
            files=files,
            data=data
        )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    # Print summary
    if response.status_code == 200:
        print(f"\n--- Summary ---")
        print(f"Total faces detected: {result.get('total_faces', 0)}")
        print(f"Processed faces: {result.get('processed_faces', 0)}")
        print(f"Recognized: {len([f for f in result.get('recognized_faces', []) if f.get('match')])}")
        print(f"Unrecognized: {result.get('unrecognized_count', 0)}")
        print(f"Processing time: {result.get('processing_time', 0)}s")
        print(f"FPS: {result.get('fps', 0)}")
        
        # Show recognized individuals
        recognized = [f for f in result.get('recognized_faces', []) if f.get('match')]
        if recognized:
            print(f"\n--- Recognized People ---")
            for face in recognized:
                print(f"  - {face['match']} (similarity: {face['similarity']}, quality: {face['quality_score']:.2f})")


def test_reset_cache():
    """Reset recognition cache"""
    print("\n=== Resetting Recognition Cache ===")
    response = requests.post(f"{BASE_URL}/api/face/recognize/reset-cache")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Crowd Face Recognition Tests")
    print("=" * 60)
    
    # Uncomment the tests you want to run
    # test_single_face_recognition()
    test_crowd_recognition()
    # test_reset_cache()
    
    print("\n" + "=" * 60)
    print("Tests Complete")
    print("=" * 60)
