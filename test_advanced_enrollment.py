"""
Test Advanced Enrollment System
Tests the new enrollment endpoints and features
"""
import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:10000"

def test_image_validation():
    """Test real-time image validation"""
    print("\n" + "="*60)
    print("Testing Real-Time Image Validation")
    print("="*60)
    
    # Replace with your test image path
    test_image = "path/to/test_face.jpg"
    
    if not os.path.exists(test_image):
        print(f"⚠️  Test image not found: {test_image}")
        print("Please provide a valid image path")
        return False
    
    with open(test_image, 'rb') as f:
        files = {'image': f}
        response = requests.post(
            f"{BASE_URL}/api/enrollment/validate-image",
            files=files
        )
    
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    print(f"\nValidation Result:")
    print(json.dumps(result, indent=2))
    
    if result.get('valid'):
        print(f"\n✅ Image is valid!")
        print(f"📊 Quality Score: {result.get('quality_score', 0)*100:.1f}%")
        
        metrics = result.get('metrics', {})
        print(f"   - Sharpness: {metrics.get('sharpness', 0)*100:.1f}%")
        print(f"   - Brightness: {metrics.get('brightness', 0)*100:.1f}%")
        print(f"   - Contrast: {metrics.get('contrast', 0)*100:.1f}%")
        print(f"   - Face Size: {metrics.get('face_size', [0,0])}")
    else:
        print(f"\n❌ Image validation failed")
        issues = result.get('issues', [])
        for issue in issues:
            print(f"   - {issue}")
    
    return result.get('valid', False)


def test_advanced_enrollment():
    """Test advanced multi-image enrollment"""
    print("\n" + "="*60)
    print("Testing Advanced Enrollment")
    print("="*60)
    
    # Replace with your test images directory
    test_images_dir = "path/to/test_images/"
    
    if not os.path.exists(test_images_dir):
        print(f"⚠️  Test images directory not found: {test_images_dir}")
        print("Please provide a valid directory with 5+ face images")
        return False
    
    # Get all image files
    image_files = list(Path(test_images_dir).glob("*.jpg")) + \
                  list(Path(test_images_dir).glob("*.png")) + \
                  list(Path(test_images_dir).glob("*.jpeg"))
    
    if len(image_files) < 3:
        print(f"⚠️  Need at least 3 images. Found: {len(image_files)}")
        return False
    
    print(f"\nFound {len(image_files)} images")
    
    # Limit to first 8 images
    image_files = image_files[:8]
    
    # Prepare form data
    data = {
        'name': 'Test User Advanced',
        'email': f'test.advanced.{hash(str(image_files))}@test.com',  # Unique email
        'password': 'test123',
        'department': 'Testing',
        'phone': '+1234567890'
    }
    
    # Prepare files
    files = []
    for img_path in image_files:
        files.append(('files', open(img_path, 'rb')))
    
    print(f"\n📤 Uploading {len(files)} images for enrollment...")
    print(f"   Name: {data['name']}")
    print(f"   Email: {data['email']}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/enrollment/enroll",
            data=data,
            files=files
        )
        
        # Close file handles
        for _, f in files:
            f.close()
        
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        print(f"\nEnrollment Result:")
        print(json.dumps(result, indent=2))
        
        if result.get('success'):
            print(f"\n✅ Enrollment successful!")
            print(f"   Name: {result.get('name')}")
            print(f"   Embedding Dimension: {result.get('embedding_dim')}")
            print(f"   Photos Used: {result.get('photos_used')}/{result.get('total_photos')}")
            print(f"   Consistency Score: {result.get('consistency_score', 0)*100:.1f}%")
            print(f"   Average Quality: {result.get('avg_quality', 0)*100:.1f}%")
            
            quality_scores = result.get('quality_scores', [])
            if quality_scores:
                print(f"\n📊 Quality Scores per Image:")
                for i, score in enumerate(quality_scores):
                    print(f"   Image {i+1}: {score*100:.1f}%")
            
            return True
        else:
            print(f"\n❌ Enrollment failed: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_face_recognition():
    """Test single face recognition"""
    print("\n" + "="*60)
    print("Testing Face Recognition")
    print("="*60)
    
    test_image = "path/to/test_face.jpg"
    
    if not os.path.exists(test_image):
        print(f"⚠️  Test image not found: {test_image}")
        return False
    
    with open(test_image, 'rb') as f:
        files = {'image': f}
        response = requests.post(
            f"{BASE_URL}/api/face/recognize",
            files=files
        )
    
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    print(f"\nRecognition Result:")
    print(json.dumps(result, indent=2))
    
    if result.get('recognized'):
        print(f"\n✅ Face recognized!")
        print(f"   Name: {result.get('name')}")
        print(f"   Similarity: {result.get('similarity', 0)*100:.1f}%")
        print(f"   Confidence: {result.get('confidence', 'unknown')}")
        return True
    else:
        print(f"\n❌ Face not recognized")
        return False


def test_crowd_recognition():
    """Test crowd recognition with multiple faces"""
    print("\n" + "="*60)
    print("Testing Crowd Recognition")
    print("="*60)
    
    test_image = "path/to/crowd_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"⚠️  Test image not found: {test_image}")
        return False
    
    with open(test_image, 'rb') as f:
        files = {'image': f}
        data = {
            'cooldown': '5',
            'enable_tracking': 'false'
        }
        response = requests.post(
            f"{BASE_URL}/api/face/recognize/crowd",
            files=files,
            data=data
        )
    
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    print(f"\nCrowd Recognition Result:")
    print(json.dumps(result, indent=2))
    
    total_faces = result.get('total_faces', 0)
    processed = result.get('processed_faces', 0)
    recognized_faces = result.get('recognized_faces', [])
    recognized_count = len([f for f in recognized_faces if f.get('match')])
    
    print(f"\n📊 Summary:")
    print(f"   Total Faces Detected: {total_faces}")
    print(f"   Faces Processed: {processed}")
    print(f"   Recognized: {recognized_count}")
    print(f"   Unrecognized: {result.get('unrecognized_count', 0)}")
    print(f"   Processing Time: {result.get('processing_time', 0)}s")
    print(f"   FPS: {result.get('fps', 0)}")
    
    if recognized_count > 0:
        print(f"\n✅ Recognized People:")
        for face in recognized_faces:
            if face.get('match'):
                print(f"   - {face['match']} (similarity: {face['similarity']:.3f}, "
                      f"quality: {face['quality_score']:.2f})")
    
    return total_faces > 0


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 ADVANCED ENROLLMENT & RECOGNITION TEST SUITE")
    print("="*70)
    
    print("\n⚠️  IMPORTANT: Update image paths in this script before running!")
    print("   - test_image: Single face image")
    print("   - test_images_dir: Directory with 5+ face images of same person")
    print("   - crowd_image: Image with multiple faces")
    
    results = {
        "Image Validation": test_image_validation(),
        "Advanced Enrollment": test_advanced_enrollment(),
        "Face Recognition": test_face_recognition(),
        "Crowd Recognition": test_crowd_recognition()
    }
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("="*70)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Configuration
    print("\n📋 Test Configuration:")
    print(f"   Backend URL: {BASE_URL}")
    print(f"   Make sure the server is running!")
    
    input("\nPress Enter to start tests...")
    
    run_all_tests()
