# Crowd Face Recognition Guide

## Overview
This system is optimized for recognizing multiple faces simultaneously in crowded environments like classrooms, events, or entry gates.

## Key Features

### ✅ Multi-Face Processing
- Detects and processes up to **50 faces** per frame
- Parallel embedding extraction
- Batch FAISS search for efficiency

### ✅ Smart Filtering
- **Size filtering**: Ignores faces < 50x50 pixels
- **Quality scoring**: Filters blurry/low-quality faces
- **Confidence thresholding**: Only processes high-confidence detections

### ✅ Deduplication
- **Time-based cooldown**: Prevents duplicate recognition (default: 5 seconds)
- **In-frame deduplication**: Single person counted once per image
- **Cache cleanup**: Automatic cleanup of old entries

### ✅ Performance Metrics
- Real-time FPS calculation
- Processing time tracking
- Face count statistics

---

## API Endpoints

### 1. Single Face Recognition
```bash
POST /api/face/recognize

Body (multipart/form-data):
  - image: Image file

Response:
{
  "recognized": true,
  "name": "John Doe",
  "similarity": 0.85,
  "confidence": "high"
}
```

### 2. Crowd Recognition
```bash
POST /api/face/recognize/crowd

Body (multipart/form-data):
  - image: Image file with multiple faces
  - enable_tracking: (optional) "true" or "false" - for video streams
  - cooldown: (optional) Integer - seconds between duplicate detections (default: 5)

Response:
{
  "total_faces": 12,
  "processed_faces": 10,
  "recognized_faces": [
    {
      "face_id": 0,
      "bbox": {"x": 100, "y": 150, "width": 120, "height": 140},
      "match": "John Doe",
      "similarity": 0.85,
      "quality_score": 0.78,
      "confidence": 0.95
    },
    {
      "face_id": 1,
      "bbox": {"x": 300, "y": 200, "width": 110, "height": 130},
      "match": null,
      "similarity": 0.0,
      "quality_score": 0.65,
      "confidence": 0.88
    }
  ],
  "unrecognized_count": 3,
  "processing_time": 2.5,
  "fps": 0.4
}
```

### 3. Reset Cache
```bash
POST /api/face/recognize/reset-cache

Response:
{
  "message": "Recognition cache cleared"
}
```

---

## Performance Optimization

### Hardware Recommendations

#### Minimal Setup (10-20 FPS)
- **CPU**: 4+ cores (Intel i5/Ryzen 5)
- **RAM**: 8GB
- **Storage**: SSD

#### Optimal Setup (30-60 FPS)
- **CPU**: 8+ cores (Intel i7/Ryzen 7) or GPU
- **GPU**: NVIDIA GTX 1660 or better (with CUDA)
- **RAM**: 16GB+
- **Storage**: NVMe SSD

#### Enterprise Setup (100+ FPS)
- **CPU**: 16+ cores (Xeon/Threadripper)
- **GPU**: NVIDIA RTX 3060+ or A100
- **RAM**: 32GB+
- **Storage**: NVMe RAID

### Software Optimizations

#### 1. Use GPU Acceleration
```bash
# Install CUDA-enabled TensorFlow
pip uninstall tensorflow
pip install tensorflow-gpu==2.15.0

# Install GPU FAISS
pip uninstall faiss-cpu
pip install faiss-gpu
```

#### 2. Adjust Detector Backend
```python
# In crowd_recognition.py line 99
detector = 'opencv'        # Fastest (30-60 FPS) - use for real-time
detector = 'ssd'           # Balanced (15-30 FPS)
detector = 'retinaface'    # Most accurate (5-15 FPS) - use for enrollment
```

#### 3. Tune Quality Thresholds
```python
# In crowd_recognition.py __init__
self.min_quality_score = 0.3  # Lower = more faces, less accuracy
self.min_quality_score = 0.5  # Higher = fewer faces, better accuracy
```

#### 4. Adjust Recognition Threshold
```python
# In face.py recognize_crowd()
# In faiss_service.search() call
threshold=0.60  # Lenient (more matches, higher false positives)
threshold=0.65  # Balanced (default for crowds)
threshold=0.70  # Strict (fewer matches, lower false positives)
```

#### 5. Optimize Max Faces
```python
# In crowd_recognition.py __init__
self.max_faces = 20   # For real-time video (faster)
self.max_faces = 50   # For static images (more comprehensive)
self.max_faces = 100  # For high-resolution photos (slower)
```

#### 6. Enable Redis Caching
Ensure Redis is running for FAISS index caching:
```bash
# Start Redis
redis-server

# Test connection
redis-cli ping
# Should return: PONG
```

---

## Best Practices

### Camera Setup
1. **Position**: Mount camera 6-8 feet high, angled 15-30° downward
2. **Lighting**: Use even, diffused lighting (avoid backlighting)
3. **Resolution**: Minimum 720p, optimal 1080p
4. **Frame rate**: 15-30 FPS for video streams
5. **Distance**: Keep faces 3-15 feet from camera

### Image Quality
- **Minimum face size**: 50x50 pixels
- **Optimal face size**: 150x150+ pixels
- **Avoid**: Motion blur, backlighting, extreme angles
- **Lighting**: 300-500 lux minimum

### Enrollment Strategy
1. **Multiple images**: Capture 5-10 images per person
2. **Varied conditions**: Different angles, lighting, expressions
3. **Quality check**: Use high-quality detector (`retinaface`) during enrollment
4. **Consistency validation**: Check embedding similarity across images

### Deduplication Settings
```python
# Adjust based on use case:

# Entry gate (people walking through)
cooldown = 3  # seconds

# Classroom (mostly stationary)
cooldown = 10  # seconds

# Event (mixed movement)
cooldown = 5  # seconds (default)
```

---

## Performance Benchmarks

### Single Face Recognition
| Hardware | FPS | Latency |
|----------|-----|---------|
| CPU (4-core) | 5-10 | 100-200ms |
| CPU (8-core) | 10-20 | 50-100ms |
| GPU (GTX 1660) | 30-60 | 15-30ms |
| GPU (RTX 3060) | 60-100 | 10-15ms |

### Crowd Recognition (10 faces)
| Hardware | FPS | Latency |
|----------|-----|---------|
| CPU (4-core) | 1-2 | 500-1000ms |
| CPU (8-core) | 3-5 | 200-300ms |
| GPU (GTX 1660) | 10-15 | 70-100ms |
| GPU (RTX 3060) | 20-30 | 30-50ms |

*Note: Benchmarks vary based on image size, face count, and quality settings*

---

## Troubleshooting

### Low FPS
1. Lower `max_faces` limit
2. Reduce `min_quality_score` to filter more faces
3. Use faster detector (`opencv` instead of `retinaface`)
4. Enable GPU acceleration
5. Reduce image resolution before processing

### Too Many False Positives
1. Increase recognition `threshold` (0.65 → 0.70)
2. Increase `min_quality_score` (0.4 → 0.5)
3. Use more accurate detector (`retinaface`)
4. Improve enrollment image quality
5. Capture more enrollment images per person

### Missing Faces
1. Lower `min_quality_score` (0.5 → 0.3)
2. Lower recognition `threshold` (0.65 → 0.60)
3. Increase `max_faces` limit
4. Use more accurate detector (`retinaface`)
5. Improve lighting conditions

### Duplicate Recognitions
1. Increase `cooldown` period (5s → 10s)
2. Check cache is not disabled
3. Verify time synchronization

---

## Advanced Features (Future Enhancements)

### 1. GPU Batch Processing
Process multiple images simultaneously on GPU

### 2. Face Tracking
Track faces across video frames to reduce processing

### 3. Attention Mechanism
Focus on high-quality faces, skip low-quality detections

### 4. Pose Estimation
Reject extreme angles (>45° rotation)

### 5. Age/Gender Classification
Additional metadata for better filtering

### 6. Mask Detection
Handle partially occluded faces (COVID-19 masks)

### 7. Live Video Streaming
WebRTC/WebSocket integration for real-time feeds

---

## API Usage Examples

### Python
```python
import requests

# Crowd recognition
with open('crowd.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:10000/api/face/recognize/crowd',
        files={'image': f},
        data={'cooldown': '5'}
    )
    result = response.json()
    
print(f"Found {result['total_faces']} faces")
print(f"Recognized {len([f for f in result['recognized_faces'] if f['match']])} people")
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('cooldown', '5');

fetch('http://localhost:10000/api/face/recognize/crowd', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => {
  console.log(`Found ${data.total_faces} faces`);
  console.log(`Processing time: ${data.processing_time}s`);
});
```

### cURL
```bash
curl -X POST http://localhost:10000/api/face/recognize/crowd \
  -F "image=@crowd.jpg" \
  -F "cooldown=5"
```

---

## Monitoring & Logging

### Enable Debug Logging
```python
# In app/__init__.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Key Metrics to Monitor
- Average FPS
- Recognition accuracy rate
- False positive rate
- Cache hit rate (Redis)
- Memory usage
- Processing latency

### Health Check
```bash
curl http://localhost:10000/health
```

---

## Security Considerations

1. **Rate Limiting**: Built-in (10 requests/min for crowd endpoint)
2. **File Size Limits**: Configure in Flask app (default: 16MB)
3. **Image Validation**: Automatic format and dimension checks
4. **Authentication**: Add JWT tokens for production
5. **HTTPS**: Use reverse proxy (nginx) with SSL

---

## Support & Debugging

For issues, check:
1. Logs: `app/logs/` directory
2. Redis connection: `redis-cli ping`
3. Model downloads: `~/.deepface/weights/`
4. FAISS index: Redis key `faiss:index:data`

Common error fixes:
- `Model not found`: Delete `~/.deepface/weights/` and restart
- `Redis connection failed`: Start Redis server
- `Out of memory`: Reduce `max_faces` or image resolution
- `Slow performance`: Enable GPU or reduce quality threshold
