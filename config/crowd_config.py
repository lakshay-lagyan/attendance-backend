
class CrowdRecognitionConfig:
    """Configuration for crowd recognition"""
    
    DETECTOR_BACKEND = 'retinaface'  
    
    
    MIN_FACE_SIZE = (50, 50)  # Smaller = detect distant faces, more false positives
    
    MAX_FACES_PER_FRAME = 50
    
    MIN_QUALITY_SCORE = 0.4
    
    MIN_DETECTION_CONFIDENCE = 0.5
    
    RECOGNITION_THRESHOLD = 0.65
    
    # Top K matches to return
    TOP_K_MATCHES = 1
    
    DEDUPLICATION_COOLDOWN_SECONDS = 5
    
    # Cache cleanup interval (hours)
    CACHE_CLEANUP_HOURS = 1
    
    
    # Embedding extraction timeout (seconds)
    EMBEDDING_TIMEOUT = 5  # Lower for crowds to avoid blocking
    
    ENABLE_PARALLEL_PROCESSING = True
    
    THREAD_POOL_SIZE = 4
    
    MAX_IMAGE_DIMENSION = 1024
    
    ENABLE_IMAGE_ENHANCEMENT = False
    
    ENABLE_FACE_TRACKING = False
    
    MAX_DISAPPEARED_FRAMES = 10
    
    
    # Enable detailed logging
    ENABLE_DEBUG_LOGGING = True
    
    # Log quality scores
    LOG_QUALITY_SCORES = True
    
    # Log recognition timings
    LOG_TIMING_INFO = True


class ProductionConfig(CrowdRecognitionConfig):
    DETECTOR_BACKEND = 'ssd'
    MAX_FACES_PER_FRAME = 30
    MIN_QUALITY_SCORE = 0.45
    RECOGNITION_THRESHOLD = 0.65
    ENABLE_DEBUG_LOGGING = False
    LOG_QUALITY_SCORES = False


class AccuracyConfig(CrowdRecognitionConfig):
    DETECTOR_BACKEND = 'retinaface'
    MAX_FACES_PER_FRAME = 50
    MIN_QUALITY_SCORE = 0.5
    RECOGNITION_THRESHOLD = 0.70
    ENABLE_IMAGE_ENHANCEMENT = True
    EMBEDDING_TIMEOUT = 10


class SpeedConfig(CrowdRecognitionConfig):
    DETECTOR_BACKEND = 'opencv'
    MAX_FACES_PER_FRAME = 20
    MIN_QUALITY_SCORE = 0.35
    RECOGNITION_THRESHOLD = 0.60
    ENABLE_IMAGE_ENHANCEMENT = False
    EMBEDDING_TIMEOUT = 3


class VideoStreamConfig(CrowdRecognitionConfig):
    """for real-time video streams"""
    DETECTOR_BACKEND = 'opencv'
    MAX_FACES_PER_FRAME = 15
    MIN_QUALITY_SCORE = 0.4
    RECOGNITION_THRESHOLD = 0.65
    DEDUPLICATION_COOLDOWN_SECONDS = 3
    ENABLE_FACE_TRACKING = True
    MAX_DISAPPEARED_FRAMES = 15
    EMBEDDING_TIMEOUT = 2


