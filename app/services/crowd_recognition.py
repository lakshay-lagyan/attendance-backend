import logging
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2 as cv
from collections import defaultdict, deque
from datetime import datetime, timedelta

from app.services.face_recognition import face_recognition_service
from app.services.faiss_service import faiss_service

logger = logging.getLogger(__name__)


class CrowdRecognitionService:
    """
    Optimized face recognition for crowded scenes
    Handles multiple faces with deduplication and tracking
    """
    
    def __init__(self):
        self.min_face_size = (50, 50)  # Minimum face dimensions
        self.max_faces = 50  # Maximum faces to process per frame
        self.min_quality_score = 0.4  # Minimum quality threshold
        
        # Deduplication: Track recent recognitions
        self.recognition_cache = {}  # {person_name: last_seen_time}
        self.cooldown_seconds = 5  # Prevent duplicate recognition within 5 seconds
        
        # Face tracking for video streams
        self.face_tracker = FaceTracker(max_disappeared=10)
        
    def process_crowd_image(self, image: np.ndarray, 
                           enable_tracking: bool = False) -> Dict:
        """
        Process image with multiple faces
        
        Args:
            image: Input image (BGR format)
            enable_tracking: Enable face tracking for video streams
            
        Returns:
            {
                "total_faces": int,
                "processed_faces": int,
                "recognized_faces": List[dict],
                "unrecognized_count": int,
                "processing_time": float
            }
        """
        start_time = time.time()
        
        try:
            # Step 1: Detect all faces
            face_detections = self._detect_multiple_faces(image)
            
            if not face_detections:
                return self._empty_result(time.time() - start_time)
            
            logger.info(f"Detected {len(face_detections)} faces")
            
            # Step 2: Filter by quality and size
            valid_faces = self._filter_valid_faces(face_detections, image)
            
            if not valid_faces:
                return self._empty_result(time.time() - start_time)
            
            logger.info(f"Valid faces after filtering: {len(valid_faces)}")
            
            # Step 3: Extract embeddings in batch
            embeddings = self._extract_batch_embeddings(valid_faces, image)
            
            # Step 4: Recognize faces in batch
            recognized_faces = self._recognize_batch(valid_faces, embeddings)
            
            # Step 5: Apply deduplication
            unique_faces = self._deduplicate_recognitions(recognized_faces)
            
            # Step 6: Update tracking (if enabled)
            if enable_tracking:
                unique_faces = self._apply_tracking(unique_faces, valid_faces)
            
            processing_time = time.time() - start_time
            
            return {
                "total_faces": len(face_detections),
                "processed_faces": len(valid_faces),
                "recognized_faces": unique_faces,
                "unrecognized_count": len([f for f in unique_faces if not f.get('match')]),
                "processing_time": round(processing_time, 3),
                "fps": round(1 / processing_time, 2) if processing_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Crowd processing error: {e}")
            return self._empty_result(time.time() - start_time, error=str(e))
    
    def _detect_multiple_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all faces in image using optimized detector
        """
        try:
            # Use faster detector for crowd scenes
            from deepface import DeepFace
            
            # Use opencv for speed, retinaface for accuracy (choose based on needs)
            detector = 'opencv'  # Faster for crowds
            # detector = 'retinaface'  # More accurate but slower
            
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=detector,
                enforce_detection=False,
                align=False  # Skip alignment for speed
            )
            
            return face_objs
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def _filter_valid_faces(self, face_detections: List[Dict], 
                           image: np.ndarray) -> List[Dict]:
        """
        Filter faces by size, quality, and confidence
        """
        valid_faces = []
        
        for i, face_obj in enumerate(face_detections):
            try:
                # Get face region
                facial_area = face_obj.get('facial_area', {})
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)
                confidence = face_obj.get('confidence', 0)
                
                # Filter 1: Size check
                if w < self.min_face_size[0] or h < self.min_face_size[1]:
                    logger.debug(f"Face {i} too small: {w}x{h}")
                    continue
                
                # Filter 2: Confidence check
                if confidence < 0.5:
                    logger.debug(f"Face {i} low confidence: {confidence}")
                    continue
                
                # Filter 3: Extract face region for quality check
                face_region = image[y:y+h, x:x+w]
                
                if face_region.size == 0:
                    continue
                
                # Filter 4: Quality check (sharpness)
                quality_score = self._calculate_face_quality(face_region)
                
                if quality_score < self.min_quality_score:
                    logger.debug(f"Face {i} low quality: {quality_score}")
                    continue
                
                # Add to valid faces with metadata
                face_obj['face_id'] = i
                face_obj['quality_score'] = quality_score
                face_obj['face_region'] = face_region
                valid_faces.append(face_obj)
                
            except Exception as e:
                logger.warning(f"Face {i} filtering error: {e}")
                continue
        
        # Limit to max_faces (keep highest quality)
        if len(valid_faces) > self.max_faces:
            valid_faces.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            valid_faces = valid_faces[:self.max_faces]
            logger.warning(f"Limited to {self.max_faces} highest quality faces")
        
        return valid_faces
    
    def _calculate_face_quality(self, face_region: np.ndarray) -> float:
        """
        Calculate quality score for face region (0-1)
        """
        try:
            gray = cv.cvtColor(face_region, cv.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
            sharpness = min(laplacian_var / 100.0, 1.0)
            
            # Brightness (avoid over/under exposure)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Combined score
            quality = (sharpness * 0.7) + (brightness_score * 0.3)
            
            return quality
            
        except Exception as e:
            logger.error(f"Quality calculation error: {e}")
            return 0.5
    
    def _extract_batch_embeddings(self, valid_faces: List[Dict], 
                                  image: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for all faces (batch processing for efficiency)
        """
        embeddings = []
        
        for face_obj in valid_faces:
            try:
                # Extract face region
                face_region = face_obj['face_region']
                
                # Extract embedding
                embedding = face_recognition_service.extract_embedding(
                    face_region, 
                    timeout=5  # Shorter timeout for crowds
                )
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.warning(f"Embedding extraction failed for face {face_obj.get('face_id')}: {e}")
                embeddings.append(None)
        
        return embeddings
    
    def _recognize_batch(self, valid_faces: List[Dict], 
                        embeddings: List[Optional[np.ndarray]]) -> List[Dict]:
        """
        Recognize all faces using FAISS batch search
        """
        recognized_faces = []
        
        for face_obj, embedding in zip(valid_faces, embeddings):
            try:
                facial_area = face_obj.get('facial_area', {})
                
                result = {
                    "face_id": face_obj.get('face_id'),
                    "bbox": {
                        "x": facial_area.get('x', 0),
                        "y": facial_area.get('y', 0),
                        "width": facial_area.get('w', 0),
                        "height": facial_area.get('h', 0)
                    },
                    "quality_score": face_obj.get('quality_score', 0),
                    "confidence": face_obj.get('confidence', 0),
                    "match": None,
                    "similarity": 0.0
                }
                
                if embedding is not None:
                    # Search in FAISS index
                    search_results = faiss_service.search(
                        embedding, 
                        threshold=0.65,  # Slightly higher threshold for crowds
                        top_k=1
                    )
                    
                    if search_results and search_results[0][0] is not None:
                        name, similarity = search_results[0]
                        result["match"] = name
                        result["similarity"] = round(similarity, 3)
                
                recognized_faces.append(result)
                
            except Exception as e:
                logger.warning(f"Recognition failed for face {face_obj.get('face_id')}: {e}")
                continue
        
        return recognized_faces
    
    def _deduplicate_recognitions(self, recognized_faces: List[Dict]) -> List[Dict]:
        """
        Remove duplicate recognitions based on time-based cooldown
        """
        current_time = datetime.now()
        unique_faces = []
        seen_in_frame = set()
        
        for face in recognized_faces:
            match = face.get('match')
            
            # Keep unrecognized faces
            if not match:
                unique_faces.append(face)
                continue
            
            # Check if already seen in this frame
            if match in seen_in_frame:
                face['duplicate'] = True
                face['reason'] = 'duplicate_in_frame'
                continue
            
            # Check cooldown period
            last_seen = self.recognition_cache.get(match)
            
            if last_seen:
                time_diff = (current_time - last_seen).total_seconds()
                
                if time_diff < self.cooldown_seconds:
                    face['duplicate'] = True
                    face['reason'] = f'seen_{int(time_diff)}s_ago'
                    logger.debug(f"Skipping {match} - seen {time_diff:.1f}s ago")
                    continue
            
            # Valid unique recognition
            self.recognition_cache[match] = current_time
            seen_in_frame.add(match)
            unique_faces.append(face)
        
        # Cleanup old cache entries (keep last hour)
        self._cleanup_cache(current_time)
        
        return unique_faces
    
    def _cleanup_cache(self, current_time: datetime):
        """Remove old entries from recognition cache"""
        cutoff_time = current_time - timedelta(hours=1)
        
        to_remove = [
            name for name, last_seen in self.recognition_cache.items()
            if last_seen < cutoff_time
        ]
        
        for name in to_remove:
            del self.recognition_cache[name]
    
    def _apply_tracking(self, recognized_faces: List[Dict], 
                       valid_faces: List[Dict]) -> List[Dict]:
        """
        Apply face tracking for video streams (optional)
        """
        # This would integrate with a face tracking algorithm
        # For now, just return as-is
        return recognized_faces
    
    def _empty_result(self, processing_time: float, error: str = None) -> Dict:
        """Return empty result"""
        result = {
            "total_faces": 0,
            "processed_faces": 0,
            "recognized_faces": [],
            "unrecognized_count": 0,
            "processing_time": round(processing_time, 3),
            "fps": 0
        }
        
        if error:
            result["error"] = error
        
        return result
    
    def reset_cache(self):
        """Reset recognition cache (useful for new session)"""
        self.recognition_cache.clear()
        logger.info("Recognition cache cleared")
    
    def set_cooldown(self, seconds: int):
        """Adjust cooldown period"""
        self.cooldown_seconds = seconds
        logger.info(f"Cooldown set to {seconds}s")


class FaceTracker:
    """
    Simple face tracker for video streams
    Tracks faces across frames to avoid duplicate processing
    """
    
    def __init__(self, max_disappeared: int = 10):
        self.next_object_id = 0
        self.objects = {}  # {object_id: centroid}
        self.disappeared = {}  # {object_id: disappeared_count}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        """Register new face"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove tracked face"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, bboxes: List[Tuple[int, int, int, int]]):
        """
        Update tracked faces with new detections
        
        Args:
            bboxes: List of (x, y, w, h) bounding boxes
        """
        # Implementation would use centroid tracking algorithm
        # This is a placeholder
        pass


# Global singleton
crowd_recognition_service = CrowdRecognitionService()
