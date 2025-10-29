"""
Optimized Face Recognition Service
High-performance face detection and recognition
"""

import logging
import io
from typing import Optional, Tuple, List
import numpy as np
import cv2 as cv
from PIL import Image
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """Optimized face recognition with caching and parallel processing"""
    
    def __init__(self):
        self.embedding_dim = 512
        self.model_name = 'ArcFace'
        self.detector_backend = 'opencv'  # Fastest detector
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._models_loaded = False
    
    def _load_models(self):
        """Lazy load DeepFace models"""
        if self._models_loaded:
            return
        
        try:
            # Pre-load models
            logger.info("Loading face recognition models...")
            DeepFace.build_model(self.model_name)
            self._models_loaded = True
            logger.info("✅ Models loaded")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def preprocess_image(self, image_data) -> Optional[np.ndarray]:
        """
        Preprocess image with optimization
        
        Args:
            image_data: File object, bytes, or numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert to numpy array
            if hasattr(image_data, 'read'):
                # File object
                image_bytes = image_data.read()
                image_data.seek(0)  # Reset for reuse
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv.imdecode(nparr, cv.IMREAD_COLOR)
            elif isinstance(image_data, bytes):
                # Bytes
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv.imdecode(nparr, cv.IMREAD_COLOR)
            elif isinstance(image_data, np.ndarray):
                # Already numpy array
                image = image_data
            else:
                logger.error(f"Unsupported image type: {type(image_data)}")
                return None
            
            if image is None:
                logger.error("Failed to decode image")
                return None
            
            # Resize if too large (optimization)
            max_size = 1024
            height, width = image.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return None
    
    def extract_embedding(self, image, timeout: int = 15) -> Optional[np.ndarray]:
        """
        Extract face embedding with timeout
        
        Args:
            image: Preprocessed image
            timeout: Maximum time in seconds
            
        Returns:
            Normalized embedding vector
        """
        try:
            self._load_models()
            
            # Run with timeout
            future = self.executor.submit(self._extract_embedding_impl, image)
            embedding = future.result(timeout=timeout)
            
            return embedding
            
        except TimeoutError:
            logger.error(f"Embedding extraction timeout ({timeout}s)")
            return None
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None
    
    def _extract_embedding_impl(self, image) -> Optional[np.ndarray]:
        """Internal embedding extraction"""
        try:
            # Extract face embedding
            embedding_objs = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Don't fail if no face detected
                align=True
            )
            
            if not embedding_objs:
                return None
            
            # Get first face
            embedding = np.array(embedding_objs[0]['embedding'])
            
            # Ensure correct dimension
            if len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            elif len(embedding) < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])
            
            # L2 normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding implementation error: {e}")
            return None
    
    def extract_multiple_embeddings(self, images: List, timeout: int = 15) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from multiple images in parallel
        
        Args:
            images: List of images
            timeout: Timeout per image
            
        Returns:
            List of embeddings
        """
        embeddings = []
        
        # Process in parallel
        futures = [
            self.executor.submit(self.extract_embedding, img, timeout)
            for img in images
        ]
        
        for future in futures:
            try:
                embedding = future.result(timeout=timeout)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Parallel embedding error: {e}")
                embeddings.append(None)
        
        return embeddings
    
    def compute_average_embedding(self, embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Compute average embedding from multiple faces
        
        Args:
            embeddings: List of embeddings
            
        Returns:
            Average normalized embedding
        """
        try:
            # Filter out None values
            valid_embeddings = [e for e in embeddings if e is not None]
            
            if not valid_embeddings:
                return None
            
            # Compute mean
            avg_embedding = np.mean(valid_embeddings, axis=0)
            
            # Normalize
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            
            return avg_embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Average embedding error: {e}")
            return None
    
    def detect_faces(self, image) -> List[dict]:
        """
        Detect faces in image
        
        Returns:
            List of face bounding boxes
        """
        try:
            self._load_models()
            
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False
            )
            
            return face_objs
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def is_real_face(self, face_image: np.ndarray, threshold: float = 50.0) -> bool:
        """
        Simple liveness detection using Laplacian variance
        
        Args:
            face_image: Face region image
            threshold: Blur threshold (higher = sharper)
            
        Returns:
            True if likely a real face
        """
        try:
            # Convert to grayscale
            gray = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
            
            # Compute Laplacian variance (blur detection)
            laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
            
            # Check if sharp enough
            is_real = laplacian_var >= threshold
            
            logger.debug(f"Liveness check: {laplacian_var:.2f} (threshold: {threshold})")
            
            return is_real
            
        except Exception as e:
            logger.error(f"Liveness detection error: {e}")
            return True  # Default to allowing
    
    def compress_image(self, image: np.ndarray, quality: int = 85, max_size: Tuple[int, int] = (1024, 1024)) -> bytes:
        """
        Compress image for storage
        
        Args:
            image: Image as numpy array
            quality: JPEG quality (1-100)
            max_size: Maximum dimensions
            
        Returns:
            Compressed image bytes
        """
        try:
            # Resize if necessary
            height, width = image.shape[:2]
            if width > max_size[0] or height > max_size[1]:
                scale = min(max_size[0] / width, max_size[1] / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
            
            # Convert to PIL Image
            if len(image.shape) == 2:
                # Grayscale
                pil_image = Image.fromarray(image, mode='L')
            else:
                # RGB
                image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            
            # Compress
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Image compression error: {e}")
            return None
    
    def validate_image(self, image_data) -> Tuple[bool, str]:
        """
        Validate uploaded image
        
        Returns:
            (is_valid, error_message)
        """
        try:
            image = self.preprocess_image(image_data)
            
            if image is None:
                return False, "Invalid image format"
            
            # Check dimensions
            height, width = image.shape[:2]
            if width < 100 or height < 100:
                return False, "Image too small (minimum 100x100)"
            
            if width > 4096 or height > 4096:
                return False, "Image too large (maximum 4096x4096)"
            
            # Check if face exists
            faces = self.detect_faces(image)
            if not faces:
                return False, "No face detected in image"
            
            if len(faces) > 1:
                return False, "Multiple faces detected. Please upload image with single face"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False, f"Validation error: {str(e)}"


# Global singleton instance
face_recognition_service = FaceRecognitionService()