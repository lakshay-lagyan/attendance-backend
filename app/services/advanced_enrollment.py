import logging
import pickle
import numpy as np
import cv2 as cv
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from app.services.face_recognition import face_recognition_service
from app.services.faiss_service import faiss_service
from deepface import DeepFace

logger = logging.getLogger(__name__)


class AdvancedEnrollmentService:
    """
    Maximum accuracy enrollment system
    - Multi-image processing with quality scoring
    - Face alignment and preprocessing
    - Embedding consistency validation
    - Best embedding selection/averaging
    """
    
    def __init__(self):
        # Quality thresholds
        self.min_quality_score = 0.50  # Minimum quality for enrollment
        self.min_images_required = 3   # Minimum images needed
        self.optimal_images = 5        # Optimal number of images
        self.max_images = 15           # Maximum images to process
        
        # Embedding consistency
        self.min_consistency = 0.75    # Minimum similarity between embeddings
        
        # Detector for enrollment (use most accurate)
        self.enrollment_detector = 'retinaface'  # Most accurate detector
        
        # Preprocessing options
        self.enable_enhancement = True  # CLAHE and preprocessing
        self.enable_alignment = True    # Face alignment
        
    def process_enrollment(self, images: List, name: str, 
                          user_data: Dict) -> Dict:
        """
        Process enrollment with multiple images
        
        Args:
            images: List of image data (bytes or file objects)
            name: Person's name
            user_data: Additional user information (email, phone, etc.)
            
        Returns:
            {
                "success": bool,
                "message": str,
                "embedding": np.ndarray (if success),
                "embedding_dim": int,
                "photos_used": int,
                "quality_scores": List[float],
                "consistency_score": float
            }
        """
        try:
            logger.info(f"Starting advanced enrollment for {name} with {len(images)} images")
            
            # Step 1: Validate input
            if len(images) < self.min_images_required:
                return self._error_result(
                    f"Minimum {self.min_images_required} images required. Provided: {len(images)}"
                )
            
            # Limit to max images
            if len(images) > self.max_images:
                logger.warning(f"Limiting to {self.max_images} images from {len(images)}")
                images = images[:self.max_images]
            
            # Step 2: Process all images and extract faces
            processed_data = self._process_images(images)
            
            if not processed_data['valid_images']:
                return self._error_result(
                    f"No valid faces detected. Issues: {processed_data['errors']}"
                )
            
            logger.info(f"Processed {len(processed_data['valid_images'])} valid images")
            
            # Step 3: Extract embeddings
            embeddings_data = self._extract_embeddings(processed_data['valid_images'])
            
            if len(embeddings_data['embeddings']) < self.min_images_required:
                return self._error_result(
                    f"Only {len(embeddings_data['embeddings'])} valid embeddings extracted. "
                    f"Minimum {self.min_images_required} required."
                )
            
            # Step 4: Validate embedding consistency
            consistency_check = self._validate_consistency(embeddings_data['embeddings'])
            
            if not consistency_check['valid']:
                return self._error_result(
                    f"Embeddings are inconsistent (similarity: {consistency_check['avg_similarity']:.2f}). "
                    f"Please ensure all images are of the same person with good quality."
                )
            
            # Step 5: Generate final embedding (weighted average of best embeddings)
            final_embedding = self._generate_final_embedding(
                embeddings_data['embeddings'],
                embeddings_data['quality_scores']
            )
            
            # Step 6: Save to database and FAISS index
            save_result = self._save_enrollment(name, final_embedding, user_data, len(images))
            
            if not save_result['success']:
                return self._error_result(save_result['message'])
            
            # Success!
            logger.info(f"✅ Enrollment successful for {name}")
            
            return {
                "success": True,
                "message": f"Successfully enrolled {name}",
                "name": name,
                "embedding_dim": len(final_embedding),
                "photos_used": len(embeddings_data['embeddings']),
                "total_photos": len(images),
                "quality_scores": [round(q, 3) for q in embeddings_data['quality_scores']],
                "consistency_score": round(consistency_check['avg_similarity'], 3),
                "avg_quality": round(np.mean(embeddings_data['quality_scores']), 3)
            }
            
        except Exception as e:
            logger.error(f"Enrollment processing error: {e}", exc_info=True)
            return self._error_result(f"Enrollment failed: {str(e)}")
    
    def _process_images(self, images: List) -> Dict:
        """
        Process and validate all images
        Returns validated images with metadata
        """
        valid_images = []
        errors = []
        
        for idx, image_data in enumerate(images):
            try:
                # Preprocess image
                image = face_recognition_service.preprocess_image(image_data)
                
                if image is None:
                    errors.append(f"Image {idx+1}: Invalid format")
                    continue
                
                # Enhance image quality
                if self.enable_enhancement:
                    image = self._enhance_image(image)
                
                # Detect faces
                faces = self._detect_faces(image)
                
                if not faces:
                    errors.append(f"Image {idx+1}: No face detected")
                    continue
                
                if len(faces) > 1:
                    errors.append(f"Image {idx+1}: Multiple faces detected")
                    continue
                
                face_data = faces[0]
                
                # Extract face region
                facial_area = face_data.get('facial_area', {})
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)
                
                # Validate face size
                if w < 80 or h < 80:
                    errors.append(f"Image {idx+1}: Face too small ({w}x{h})")
                    continue
                
                # Extract and align face
                face_region = image[y:y+h, x:x+w]
                
                if self.enable_alignment:
                    face_region = self._align_face(face_region, face_data)
                
                # Calculate quality
                quality_score = self._calculate_detailed_quality(face_region, image)
                
                if quality_score < self.min_quality_score:
                    errors.append(f"Image {idx+1}: Low quality ({quality_score:.2f})")
                    continue
                
                # Store valid image data
                valid_images.append({
                    'image': face_region,
                    'full_image': image,
                    'quality': quality_score,
                    'face_data': face_data,
                    'index': idx
                })
                
                logger.debug(f"Image {idx+1}: Valid (quality: {quality_score:.2f})")
                
            except Exception as e:
                errors.append(f"Image {idx+1}: Processing error - {str(e)}")
                logger.warning(f"Error processing image {idx+1}: {e}")
        
        return {
            'valid_images': valid_images,
            'errors': errors
        }
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using high-accuracy detector"""
        try:
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.enrollment_detector,
                enforce_detection=False,
                align=True
            )
            return faces
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced image enhancement"""
        try:
            # Convert to LAB color space
            lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced_lab = cv.merge([l, a, b])
            enhanced = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)
            
            # Optional: Denoise (adds latency)
            # enhanced = cv.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            return image
    
    def _align_face(self, face_region: np.ndarray, face_data: Dict) -> np.ndarray:
        """Align face for better recognition"""
        try:
            # Simple resize to standard size
            # More advanced alignment would use facial landmarks
            target_size = (224, 224)
            aligned = cv.resize(face_region, target_size, interpolation=cv.INTER_LANCZOS4)
            return aligned
        except Exception as e:
            logger.warning(f"Alignment failed: {e}")
            return face_region
    
    def _calculate_detailed_quality(self, face_region: np.ndarray, 
                                    full_image: np.ndarray) -> float:
        """
        Calculate comprehensive quality score
        Returns: 0-1 score
        """
        try:
            scores = []
            
            # 1. Sharpness (Laplacian variance)
            gray = cv.cvtColor(face_region, cv.COLOR_BGR2GRAY)
            laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
            sharpness_score = min(laplacian_var / 150.0, 1.0)
            scores.append(sharpness_score * 0.30)  # 30% weight
            
            # 2. Brightness (optimal range: 100-160)
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 130) / 130.0
            brightness_score = max(0, brightness_score)
            scores.append(brightness_score * 0.20)  # 20% weight
            
            # 3. Contrast
            contrast = np.std(gray)
            contrast_score = min(contrast / 50.0, 1.0)
            scores.append(contrast_score * 0.20)  # 20% weight
            
            # 4. Face size (larger is better)
            face_area = face_region.shape[0] * face_region.shape[1]
            image_area = full_image.shape[0] * full_image.shape[1]
            size_ratio = face_area / image_area if image_area > 0 else 0
            size_score = min(size_ratio * 5, 1.0)  # Face should be ~20% of image
            scores.append(size_score * 0.15)  # 15% weight
            
            # 5. Edge density (detail preservation)
            edges = cv.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(edge_density * 20, 1.0)
            scores.append(edge_score * 0.15)  # 15% weight
            
            final_score = sum(scores)
            
            logger.debug(f"Quality: {final_score:.3f} (sharp:{sharpness_score:.2f}, "
                        f"bright:{brightness_score:.2f}, contrast:{contrast_score:.2f})")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Quality calculation error: {e}")
            return 0.5
    
    def _extract_embeddings(self, valid_images: List[Dict]) -> Dict:
        """Extract embeddings from all valid images"""
        embeddings = []
        quality_scores = []
        
        for img_data in valid_images:
            try:
                # Extract embedding with shorter timeout
                embedding = face_recognition_service.extract_embedding(
                    img_data['image'],
                    timeout=10
                )
                
                if embedding is not None:
                    embeddings.append(embedding)
                    quality_scores.append(img_data['quality'])
                    logger.debug(f"Extracted embedding from image {img_data['index']+1}")
                else:
                    logger.warning(f"Failed to extract embedding from image {img_data['index']+1}")
                    
            except Exception as e:
                logger.warning(f"Embedding extraction failed for image {img_data['index']+1}: {e}")
        
        return {
            'embeddings': embeddings,
            'quality_scores': quality_scores
        }
    
    def _validate_consistency(self, embeddings: List[np.ndarray]) -> Dict:
        """
        Validate that all embeddings are from the same person
        Returns consistency metrics
        """
        try:
            if len(embeddings) < 2:
                return {'valid': True, 'avg_similarity': 1.0, 'min_similarity': 1.0}
            
            # Calculate pairwise similarities (cosine similarity)
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # Cosine similarity (dot product of normalized vectors)
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            min_similarity = np.min(similarities)
            
            # Check if consistent
            valid = min_similarity >= self.min_consistency
            
            logger.info(f"Consistency: avg={avg_similarity:.3f}, min={min_similarity:.3f}, valid={valid}")
            
            return {
                'valid': valid,
                'avg_similarity': avg_similarity,
                'min_similarity': min_similarity,
                'similarities': similarities
            }
            
        except Exception as e:
            logger.error(f"Consistency validation error: {e}")
            return {'valid': False, 'avg_similarity': 0.0, 'min_similarity': 0.0}
    
    def _generate_final_embedding(self, embeddings: List[np.ndarray], 
                                  quality_scores: List[float]) -> np.ndarray:
        """
        Generate final embedding using weighted average
        Higher quality images get more weight
        """
        try:
            # Normalize quality scores to weights
            quality_array = np.array(quality_scores)
            weights = quality_array / np.sum(quality_array)
            
            # Weighted average
            final_embedding = np.average(embeddings, axis=0, weights=weights)
            
            # Normalize
            final_embedding = final_embedding / (np.linalg.norm(final_embedding) + 1e-8)
            
            logger.info(f"Generated final embedding using {len(embeddings)} embeddings "
                       f"with quality weights: {[f'{w:.2f}' for w in weights]}")
            
            return final_embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Final embedding generation error: {e}")
            # Fallback to simple average
            return np.mean(embeddings, axis=0).astype(np.float32)
    
    def _save_enrollment(self, name: str, embedding: np.ndarray, 
                        user_data: Dict, photos_count: int) -> Dict:
        """Save enrollment to database and FAISS index"""
        try:
            from app import db
            from app.models import Person, User, Profile
            from werkzeug.security import generate_password_hash
            
            # Check if person already exists
            existing_person = Person.query.filter_by(name=name).first()
            if existing_person:
                return {'success': False, 'message': f'Person {name} already enrolled'}
            
            # Serialize embedding
            embedding_bytes = pickle.dumps(embedding)
            
            # Create Person record
            person = Person(
                name=name,
                embedding=embedding_bytes,
                embedding_dim=len(embedding),
                photos_count=photos_count,
                status='active',
                enrollment_date=datetime.utcnow()
            )
            db.session.add(person)
            
            # Create User record (if credentials provided)
            if user_data.get('email') and user_data.get('password'):
                user = User(
                    name=name,
                    email=user_data['email'],
                    password_hash=generate_password_hash(user_data['password']),
                    department=user_data.get('department', ''),
                    phone=user_data.get('phone', ''),
                    status='active'
                )
                db.session.add(user)
            
            # Create Profile record
            profile = Profile(
                name=name,
                email=user_data.get('email', ''),
                department=user_data.get('department', ''),
                phone=user_data.get('phone', '')
            )
            db.session.add(profile)
            
            db.session.commit()
            
            # Add to FAISS index
            faiss_service.add_person(name, embedding)
            
            logger.info(f"✅ Saved {name} to database and FAISS index")
            
            return {'success': True, 'message': 'Enrollment saved successfully'}
            
        except Exception as e:
            logger.error(f"Save enrollment error: {e}", exc_info=True)
            db.session.rollback()
            return {'success': False, 'message': f'Failed to save: {str(e)}'}
    
    def _error_result(self, message: str) -> Dict:
        """Return error result"""
        logger.error(f"Enrollment error: {message}")
        return {
            "success": False,
            "message": message,
            "embedding_dim": 0,
            "photos_used": 0
        }


# Global singleton
advanced_enrollment_service = AdvancedEnrollmentService()
