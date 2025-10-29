import logging
import pickle
import threading
from typing import Optional, Tuple, List
import numpy as np
import faiss
from app.services.cache_service import cache_service

logger = logging.getLogger(__name__)

class FAISSService:
    """Optimized FAISS vector search service with Redis caching"""
    
    def __init__(self):
        self.index = None
        self.person_map = []
        self.embedding_dim = 512
        self.initialized = False
        self.lock = threading.Lock()  # Thread-safe operations
        
        # Cache keys
        self.CACHE_INDEX_KEY = 'faiss:index:data'
        self.CACHE_MAP_KEY = 'faiss:person:map'
        self.CACHE_VERSION_KEY = 'faiss:version'
        self.CACHE_TTL = 3600  # 1 hour
        
        # Performance metrics
        self.rebuild_counter = 0
        self.search_counter = 0
    
    def initialize(self):
        """Initialize FAISS index"""
        with self.lock:
            if self.initialized:
                return
            
            # Try loading from cache first
            if self._load_from_cache():
                logger.info("✅ FAISS index loaded from cache")
                self.initialized = True
                return
            
            # Fallback to database
            if self._rebuild_from_database():
                logger.info("✅ FAISS index built from database")
                self.initialized = True
                return
            
            # Create empty index
            self._create_empty_index()
            logger.info("✅ FAISS empty index created")
            self.initialized = True
    
    def _create_empty_index(self):
        """Create empty FAISS index"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.person_map = []
        faiss.normalize_L2(self.index.index)  # Use L2 normalized vectors
    
    def _load_from_cache(self) -> bool:
        """Load FAISS index from Redis cache"""
        try:
            if not cache_service.is_available():
                return False
            
            # Get cached data
            index_data = cache_service.get(self.CACHE_INDEX_KEY)
            map_data = cache_service.get(self.CACHE_MAP_KEY)
            
            if not index_data or not map_data:
                return False
            
            # Deserialize
            self.index = faiss.deserialize_index(index_data)
            self.person_map = pickle.loads(map_data)
            
            logger.info(f"Loaded FAISS index: {len(self.person_map)} persons")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return False
    
    def _save_to_cache(self):
        """Save FAISS index to Redis cache"""
        try:
            if not cache_service.is_available():
                return
            
            # Serialize index
            index_data = faiss.serialize_index(self.index)
            map_data = pickle.dumps(self.person_map)
            
            # Save to cache
            cache_service.set(self.CACHE_INDEX_KEY, index_data, self.CACHE_TTL)
            cache_service.set(self.CACHE_MAP_KEY, map_data, self.CACHE_TTL)
            
            # Increment version
            cache_service.incr(self.CACHE_VERSION_KEY)
            
            logger.info(f"Saved FAISS index to cache: {len(self.person_map)} persons")
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
    
    def _rebuild_from_database(self) -> bool:
        """Rebuild FAISS index from database"""
        try:
            from app import db
            from app.models.user import Person
            
            # Create new index
            self._create_empty_index()
            
            # Load all active persons
            persons = Person.query.filter_by(status='active').all()
            
            if not persons:
                logger.warning("No persons found in database")
                return True
            
            # Build index
            embeddings_list = []
            names_list = []
            
            for person in persons:
                try:
                    # Deserialize embedding
                    embedding = pickle.loads(person.embedding)
                    
                    # Ensure correct dimension
                    if len(embedding) != self.embedding_dim:
                        if len(embedding) > self.embedding_dim:
                            embedding = embedding[:self.embedding_dim]
                        else:
                            padding = np.zeros(self.embedding_dim - len(embedding))
                            embedding = np.concatenate([embedding, padding])
                    
                    # Normalize
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    
                    embeddings_list.append(embedding)
                    names_list.append(person.name)
                    
                except Exception as e:
                    logger.warning(f"Failed to load embedding for {person.name}: {e}")
                    continue
            
            if not embeddings_list:
                logger.warning("No valid embeddings found")
                return True
            
            # Add to index in batch (more efficient)
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            self.index.add(embeddings_array)
            self.person_map = names_list
            
            # Save to cache
            self._save_to_cache()
            
            self.rebuild_counter += 1
            logger.info(f"✅ FAISS index rebuilt: {len(self.person_map)} persons")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild from database: {e}")
            return False
    
    def add_person(self, name: str, embedding: np.ndarray):
        """Add person to index"""
        with self.lock:
            if not self.initialized:
                self.initialize()
            
            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Add to index
            self.index.add(np.array([embedding], dtype=np.float32))
            self.person_map.append(name)
            
            # Update cache
            self._save_to_cache()
            
            logger.info(f"Added person to FAISS: {name}")
    
    def search(self, embedding: np.ndarray, threshold: float = 0.6, top_k: int = 1) -> List[Tuple[Optional[str], float]]:
        
        with self.lock:
            if not self.initialized:
                self.initialize()
            
            if self.index.ntotal == 0:
                return [(None, 0.0)]
            
            # Normalize query
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Search
            distances, indices = self.index.search(
                np.array([embedding], dtype=np.float32), 
                k=min(top_k, self.index.ntotal)
            )
            
            # Filter by threshold
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if dist >= threshold:
                    name = self.person_map[idx]
                    results.append((name, float(dist)))
                else:
                    results.append((None, float(dist)))
            
            self.search_counter += 1
            
            return results if results else [(None, 0.0)]
    
    def remove_person(self, name: str):
        """Remove person from index (requires rebuild)"""
        with self.lock:
            if name in self.person_map:
                # FAISS doesn't support deletion, need to rebuild
                self._rebuild_from_database()
                logger.info(f"Removed person from FAISS: {name}")
    
    def get_total_persons(self) -> int:
        """Get total number of persons in index"""
        return self.index.ntotal if self.index else 0
    
    def get_stats(self) -> dict:
        """Get FAISS service statistics"""
        return {
            'total_persons': self.get_total_persons(),
            'initialized': self.initialized,
            'embedding_dim': self.embedding_dim,
            'rebuild_count': self.rebuild_counter,
            'search_count': self.search_counter,
            'index_type': type(self.index).__name__ if self.index else None
        }
    
    def rebuild_from_database(self):
        """Public method to force rebuild"""
        with self.lock:
            self._rebuild_from_database()
    
    def clear_cache(self):
        """Clear cached FAISS data"""
        try:
            if cache_service.is_available():
                cache_service.delete(self.CACHE_INDEX_KEY)
                cache_service.delete(self.CACHE_MAP_KEY)
                cache_service.delete(self.CACHE_VERSION_KEY)
                logger.info("Cleared FAISS cache")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Global singleton instance
faiss_service = FAISSService()