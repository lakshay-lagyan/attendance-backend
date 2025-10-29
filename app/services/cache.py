import logging
import json
import pickle
from typing import Any, Optional
import redis
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)

class CacheService:
    
    def __init__(self):
        self.client = None
        self.available = False
        self.memory_cache = {}  
        self.hit_count = 0
        self.miss_count = 0
    
    def initialize(self, redis_url: Optional[str] = None):
        """Initialize Redis connection"""
        if not redis_url:
            logger.warning("No Redis URL provided, using memory cache")
            return
        
        try:
            self.client = redis.from_url(
                redis_url,
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.client.ping()
            self.available = True
            logger.info("✅ Redis cache connected")
            
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable: {e}. Using memory cache.")
            self.available = False
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self.available and self.client is not None
    
    def ping(self) -> bool:
        """Ping Redis server"""
        try:
            if self.is_available():
                return self.client.ping()
        except Exception:
            self.available = False
        return False
    
    def get(self, key: str, default=None) -> Any:
        """Get value from cache"""
        try:
            if self.is_available():
                value = self.client.get(key)
                if value is not None:
                    self.hit_count += 1
                    return value
                self.miss_count += 1
                return default
            else:
                # Fallback to memory cache
                self.miss_count += 1
                return self.memory_cache.get(key, default)
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.miss_count += 1
            return default
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            if self.is_available():
                return self.client.setex(key, ttl, value)
            else:
                # Fallback to memory cache 
                self.memory_cache[key] = value
                return True
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.is_available():
                return bool(self.client.delete(key))
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            if self.is_available():
                return bool(self.client.exists(key))
            else:
                return key in self.memory_cache
                
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter"""
        try:
            if self.is_available():
                return self.client.incr(key, amount)
            else:
                current = self.memory_cache.get(key, 0)
                self.memory_cache[key] = current + amount
                return self.memory_cache[key]
                
        except Exception as e:
            logger.error(f"Cache incr error: {e}")
            return None
    
    def get_json(self, key: str, default=None) -> Any:
        """Get JSON value from cache"""
        value = self.get(key)
        if value:
            try:
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                return json.loads(value)
            except Exception as e:
                logger.error(f"JSON decode error: {e}")
        return default
    
    def set_json(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set JSON value in cache"""
        try:
            json_str = json.dumps(value)
            return self.set(key, json_str, ttl)
        except Exception as e:
            logger.error(f"JSON encode error: {e}")
            return False
    
    def get_pickle(self, key: str, default=None) -> Any:
        """Get pickled value from cache"""
        value = self.get(key)
        if value:
            try:
                return pickle.loads(value)
            except Exception as e:
                logger.error(f"Pickle decode error: {e}")
        return default
    
    def set_pickle(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set pickled value in cache"""
        try:
            pickled = pickle.dumps(value)
            return self.set(key, pickled, ttl)
        except Exception as e:
            logger.error(f"Pickle encode error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            if self.is_available():
                keys = self.client.keys(pattern)
                if keys:
                    return self.client.delete(*keys)
                return 0
            else:
                # Clear memory cache
                count = 0
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    count += 1
                return count
                
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            'available': self.available,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
        }
        
        if self.is_available():
            try:
                info = self.client.info('stats')
                stats.update({
                    'connected_clients': info.get('connected_clients', 0),
                    'total_connections_received': info.get('total_connections_received', 0),
                    'used_memory_human': info.get('used_memory_human', 'N/A')
                })
            except Exception as e:
                logger.error(f"Failed to get Redis stats: {e}")
        
        return stats
    
    def flush_all(self):
        """Clear entire cache (use with caution!)"""
        try:
            if self.is_available():
                self.client.flushdb()
            else:
                self.memory_cache.clear()
            logger.warning("Cache flushed!")
        except Exception as e:
            logger.error(f"Cache flush error: {e}")


# Decorator for caching function results
def cached(ttl: int = 3600, key_prefix: str = 'func'):
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            cache_key = ':'.join(key_parts)
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            # Try to get from cache
            result = cache_service.get_pickle(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache_service.set_pickle(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# Global singleton instance
cache_service = CacheService()