# axiom/politeness/robots.py
"""
Intelligent Rate Limiting & Politeness for axiom
Adaptive rate limiting, robots.txt compliance, automatic backoff, and domain-specific throttling.
"""

import asyncio
import time
import re
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple, List, Any, Set
from urllib.parse import urlparse, urljoin
from collections import defaultdict, deque
from enum import Enum
import logging

# ML imports for adaptive learning
try:
    import numpy as np
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

# Import from existing axiom modules
from axiom.core.custom_types import RequestResult, Response
from axiom.core.utils._utils import get_logger, exponential_backoff
from axiom.core.storage import StorageManager


class PolitenessError(Exception):
    """Base exception for politeness-related errors."""
    pass


class RobotsTxtDisallowedError(PolitenessError):
    """Raised when a URL is disallowed by robots.txt."""
    pass


class RateLimitExceededError(PolitenessError):
    """Raised when rate limit is exceeded."""
    pass


class DomainThrottledError(PolitenessError):
    """Raised when domain is temporarily throttled."""
    pass


class BackoffRequiredError(PolitenessError):
    """Raised when backoff is required due to 429/503 responses."""
    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(f"Backoff required. Retry after: {retry_after}s")


class TokenBucket:
    """
    Token bucket algorithm for rate limiting with dynamic adjustment.
    Supports burst capacity and adaptive refill rates.
    """
    
    def __init__(self, 
                 capacity: float = 10.0, 
                 refill_rate: float = 1.0,
                 initial_tokens: Optional[float] = None):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_rate: Tokens added per second
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(initial_tokens if initial_tokens is not None else capacity)
        self.last_refill_time = time.time()
        self._lock = asyncio.Lock()
        
    async def consume(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait for tokens
            
        Returns:
            True if tokens were consumed, False if timeout occurred
            
        Raises:
            RateLimitExceededError: If timeout is None and insufficient tokens
        """
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            if timeout is None:
                raise RateLimitExceededError(
                    f"Insufficient tokens: {self.tokens:.2f} < {tokens:.2f}"
                )
            
            # Calculate wait time
            deficit = tokens - self.tokens
            wait_time = deficit / self.refill_rate if self.refill_rate > 0 else float('inf')
            
            if wait_time > timeout:
                return False
            
            # Wait for tokens
            await asyncio.sleep(wait_time)
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        tokens_to_add = elapsed * self.refill_rate
        
        if tokens_to_add > 0:
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill_time = now
    
    def adjust_rate(self, new_rate: float):
        """Dynamically adjust the refill rate."""
        self._refill()  # Refill at old rate first
        self.refill_rate = max(0.001, new_rate)  # Minimum rate to prevent division by zero
    
    def adjust_capacity(self, new_capacity: float):
        """Dynamically adjust the bucket capacity."""
        self._refill()
        self.capacity = max(1.0, new_capacity)  # Minimum capacity of 1
        self.tokens = min(self.tokens, self.capacity)
    
    def get_wait_time(self, tokens: float = 1.0) -> float:
        """Calculate wait time for requested tokens."""
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        deficit = tokens - self.tokens
        return deficit / self.refill_rate if self.refill_rate > 0 else float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize bucket state."""
        return {
            'capacity': self.capacity,
            'refill_rate': self.refill_rate,
            'tokens': self.tokens,
            'last_refill_time': self.last_refill_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenBucket':
        """Deserialize bucket state."""
        bucket = cls(
            capacity=data['capacity'],
            refill_rate=data['refill_rate'],
            initial_tokens=data.get('tokens')
        )
        bucket.last_refill_time = data.get('last_refill_time', time.time())
        return bucket


@dataclass
class RobotsTxtRule:
    """Parsed robots.txt rule."""
    user_agents: Set[str]
    disallowed_paths: List[str]
    allowed_paths: List[str]
    crawl_delay: Optional[float] = None
    request_rate: Optional[Tuple[int, int]] = None  # (requests, seconds)
    sitemaps: List[str] = field(default_factory=list)
    
    def is_allowed(self, path: str, user_agent: str = '*') -> bool:
        """Check if path is allowed for given user agent."""
        # Check if rule applies to this user agent
        if '*' not in self.user_agents and user_agent not in self.user_agents:
            return False
        
        # Check explicit allows first
        for allowed in self.allowed_paths:
            if path.startswith(allowed):
                return True
        
        # Check disallows
        for disallowed in self.disallowed_paths:
            if path.startswith(disallowed):
                return False
        
        # Default allow if no matching rules
        return True


class RobotsTxtParser:
    """
    Parser and cache for robots.txt files.
    Supports standard robots.txt format and extensions.
    """
    
    def __init__(self, cache_ttl: int = 3600):
        """
        Initialize parser.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, RobotsTxtRule]] = {}
        self._storage = StorageManager()
        self.logger = get_logger('RobotsTxtParser')
    
    async def get_rules(self, 
                       domain: str, 
                       fetch_func: Any,
                       user_agent: str = 'axiom') -> RobotsTxtRule:
        """
        Get robots.txt rules for domain.
        
        Args:
            domain: Domain to get rules for
            fetch_func: Async function to fetch robots.txt content
            user_agent: User agent to use when fetching
            
        Returns:
            RobotsTxtRule object
            
        Raises:
            RobotsTxtDisallowedError: If robots.txt cannot be fetched
        """
        cache_key = f"robots:{domain}"
        
        # Check memory cache
        if cache_key in self._cache:
            cached_time, rules = self._cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return rules
        
        # Check persistent storage
        stored = await self._storage.get(cache_key)
        if stored:
            try:
                data = json.loads(stored)
                if time.time() - data['timestamp'] < self.cache_ttl:
                    rules = self._dict_to_rules(data['rules'])
                    self._cache[cache_key] = (data['timestamp'], rules)
                    return rules
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Fetch robots.txt
        robots_url = f"https://{domain}/robots.txt"
        try:
            response = await fetch_func(robots_url)
            if response.status == 200:
                content = response.text
            elif response.status == 404:
                # No robots.txt means everything is allowed
                content = ""
            else:
                raise RobotsTxtDisallowedError(
                    f"Failed to fetch robots.txt: HTTP {response.status}"
                )
        except Exception as e:
            self.logger.warning(f"Failed to fetch robots.txt for {domain}: {e}")
            # Default to allowing everything on fetch failure
            content = ""
        
        # Parse content
        rules = self._parse_robots_txt(content, domain)
        
        # Cache results
        cache_time = time.time()
        self._cache[cache_key] = (cache_time, rules)
        
        # Store persistently
        await self._storage.set(
            cache_key,
            json.dumps({
                'timestamp': cache_time,
                'rules': self._rules_to_dict(rules)
            }),
            ttl=self.cache_ttl
        )
        
        return rules
    
    def _parse_robots_txt(self, content: str, domain: str) -> RobotsTxtRule:
        """Parse robots.txt content."""
        user_agents = set()
        disallowed_paths = []
        allowed_paths = []
        crawl_delay = None
        request_rate = None
        sitemaps = []
        
        current_agents = set()
        is_default_agent = True
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse directive
            if ':' in line:
                directive, _, value = line.partition(':')
                directive = directive.strip().lower()
                value = value.strip()
                
                if directive == 'user-agent':
                    # Start new group
                    if current_agents and is_default_agent:
                        # Save rules for default agent
                        pass
                    
                    current_agents = {value.lower()}
                    is_default_agent = (value == '*')
                    
                elif directive == 'disallow':
                    if value:
                        disallowed_paths.append(value)
                        
                elif directive == 'allow':
                    if value:
                        allowed_paths.append(value)
                        
                elif directive == 'crawl-delay':
                    try:
                        crawl_delay = float(value)
                    except ValueError:
                        pass
                        
                elif directive == 'request-rate':
                    try:
                        parts = value.split('/')
                        if len(parts) == 2:
                            request_rate = (int(parts[0]), int(parts[1]))
                    except (ValueError, IndexError):
                        pass
                        
                elif directive == 'sitemap':
                    sitemaps.append(value)
        
        # Create rule
        return RobotsTxtRule(
            user_agents=current_agents or {'*'},
            disallowed_paths=disallowed_paths,
            allowed_paths=allowed_paths,
            crawl_delay=crawl_delay,
            request_rate=request_rate,
            sitemaps=sitemaps
        )
    
    def _rules_to_dict(self, rules: RobotsTxtRule) -> Dict[str, Any]:
        """Convert rules to dictionary for serialization."""
        return {
            'user_agents': list(rules.user_agents),
            'disallowed_paths': rules.disallowed_paths,
            'allowed_paths': rules.allowed_paths,
            'crawl_delay': rules.crawl_delay,
            'request_rate': rules.request_rate,
            'sitemaps': rules.sitemaps
        }
    
    def _dict_to_rules(self, data: Dict[str, Any]) -> RobotsTxtRule:
        """Convert dictionary to rules."""
        return RobotsTxtRule(
            user_agents=set(data.get('user_agents', ['*'])),
            disallowed_paths=data.get('disallowed_paths', []),
            allowed_paths=data.get('allowed_paths', []),
            crawl_delay=data.get('crawl_delay'),
            request_rate=tuple(data['request_rate']) if data.get('request_rate') else None,
            sitemaps=data.get('sitemaps', [])
        )
    
    def clear_cache(self, domain: Optional[str] = None):
        """Clear cache for domain or all domains."""
        if domain:
            cache_key = f"robots:{domain}"
            self._cache.pop(cache_key, None)
        else:
            self._cache.clear()


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter with machine learning for optimal request intervals.
    Uses online learning to adjust based on response patterns.
    """
    
    def __init__(self, 
                 initial_rate: float = 1.0,
                 min_rate: float = 0.1,
                 max_rate: float = 10.0,
                 learning_rate: float = 0.01):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_rate: Initial requests per second
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
            learning_rate: Learning rate for ML model
        """
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        
        # Response time tracking
        self.response_times: deque = deque(maxlen=100)
        self.error_counts: Dict[int, int] = defaultdict(int)
        self.last_error_time: Dict[int, float] = {}
        
        # ML model for predicting optimal interval
        self.ml_enabled = ML_AVAILABLE
        if self.ml_enabled:
            self.model = SGDRegressor(learning_rate='constant', eta0=learning_rate)
            self.scaler = StandardScaler()
            self._initialize_ml_model()
        else:
            self.logger = get_logger('AdaptiveRateLimiter')
            self.logger.warning("ML libraries not available. Using heuristic adaptation.")
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.throttled_requests = 0
        
    def _initialize_ml_model(self):
        """Initialize ML model with synthetic data."""
        # Features: [avg_response_time, error_rate, time_since_last_error]
        # Target: optimal_interval (in seconds)
        X_init = np.array([
            [0.1, 0.0, 10.0],  # Fast, no errors
            [1.0, 0.1, 5.0],   # Medium, some errors
            [5.0, 0.5, 1.0],   # Slow, many errors
        ])
        y_init = np.array([0.1, 1.0, 5.0])  # Corresponding intervals
        
        # Scale features
        self.scaler.fit(X_init)
        X_scaled = self.scaler.transform(X_init)
        
        # Train initial model
        self.model.partial_fit(X_scaled, y_init)
    
    def record_response(self, 
                       status_code: int, 
                       response_time: float,
                       domain: str = ''):
        """
        Record response for adaptive learning.
        
        Args:
            status_code: HTTP status code
            response_time: Response time in seconds
            domain: Domain that was requested
        """
        self.total_requests += 1
        
        # Track response time
        self.response_times.append(response_time)
        
        # Track errors
        if status_code >= 400:
            self.error_counts[status_code] += 1
            self.last_error_time[domain] = time.time()
            
            if status_code in (429, 503):
                self.throttled_requests += 1
        else:
            self.successful_requests += 1
        
        # Update ML model if enabled
        if self.ml_enabled and len(self.response_times) >= 10:
            self._update_ml_model(domain)
    
    def _update_ml_model(self, domain: str):
        """Update ML model with recent data."""
        try:
            # Calculate features
            avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0.5
            error_rate = self.throttled_requests / max(1, self.total_requests)
            time_since_error = time.time() - self.last_error_time.get(domain, 0)
            
            # Predict optimal interval
            features = np.array([[avg_response_time, error_rate, time_since_error]])
            features_scaled = self.scaler.transform(features)
            
            # Current optimal interval based on model
            predicted_interval = self.model.predict(features_scaled)[0]
            
            # Calculate actual optimal interval based on recent success
            if self.successful_requests > 0:
                success_rate = self.successful_requests / self.total_requests
                actual_interval = 1.0 / self.current_rate
                
                # Adjust based on success rate
                if success_rate > 0.95:  # Very successful, can speed up
                    target_interval = actual_interval * 0.9
                elif success_rate < 0.8:  # Too many errors, slow down
                    target_interval = actual_interval * 1.5
                else:
                    target_interval = actual_interval
                
                # Update model with actual data
                self.model.partial_fit(features_scaled, [target_interval])
                
                # Update current rate based on prediction blend
                blend_factor = 0.3  # Weight of ML prediction
                new_interval = (blend_factor * predicted_interval + 
                               (1 - blend_factor) * target_interval)
                new_rate = 1.0 / max(0.1, new_interval)
                
                # Clamp to bounds
                self.current_rate = max(self.min_rate, min(self.max_rate, new_rate))
        
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"ML model update failed: {e}")
    
    def get_optimal_interval(self) -> float:
        """Get optimal interval between requests in seconds."""
        return 1.0 / self.current_rate if self.current_rate > 0 else float('inf')
    
    def adjust_for_backoff(self, backoff_factor: float = 2.0):
        """Adjust rate based on backoff requirement."""
        self.current_rate = max(self.min_rate, self.current_rate / backoff_factor)
    
    def adjust_for_success(self):
        """Gradually increase rate after successful requests."""
        if self.successful_requests > 0:
            success_rate = self.successful_requests / self.total_requests
            if success_rate > 0.95:
                # Gradually increase rate (max 10% increase)
                self.current_rate = min(
                    self.max_rate,
                    self.current_rate * 1.1
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'current_rate': self.current_rate,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'throttled_requests': self.throttled_requests,
            'success_rate': self.successful_requests / max(1, self.total_requests),
            'avg_response_time': np.mean(list(self.response_times)) if self.response_times else 0,
            'ml_enabled': self.ml_enabled
        }


@dataclass
class DomainState:
    """State tracking for a specific domain."""
    domain: str
    token_bucket: TokenBucket
    robots_rules: Optional[RobotsTxtRule] = None
    rate_limiter: AdaptiveRateLimiter = field(default_factory=AdaptiveRateLimiter)
    last_request_time: float = 0.0
    consecutive_errors: int = 0
    backoff_until: float = 0.0
    crawl_delay: float = 0.0
    request_history: deque = field(default_factory=lambda: deque(maxlen=100))
    is_throttled: bool = False
    throttle_until: float = 0.0
    
    def update_from_response(self, 
                            status_code: int, 
                            response_time: float,
                            headers: Optional[Dict] = None):
        """Update state based on response."""
        now = time.time()
        
        # Record request
        self.request_history.append({
            'time': now,
            'status': status_code,
            'response_time': response_time
        })
        
        # Update rate limiter
        self.rate_limiter.record_response(status_code, response_time, self.domain)
        
        # Handle specific status codes
        if status_code == 429:  # Too Many Requests
            self.consecutive_errors += 1
            retry_after = self._parse_retry_after(headers)
            backoff_time = retry_after if retry_after else exponential_backoff(self.consecutive_errors)
            self.backoff_until = now + backoff_time
            
            # Adjust rate limiter
            self.rate_limiter.adjust_for_backoff(2.0 ** self.consecutive_errors)
            
        elif status_code == 503:  # Service Unavailable
            self.consecutive_errors += 1
            retry_after = self._parse_retry_after(headers)
            backoff_time = retry_after if retry_after else exponential_backoff(self.consecutive_errors)
            self.backoff_until = now + backoff_time
            
            # Adjust rate limiter
            self.rate_limiter.adjust_for_backoff(2.0 ** self.consecutive_errors)
            
        elif status_code >= 500:  # Server error
            self.consecutive_errors += 1
            if self.consecutive_errors >= 3:
                # Throttle after multiple server errors
                self.is_throttled = True
                self.throttle_until = now + 60  # Throttle for 1 minute
                
        elif status_code < 400:  # Success
            self.consecutive_errors = 0
            self.rate_limiter.adjust_for_success()
            
            # Clear backoff if successful
            if now > self.backoff_until:
                self.backoff_until = 0.0
    
    def _parse_retry_after(self, headers: Optional[Dict]) -> Optional[float]:
        """Parse Retry-After header."""
        if not headers:
            return None
        
        retry_after = headers.get('Retry-After') or headers.get('retry-after')
        if not retry_after:
            return None
        
        try:
            # Could be seconds or HTTP date
            if retry_after.isdigit():
                return float(retry_after)
            # Try parsing as HTTP date (simplified)
            # In production, use email.utils.parsedate_to_datetime
            return 60.0  # Default to 60 seconds
        except (ValueError, AttributeError):
            return None
    
    def get_required_delay(self) -> float:
        """Get required delay before next request."""
        now = time.time()
        
        # Check backoff
        if now < self.backoff_until:
            return self.backoff_until - now
        
        # Check throttling
        if self.is_throttled and now < self.throttle_until:
            return self.throttle_until - now
        
        # Check crawl delay from robots.txt
        if self.crawl_delay > 0:
            time_since_last = now - self.last_request_time
            if time_since_last < self.crawl_delay:
                return self.crawl_delay - time_since_last
        
        # Check rate limiter
        return self.rate_limiter.get_optimal_interval()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize domain state."""
        return {
            'domain': self.domain,
            'token_bucket': self.token_bucket.to_dict(),
            'last_request_time': self.last_request_time,
            'consecutive_errors': self.consecutive_errors,
            'backoff_until': self.backoff_until,
            'crawl_delay': self.crawl_delay,
            'is_throttled': self.is_throttled,
            'throttle_until': self.throttle_until,
            'rate_limiter_stats': self.rate_limiter.get_stats()
        }


class PolitenessPolicy:
    """
    Main politeness policy coordinator.
    Manages rate limiting, robots.txt compliance, and adaptive throttling.
    """
    
    def __init__(self,
                 default_capacity: float = 10.0,
                 default_refill_rate: float = 1.0,
                 respect_robots_txt: bool = True,
                 cache_ttl: int = 3600,
                 user_agent: str = 'axiom',
                 storage_path: Optional[str] = None):
        """
        Initialize politeness policy.
        
        Args:
            default_capacity: Default token bucket capacity
            default_refill_rate: Default token refill rate (tokens/second)
            respect_robots_txt: Whether to respect robots.txt
            cache_ttl: Cache TTL for robots.txt in seconds
            user_agent: User agent string for robots.txt
            storage_path: Path for persistent storage
        """
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self.respect_robots_txt = respect_robots_txt
        self.user_agent = user_agent
        
        # Domain state management
        self.domain_states: Dict[str, DomainState] = {}
        self._state_lock = asyncio.Lock()
        
        # Robots.txt parser
        self.robots_parser = RobotsTxtParser(cache_ttl=cache_ttl)
        
        # Global statistics
        self.global_stats = {
            'total_requests': 0,
            'blocked_by_robots': 0,
            'rate_limited': 0,
            'backoffs': 0,
            'domains_tracked': 0
        }
        
        # Storage for persistence
        self.storage = StorageManager(storage_path) if storage_path else None
        
        self.logger = get_logger('PolitenessPolicy')
        
        # Load persisted state
        if self.storage:
            asyncio.create_task(self._load_state())
    
    async def _load_state(self):
        """Load persisted state from storage."""
        if not self.storage:
            return
        
        try:
            state_data = await self.storage.get('politeness_state')
            if state_data:
                data = json.loads(state_data)
                for domain, domain_data in data.get('domains', {}).items():
                    bucket = TokenBucket.from_dict(domain_data['token_bucket'])
                    state = DomainState(
                        domain=domain,
                        token_bucket=bucket,
                        last_request_time=domain_data.get('last_request_time', 0),
                        consecutive_errors=domain_data.get('consecutive_errors', 0),
                        backoff_until=domain_data.get('backoff_until', 0),
                        crawl_delay=domain_data.get('crawl_delay', 0),
                        is_throttled=domain_data.get('is_throttled', False),
                        throttle_until=domain_data.get('throttle_until', 0)
                    )
                    self.domain_states[domain] = state
                
                self.global_stats.update(data.get('global_stats', {}))
                self.logger.info(f"Loaded state for {len(self.domain_states)} domains")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    async def _save_state(self):
        """Persist state to storage."""
        if not self.storage:
            return
        
        try:
            domains_data = {}
            for domain, state in self.domain_states.items():
                domains_data[domain] = state.to_dict()
            
            state_data = {
                'domains': domains_data,
                'global_stats': self.global_stats,
                'timestamp': time.time()
            }
            
            await self.storage.set(
                'politeness_state',
                json.dumps(state_data),
                ttl=86400  # 24 hours
            )
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    async def get_domain_state(self, 
                              domain: str,
                              fetch_robots: bool = True) -> DomainState:
        """
        Get or create state for domain.
        
        Args:
            domain: Domain name
            fetch_robots: Whether to fetch robots.txt
            
        Returns:
            DomainState object
        """
        async with self._state_lock:
            if domain not in self.domain_states:
                # Create new domain state
                bucket = TokenBucket(
                    capacity=self.default_capacity,
                    refill_rate=self.default_refill_rate
                )
                
                state = DomainState(
                    domain=domain,
                    token_bucket=bucket
                )
                
                # Fetch robots.txt if enabled
                if self.respect_robots_txt and fetch_robots:
                    try:
                        # We'll need a fetch function - this will be provided by the session
                        # For now, we'll create a placeholder
                        async def dummy_fetch(url):
                            # This should be replaced with actual fetch function
                            raise NotImplementedError("Fetch function not provided")
                        
                        state.robots_rules = await self.robots_parser.get_rules(
                            domain, dummy_fetch, self.user_agent
                        )
                        
                        # Apply crawl delay from robots.txt
                        if state.robots_rules.crawl_delay:
                            state.crawl_delay = state.robots_rules.crawl_delay
                            # Adjust bucket refill rate based on crawl delay
                            if state.crawl_delay > 0:
                                state.token_bucket.adjust_rate(1.0 / state.crawl_delay)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch robots.txt for {domain}: {e}")
                
                self.domain_states[domain] = state
                self.global_stats['domains_tracked'] = len(self.domain_states)
            
            return self.domain_states[domain]
    
    async def check_robots_txt(self, 
                              url: str,
                              domain_state: Optional[DomainState] = None) -> bool:
        """
        Check if URL is allowed by robots.txt.
        
        Args:
            url: URL to check
            domain_state: Optional pre-fetched domain state
            
        Returns:
            True if allowed, False otherwise
            
        Raises:
            RobotsTxtDisallowedError: If disallowed and strict mode is enabled
        """
        if not self.respect_robots_txt:
            return True
        
        parsed = urlparse(url)
        domain = parsed.netloc
        
        if domain_state is None:
            domain_state = await self.get_domain_state(domain)
        
        if not domain_state.robots_rules:
            # No rules means allowed
            return True
        
        path = parsed.path or '/'
        
        # Check if allowed
        is_allowed = domain_state.robots_rules.is_allowed(path, self.user_agent)
        
        if not is_allowed:
            self.global_stats['blocked_by_robots'] += 1
            raise RobotsTxtDisallowedError(
                f"URL {url} is disallowed by robots.txt for user agent '{self.user_agent}'"
            )
        
        return True
    
    async def wait_for_permission(self,
                                 url: str,
                                 timeout: Optional[float] = 30.0) -> bool:
        """
        Wait for permission to make request to URL.
        
        Args:
            url: URL to request
            timeout: Maximum time to wait
            
        Returns:
            True if permission granted
            
        Raises:
            BackoffRequiredError: If backoff is required
            DomainThrottledError: If domain is throttled
        """
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Get domain state
        domain_state = await self.get_domain_state(domain)
        
        # Check robots.txt
        await self.check_robots_txt(url, domain_state)
        
        # Check if we need to backoff
        now = time.time()
        if now < domain_state.backoff_until:
            wait_time = domain_state.backoff_until - now
            if timeout and wait_time > timeout:
                raise BackoffRequiredError(retry_after=wait_time)
            await asyncio.sleep(wait_time)
            self.global_stats['backoffs'] += 1
        
        # Check if domain is throttled
        if domain_state.is_throttled and now < domain_state.throttle_until:
            wait_time = domain_state.throttle_until - now
            if timeout and wait_time > timeout:
                raise DomainThrottledError(
                    f"Domain {domain} is throttled for {wait_time:.1f}s"
                )
            await asyncio.sleep(wait_time)
        
        # Calculate required delay
        required_delay = domain_state.get_required_delay()
        
        if required_delay > 0:
            if timeout and required_delay > timeout:
                self.global_stats['rate_limited'] += 1
                return False
            
            await asyncio.sleep(required_delay)
        
        # Wait for token from bucket
        try:
            await domain_state.token_bucket.consume(1.0, timeout=timeout)
        except RateLimitExceededError:
            self.global_stats['rate_limited'] += 1
            raise
        
        # Update last request time
        domain_state.last_request_time = time.time()
        self.global_stats['total_requests'] += 1
        
        return True
    
    async def record_response(self,
                             url: str,
                             status_code: int,
                             response_time: float,
                             headers: Optional[Dict] = None):
        """
        Record response for adaptive learning.
        
        Args:
            url: URL that was requested
            status_code: HTTP status code
            response_time: Response time in seconds
            headers: Response headers
        """
        parsed = urlparse(url)
        domain = parsed.netloc
        
        domain_state = await self.get_domain_state(domain, fetch_robots=False)
        domain_state.update_from_response(status_code, response_time, headers)
        
        # Check for Retry-After header on 429/503
        if status_code in (429, 503) and headers:
            retry_after = domain_state._parse_retry_after(headers)
            if retry_after:
                domain_state.backoff_until = time.time() + retry_after
        
        # Periodically save state
        if self.global_stats['total_requests'] % 100 == 0:
            await self._save_state()
    
    async def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for specific domain."""
        if domain not in self.domain_states:
            return {}
        
        state = self.domain_states[domain]
        return {
            'domain': domain,
            'token_bucket': state.token_bucket.to_dict(),
            'rate_limiter': state.rate_limiter.get_stats(),
            'consecutive_errors': state.consecutive_errors,
            'crawl_delay': state.crawl_delay,
            'is_throttled': state.is_throttled,
            'backoff_until': state.backoff_until,
            'last_request_time': state.last_request_time,
            'request_history_size': len(state.request_history)
        }
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics."""
        stats = self.global_stats.copy()
        stats['domains_tracked'] = len(self.domain_states)
        
        # Aggregate domain stats
        total_rate = 0.0
        total_capacity = 0.0
        for state in self.domain_states.values():
            total_rate += state.token_bucket.refill_rate
            total_capacity += state.token_bucket.capacity
        
        stats['average_rate'] = total_rate / max(1, len(self.domain_states))
        stats['average_capacity'] = total_capacity / max(1, len(self.domain_states))
        
        return stats
    
    async def update_domain_policy(self,
                                  domain: str,
                                  capacity: Optional[float] = None,
                                  refill_rate: Optional[float] = None,
                                  crawl_delay: Optional[float] = None):
        """
        Update policy for specific domain.
        
        Args:
            domain: Domain to update
            capacity: New token bucket capacity
            refill_rate: New refill rate
            crawl_delay: New crawl delay (overrides robots.txt)
        """
        state = await self.get_domain_state(domain, fetch_robots=False)
        
        if capacity is not None:
            state.token_bucket.adjust_capacity(capacity)
        
        if refill_rate is not None:
            state.token_bucket.adjust_rate(refill_rate)
        
        if crawl_delay is not None:
            state.crawl_delay = crawl_delay
            if crawl_delay > 0:
                state.token_bucket.adjust_rate(1.0 / crawl_delay)
    
    async def clear_domain_backoff(self, domain: str):
        """Clear backoff state for domain."""
        if domain in self.domain_states:
            state = self.domain_states[domain]
            state.backoff_until = 0.0
            state.consecutive_errors = 0
            state.is_throttled = False
            state.throttle_until = 0.0
    
    async def clear_all_state(self):
        """Clear all state."""
        async with self._state_lock:
            self.domain_states.clear()
            self.global_stats = {
                'total_requests': 0,
                'blocked_by_robots': 0,
                'rate_limited': 0,
                'backoffs': 0,
                'domains_tracked': 0
            }
            self.robots_parser.clear_cache()
            
            if self.storage:
                await self.storage.delete('politeness_state')
    
    async def export_state(self) -> Dict[str, Any]:
        """Export complete state for backup or analysis."""
        domains_data = {}
        for domain, state in self.domain_states.items():
            domains_data[domain] = state.to_dict()
        
        return {
            'domains': domains_data,
            'global_stats': self.global_stats,
            'robots_cache': {
                domain: self.robots_parser._cache.get(f"robots:{domain}")
                for domain in self.domain_states.keys()
            },
            'export_timestamp': time.time()
        }
    
    async def import_state(self, state_data: Dict[str, Any]):
        """Import state from backup."""
        async with self._state_lock:
            # Clear current state
            self.domain_states.clear()
            
            # Import domains
            for domain, domain_data in state_data.get('domains', {}).items():
                bucket = TokenBucket.from_dict(domain_data['token_bucket'])
                state = DomainState(
                    domain=domain,
                    token_bucket=bucket,
                    last_request_time=domain_data.get('last_request_time', 0),
                    consecutive_errors=domain_data.get('consecutive_errors', 0),
                    backoff_until=domain_data.get('backoff_until', 0),
                    crawl_delay=domain_data.get('crawl_delay', 0),
                    is_throttled=domain_data.get('is_throttled', False),
                    throttle_until=domain_data.get('throttle_until', 0)
                )
                self.domain_states[domain] = state
            
            # Import global stats
            self.global_stats.update(state_data.get('global_stats', {}))
            self.global_stats['domains_tracked'] = len(self.domain_states)
            
            # Import robots cache
            for domain, cache_entry in state_data.get('robots_cache', {}).items():
                if cache_entry:
                    cache_key = f"robots:{domain}"
                    self.robots_parser._cache[cache_key] = cache_entry
            
            self.logger.info(f"Imported state for {len(self.domain_states)} domains")


# Integration helper for existing axiom sessions
class PolitenessMiddleware:
    """
    Middleware for integrating politeness policy with existing axiom sessions.
    Can be used as a context manager or decorator.
    """
    
    def __init__(self, 
                 politeness_policy: PolitenessPolicy,
                 session: Any = None):
        """
        Initialize middleware.
        
        Args:
            politeness_policy: PolitenessPolicy instance
            session: Optional session to wrap
        """
        self.policy = politeness_policy
        self.session = session
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Save state on exit
        await self.policy._save_state()
    
    async def fetch(self, url: str, **kwargs) -> RequestResult:
        """
        Fetch URL with politeness policy applied.
        
        Args:
            url: URL to fetch
            **kwargs: Additional arguments for session.fetch
            
        Returns:
            RequestResult with response
        """
        # Wait for permission
        await self.policy.wait_for_permission(url)
        
        # Make request
        if self.session:
            start_time = time.time()
            try:
                response = await self.session.fetch(url, **kwargs)
                response_time = time.time() - start_time
                
                # Record response
                await self.policy.record_response(
                    url=url,
                    status_code=response.status,
                    response_time=response_time,
                    headers=dict(response.headers) if hasattr(response, 'headers') else None
                )
                
                return RequestResult(
                    url=url,
                    response=response,
                    success=response.status < 400,
                    status_code=response.status,
                    response_time=response_time
                )
            
            except Exception as e:
                response_time = time.time() - start_time
                # Record error
                await self.policy.record_response(
                    url=url,
                    status_code=500,  # Treat as server error
                    response_time=response_time
                )
                raise
        else:
            raise ValueError("No session provided for fetching")
    
    async def get(self, url: str, **kwargs) -> RequestResult:
        """Alias for fetch for compatibility."""
        return await self.fetch(url, **kwargs)


# Factory function for easy integration
def create_politeness_policy(**kwargs) -> PolitenessPolicy:
    """
    Create a PolitenessPolicy with sensible defaults.
    
    Returns:
        Configured PolitenessPolicy instance
    """
    defaults = {
        'default_capacity': 10.0,
        'default_refill_rate': 1.0,
        'respect_robots_txt': True,
        'cache_ttl': 3600,
        'user_agent': 'axiom (+https://github.com/user/axiom)'
    }
    defaults.update(kwargs)
    
    return PolitenessPolicy(**defaults)


# Example usage with existing axiom code
"""
# Example 1: Using with fetcher session
from axiom import Fetcher
from axiom.politeness.robots import create_politeness_policy, PolitenessMiddleware

async def example_with_fetcher():
    # Create politeness policy
    policy = create_politeness_policy(
        default_capacity=5.0,
        default_refill_rate=0.5,  # 2 seconds between requests
        user_agent='MyBot/1.0'
    )
    
    # Create fetcher session
    async with Fetcher() as session:
        # Wrap with politeness middleware
        async with PolitenessMiddleware(policy, session) as polite_session:
            # Fetch with automatic rate limiting
            result = await polite_session.fetch('https://example.com')
            print(f"Status: {result.status_code}")
            
            # Fetch another URL (will respect rate limits)
            result2 = await polite_session.fetch('https://example.com/page2')
            print(f"Status: {result2.status_code}")
    
    # Check statistics
    stats = await policy.get_global_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Rate limited: {stats['rate_limited']}")


# Example 2: Using with spider
from axiom import Spider
from axiom.politeness.robots import create_politeness_policy

class PoliteSpider(Spider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.politeness = create_politeness_policy(
            respect_robots_txt=True,
            default_refill_rate=0.2  # 5 seconds between requests
        )
    
    async def parse(self, response):
        # Check robots.txt before parsing links
        await self.politeness.check_robots_txt(str(response.url))
        
        # Process response
        for link in response.css('a::attr(href)').getall():
            # Wait for permission before following link
            await self.politeness.wait_for_permission(link)
            yield response.follow(link, self.parse)
"""


# Command-line interface for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_politeness():
        """Test the politeness policy."""
        policy = create_politeness_policy()
        
        # Test with a few URLs
        test_urls = [
            'https://example.com',
            'https://example.com/page1',
            'https://example.com/page2'
        ]
        
        for url in test_urls:
            try:
                print(f"Requesting: {url}")
                await policy.wait_for_permission(url)
                print(f"  ✓ Permission granted")
                
                # Simulate response
                await policy.record_response(
                    url=url,
                    status_code=200,
                    response_time=0.5
                )
                
            except RobotsTxtDisallowedError as e:
                print(f"  ✗ Blocked by robots.txt: {e}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Print statistics
        stats = await policy.get_global_stats()
        print("\nGlobal Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Run test
    asyncio.run(test_politeness())