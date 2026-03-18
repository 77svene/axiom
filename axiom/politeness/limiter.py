"""
Intelligent Rate Limiting & Politeness Module for axiom
Adaptive rate limiting based on target site response patterns,
robots.txt compliance with caching, automatic backoff on 429/503 errors,
and domain-specific throttling.
"""

import asyncio
import time
import re
import hashlib
import logging
import json
from typing import Dict, Optional, Tuple, List, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from urllib.parse import urlparse
from datetime import datetime, timedelta
import aiohttp
from pathlib import Path

from axiom.core.custom_types import SessionConfig
from axiom.core.storage import StorageManager
from axiom.core.utils._utils import get_logger

logger = get_logger(__name__)


class ThrottleState(Enum):
    """Current throttling state for a domain"""
    NORMAL = "normal"
    BACKOFF = "backoff"
    AGGRESSIVE = "aggressive"
    BLOCKED = "blocked"


@dataclass
class DomainStats:
    """Statistics for a specific domain"""
    domain: str
    requests_made: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_request_time: float = 0.0
    last_response_time: float = 0.0
    average_response_time: float = 0.0
    error_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    throttle_state: ThrottleState = ThrottleState.NORMAL
    backoff_until: float = 0.0
    crawl_delay: Optional[float] = None
    robots_rules: Dict[str, List[str]] = field(default_factory=dict)
    robots_loaded: bool = False
    robots_hash: Optional[str] = None
    last_robots_check: float = 0.0


class TokenBucket:
    """Token bucket algorithm implementation with dynamic adjustment"""
    
    def __init__(self, capacity: float = 10.0, refill_rate: float = 1.0):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens the bucket can hold
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def adjust_rate(self, new_rate: float):
        """Dynamically adjust the refill rate"""
        self.refill_rate = max(0.001, new_rate)  # Minimum 0.001 tokens/sec
    
    def get_wait_time(self, tokens: float = 1.0) -> float:
        """Calculate wait time needed to accumulate required tokens"""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        deficit = tokens - self.tokens
        return deficit / self.refill_rate


class RobotsParser:
    """Parser for robots.txt files with caching"""
    
    def __init__(self):
        self.rules: Dict[str, List[str]] = {}
        self.sitemaps: List[str] = []
        self.crawl_delay: Optional[float] = None
        self.request_rate: Optional[float] = None
        self.user_agents: Set[str] = set()
    
    def parse(self, content: str, user_agent: str = "*"):
        """Parse robots.txt content"""
        self.rules.clear()
        self.sitemaps.clear()
        current_agents = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'user-agent':
                    current_agents = [value.lower()]
                    self.user_agents.add(value.lower())
                elif key == 'disallow' and current_agents:
                    for agent in current_agents:
                        if agent not in self.rules:
                            self.rules[agent] = []
                        if value:
                            self.rules[agent].append(value)
                elif key == 'allow' and current_agents:
                    # For simplicity, we treat allow as higher priority than disallow
                    pass
                elif key == 'crawl-delay' and current_agents:
                    try:
                        delay = float(value)
                        if user_agent.lower() in [a.lower() for a in current_agents]:
                            self.crawl_delay = delay
                    except ValueError:
                        pass
                elif key == 'sitemap':
                    self.sitemaps.append(value)
                elif key == 'request-rate':
                    # Format: pages/seconds
                    try:
                        if '/' in value:
                            pages, seconds = value.split('/')
                            rate = float(pages) / float(seconds)
                            if user_agent.lower() in [a.lower() for a in current_agents]:
                                self.request_rate = rate
                    except (ValueError, ZeroDivisionError):
                        pass
    
    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if a URL can be fetched according to robots.txt rules"""
        parsed = urlparse(url)
        path = parsed.path or '/'
        
        # Check specific user agent first, then wildcard
        agents_to_check = [user_agent.lower(), '*']
        
        for agent in agents_to_check:
            if agent in self.rules:
                for pattern in self.rules[agent]:
                    # Convert robots.txt pattern to regex
                    regex_pattern = pattern.replace('*', '.*').replace('?', '.')
                    if regex_pattern.endswith('$'):
                        regex_pattern = regex_pattern[:-1] + '$'
                    else:
                        regex_pattern = regex_pattern + '.*'
                    
                    if re.match(regex_pattern, path):
                        return False
        
        return True


class AdaptiveRateLimiter:
    """Adaptive rate limiter with machine learning-inspired adjustments"""
    
    def __init__(self, storage_manager: Optional[StorageManager] = None):
        """
        Initialize adaptive rate limiter.
        
        Args:
            storage_manager: Optional storage manager for persistence
        """
        self.storage_manager = storage_manager or StorageManager()
        self.domain_stats: Dict[str, DomainStats] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.robots_parsers: Dict[str, RobotsParser] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.user_agent = "axiom/1.0 (+https://github.com/axiom/axiom)"
        
        # Configuration
        self.default_capacity = 10.0
        self.default_refill_rate = 1.0  # 1 request per second
        self.min_refill_rate = 0.1
        self.max_refill_rate = 10.0
        self.backoff_factor = 0.5
        self.recovery_factor = 1.1
        self.aggressive_threshold = 3  # 429/503 errors before aggressive throttling
        self.block_threshold = 10  # 429/503 errors before blocking
        
        # Response time tracking for ML-like adjustment
        self.response_time_window = 100  # Last N response times to consider
        self.target_response_time = 1.0  # Target response time in seconds
        
        # Load persisted data
        self._load_persisted_data()
    
    async def initialize(self):
        """Initialize async components"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self._persist_data()
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc.lower()
    
    def _get_domain_stats(self, domain: str) -> DomainStats:
        """Get or create domain statistics"""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = DomainStats(domain=domain)
        return self.domain_stats[domain]
    
    def _get_token_bucket(self, domain: str) -> TokenBucket:
        """Get or create token bucket for domain"""
        if domain not in self.token_buckets:
            stats = self._get_domain_stats(domain)
            capacity = self.default_capacity
            refill_rate = self.default_refill_rate
            
            # Apply crawl delay if available
            if stats.crawl_delay and stats.crawl_delay > 0:
                refill_rate = 1.0 / stats.crawl_delay
            
            self.token_buckets[domain] = TokenBucket(capacity, refill_rate)
        
        return self.token_buckets[domain]
    
    async def _fetch_robots_txt(self, domain: str) -> Optional[str]:
        """Fetch robots.txt for a domain"""
        if not self.session:
            await self.initialize()
        
        robots_url = f"https://{domain}/robots.txt"
        try:
            async with self.session.get(
                robots_url,
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": self.user_agent}
            ) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logger.debug(f"Failed to fetch robots.txt for {domain}: {e}")
        
        return None
    
    async def _load_robots_rules(self, domain: str, user_agent: str = "*"):
        """Load and parse robots.txt for a domain"""
        stats = self._get_domain_stats(domain)
        
        # Check cache validity (refresh every 24 hours)
        current_time = time.time()
        if (stats.robots_loaded and 
            stats.last_robots_check and 
            current_time - stats.last_robots_check < 86400):
            return
        
        content = await self._fetch_robots_txt(domain)
        if content:
            # Check if content has changed
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash != stats.robots_hash:
                parser = RobotsParser()
                parser.parse(content, user_agent)
                self.robots_parsers[domain] = parser
                stats.robots_hash = content_hash
                stats.crawl_delay = parser.crawl_delay
                stats.robots_rules = parser.rules
                
                # Update token bucket if crawl delay is specified
                if parser.crawl_delay and parser.crawl_delay > 0:
                    bucket = self._get_token_bucket(domain)
                    bucket.adjust_rate(1.0 / parser.crawl_delay)
                    logger.info(f"Applied crawl-delay of {parser.crawl_delay}s for {domain}")
        
        stats.robots_loaded = True
        stats.last_robots_check = current_time
    
    async def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """
        Check if a URL can be fetched according to robots.txt.
        
        Args:
            url: URL to check
            user_agent: User agent string
            
        Returns:
            True if fetch is allowed, False otherwise
        """
        domain = self._get_domain(url)
        stats = self._get_domain_stats(domain)
        
        # Check if domain is blocked
        if stats.throttle_state == ThrottleState.BLOCKED:
            logger.warning(f"Domain {domain} is blocked due to excessive errors")
            return False
        
        # Load robots.txt if not already loaded
        if not stats.robots_loaded:
            await self._load_robots_rules(domain, user_agent)
        
        # Check robots.txt rules
        if domain in self.robots_parsers:
            parser = self.robots_parsers[domain]
            if not parser.can_fetch(url, user_agent):
                logger.debug(f"robots.txt disallows fetching {url}")
                return False
        
        return True
    
    async def acquire(self, url: str, user_agent: str = "*") -> Tuple[bool, float]:
        """
        Attempt to acquire permission to make a request.
        
        Args:
            url: URL to request
            user_agent: User agent string
            
        Returns:
            Tuple of (success, wait_time)
            - success: True if request is allowed
            - wait_time: Time to wait before making request (0 if immediate)
        """
        domain = self._get_domain(url)
        stats = self._get_domain_stats(domain)
        bucket = self._get_token_bucket(domain)
        
        # Check robots.txt
        if not await self.can_fetch(url, user_agent):
            return False, 0.0
        
        # Check if in backoff period
        current_time = time.time()
        if stats.backoff_until > current_time:
            wait_time = stats.backoff_until - current_time
            logger.debug(f"Domain {domain} in backoff, waiting {wait_time:.2f}s")
            return False, wait_time
        
        # Try to consume token
        if await bucket.consume(1.0):
            stats.requests_made += 1
            stats.last_request_time = current_time
            return True, 0.0
        else:
            wait_time = bucket.get_wait_time(1.0)
            return False, wait_time
    
    def record_response(self, url: str, status_code: int, response_time: float):
        """
        Record response for adaptive rate limiting.
        
        Args:
            url: Requested URL
            status_code: HTTP status code
            response_time: Response time in seconds
        """
        domain = self._get_domain(url)
        stats = self._get_domain_stats(domain)
        bucket = self._get_token_bucket(domain)
        
        # Update statistics
        stats.last_response_time = response_time
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if stats.average_response_time == 0:
            stats.average_response_time = response_time
        else:
            stats.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * stats.average_response_time
            )
        
        # Handle error responses
        if status_code == 429 or status_code == 503:
            stats.error_counts[status_code] += 1
            stats.failed_requests += 1
            
            # Calculate total errors for this domain
            total_errors = sum(stats.error_counts.values())
            
            # Apply progressive throttling
            if total_errors >= self.block_threshold:
                stats.throttle_state = ThrottleState.BLOCKED
                logger.warning(f"Domain {domain} blocked after {total_errors} errors")
            elif total_errors >= self.aggressive_threshold:
                stats.throttle_state = ThrottleState.AGGRESSIVE
                # Apply exponential backoff
                backoff_seconds = min(300, 2 ** (total_errors - self.aggressive_threshold))
                stats.backoff_until = time.time() + backoff_seconds
                logger.warning(f"Domain {domain} in aggressive throttling, backoff for {backoff_seconds}s")
            else:
                stats.throttle_state = ThrottleState.BACKOFF
                # Apply backoff with jitter
                backoff_seconds = min(60, 2 ** total_errors)
                stats.backoff_until = time.time() + backoff_seconds
                logger.info(f"Domain {domain} backing off for {backoff_seconds}s")
            
            # Reduce request rate
            new_rate = bucket.refill_rate * self.backoff_factor
            bucket.adjust_rate(max(self.min_refill_rate, new_rate))
            
        elif 200 <= status_code < 300:
            stats.successful_requests += 1
            
            # Gradually recover from throttling states
            if stats.throttle_state in [ThrottleState.BACKOFF, ThrottleState.AGGRESSIVE]:
                # Reset error counts on success
                stats.error_counts.clear()
                stats.throttle_state = ThrottleState.NORMAL
                stats.backoff_until = 0.0
            
            # Adaptive rate adjustment based on response time
            if stats.average_response_time > self.target_response_time * 1.5:
                # Slow response, reduce rate
                new_rate = bucket.refill_rate * 0.9
                bucket.adjust_rate(max(self.min_refill_rate, new_rate))
            elif stats.average_response_time < self.target_response_time * 0.5:
                # Fast response, increase rate
                new_rate = bucket.refill_rate * self.recovery_factor
                bucket.adjust_rate(min(self.max_refill_rate, new_rate))
    
    def get_domain_stats(self, domain: str) -> Optional[DomainStats]:
        """Get statistics for a domain"""
        return self.domain_stats.get(domain)
    
    def get_all_stats(self) -> Dict[str, DomainStats]:
        """Get statistics for all domains"""
        return self.domain_stats.copy()
    
    def _persist_data(self):
        """Persist rate limiting data to storage"""
        if not self.storage_manager:
            return
        
        try:
            data = {
                "domain_stats": {},
                "token_buckets": {}
            }
            
            for domain, stats in self.domain_stats.items():
                data["domain_stats"][domain] = {
                    "requests_made": stats.requests_made,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "average_response_time": stats.average_response_time,
                    "error_counts": dict(stats.error_counts),
                    "throttle_state": stats.throttle_state.value,
                    "crawl_delay": stats.crawl_delay,
                    "robots_hash": stats.robots_hash,
                    "last_robots_check": stats.last_robots_check
                }
            
            for domain, bucket in self.token_buckets.items():
                data["token_buckets"][domain] = {
                    "capacity": bucket.capacity,
                    "refill_rate": bucket.refill_rate,
                    "tokens": bucket.tokens
                }
            
            self.storage_manager.set("rate_limiter_data", data)
            logger.debug("Persisted rate limiter data")
        except Exception as e:
            logger.error(f"Failed to persist rate limiter data: {e}")
    
    def _load_persisted_data(self):
        """Load persisted rate limiting data from storage"""
        if not self.storage_manager:
            return
        
        try:
            data = self.storage_manager.get("rate_limiter_data")
            if not data:
                return
            
            # Restore domain stats
            for domain, stats_data in data.get("domain_stats", {}).items():
                stats = DomainStats(domain=domain)
                stats.requests_made = stats_data.get("requests_made", 0)
                stats.successful_requests = stats_data.get("successful_requests", 0)
                stats.failed_requests = stats_data.get("failed_requests", 0)
                stats.average_response_time = stats_data.get("average_response_time", 0.0)
                stats.error_counts = defaultdict(int, stats_data.get("error_counts", {}))
                stats.throttle_state = ThrottleState(stats_data.get("throttle_state", "normal"))
                stats.crawl_delay = stats_data.get("crawl_delay")
                stats.robots_hash = stats_data.get("robots_hash")
                stats.last_robots_check = stats_data.get("last_robots_check", 0.0)
                self.domain_stats[domain] = stats
            
            # Restore token buckets
            for domain, bucket_data in data.get("token_buckets", {}).items():
                bucket = TokenBucket(
                    capacity=bucket_data.get("capacity", self.default_capacity),
                    refill_rate=bucket_data.get("refill_rate", self.default_refill_rate)
                )
                bucket.tokens = bucket_data.get("tokens", bucket.capacity)
                self.token_buckets[domain] = bucket
            
            logger.info(f"Loaded rate limiter data for {len(self.domain_stats)} domains")
        except Exception as e:
            logger.error(f"Failed to load rate limiter data: {e}")


class PolitenessManager:
    """
    High-level politeness manager that coordinates rate limiting,
    robots.txt compliance, and adaptive throttling.
    """
    
    def __init__(
        self,
        storage_manager: Optional[StorageManager] = None,
        user_agent: str = "axiom/1.0 (+https://github.com/axiom/axiom)",
        respect_robots_txt: bool = True,
        default_delay: float = 1.0,
        max_concurrent_requests: int = 5
    ):
        """
        Initialize politeness manager.
        
        Args:
            storage_manager: Storage manager for persistence
            user_agent: User agent string to use
            respect_robots_txt: Whether to respect robots.txt
            default_delay: Default delay between requests in seconds
            max_concurrent_requests: Maximum concurrent requests per domain
        """
        self.storage_manager = storage_manager or StorageManager()
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt
        self.default_delay = default_delay
        self.max_concurrent_requests = max_concurrent_requests
        
        self.rate_limiter = AdaptiveRateLimiter(storage_manager)
        self.rate_limiter.user_agent = user_agent
        
        # Track active requests per domain
        self.active_requests: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        
        # Domain-specific settings
        self.domain_settings: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize async components"""
        await self.rate_limiter.initialize()
    
    async def close(self):
        """Clean up resources"""
        await self.rate_limiter.close()
    
    async def prepare_request(self, url: str) -> Dict[str, Any]:
        """
        Prepare a request with politeness checks.
        
        Args:
            url: URL to request
            
        Returns:
            Dictionary with request preparation info:
            - allowed: Whether request is allowed
            - wait_time: Time to wait before request
            - headers: Headers to include
            - error: Error message if not allowed
        """
        domain = self._get_domain(url)
        
        # Check robots.txt if enabled
        if self.respect_robots_txt:
            allowed = await self.rate_limiter.can_fetch(url, self.user_agent)
            if not allowed:
                return {
                    "allowed": False,
                    "wait_time": 0,
                    "headers": {},
                    "error": f"robots.txt disallows fetching {url}"
                }
        
        # Check concurrent request limit
        async with self._lock:
            if self.active_requests[domain] >= self.max_concurrent_requests:
                return {
                    "allowed": False,
                    "wait_time": 1.0,  # Retry after 1 second
                    "headers": {},
                    "error": f"Too many concurrent requests for {domain}"
                }
        
        # Acquire rate limit permission
        allowed, wait_time = await self.rate_limiter.acquire(url, self.user_agent)
        
        if allowed:
            async with self._lock:
                self.active_requests[domain] += 1
        
        # Prepare headers
        headers = {"User-Agent": self.user_agent}
        
        # Add domain-specific headers
        if domain in self.domain_settings:
            headers.update(self.domain_settings[domain].get("headers", {}))
        
        return {
            "allowed": allowed,
            "wait_time": wait_time,
            "headers": headers,
            "error": None
        }
    
    def record_response(self, url: str, status_code: int, response_time: float):
        """
        Record response for politeness adjustments.
        
        Args:
            url: Requested URL
            status_code: HTTP status code
            response_time: Response time in seconds
        """
        domain = self._get_domain(url)
        
        # Update rate limiter
        self.rate_limiter.record_response(url, status_code, response_time)
        
        # Decrement active requests
        async with self._lock:
            if self.active_requests[domain] > 0:
                self.active_requests[domain] -= 1
    
    def set_domain_setting(self, domain: str, setting: str, value: Any):
        """Set a domain-specific setting"""
        if domain not in self.domain_settings:
            self.domain_settings[domain] = {}
        self.domain_settings[domain][setting] = value
    
    def get_domain_setting(self, domain: str, setting: str, default: Any = None) -> Any:
        """Get a domain-specific setting"""
        return self.domain_settings.get(domain, {}).get(setting, default)
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc.lower()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get politeness manager statistics"""
        stats = {
            "total_domains": len(self.rate_limiter.domain_stats),
            "active_requests": dict(self.active_requests),
            "domain_stats": {}
        }
        
        for domain, domain_stats in self.rate_limiter.get_all_stats().items():
            stats["domain_stats"][domain] = {
                "requests_made": domain_stats.requests_made,
                "successful_requests": domain_stats.successful_requests,
                "failed_requests": domain_stats.failed_requests,
                "throttle_state": domain_stats.throttle_state.value,
                "average_response_time": domain_stats.average_response_time,
                "crawl_delay": domain_stats.crawl_delay
            }
        
        return stats


# Factory function for easy integration
def create_politeness_manager(
    storage_path: Optional[str] = None,
    **kwargs
) -> PolitenessManager:
    """
    Create a politeness manager with optional storage.
    
    Args:
        storage_path: Path for persistent storage
        **kwargs: Additional arguments for PolitenessManager
        
    Returns:
        Configured PolitenessManager instance
    """
    storage_manager = StorageManager(storage_path) if storage_path else StorageManager()
    return PolitenessManager(storage_manager=storage_manager, **kwargs)


# Integration with existing axiom sessions
class PoliteSessionMixin:
    """Mixin to add politeness to axiom sessions"""
    
    def __init__(self, *args, politeness_manager: Optional[PolitenessManager] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.politeness_manager = politeness_manager or create_politeness_manager()
        self._politeness_initialized = False
    
    async def _ensure_politeness(self):
        """Ensure politeness manager is initialized"""
        if not self._politeness_initialized:
            await self.politeness_manager.initialize()
            self._politeness_initialized = True
    
    async def polite_request(self, url: str, **kwargs):
        """
        Make a polite request with rate limiting and robots.txt compliance.
        
        Args:
            url: URL to request
            **kwargs: Additional request arguments
            
        Returns:
            Response object or None if request not allowed
        """
        await self._ensure_politeness()
        
        # Prepare request with politeness checks
        prep = await self.politeness_manager.prepare_request(url)
        
        if not prep["allowed"]:
            if prep["wait_time"] > 0:
                await asyncio.sleep(prep["wait_time"])
                # Retry after waiting
                return await self.polite_request(url, **kwargs)
            else:
                logger.warning(f"Request not allowed: {prep['error']}")
                return None
        
        # Merge headers
        headers = kwargs.get("headers", {})
        headers.update(prep["headers"])
        kwargs["headers"] = headers
        
        # Make request and record response
        start_time = time.time()
        try:
            response = await self.request(url, **kwargs)
            response_time = time.time() - start_time
            
            # Record response for adaptive rate limiting
            if hasattr(response, "status"):
                self.politeness_manager.record_response(
                    url, response.status, response_time
                )
            
            return response
        except Exception as e:
            response_time = time.time() - start_time
            # Record error (assuming 500 for exceptions)
            self.politeness_manager.record_response(url, 500, response_time)
            raise


# Example usage with existing axiom components
async def example_integration():
    """Example showing integration with existing axiom sessions"""
    from axiom.core.ai import AdaptiveSession
    
    # Create politeness manager
    politeness_manager = create_politeness_manager(
        user_agent="MyBot/1.0",
        respect_robots_txt=True,
        default_delay=1.5
    )
    
    # Create session with politeness
    class PoliteAdaptiveSession(PoliteSessionMixin, AdaptiveSession):
        pass
    
    session = PoliteAdaptiveSession(politeness_manager=politeness_manager)
    
    try:
        # Make polite requests
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3"
        ]
        
        for url in urls:
            response = await session.polite_request(url)
            if response:
                print(f"Successfully fetched {url}: {response.status}")
            else:
                print(f"Failed to fetch {url}")
        
        # Get statistics
        stats = politeness_manager.get_stats()
        print(f"Politeness stats: {json.dumps(stats, indent=2)}")
    
    finally:
        await politeness_manager.close()
        await session.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_integration())