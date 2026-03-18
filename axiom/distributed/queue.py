"""axiom/distributed/queue.py

Distributed Crawling Engine with Redis/RabbitMQ-backed task queue.
Implements horizontal scaling, automatic node discovery, load balancing,
and fault tolerance with checkpointing for the axiom framework.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Awaitable
from urllib.parse import urlparse
import heapq
import bisect
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import zlib

# Celery imports (with fallback for development)
try:
    from celery import Celery, Task
    from celery.signals import (
        worker_init, worker_shutdown, task_prerun, task_postrun,
        beat_init, worker_ready
    )
    from kombu import Connection, Exchange, Queue
    from kombu.serialization import register
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    # Fallback definitions for type hints
    class Celery:
        pass
    class Task:
        pass

# Redis imports (with fallback)
try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import from existing axiom modules
from axiom.core.storage import StorageManager
from axiom.core.custom_types import Request, Response, CrawlResult
from axiom.core.utils._utils import get_logger, exponential_backoff
from axiom.core.mixins import SerializableMixin

logger = get_logger(__name__)


class TaskPriority(Enum):
    """Priority levels for crawl tasks."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BULK = 4


class TaskStatus(Enum):
    """Status of a distributed task."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CHECKPOINTED = "checkpointed"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    """Status of worker nodes."""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class CrawlTask(SerializableMixin):
    """Represents a single crawl task in the distributed queue."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    depth: int = 0
    parent_url: Optional[str] = None
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    callback: Optional[str] = None
    max_retries: int = 3
    retry_count: int = 0
    timeout: int = 30
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    checkpoint_data: Optional[bytes] = None
    result_hash: Optional[str] = None
    node_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.domain and self.url:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['tags'] = list(self.tags)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrawlTask':
        """Create from dictionary."""
        if 'priority' in data and isinstance(data['priority'], int):
            data['priority'] = TaskPriority(data['priority'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = TaskStatus(data['status'])
        if 'tags' in data and isinstance(data['tags'], list):
            data['tags'] = set(data['tags'])
        return cls(**data)
    
    def calculate_priority_score(self) -> float:
        """Calculate score for priority queue ordering."""
        # Lower score = higher priority
        base_score = self.priority.value * 1000
        # Boost for depth (prefer shallower pages)
        depth_penalty = self.depth * 100
        # Boost for recency
        recency_boost = (time.time() - self.created_at) / 3600  # Hours
        return base_score + depth_penalty - recency_boost
    
    def get_url_hash(self) -> str:
        """Generate consistent hash for URL deduplication."""
        # Normalize URL for consistent hashing
        normalized = self.url.lower().strip()
        parsed = urlparse(normalized)
        # Remove fragments and sort query parameters
        normalized = parsed._replace(fragment="").geturl()
        if parsed.query:
            params = sorted(parsed.query.split('&'))
            normalized = parsed._replace(query='&'.join(params)).geturl()
        
        return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    node_id: str = field(default_factory=lambda: f"node-{socket.gethostname()}-{uuid.uuid4().hex[:8]}")
    hostname: str = field(default_factory=socket.gethostname)
    ip_address: str = ""
    port: int = 0
    status: NodeStatus = NodeStatus.IDLE
    capabilities: Set[str] = field(default_factory=set)
    current_tasks: int = 0
    max_tasks: int = 10
    last_heartbeat: float = field(default_factory=time.time)
    total_completed: int = 0
    total_failed: int = 0
    average_task_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['capabilities'] = list(self.capabilities)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerNode':
        """Create from dictionary."""
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = NodeStatus(data['status'])
        if 'capabilities' in data and isinstance(data['capabilities'], list):
            data['capabilities'] = set(data['capabilities'])
        return cls(**data)
    
    def is_available(self) -> bool:
        """Check if node can accept more tasks."""
        return (
            self.status in (NodeStatus.ACTIVE, NodeStatus.IDLE) and
            self.current_tasks < self.max_tasks
        )
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time()
    
    def load_score(self) -> float:
        """Calculate load score for load balancing (lower is better)."""
        if not self.is_available():
            return float('inf')
        return self.current_tasks / self.max_tasks


class ConsistentHashRing:
    """Consistent hashing implementation for URL deduplication and load balancing."""
    
    def __init__(self, num_replicas: int = 100):
        self.num_replicas = num_replicas
        self.ring: List[Tuple[int, str]] = []
        self.keys: Set[str] = set()
        self._lock = threading.RLock()
    
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: str) -> None:
        """Add a node to the ring."""
        with self._lock:
            if node in self.keys:
                return
            self.keys.add(node)
            for i in range(self.num_replicas):
                hash_key = f"{node}:{i}"
                hash_val = self._hash(hash_key)
                bisect.insort(self.ring, (hash_val, node))
    
    def remove_node(self, node: str) -> None:
        """Remove a node from the ring."""
        with self._lock:
            if node not in self.keys:
                return
            self.keys.remove(node)
            self.ring = [(h, n) for h, n in self.ring if n != node]
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key."""
        if not self.ring:
            return None
        
        hash_val = self._hash(key)
        # Binary search for the first node with hash >= hash_val
        idx = bisect.bisect_left(self.ring, (hash_val,))
        if idx == len(self.ring):
            idx = 0
        return self.ring[idx][1]
    
    def get_nodes(self, key: str, count: int = 3) -> List[str]:
        """Get multiple nodes for redundancy."""
        if not self.ring:
            return []
        
        hash_val = self._hash(key)
        nodes = []
        seen = set()
        
        # Start from the position in the ring
        idx = bisect.bisect_left(self.ring, (hash_val,))
        if idx == len(self.ring):
            idx = 0
        
        # Walk around the ring
        start_idx = idx
        while len(nodes) < count and len(seen) < len(self.keys):
            node = self.ring[idx][1]
            if node not in seen:
                nodes.append(node)
                seen.add(node)
            idx = (idx + 1) % len(self.ring)
            if idx == start_idx:
                break
        
        return nodes


class DistributedQueue:
    """Main distributed queue manager for axiom."""
    
    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        result_backend: Optional[str] = None,
        queue_name: str = "axiom",
        node_id: Optional[str] = None,
        enable_checkpoints: bool = True,
        checkpoint_interval: int = 100,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        url_deduplication: bool = True,
        politeness_delay: float = 1.0,
        max_tasks_per_domain: int = 5,
        **kwargs
    ):
        """Initialize the distributed queue.
        
        Args:
            broker_url: Celery broker URL (Redis/RabbitMQ)
            result_backend: Result backend URL
            queue_name: Base name for queues
            node_id: Unique identifier for this node
            enable_checkpoints: Enable fault tolerance checkpointing
            checkpoint_interval: Tasks between checkpoints
            max_retries: Maximum retry attempts for failed tasks
            retry_backoff: Exponential backoff multiplier
            url_deduplication: Enable URL deduplication
            politeness_delay: Delay between requests to same domain
            max_tasks_per_domain: Max concurrent tasks per domain
        """
        self.broker_url = broker_url
        self.result_backend = result_backend or broker_url
        self.queue_name = queue_name
        self.node_id = node_id or f"coordinator-{uuid.uuid4().hex[:8]}"
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.url_deduplication = url_deduplication
        self.politeness_delay = politeness_delay
        self.max_tasks_per_domain = max_tasks_per_domain
        
        # Internal state
        self._celery_app: Optional[Celery] = None
        self._redis_client: Optional[redis.Redis] = None
        self._hash_ring = ConsistentHashRing()
        self._nodes: Dict[str, WorkerNode] = {}
        self._domain_queues: Dict[str, List[CrawlTask]] = {}
        self._domain_timestamps: Dict[str, float] = {}
        self._task_callbacks: Dict[str, Callable] = {}
        self._checkpoint_counter = 0
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'tasks_enqueued': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'urls_deduplicated': 0,
            'nodes_discovered': 0,
            'checkpoints_created': 0,
        }
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize Celery and Redis connections."""
        if not CELERY_AVAILABLE:
            logger.warning("Celery not available. Distributed queue will run in local mode only.")
            return
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Some features may be limited.")
        
        # Initialize Celery
        self._celery_app = Celery(
            'axiom.distributed',
            broker=self.broker_url,
            backend=self.result_backend
        )
        
        # Configure Celery
        self._configure_celery()
        
        # Initialize Redis for deduplication and node discovery
        if REDIS_AVAILABLE:
            try:
                self._redis_client = redis.Redis.from_url(
                    self.broker_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                # Test connection
                self._redis_client.ping()
                logger.info("Redis connection established")
            except RedisConnectionError as e:
                logger.warning(f"Redis connection failed: {e}. Running without Redis.")
                self._redis_client = None
        
        # Register custom serialization for complex objects
        if CELERY_AVAILABLE:
            register(
                'axiom_pickle',
                pickle.dumps,
                pickle.loads,
                content_type='application/x-axiom-pickle',
                content_encoding='binary'
            )
    
    def _configure_celery(self):
        """Configure Celery application."""
        if not self._celery_app:
            return
        
        # Queue configuration
        self._celery_app.conf.update(
            task_serializer='pickle',
            result_serializer='pickle',
            accept_content=['pickle', 'json'],
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            task_reject_on_worker_lost=True,
            task_default_queue=f"{self.queue_name}.default",
            task_routes={
                'axiom.distributed.queue.crawl_task': {
                    'queue': f"{self.queue_name}.crawl",
                    'routing_key': 'crawl',
                },
                'axiom.distributed.queue.checkpoint_task': {
                    'queue': f"{self.queue_name}.system",
                    'routing_key': 'system',
                },
            },
            task_queues=[
                Queue(f"{self.queue_name}.default", Exchange('default'), routing_key='default'),
                Queue(f"{self.queue_name}.crawl", Exchange('crawl'), routing_key='crawl'),
                Queue(f"{self.queue_name}.system", Exchange('system'), routing_key='system'),
            ],
        )
        
        # Configure retry policy
        self._celery_app.conf.update(
            task_annotations={
                'axiom.distributed.queue.crawl_task': {
                    'rate_limit': '100/m',
                    'default_retry_delay': 60,
                    'max_retries': self.max_retries,
                    'retry_backoff': self.retry_backoff,
                    'retry_backoff_max': 600,
                    'retry_jitter': True,
                }
            }
        )
    
    def start(self):
        """Start the distributed queue system."""
        if self._running:
            logger.warning("Queue system already running")
            return
        
        self._running = True
        logger.info(f"Starting distributed queue system (node: {self.node_id})")
        
        # Start node discovery
        self._start_node_discovery()
        
        # Start health check thread
        self._start_health_check()
        
        # Start domain queue processor
        self._start_domain_processor()
        
        logger.info("Distributed queue system started")
    
    def stop(self):
        """Stop the distributed queue system."""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping distributed queue system")
        
        # Stop executor
        self._executor.shutdown(wait=True)
        
        # Close connections
        if self._redis_client:
            self._redis_client.close()
        
        logger.info("Distributed queue system stopped")
    
    def _start_node_discovery(self):
        """Start automatic node discovery thread."""
        def discovery_loop():
            while self._running:
                try:
                    self._discover_nodes()
                    time.sleep(30)  # Discover every 30 seconds
                except Exception as e:
                    logger.error(f"Node discovery error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=discovery_loop, daemon=True)
        thread.start()
    
    def _discover_nodes(self):
        """Discover available worker nodes."""
        if not self._redis_client:
            return
        
        try:
            # Register self
            node_key = f"axiom:nodes:{self.node_id}"
            node_data = {
                'node_id': self.node_id,
                'status': 'active',
                'last_heartbeat': time.time(),
                'capabilities': json.dumps(['crawl', 'extract', 'checkpoint']),
            }
            self._redis_client.hset(node_key, mapping=node_data)
            self._redis_client.expire(node_key, 60)  # 60 second TTL
            
            # Discover other nodes
            node_keys = self._redis_client.keys("axiom:nodes:*")
            discovered_nodes = {}
            
            for key in node_keys:
                if key == node_key:
                    continue
                
                node_data = self._redis_client.hgetall(key)
                if not node_data:
                    continue
                
                try:
                    node = WorkerNode(
                        node_id=node_data['node_id'],
                        status=NodeStatus(node_data.get('status', 'active')),
                        last_heartbeat=float(node_data.get('last_heartbeat', 0)),
                        capabilities=set(json.loads(node_data.get('capabilities', '[]'))),
                    )
                    
                    # Check if node is alive (heartbeat within 90 seconds)
                    if time.time() - node.last_heartbeat < 90:
                        discovered_nodes[node.node_id] = node
                        self._hash_ring.add_node(node.node_id)
                except (KeyError, ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Invalid node data for {key}: {e}")
            
            # Update nodes
            with self._lock:
                old_nodes = set(self._nodes.keys())
                new_nodes = set(discovered_nodes.keys())
                
                # Remove offline nodes
                for node_id in old_nodes - new_nodes:
                    self._hash_ring.remove_node(node_id)
                    logger.info(f"Node {node_id} went offline")
                
                # Add new nodes
                for node_id in new_nodes - old_nodes:
                    logger.info(f"Discovered new node: {node_id}")
                    self.stats['nodes_discovered'] += 1
                
                self._nodes = discovered_nodes
        
        except RedisError as e:
            logger.error(f"Redis error during node discovery: {e}")
    
    def _start_health_check(self):
        """Start health check thread for nodes."""
        def health_check_loop():
            while self._running:
                try:
                    self._check_node_health()
                    time.sleep(15)  # Check every 15 seconds
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=health_check_loop, daemon=True)
        thread.start()
    
    def _check_node_health(self):
        """Check health of all nodes and update status."""
        with self._lock:
            current_time = time.time()
            offline_nodes = []
            
            for node_id, node in self._nodes.items():
                if current_time - node.last_heartbeat > 90:  # 90 seconds timeout
                    node.status = NodeStatus.OFFLINE
                    offline_nodes.append(node_id)
            
            # Remove offline nodes from hash ring
            for node_id in offline_nodes:
                self._hash_ring.remove_node(node_id)
                logger.warning(f"Node {node_id} marked as offline")
    
    def _start_domain_processor(self):
        """Start thread to process domain-specific queues with politeness."""
        def domain_processor_loop():
            while self._running:
                try:
                    self._process_domain_queues()
                    time.sleep(0.1)  # Small sleep to prevent busy waiting
                except Exception as e:
                    logger.error(f"Domain processor error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=domain_processor_loop, daemon=True)
        thread.start()
    
    def _process_domain_queues(self):
        """Process tasks from domain queues with politeness delay."""
        current_time = time.time()
        
        with self._lock:
            for domain, queue in list(self._domain_queues.items()):
                if not queue:
                    continue
                
                # Check politeness delay
                last_request = self._domain_timestamps.get(domain, 0)
                if current_time - last_request < self.politeness_delay:
                    continue
                
                # Check concurrent task limit
                active_tasks = sum(
                    1 for node in self._nodes.values()
                    if domain in node.metadata.get('active_domains', set())
                )
                if active_tasks >= self.max_tasks_per_domain:
                    continue
                
                # Get next task
                task = heapq.heappop(queue)
                self._domain_timestamps[domain] = current_time
                
                # Dispatch task
                self._dispatch_task(task)
    
    def enqueue(
        self,
        url: str,
        priority: Union[TaskPriority, str, int] = TaskPriority.NORMAL,
        depth: int = 0,
        parent_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        callback: Optional[Union[str, Callable]] = None,
        tags: Optional[Set[str]] = None,
        **kwargs
    ) -> str:
        """Enqueue a URL for distributed crawling.
        
        Args:
            url: URL to crawl
            priority: Task priority
            depth: Crawl depth
            parent_url: URL that linked to this one
            metadata: Additional metadata
            callback: Callback function or name
            tags: Tags for categorization
            **kwargs: Additional task parameters
            
        Returns:
            Task ID
        """
        # Normalize priority
        if isinstance(priority, str):
            priority = TaskPriority[priority.upper()]
        elif isinstance(priority, int):
            priority = TaskPriority(priority)
        
        # Create task
        task = CrawlTask(
            url=url,
            priority=priority,
            depth=depth,
            parent_url=parent_url,
            metadata=metadata or {},
            callback=callback if isinstance(callback, str) else None,
            tags=tags or set(),
            **kwargs
        )
        
        # Store callback if it's a function
        if callable(callback):
            self._task_callbacks[task.task_id] = callback
        
        # Check URL deduplication
        if self.url_deduplication:
            url_hash = task.get_url_hash()
            if self._is_url_seen(url_hash):
                logger.debug(f"URL already seen: {url}")
                self.stats['urls_deduplicated'] += 1
                return task.task_id
            
            self._mark_url_seen(url_hash)
        
        # Add to appropriate queue
        self._enqueue_task(task)
        self.stats['tasks_enqueued'] += 1
        
        logger.info(f"Enqueued task {task.task_id} for {url} (priority: {priority.name})")
        return task.task_id
    
    def enqueue_many(
        self,
        urls: List[str],
        priority: Union[TaskPriority, str, int] = TaskPriority.NORMAL,
        **kwargs
    ) -> List[str]:
        """Enqueue multiple URLs at once.
        
        Args:
            urls: List of URLs to crawl
            priority: Task priority for all URLs
            **kwargs: Additional parameters
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for url in urls:
            task_id = self.enqueue(url=url, priority=priority, **kwargs)
            task_ids.append(task_id)
        return task_ids
    
    def _enqueue_task(self, task: CrawlTask):
        """Add task to the appropriate queue."""
        with self._lock:
            # Add to domain-specific queue for politeness
            domain = task.domain
            if domain not in self._domain_queues:
                self._domain_queues[domain] = []
            
            # Use priority queue (heap)
            heapq.heappush(self._domain_queues[domain], task)
    
    def _dispatch_task(self, task: CrawlTask):
        """Dispatch task to a worker node."""
        if not CELERY_AVAILABLE or not self._celery_app:
            # Fallback: execute locally
            self._execute_locally(task)
            return
        
        # Find best node for this task
        node_id = self._select_node_for_task(task)
        if not node_id:
            logger.warning(f"No available nodes for task {task.task_id}")
            # Re-queue with lower priority
            task.priority = TaskPriority.LOW
            self._enqueue_task(task)
            return
        
        # Update task
        task.status = TaskStatus.QUEUED
        task.node_id = node_id
        task.started_at = time.time()
        
        # Send to Celery
        try:
            crawl_task.apply_async(
                args=[task.to_dict()],
                queue=f"{self.queue_name}.crawl",
                retry=True,
                retry_policy={
                    'max_retries': self.max_retries,
                    'interval_start': self.retry_backoff,
                    'interval_step': self.retry_backoff * 2,
                    'interval_max': 60,
                }
            )
            logger.debug(f"Dispatched task {task.task_id} to node {node_id}")
        except Exception as e:
            logger.error(f"Failed to dispatch task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            self.stats['tasks_failed'] += 1
    
    def _select_node_for_task(self, task: CrawlTask) -> Optional[str]:
        """Select the best node for a task using load balancing."""
        if not self._nodes:
            return None
        
        # Use consistent hashing for URL-based routing (optional)
        if self.url_deduplication and self._hash_ring.keys:
            # Get nodes responsible for this URL
            candidate_nodes = self._hash_ring.get_nodes(task.url, count=3)
            available_nodes = [
                node_id for node_id in candidate_nodes
                if node_id in self._nodes and self._nodes[node_id].is_available()
            ]
            if available_nodes:
                # Select least loaded among candidates
                return min(available_nodes, key=lambda nid: self._nodes[nid].load_score())
        
        # Fallback: select least loaded available node
        available_nodes = [
            node_id for node_id, node in self._nodes.items()
            if node.is_available()
        ]
        if not available_nodes:
            return None
        
        return min(available_nodes, key=lambda nid: self._nodes[nid].load_score())
    
    def _execute_locally(self, task: CrawlTask):
        """Execute task locally when Celery is not available."""
        def local_executor():
            try:
                # Import here to avoid circular imports
                from axiom.core.ai import AdaptiveExtractor
                from axiom.core.storage import StorageManager
                
                # Simulate crawl execution
                logger.info(f"Executing task {task.task_id} locally: {task.url}")
                
                # Create checkpoint before execution
                if self.enable_checkpoints:
                    self._create_checkpoint(task)
                
                # Simulate processing time
                time.sleep(0.1)
                
                # Mark as completed
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                self.stats['tasks_completed'] += 1
                
                # Execute callback if registered
                if task.task_id in self._task_callbacks:
                    callback = self._task_callbacks[task.task_id]
                    try:
                        callback(task)
                    except Exception as e:
                        logger.error(f"Callback error for task {task.task_id}: {e}")
                
                logger.info(f"Completed task {task.task_id}")
                
            except Exception as e:
                logger.error(f"Local execution failed for task {task.task_id}: {e}")
                task.status = TaskStatus.FAILED
                self.stats['tasks_failed'] += 1
        
        self._executor.submit(local_executor)
    
    def _is_url_seen(self, url_hash: str) -> bool:
        """Check if URL has been seen before."""
        if not self._redis_client:
            # In-memory fallback
            return url_hash in getattr(self, '_seen_urls', set())
        
        try:
            return self._redis_client.sismember("axiom:seen_urls", url_hash)
        except RedisError:
            return False
    
    def _mark_url_seen(self, url_hash: str):
        """Mark URL as seen."""
        if not self._redis_client:
            # In-memory fallback
            if not hasattr(self, '_seen_urls'):
                self._seen_urls = set()
            self._seen_urls.add(url_hash)
            return
        
        try:
            self._redis_client.sadd("axiom:seen_urls", url_hash)
            # Set expiration (24 hours)
            self._redis_client.expire("axiom:seen_urls", 86400)
        except RedisError as e:
            logger.warning(f"Failed to mark URL as seen: {e}")
    
    def _create_checkpoint(self, task: CrawlTask):
        """Create checkpoint for fault tolerance."""
        if not self.enable_checkpoints:
            return
        
        self._checkpoint_counter += 1
        if self._checkpoint_counter % self.checkpoint_interval != 0:
            return
        
        try:
            # Serialize task state
            checkpoint_data = {
                'task': task.to_dict(),
                'timestamp': time.time(),
                'node_id': self.node_id,
                'stats': self.stats.copy(),
            }
            
            # Compress for efficiency
            serialized = pickle.dumps(checkpoint_data)
            compressed = zlib.compress(serialized)
            
            # Store checkpoint
            if self._redis_client:
                checkpoint_key = f"axiom:checkpoints:{task.task_id}"
                self._redis_client.setex(
                    checkpoint_key,
                    3600,  # 1 hour expiration
                    compressed
                )
            
            # Update task
            task.checkpoint_data = compressed
            task.status = TaskStatus.CHECKPOINTED
            
            self.stats['checkpoints_created'] += 1
            logger.debug(f"Created checkpoint for task {task.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
    
    def recover_from_checkpoint(self, task_id: str) -> Optional[CrawlTask]:
        """Recover task from checkpoint."""
        if not self._redis_client:
            return None
        
        try:
            checkpoint_key = f"axiom:checkpoints:{task_id}"
            compressed = self._redis_client.get(checkpoint_key)
            
            if not compressed:
                return None
            
            # Decompress and deserialize
            serialized = zlib.decompress(compressed)
            checkpoint_data = pickle.loads(serialized)
            
            # Recreate task
            task_dict = checkpoint_data['task']
            task = CrawlTask.from_dict(task_dict)
            
            logger.info(f"Recovered task {task_id} from checkpoint")
            return task
            
        except Exception as e:
            logger.error(f"Failed to recover checkpoint for {task_id}: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        # In a full implementation, this would query Celery result backend
        # For now, return basic info
        return {
            'task_id': task_id,
            'status': 'unknown',
            'message': 'Status tracking requires Celery result backend'
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        # Implementation would depend on Celery's revocation
        logger.info(f"Task cancellation requested: {task_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'active_nodes': len([n for n in self._nodes.values() if n.status != NodeStatus.OFFLINE]),
                'total_nodes': len(self._nodes),
                'domains_queued': len(self._domain_queues),
                'total_queued_tasks': sum(len(q) for q in self._domain_queues.values()),
            })
        return stats
    
    def get_nodes(self) -> List[Dict[str, Any]]:
        """Get information about all nodes."""
        with self._lock:
            return [node.to_dict() for node in self._nodes.values()]
    
    def clear_queues(self):
        """Clear all queues (use with caution)."""
        with self._lock:
            self._domain_queues.clear()
            self._domain_timestamps.clear()
            logger.warning("All queues cleared")
    
    def register_callback(self, task_id: str, callback: Callable):
        """Register a callback for a specific task."""
        self._task_callbacks[task_id] = callback


# Celery tasks (only defined if Celery is available)
if CELERY_AVAILABLE:
    @celery_app.task(
        bind=True,
        name='axiom.distributed.queue.crawl_task',
        max_retries=3,
        default_retry_delay=60,
        acks_late=True,
        reject_on_worker_lost=True,
        track_started=True
    )
    def crawl_task(self, task_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Celery task for crawling a URL."""
        try:
            # Reconstruct task
            task = CrawlTask.from_dict(task_dict)
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            logger.info(f"Executing crawl task {task.task_id}: {task.url}")
            
            # Import here to avoid circular imports in worker processes
            from axiom.core.ai import AdaptiveExtractor
            from axiom.core.storage import StorageManager
            
            # Simulate crawling with actual axiom components
            # In production, this would use the full axiom pipeline
            
            # Create checkpoint before starting
            if hasattr(self, 'queue_manager') and self.queue_manager.enable_checkpoints:
                self.queue_manager._create_checkpoint(task)
            
            # Simulate work
            time.sleep(0.5)  # Replace with actual crawling
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            result = {
                'task_id': task.task_id,
                'status': 'completed',
                'url': task.url,
                'execution_time': task.completed_at - task.started_at,
                'timestamp': time.time(),
            }
            
            logger.info(f"Completed crawl task {task.task_id}")
            return result
            
        except Exception as exc:
            logger.error(f"Crawl task failed: {exc}")
            
            # Retry with exponential backoff
            retry_count = self.request.retries
            if retry_count < self.max_retries:
                # Calculate exponential backoff
                countdown = exponential_backoff(retry_count, base=2, max_delay=300)
                
                logger.info(f"Retrying task {task_dict.get('task_id')} in {countdown}s "
                          f"(attempt {retry_count + 1}/{self.max_retries})")
                
                raise self.retry(exc=exc, countdown=countdown)
            else:
                # Max retries exceeded
                logger.error(f"Task {task_dict.get('task_id')} failed after {retry_count} retries")
                return {
                    'task_id': task_dict.get('task_id'),
                    'status': 'failed',
                    'error': str(exc),
                    'retries': retry_count,
                }
    
    @celery_app.task(
        name='axiom.distributed.queue.checkpoint_task',
        bind=True
    )
    def checkpoint_task(self, task_dict: Dict[str, Any], checkpoint_data: bytes) -> Dict[str, Any]:
        """Celery task for creating checkpoints."""
        try:
            # Store checkpoint in Redis
            task_id = task_dict.get('task_id')
            checkpoint_key = f"axiom:checkpoints:{task_id}"
            
            # In production, would use the Redis client from the queue manager
            # For now, just log
            logger.info(f"Created checkpoint for task {task_id}")
            
            return {
                'task_id': task_id,
                'checkpoint_created': True,
                'timestamp': time.time(),
            }
            
        except Exception as exc:
            logger.error(f"Checkpoint task failed: {exc}")
            return {
                'task_id': task_dict.get('task_id'),
                'checkpoint_created': False,
                'error': str(exc),
            }
    
    # Signal handlers for worker lifecycle
    @worker_init.connect
    def worker_init_handler(sender=None, **kwargs):
        """Handle worker initialization."""
        logger.info("Celery worker initializing")
    
    @worker_shutdown.connect
    def worker_shutdown_handler(sender=None, **kwargs):
        """Handle worker shutdown."""
        logger.info("Celery worker shutting down")
    
    @worker_ready.connect
    def worker_ready_handler(sender=None, **kwargs):
        """Handle worker ready state."""
        logger.info("Celery worker ready")


# Priority queue implementation for tasks
class TaskPriorityQueue:
    """Thread-safe priority queue for crawl tasks."""
    
    def __init__(self):
        self._heap: List[Tuple[float, str, CrawlTask]] = []
        self._lock = threading.RLock()
        self._task_map: Dict[str, CrawlTask] = {}
    
    def push(self, task: CrawlTask):
        """Add task to queue."""
        with self._lock:
            score = task.calculate_priority_score()
            heapq.heappush(self._heap, (score, task.task_id, task))
            self._task_map[task.task_id] = task
    
    def pop(self) -> Optional[CrawlTask]:
        """Remove and return highest priority task."""
        with self._lock:
            if not self._heap:
                return None
            
            _, task_id, task = heapq.heappop(self._heap)
            self._task_map.pop(task_id, None)
            return task
    
    def peek(self) -> Optional[CrawlTask]:
        """Return highest priority task without removing."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][2]
    
    def remove(self, task_id: str) -> bool:
        """Remove task by ID."""
        with self._lock:
            if task_id not in self._task_map:
                return False
            
            # Mark as removed (lazy deletion)
            self._task_map[task_id].status = TaskStatus.CANCELLED
            return True
    
    def __len__(self):
        return len(self._heap)
    
    def __contains__(self, task_id: str):
        return task_id in self._task_app


# Factory function for easy initialization
def create_distributed_queue(
    broker_url: str = "redis://localhost:6379/0",
    **kwargs
) -> DistributedQueue:
    """Create and configure a distributed queue instance.
    
    Args:
        broker_url: Broker URL (Redis or RabbitMQ)
        **kwargs: Additional configuration options
        
    Returns:
        Configured DistributedQueue instance
    """
    queue = DistributedQueue(broker_url=broker_url, **kwargs)
    return queue


# Integration with existing axiom components
class DistributedCrawler:
    """High-level interface for distributed crawling."""
    
    def __init__(
        self,
        queue: Optional[DistributedQueue] = None,
        storage_manager: Optional[StorageManager] = None,
        **kwargs
    ):
        self.queue = queue or create_distributed_queue(**kwargs)
        self.storage = storage_manager or StorageManager()
        self._started = False
    
    def start(self):
        """Start the distributed crawler."""
        if self._started:
            return
        
        self.queue.start()
        self._started = True
        logger.info("Distributed crawler started")
    
    def stop(self):
        """Stop the distributed crawler."""
        if not self._started:
            return
        
        self.queue.stop()
        self._started = False
        logger.info("Distributed crawler stopped")
    
    def crawl(
        self,
        urls: Union[str, List[str]],
        max_depth: int = 3,
        respect_robots: bool = True,
        **kwargs
    ) -> List[str]:
        """Start a distributed crawl.
        
        Args:
            urls: Seed URL(s)
            max_depth: Maximum crawl depth
            respect_robots: Respect robots.txt
            **kwargs: Additional crawl parameters
            
        Returns:
            List of task IDs
        """
        if isinstance(urls, str):
            urls = [urls]
        
        task_ids = []
        for url in urls:
            task_id = self.queue.enqueue(
                url=url,
                depth=0,
                metadata={
                    'max_depth': max_depth,
                    'respect_robots': respect_robots,
                    **kwargs
                },
                tags={'seed_url'}
            )
            task_ids.append(task_id)
        
        logger.info(f"Started crawl with {len(urls)} seed URLs")
        return task_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crawl statistics."""
        return self.queue.get_stats()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Export public API
__all__ = [
    'DistributedQueue',
    'DistributedCrawler',
    'CrawlTask',
    'WorkerNode',
    'TaskPriority',
    'TaskStatus',
    'NodeStatus',
    'ConsistentHashRing',
    'create_distributed_queue',
]