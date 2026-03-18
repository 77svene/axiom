"""
Distributed Crawling Engine for axiom
Redis/RabbitMQ-backed distributed task queue with horizontal scaling,
automatic node discovery, load balancing, and fault tolerance.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

# Try to import optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    pika = None

from axiom.core.storage import Storage
from axiom.core.utils._utils import get_logger, retry_with_backoff

logger = get_logger(__name__)


class QueueBackend(Enum):
    """Supported message queue backends."""
    REDIS = "redis"
    RABBITMQ = "rabbitmq"


class TaskPriority(Enum):
    """Task priority levels for politeness and scheduling."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class DistributedConfig:
    """Configuration for distributed crawling."""
    backend: QueueBackend = QueueBackend.REDIS
    redis_url: str = "redis://localhost:6379/0"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    queue_prefix: str = "axiom"
    node_discovery_interval: int = 30
    checkpoint_interval: int = 60
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    task_timeout: int = 300
    max_tasks_per_node: int = 100
    enable_priority_queues: bool = True
    enable_deduplication: bool = True
    deduplication_ttl: int = 86400  # 24 hours


@dataclass
class CrawlTask:
    """Represents a distributed crawl task."""
    url: str
    priority: TaskPriority = TaskPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    last_error: Optional[str] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = self._generate_task_id()
    
    def _generate_task_id(self) -> str:
        """Generate deterministic task ID based on URL and metadata."""
        content = f"{self.url}:{json.dumps(self.metadata, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "url": self.url,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "attempts": self.attempts,
            "last_error": self.last_error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrawlTask":
        """Deserialize task from dictionary."""
        return cls(
            url=data["url"],
            priority=TaskPriority(data["priority"]),
            metadata=data.get("metadata", {}),
            task_id=data.get("task_id"),
            created_at=data.get("created_at", time.time()),
            attempts=data.get("attempts", 0),
            last_error=data.get("last_error")
        )


@dataclass
class NodeInfo:
    """Information about a worker node."""
    node_id: str
    hostname: str
    ip_address: str
    last_heartbeat: float
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    capabilities: Set[str] = field(default_factory=set)
    load_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node info to dictionary."""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "last_heartbeat": self.last_heartbeat,
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "capabilities": list(self.capabilities),
            "load_score": self.load_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        """Deserialize node info from dictionary."""
        return cls(
            node_id=data["node_id"],
            hostname=data["hostname"],
            ip_address=data["ip_address"],
            last_heartbeat=data["last_heartbeat"],
            active_tasks=data.get("active_tasks", 0),
            completed_tasks=data.get("completed_tasks", 0),
            failed_tasks=data.get("failed_tasks", 0),
            capabilities=set(data.get("capabilities", [])),
            load_score=data.get("load_score", 0.0)
        )


class ConsistentHashRing:
    """Consistent hashing for URL deduplication and load distribution."""
    
    def __init__(self, nodes: List[str], replicas: int = 100):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        
        for node in nodes:
            self.add_node(node)
    
    def add_node(self, node: str) -> None:
        """Add a node to the hash ring."""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)
        self.sorted_keys.sort()
    
    def remove_node(self, node: str) -> None:
        """Remove a node from the hash ring."""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)
    
    def get_node(self, url: str) -> str:
        """Get the node responsible for a URL."""
        if not self.ring:
            raise ValueError("Hash ring is empty")
        
        hash_key = self._hash(url)
        
        # Find the first node with hash >= hash_key
        for key in self.sorted_keys:
            if key >= hash_key:
                return self.ring[key]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class RedisBackend:
    """Redis backend for distributed task queue."""
    
    def __init__(self, config: DistributedConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not installed. Install with: pip install redis")
        
        self.config = config
        self.redis_client = redis.Redis.from_url(config.redis_url)
        self._prefix = config.queue_prefix
        
    def _key(self, *parts: str) -> str:
        """Generate Redis key with prefix."""
        return f"{self._prefix}:{':'.join(parts)}"
    
    async def enqueue_task(self, task: CrawlTask) -> bool:
        """Add task to appropriate priority queue."""
        try:
            queue_key = self._key("queue", str(task.priority.value))
            task_data = json.dumps(task.to_dict())
            
            # Use Redis list for FIFO queue
            self.redis_client.lpush(queue_key, task_data)
            
            # Track task in global set
            self.redis_client.sadd(self._key("tasks", "all"), task.task_id)
            
            # Set task metadata
            self.redis_client.hset(
                self._key("task", task.task_id),
                mapping={
                    "status": "pending",
                    "enqueued_at": str(time.time()),
                    "priority": str(task.priority.value)
                }
            )
            
            logger.debug(f"Task {task.task_id} enqueued with priority {task.priority.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False
    
    async def dequeue_task(self, node_id: str) -> Optional[CrawlTask]:
        """Get next task from highest priority queue."""
        try:
            # Try queues in priority order
            for priority in TaskPriority:
                queue_key = self._key("queue", str(priority.value))
                
                # Use blocking pop with timeout
                result = self.redis_client.brpop(queue_key, timeout=1)
                if result:
                    _, task_data = result
                    task = CrawlTask.from_dict(json.loads(task_data))
                    
                    # Update task status
                    self.redis_client.hset(
                        self._key("task", task.task_id),
                        mapping={
                            "status": "processing",
                            "node_id": node_id,
                            "started_at": str(time.time())
                        }
                    )
                    
                    # Update node's active tasks
                    self.redis_client.hincrby(
                        self._key("node", node_id),
                        "active_tasks", 1
                    )
                    
                    logger.debug(f"Task {task.task_id} dequeued by node {node_id}")
                    return task
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to dequeue task for node {node_id}: {e}")
            return None
    
    async def complete_task(self, task_id: str, node_id: str, 
                           result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark task as completed."""
        try:
            # Update task status
            self.redis_client.hset(
                self._key("task", task_id),
                mapping={
                    "status": "completed",
                    "completed_at": str(time.time()),
                    "result": json.dumps(result) if result else ""
                }
            )
            
            # Move to completed set
            self.redis_client.sadd(self._key("tasks", "completed"), task_id)
            self.redis_client.srem(self._key("tasks", "all"), task_id)
            
            # Update node stats
            self.redis_client.hincrby(
                self._key("node", node_id),
                "active_tasks", -1
            )
            self.redis_client.hincrby(
                self._key("node", node_id),
                "completed_tasks", 1
            )
            
            logger.debug(f"Task {task_id} completed by node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False
    
    async def fail_task(self, task_id: str, node_id: str, error: str) -> bool:
        """Mark task as failed and handle retry logic."""
        try:
            # Get current task data
            task_data = self.redis_client.hgetall(self._key("task", task_id))
            if not task_data:
                return False
            
            attempts = int(task_data.get(b"attempts", 0)) + 1
            
            # Update task status
            self.redis_client.hset(
                self._key("task", task_id),
                mapping={
                    "status": "failed",
                    "failed_at": str(time.time()),
                    "error": error,
                    "attempts": str(attempts)
                }
            )
            
            # Update node stats
            self.redis_client.hincrby(
                self._key("node", node_id),
                "active_tasks", -1
            )
            self.redis_client.hincrby(
                self._key("node", node_id),
                "failed_tasks", 1
            )
            
            # Check if we should retry
            if attempts < self.config.max_retries:
                # Calculate exponential backoff
                backoff = self.config.retry_backoff_base ** attempts
                retry_at = time.time() + backoff
                
                # Schedule retry
                self.redis_client.zadd(
                    self._key("retry_queue"),
                    {task_id: retry_at}
                )
                
                logger.info(f"Task {task_id} scheduled for retry {attempts}/{self.config.max_retries} "
                          f"in {backoff:.1f}s")
            else:
                # Move to failed set
                self.redis_client.sadd(self._key("tasks", "failed"), task_id)
                self.redis_client.srem(self._key("tasks", "all"), task_id)
                logger.error(f"Task {task_id} permanently failed after {attempts} attempts")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark task {task_id} as failed: {e}")
            return False
    
    async def check_duplicate(self, url: str) -> bool:
        """Check if URL has been processed recently."""
        if not self.config.enable_deduplication:
            return False
        
        try:
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            key = self._key("dedup", url_hash)
            
            # Check if URL exists in deduplication set
            exists = self.redis_client.exists(key)
            
            if not exists:
                # Add to deduplication set with TTL
                self.redis_client.setex(key, self.config.deduplication_ttl, "1")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check duplicate for {url}: {e}")
            return False
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """Register a worker node."""
        try:
            self.redis_client.hset(
                self._key("nodes"),
                node_info.node_id,
                json.dumps(node_info.to_dict())
            )
            
            # Set TTL for node (auto-cleanup if node dies)
            self.redis_client.expire(
                self._key("node", node_info.node_id, "heartbeat"),
                self.config.node_discovery_interval * 2
            )
            
            logger.info(f"Node {node_info.node_id} registered")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node {node_info.node_id}: {e}")
            return False
    
    async def update_node_heartbeat(self, node_id: str) -> bool:
        """Update node heartbeat timestamp."""
        try:
            self.redis_client.setex(
                self._key("node", node_id, "heartbeat"),
                self.config.node_discovery_interval * 2,
                str(time.time())
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for node {node_id}: {e}")
            return False
    
    async def get_active_nodes(self) -> List[NodeInfo]:
        """Get list of active worker nodes."""
        try:
            nodes_data = self.redis_client.hgetall(self._key("nodes"))
            active_nodes = []
            
            for node_id, node_json in nodes_data.items():
                node_id_str = node_id.decode()
                
                # Check if node is alive (has recent heartbeat)
                heartbeat_key = self._key("node", node_id_str, "heartbeat")
                if self.redis_client.exists(heartbeat_key):
                    node_info = NodeInfo.from_dict(json.loads(node_json))
                    active_nodes.append(node_info)
                else:
                    # Clean up dead node
                    self.redis_client.hdel(self._key("nodes"), node_id_str)
                    logger.warning(f"Removed dead node {node_id_str}")
            
            return active_nodes
            
        except Exception as e:
            logger.error(f"Failed to get active nodes: {e}")
            return []
    
    async def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Save crawl checkpoint for fault tolerance."""
        try:
            checkpoint_json = json.dumps(checkpoint_data)
            self.redis_client.set(
                self._key("checkpoint", "latest"),
                checkpoint_json
            )
            
            # Also save with timestamp for history
            timestamp = int(time.time())
            self.redis_client.set(
                self._key("checkpoint", str(timestamp)),
                checkpoint_json
            )
            
            logger.info("Checkpoint saved")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    async def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint."""
        try:
            checkpoint_json = self.redis_client.get(self._key("checkpoint", "latest"))
            if checkpoint_json:
                return json.loads(checkpoint_json)
            return None
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


class RabbitMQBackend:
    """RabbitMQ backend for distributed task queue."""
    
    def __init__(self, config: DistributedConfig):
        if not RABBITMQ_AVAILABLE:
            raise ImportError("pika package not installed. Install with: pip install pika")
        
        self.config = config
        self.connection = None
        self.channel = None
        self._prefix = config.queue_prefix
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            parameters = pika.URLParameters(self.config.rabbitmq_url)
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare priority queues
            for priority in TaskPriority:
                queue_name = self._queue_name(priority)
                self.channel.queue_declare(
                    queue=queue_name,
                    durable=True,
                    arguments={
                        'x-max-priority': 5,  # RabbitMQ priority levels
                        'x-message-ttl': self.config.task_timeout * 1000
                    }
                )
            
            logger.info("Connected to RabbitMQ")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    def _queue_name(self, priority: TaskPriority) -> str:
        """Generate queue name for priority level."""
        return f"{self._prefix}_priority_{priority.value}"
    
    def _disconnect(self) -> None:
        """Close RabbitMQ connection."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
    
    async def enqueue_task(self, task: CrawlTask) -> bool:
        """Add task to RabbitMQ queue."""
        try:
            if not self.channel or self.channel.is_closed:
                self._connect()
            
            queue_name = self._queue_name(task.priority)
            task_data = json.dumps(task.to_dict())
            
            self.channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=task_data,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistent message
                    priority=task.priority.value,
                    message_id=task.task_id
                )
            )
            
            logger.debug(f"Task {task.task_id} published to {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish task {task.task_id}: {e}")
            self._disconnect()
            return False
    
    # Note: RabbitMQ consumption is typically handled differently with callbacks
    # This is a simplified version for the interface
    async def dequeue_task(self, node_id: str) -> Optional[CrawlTask]:
        """Get next task from RabbitMQ (simplified implementation)."""
        # In production, you'd use a consumer with callbacks
        # This is just for interface compatibility
        logger.warning("RabbitMQ dequeue requires consumer setup. Use start_consumer() instead.")
        return None
    
    def start_consumer(self, callback: Callable[[CrawlTask], None]) -> None:
        """Start consuming tasks from RabbitMQ."""
        def _on_message(channel, method, properties, body):
            try:
                task = CrawlTask.from_dict(json.loads(body))
                callback(task)
                channel.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        # Consume from all priority queues
        for priority in TaskPriority:
            queue_name = self._queue_name(priority)
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=_on_message
            )
        
        logger.info("Started RabbitMQ consumer")
        self.channel.start_consuming()


class DistributedCoordinator:
    """
    Main coordinator for distributed crawling.
    Manages task distribution, node discovery, load balancing, and fault tolerance.
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.node_id = self._generate_node_id()
        self.backend = self._create_backend()
        self.hash_ring = ConsistentHashRing([])
        self.active_nodes: Dict[str, NodeInfo] = {}
        self.task_callbacks: Dict[str, Callable] = {}
        self._running = False
        self._tasks_processed = 0
        self._start_time = time.time()
        
        # Initialize storage for checkpointing
        self.storage = Storage()
        
        logger.info(f"Distributed coordinator initialized with {self.config.backend.value} backend")
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        import socket
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        timestamp = str(int(time.time()))
        content = f"{hostname}:{ip}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _create_backend(self) -> Union[RedisBackend, RabbitMQBackend]:
        """Create appropriate backend based on configuration."""
        if self.config.backend == QueueBackend.REDIS:
            return RedisBackend(self.config)
        elif self.config.backend == QueueBackend.RABBITMQ:
            return RabbitMQBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
    
    async def start(self) -> None:
        """Start the distributed coordinator."""
        self._running = True
        
        # Register this node
        node_info = NodeInfo(
            node_id=self.node_id,
            hostname=self._get_hostname(),
            ip_address=self._get_ip_address(),
            last_heartbeat=time.time(),
            capabilities={"fetch", "parse", "storage"}
        )
        
        await self.backend.register_node(node_info)
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._node_discovery_loop())
        asyncio.create_task(self._checkpoint_loop())
        asyncio.create_task(self._retry_loop())
        
        logger.info(f"Distributed coordinator started on node {self.node_id}")
    
    async def stop(self) -> None:
        """Stop the distributed coordinator."""
        self._running = False
        
        # Save final checkpoint
        await self._save_checkpoint()
        
        logger.info(f"Distributed coordinator stopped on node {self.node_id}")
    
    async def submit_url(self, url: str, priority: TaskPriority = TaskPriority.NORMAL,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Submit a URL for distributed crawling.
        
        Args:
            url: URL to crawl
            priority: Task priority level
            metadata: Additional metadata for the task
            
        Returns:
            Task ID if successful, None otherwise
        """
        # Check for duplicates
        if await self.backend.check_duplicate(url):
            logger.debug(f"URL {url} already processed, skipping")
            return None
        
        # Create task
        task = CrawlTask(
            url=url,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Enqueue task
        if await self.backend.enqueue_task(task):
            logger.info(f"URL submitted for crawling: {url} (task_id: {task.task_id})")
            return task.task_id
        
        return None
    
    async def submit_urls(self, urls: List[str], priority: TaskPriority = TaskPriority.NORMAL,
                         metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Submit multiple URLs for distributed crawling.
        
        Args:
            urls: List of URLs to crawl
            priority: Task priority level
            metadata: Additional metadata for all tasks
            
        Returns:
            List of task IDs for successfully submitted URLs
        """
        task_ids = []
        for url in urls:
            task_id = await self.submit_url(url, priority, metadata)
            if task_id:
                task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)}/{len(urls)} URLs for crawling")
        return task_ids
    
    async def process_tasks(self, worker_callback: Callable[[CrawlTask], Any]) -> None:
        """
        Process tasks from the queue (worker mode).
        
        Args:
            worker_callback: Async function to process a task
        """
        logger.info(f"Node {self.node_id} starting task processing")
        
        while self._running:
            try:
                # Get next task
                task = await self.backend.dequeue_task(self.node_id)
                
                if task:
                    # Process task with retry logic
                    result = await self._process_with_retry(task, worker_callback)
                    
                    if result["success"]:
                        await self.backend.complete_task(
                            task.task_id, 
                            self.node_id, 
                            result.get("data")
                        )
                        self._tasks_processed += 1
                    else:
                        await self.backend.fail_task(
                            task.task_id,
                            self.node_id,
                            result.get("error", "Unknown error")
                        )
                else:
                    # No tasks available, wait before retrying
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_with_retry(self, task: CrawlTask, 
                                 callback: Callable[[CrawlTask], Any]) -> Dict[str, Any]:
        """Process task with exponential backoff retry."""
        for attempt in range(self.config.max_retries):
            try:
                result = await callback(task)
                return {"success": True, "data": result}
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.config.max_retries} failed: {str(e)}"
                logger.warning(f"Task {task.task_id} {error_msg}")
                
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    backoff = self.config.retry_backoff_base ** attempt
                    await asyncio.sleep(backoff)
                else:
                    return {"success": False, "error": error_msg}
        
        return {"success": False, "error": "Max retries exceeded"}
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to indicate node is alive."""
        while self._running:
            try:
                await self.backend.update_node_heartbeat(self.node_id)
                await asyncio.sleep(self.config.node_discovery_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _node_discovery_loop(self) -> None:
        """Discover and update active nodes."""
        while self._running:
            try:
                active_nodes = await self.backend.get_active_nodes()
                
                # Update hash ring with active nodes
                node_ids = [node.node_id for node in active_nodes]
                self.hash_ring = ConsistentHashRing(node_ids)
                
                # Update active nodes cache
                self.active_nodes = {node.node_id: node for node in active_nodes}
                
                logger.debug(f"Discovered {len(active_nodes)} active nodes")
                
                await asyncio.sleep(self.config.node_discovery_interval)
                
            except Exception as e:
                logger.error(f"Node discovery error: {e}")
                await asyncio.sleep(10)
    
    async def _checkpoint_loop(self) -> None:
        """Periodically save checkpoint for fault tolerance."""
        while self._running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval)
                await self._save_checkpoint()
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")
    
    async def _retry_loop(self) -> None:
        """Process tasks scheduled for retry."""
        while self._running:
            try:
                # This would be implemented based on backend capabilities
                # For Redis, we'd check the retry_queue sorted set
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Retry loop error: {e}")
    
    async def _save_checkpoint(self) -> None:
        """Save current state checkpoint."""
        checkpoint_data = {
            "node_id": self.node_id,
            "timestamp": time.time(),
            "tasks_processed": self._tasks_processed,
            "uptime": time.time() - self._start_time,
            "active_nodes": len(self.active_nodes),
            "config": {
                "backend": self.config.backend.value,
                "max_retries": self.config.max_retries,
                "deduplication_enabled": self.config.enable_deduplication
            }
        }
        
        await self.backend.save_checkpoint(checkpoint_data)
    
    async def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint."""
        return await self.backend.load_checkpoint()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "node_id": self.node_id,
            "tasks_processed": self._tasks_processed,
            "uptime": time.time() - self._start_time,
            "active_nodes": len(self.active_nodes),
            "backend": self.config.backend.value,
            "queue_prefix": self.config.queue_prefix
        }
    
    def _get_hostname(self) -> str:
        """Get current hostname."""
        import socket
        return socket.gethostname()
    
    def _get_ip_address(self) -> str:
        """Get current IP address."""
        import socket
        return socket.gethostbyname(self._get_hostname())
    
    def register_task_callback(self, task_type: str, callback: Callable) -> None:
        """Register callback for specific task types."""
        self.task_callbacks[task_type] = callback


# Factory function for easy initialization
def create_distributed_coordinator(
    backend: str = "redis",
    redis_url: str = "redis://localhost:6379/0",
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
    **kwargs
) -> DistributedCoordinator:
    """
    Factory function to create a distributed coordinator.
    
    Args:
        backend: Queue backend ("redis" or "rabbitmq")
        redis_url: Redis connection URL
        rabbitmq_url: RabbitMQ connection URL
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured DistributedCoordinator instance
    """
    config = DistributedConfig(
        backend=QueueBackend(backend),
        redis_url=redis_url,
        rabbitmq_url=rabbitmq_url,
        **kwargs
    )
    
    return DistributedCoordinator(config)


# Integration with existing axiom modules
class DistributedFetcherSession:
    """
    Distributed version of FetcherSession that uses the coordinator.
    Integrates with existing axiom fetcher sessions.
    """
    
    def __init__(self, coordinator: DistributedCoordinator, 
                 session_factory: Callable, **session_kwargs):
        self.coordinator = coordinator
        self.session_factory = session_factory
        self.session_kwargs = session_kwargs
    
    async def fetch_distributed(self, urls: List[str], 
                               priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """
        Fetch URLs using distributed crawling.
        
        Args:
            urls: List of URLs to fetch
            priority: Task priority
            
        Returns:
            List of task IDs
        """
        return await self.coordinator.submit_urls(urls, priority)
    
    async def start_worker(self) -> None:
        """Start processing tasks as a worker."""
        async def process_task(task: CrawlTask):
            # Create a session for this task
            async with self.session_factory(**self.session_kwargs) as session:
                # Fetch the URL
                response = await session.get(task.url)
                
                # Process based on metadata
                if "callback" in task.metadata:
                    callback = task.metadata["callback"]
                    if callable(callback):
                        return await callback(response)
                
                # Default: return response data
                return {
                    "url": task.url,
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body": await response.text()
                }
        
        await self.coordinator.process_tasks(process_task)


# Example usage and CLI integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="axiom Distributed Crawler")
    parser.add_argument("--backend", choices=["redis", "rabbitmq"], default="redis",
                       help="Queue backend to use")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0",
                       help="Redis connection URL")
    parser.add_argument("--rabbitmq-url", default="amqp://guest:guest@localhost:5672/",
                       help="RabbitMQ connection URL")
    parser.add_argument("--mode", choices=["coordinator", "worker", "submit"], 
                       default="coordinator",
                       help="Operation mode")
    parser.add_argument("--urls", nargs="+", help="URLs to submit (for submit mode)")
    
    args = parser.parse_args()
    
    async def main():
        # Create coordinator
        coordinator = create_distributed_coordinator(
            backend=args.backend,
            redis_url=args.redis_url,
            rabbitmq_url=args.rabbitmq_url
        )
        
        await coordinator.start()
        
        if args.mode == "submit":
            if not args.urls:
                print("Error: --urls required for submit mode")
                return
            
            task_ids = await coordinator.submit_urls(args.urls)
            print(f"Submitted {len(task_ids)} tasks")
            
        elif args.mode == "worker":
            print(f"Starting worker node {coordinator.node_id}")
            
            async def worker_callback(task: CrawlTask):
                print(f"Processing task {task.task_id}: {task.url}")
                # Simulate work
                await asyncio.sleep(1)
                return {"status": "success", "url": task.url}
            
            await coordinator.process_tasks(worker_callback)
        
        else:  # coordinator mode
            print(f"Coordinator running on node {coordinator.node_id}")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    stats = coordinator.get_stats()
                    print(f"\rTasks processed: {stats['tasks_processed']}, "
                          f"Active nodes: {stats['active_nodes']}", end="")
                    await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\nShutting down...")
        
        await coordinator.stop()
    
    asyncio.run(main())