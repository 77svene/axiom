"""
Distributed Crawling Engine for axiom
Redis/RabbitMQ-backed distributed task queue for horizontal scaling
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import pika
    from pika.exceptions import AMQPConnectionError
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

from axiom.core.ai import AIEngine
from axiom.core.custom_types import FetcherSession, Spider
from axiom.core.storage import Storage
from axiom.core.utils._utils import get_logger, retry_with_backoff

logger = get_logger(__name__)


class TaskPriority(Enum):
    """Task priority levels for politeness and scheduling"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class CrawlTask:
    """Represents a distributed crawling task"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    depth: int = 0
    max_depth: int = 3
    callback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    checkpoint: Optional[Dict[str, Any]] = None
    worker_id: Optional[str] = None
    domain_hash: Optional[str] = None
    
    def __post_init__(self):
        if not self.domain_hash and self.url:
            parsed = urlparse(self.url)
            self.domain_hash = hashlib.md5(parsed.netloc.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrawlTask':
        """Create task from dictionary"""
        if 'priority' in data:
            data['priority'] = TaskPriority(data['priority'])
        if 'status' in data:
            data['status'] = TaskStatus(data['status'])
        return cls(**data)


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hostname: str = ""
    ip_address: str = ""
    capacity: int = 10
    current_load: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "active"
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerNode':
        return cls(**data)


class ConsistentHashRing:
    """Consistent hashing for URL distribution across workers"""
    
    def __init__(self, nodes: Optional[List[str]] = None, replicas: int = 100):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def add_node(self, node: str) -> None:
        """Add a node to the hash ring"""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)
        self.sorted_keys.sort()
    
    def remove_node(self, node: str) -> None:
        """Remove a node from the hash ring"""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a given key"""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        idx = self._find_closest_node(hash_key)
        return self.ring[self.sorted_keys[idx]]
    
    def _find_closest_node(self, hash_key: int) -> int:
        """Find the closest node clockwise in the ring"""
        for i, key in enumerate(self.sorted_keys):
            if key >= hash_key:
                return i
        return 0
    
    @staticmethod
    def _hash(key: str) -> int:
        """Generate hash for a key"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class DistributedTaskQueue:
    """Base class for distributed task queue implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queue_name = config.get('queue_name', 'axiom_tasks')
        self.priority_queues = {
            TaskPriority.CRITICAL: f"{self.queue_name}_critical",
            TaskPriority.HIGH: f"{self.queue_name}_high",
            TaskPriority.NORMAL: f"{self.queue_name}_normal",
            TaskPriority.LOW: f"{self.queue_name}_low",
            TaskPriority.BACKGROUND: f"{self.queue_name}_background",
        }
    
    async def enqueue(self, task: CrawlTask) -> bool:
        """Enqueue a task"""
        raise NotImplementedError
    
    async def dequeue(self, worker_id: str) -> Optional[CrawlTask]:
        """Dequeue a task for a worker"""
        raise NotImplementedError
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark task as completed"""
        raise NotImplementedError
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed"""
        raise NotImplementedError
    
    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        raise NotImplementedError
    
    async def get_task(self, task_id: str) -> Optional[CrawlTask]:
        """Get task by ID"""
        raise NotImplementedError
    
    async def get_pending_tasks(self, limit: int = 100) -> List[CrawlTask]:
        """Get pending tasks"""
        raise NotImplementedError
    
    async def register_worker(self, worker: WorkerNode) -> bool:
        """Register a worker node"""
        raise NotImplementedError
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker node"""
        raise NotImplementedError
    
    async def get_workers(self) -> List[WorkerNode]:
        """Get all registered workers"""
        raise NotImplementedError
    
    async def update_worker_heartbeat(self, worker_id: str) -> bool:
        """Update worker heartbeat"""
        raise NotImplementedError


class RedisTaskQueue(DistributedTaskQueue):
    """Redis-backed distributed task queue"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not installed. Install with: pip install redis")
        
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0),
            password=config.get('redis_password'),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # Initialize hash ring for URL deduplication
        self.hash_ring = ConsistentHashRing(replicas=150)
        self._init_queues()
    
    def _init_queues(self) -> None:
        """Initialize Redis queues"""
        for priority, queue_name in self.priority_queues.items():
            # Create sorted set for priority queue (score = timestamp + priority_weight)
            if not self.redis_client.exists(queue_name):
                self.redis_client.zadd(queue_name, {})
    
    async def enqueue(self, task: CrawlTask) -> bool:
        """Enqueue a task with priority"""
        try:
            # Store task data
            task_key = f"task:{task.task_id}"
            self.redis_client.hset(task_key, mapping=task.to_dict())
            
            # Add to priority queue
            priority_score = time.time() + (task.priority.value * 1000)
            queue_name = self.priority_queues[task.priority]
            self.redis_client.zadd(queue_name, {task.task_id: priority_score})
            
            # Add to URL deduplication set
            url_hash = hashlib.md5(task.url.encode()).hexdigest()
            self.redis_client.sadd("seen_urls", url_hash)
            
            logger.info(f"Enqueued task {task.task_id} for URL {task.url}")
            return True
        except RedisError as e:
            logger.error(f"Failed to enqueue task: {e}")
            return False
    
    async def dequeue(self, worker_id: str) -> Optional[CrawlTask]:
        """Dequeue highest priority task for worker"""
        try:
            # Check all priority queues in order
            for priority in TaskPriority:
                queue_name = self.priority_queues[priority]
                
                # Get task with highest priority (lowest score)
                task_ids = self.redis_client.zrange(queue_name, 0, 0)
                
                if task_ids:
                    task_id = task_ids[0]
                    
                    # Remove from queue atomically
                    removed = self.redis_client.zrem(queue_name, task_id)
                    
                    if removed:
                        # Get task data
                        task_key = f"task:{task_id}"
                        task_data = self.redis_client.hgetall(task_key)
                        
                        if task_data:
                            task = CrawlTask.from_dict(task_data)
                            task.status = TaskStatus.RUNNING
                            task.started_at = time.time()
                            task.worker_id = worker_id
                            
                            # Update task
                            self.redis_client.hset(task_key, mapping=task.to_dict())
                            
                            logger.info(f"Dequeued task {task_id} for worker {worker_id}")
                            return task
            
            return None
        except RedisError as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark task as completed"""
        try:
            task_key = f"task:{task_id}"
            task_data = self.redis_client.hgetall(task_key)
            
            if task_data:
                task = CrawlTask.from_dict(task_data)
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                
                # Store result
                result_key = f"result:{task_id}"
                self.redis_client.hset(result_key, mapping=result)
                
                # Update task
                self.redis_client.hset(task_key, mapping=task.to_dict())
                
                # Move to completed set
                self.redis_client.sadd("completed_tasks", task_id)
                
                logger.info(f"Task {task_id} completed successfully")
                return True
            
            return False
        except RedisError as e:
            logger.error(f"Failed to complete task: {e}")
            return False
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed"""
        try:
            task_key = f"task:{task_id}"
            task_data = self.redis_client.hgetall(task_key)
            
            if task_data:
                task = CrawlTask.from_dict(task_data)
                task.status = TaskStatus.FAILED
                task.error = error
                task.completed_at = time.time()
                
                # Update task
                self.redis_client.hset(task_key, mapping=task.to_dict())
                
                # Move to failed set
                self.redis_client.sadd("failed_tasks", task_id)
                
                logger.error(f"Task {task_id} failed: {error}")
                return True
            
            return False
        except RedisError as e:
            logger.error(f"Failed to mark task as failed: {e}")
            return False
    
    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task with exponential backoff"""
        try:
            task_key = f"task:{task_id}"
            task_data = self.redis_client.hgetall(task_key)
            
            if task_data:
                task = CrawlTask.from_dict(task_data)
                
                if task.retries >= task.max_retries:
                    logger.warning(f"Task {task_id} exceeded max retries")
                    return False
                
                task.retries += 1
                task.status = TaskStatus.RETRYING
                task.error = None
                
                # Calculate backoff delay (exponential)
                backoff_delay = min(300, 2 ** task.retries)  # Max 5 minutes
                task.scheduled_at = time.time() + backoff_delay
                
                # Update task
                self.redis_client.hset(task_key, mapping=task.to_dict())
                
                # Re-enqueue with delay
                priority_score = task.scheduled_at + (task.priority.value * 1000)
                queue_name = self.priority_queues[task.priority]
                self.redis_client.zadd(queue_name, {task_id: priority_score})
                
                logger.info(f"Retrying task {task_id} (attempt {task.retries})")
                return True
            
            return False
        except RedisError as e:
            logger.error(f"Failed to retry task: {e}")
            return False
    
    async def get_task(self, task_id: str) -> Optional[CrawlTask]:
        """Get task by ID"""
        try:
            task_key = f"task:{task_id}"
            task_data = self.redis_client.hgetall(task_key)
            
            if task_data:
                return CrawlTask.from_dict(task_data)
            
            return None
        except RedisError as e:
            logger.error(f"Failed to get task: {e}")
            return None
    
    async def get_pending_tasks(self, limit: int = 100) -> List[CrawlTask]:
        """Get pending tasks"""
        try:
            tasks = []
            
            for priority in TaskPriority:
                queue_name = self.priority_queues[priority]
                task_ids = self.redis_client.zrange(queue_name, 0, limit - 1)
                
                for task_id in task_ids:
                    task = await self.get_task(task_id)
                    if task and task.status == TaskStatus.PENDING:
                        tasks.append(task)
                
                if len(tasks) >= limit:
                    break
            
            return tasks[:limit]
        except RedisError as e:
            logger.error(f"Failed to get pending tasks: {e}")
            return []
    
    async def register_worker(self, worker: WorkerNode) -> bool:
        """Register a worker node"""
        try:
            worker_key = f"worker:{worker.node_id}"
            self.redis_client.hset(worker_key, mapping=worker.to_dict())
            
            # Add to worker set
            self.redis_client.sadd("workers", worker.node_id)
            
            # Update hash ring
            self.hash_ring.add_node(worker.node_id)
            
            logger.info(f"Registered worker {worker.node_id}")
            return True
        except RedisError as e:
            logger.error(f"Failed to register worker: {e}")
            return False
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker node"""
        try:
            worker_key = f"worker:{worker_id}"
            self.redis_client.delete(worker_key)
            
            # Remove from worker set
            self.redis_client.srem("workers", worker_id)
            
            # Update hash ring
            self.hash_ring.remove_node(worker_id)
            
            logger.info(f"Unregistered worker {worker_id}")
            return True
        except RedisError as e:
            logger.error(f"Failed to unregister worker: {e}")
            return False
    
    async def get_workers(self) -> List[WorkerNode]:
        """Get all registered workers"""
        try:
            worker_ids = self.redis_client.smembers("workers")
            workers = []
            
            for worker_id in worker_ids:
                worker_key = f"worker:{worker_id}"
                worker_data = self.redis_client.hgetall(worker_key)
                
                if worker_data:
                    workers.append(WorkerNode.from_dict(worker_data))
            
            return workers
        except RedisError as e:
            logger.error(f"Failed to get workers: {e}")
            return []
    
    async def update_worker_heartbeat(self, worker_id: str) -> bool:
        """Update worker heartbeat"""
        try:
            worker_key = f"worker:{worker_id}"
            self.redis_client.hset(worker_key, "last_heartbeat", time.time())
            return True
        except RedisError as e:
            logger.error(f"Failed to update worker heartbeat: {e}")
            return False
    
    async def is_url_seen(self, url: str) -> bool:
        """Check if URL has been seen before"""
        try:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            return self.redis_client.sismember("seen_urls", url_hash)
        except RedisError as e:
            logger.error(f"Failed to check URL: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            stats = {
                "pending_tasks": 0,
                "completed_tasks": self.redis_client.scard("completed_tasks"),
                "failed_tasks": self.redis_client.scard("failed_tasks"),
                "active_workers": self.redis_client.scard("workers"),
                "seen_urls": self.redis_client.scard("seen_urls"),
            }
            
            for priority in TaskPriority:
                queue_name = self.priority_queues[priority]
                count = self.redis_client.zcard(queue_name)
                stats[f"pending_{priority.name.lower()}"] = count
                stats["pending_tasks"] += count
            
            return stats
        except RedisError as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


class DistributedWorker:
    """Distributed crawling worker node"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.worker_id = config.get('worker_id', str(uuid.uuid4()))
        self.hostname = config.get('hostname', 'localhost')
        self.capacity = config.get('capacity', 10)
        
        # Initialize task queue
        queue_type = config.get('queue_type', 'redis')
        if queue_type == 'redis':
            self.task_queue = RedisTaskQueue(config)
        else:
            raise ValueError(f"Unsupported queue type: {queue_type}")
        
        # Worker state
        self.worker = WorkerNode(
            node_id=self.worker_id,
            hostname=self.hostname,
            capacity=self.capacity,
            capabilities=['crawling', 'parsing', 'extraction']
        )
        
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        self.heartbeat_interval = config.get('heartbeat_interval', 30)
        self.poll_interval = config.get('poll_interval', 1)
        
        # Load existing sessions from config
        self.sessions: Dict[str, FetcherSession] = {}
        self._init_sessions()
        
        # AI engine for intelligent crawling
        self.ai_engine = AIEngine(config.get('ai_config', {}))
        
        logger.info(f"Initialized worker {self.worker_id}")
    
    def _init_sessions(self) -> None:
        """Initialize fetcher sessions"""
        session_configs = self.config.get('sessions', [])
        
        for session_config in session_configs:
            session_type = session_config.get('type', 'static')
            session_name = session_config.get('name', f"session_{len(self.sessions)}")
            
            # Create session based on type
            if session_type == 'dynamic':
                from axiom.core.custom_types import DynamicSession
                session = DynamicSession(**session_config.get('options', {}))
            elif session_type == 'stealthy':
                from axiom.core.custom_types import StealthySession
                session = StealthySession(**session_config.get('options', {}))
            else:
                from axiom.core.custom_types import FetcherSession
                session = FetcherSession(**session_config.get('options', {}))
            
            self.sessions[session_name] = session
    
    async def start(self) -> None:
        """Start the worker"""
        if self.is_running:
            logger.warning("Worker is already running")
            return
        
        self.is_running = True
        
        # Register worker
        await self.task_queue.register_worker(self.worker)
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._process_tasks_loop())
        asyncio.create_task(self._monitor_tasks_loop())
        
        logger.info(f"Worker {self.worker_id} started")
    
    async def stop(self) -> None:
        """Stop the worker"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel active tasks
        for task_id, task in self.active_tasks.items():
            task.cancel()
            logger.info(f"Cancelled task {task_id}")
        
        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Unregister worker
        await self.task_queue.unregister_worker(self.worker_id)
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats"""
        while self.is_running:
            try:
                await self.task_queue.update_worker_heartbeat(self.worker_id)
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _process_tasks_loop(self) -> None:
        """Main task processing loop"""
        while self.is_running:
            try:
                # Check capacity
                if len(self.active_tasks) >= self.capacity:
                    await asyncio.sleep(self.poll_interval)
                    continue
                
                # Dequeue task
                task = await self.task_queue.dequeue(self.worker_id)
                
                if task:
                    # Process task asynchronously
                    task_coroutine = self._process_task(task)
                    asyncio_task = asyncio.create_task(task_coroutine)
                    self.active_tasks[task.task_id] = asyncio_task
                    
                    # Update worker load
                    self.worker.current_load = len(self.active_tasks)
                    await self.task_queue.register_worker(self.worker)
                else:
                    await asyncio.sleep(self.poll_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task processing error: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_tasks_loop(self) -> None:
        """Monitor active tasks and handle completion"""
        while self.is_running:
            try:
                completed_tasks = []
                
                for task_id, task in self.active_tasks.items():
                    if task.done():
                        completed_tasks.append(task_id)
                        
                        try:
                            result = task.result()
                            await self.task_queue.complete_task(task_id, result)
                        except Exception as e:
                            logger.error(f"Task {task_id} failed: {e}")
                            await self.task_queue.fail_task(task_id, str(e))
                
                # Remove completed tasks
                for task_id in completed_tasks:
                    del self.active_tasks[task_id]
                
                # Update worker load
                if completed_tasks:
                    self.worker.current_load = len(self.active_tasks)
                    await self.task_queue.register_worker(self.worker)
                
                await asyncio.sleep(1)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _process_task(self, task: CrawlTask) -> Dict[str, Any]:
        """Process a single crawl task"""
        logger.info(f"Processing task {task.task_id}: {task.url}")
        
        try:
            # Select appropriate session
            session = self._select_session(task)
            
            # Create checkpoint
            checkpoint = {
                'url': task.url,
                'depth': task.depth,
                'started_at': time.time(),
                'worker_id': self.worker_id
            }
            
            # Update task with checkpoint
            task_key = f"task:{task.task_id}"
            self.task_queue.redis_client.hset(task_key, "checkpoint", json.dumps(checkpoint))
            
            # Fetch URL
            response = await session.fetch(task.url)
            
            if not response:
                raise Exception(f"Failed to fetch {task.url}")
            
            # Parse content
            parsed_data = await self._parse_content(response, task)
            
            # Extract links for further crawling
            if task.depth < task.max_depth:
                links = await self._extract_links(response, task)
                
                # Enqueue discovered links
                for link in links:
                    if not await self.task_queue.is_url_seen(link):
                        new_task = CrawlTask(
                            url=link,
                            priority=TaskPriority.NORMAL,
                            depth=task.depth + 1,
                            max_depth=task.max_depth,
                            metadata={'parent_task': task.task_id}
                        )
                        await self.task_queue.enqueue(new_task)
            
            # Prepare result
            result = {
                'url': task.url,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'content_length': len(response.content),
                'parsed_data': parsed_data,
                'processing_time': time.time() - checkpoint['started_at'],
                'worker_id': self.worker_id
            }
            
            logger.info(f"Completed task {task.task_id}")
            return result
        
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            raise
    
    def _select_session(self, task: CrawlTask) -> FetcherSession:
        """Select appropriate session for task"""
        # Use AI to select optimal session if available
        if self.ai_engine and task.metadata.get('use_ai', True):
            session_type = self.ai_engine.select_session_type(task.url, task.metadata)
            
            for session_name, session in self.sessions.items():
                if session_type.lower() in session_name.lower():
                    return session
        
        # Default to first available session
        if self.sessions:
            return next(iter(self.sessions.values()))
        
        # Create default session
        from axiom.core.custom_types import FetcherSession
        return FetcherSession()
    
    async def _parse_content(self, response: Any, task: CrawlTask) -> Dict[str, Any]:
        """Parse fetched content"""
        # Use AI for intelligent parsing if enabled
        if self.ai_engine and task.metadata.get('use_ai_parsing', True):
            return await self.ai_engine.parse_content(response, task.metadata)
        
        # Basic parsing
        from axiom.core.custom_types import ParsedContent
        
        parsed = ParsedContent(
            url=task.url,
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.text,
            encoding=response.encoding
        )
        
        return parsed.to_dict()
    
    async def _extract_links(self, response: Any, task: CrawlTask) -> List[str]:
        """Extract links from response"""
        links = []
        
        try:
            # Basic link extraction
            from urllib.parse import urljoin, urlparse
            
            base_url = task.url
            content = response.text
            
            # Simple regex for href attributes
            import re
            href_pattern = re.compile(r'href=["\'](.*?)["\']', re.IGNORECASE)
            
            for match in href_pattern.finditer(content):
                link = match.group(1)
                
                # Convert relative URLs to absolute
                absolute_link = urljoin(base_url, link)
                
                # Filter links
                parsed = urlparse(absolute_link)
                if parsed.scheme in ['http', 'https']:
                    # Apply domain restrictions if specified
                    allowed_domains = task.metadata.get('allowed_domains', [])
                    if not allowed_domains or parsed.netloc in allowed_domains:
                        links.append(absolute_link)
        
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
        
        return links
    
    async def submit_task(self, url: str, **kwargs) -> str:
        """Submit a new crawl task"""
        task = CrawlTask(
            url=url,
            priority=TaskPriority(kwargs.get('priority', TaskPriority.NORMAL.value)),
            depth=kwargs.get('depth', 0),
            max_depth=kwargs.get('max_depth', 3),
            metadata=kwargs.get('metadata', {})
        )
        
        success = await self.task_queue.enqueue(task)
        
        if success:
            return task.task_id
        else:
            raise Exception("Failed to enqueue task")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task"""
        task = await self.task_queue.get_task(task_id)
        
        if task:
            return {
                'task_id': task.task_id,
                'url': task.url,
                'status': task.status.value,
                'retries': task.retries,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'error': task.error,
                'worker_id': task.worker_id
            }
        
        return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        queue_stats = await self.task_queue.get_stats()
        
        return {
            'worker_id': self.worker_id,
            'active_tasks': len(self.active_tasks),
            'capacity': self.capacity,
            'utilization': len(self.active_tasks) / self.capacity if self.capacity > 0 else 0,
            'queue_stats': queue_stats,
            'sessions': list(self.sessions.keys())
        }


class DistributedCrawler:
    """Main distributed crawling coordinator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workers: Dict[str, DistributedWorker] = {}
        self.task_queue: Optional[DistributedTaskQueue] = None
        self.is_running = False
        
        # Initialize task queue
        queue_type = config.get('queue_type', 'redis')
        if queue_type == 'redis':
            self.task_queue = RedisTaskQueue(config)
        else:
            raise ValueError(f"Unsupported queue type: {queue_type}")
        
        # Load balancing strategy
        self.load_balancing = config.get('load_balancing', 'round_robin')
        self.worker_index = 0
        
        logger.info("Initialized distributed crawler")
    
    async def start(self) -> None:
        """Start the distributed crawler"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring
        asyncio.create_task(self._monitor_workers())
        
        logger.info("Distributed crawler started")
    
    async def stop(self) -> None:
        """Stop the distributed crawler"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all workers
        for worker_id, worker in self.workers.items():
            await worker.stop()
        
        logger.info("Distributed crawler stopped")
    
    async def add_worker(self, worker_config: Dict[str, Any]) -> str:
        """Add a new worker node"""
        worker = DistributedWorker(worker_config)
        worker_id = worker.worker_id
        
        await worker.start()
        self.workers[worker_id] = worker
        
        logger.info(f"Added worker {worker_id}")
        return worker_id
    
    async def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker node"""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            await worker.stop()
            del self.workers[worker_id]
            
            logger.info(f"Removed worker {worker_id}")
            return True
        
        return False
    
    async def submit_url(self, url: str, **kwargs) -> str:
        """Submit URL for crawling"""
        if not self.workers:
            raise Exception("No workers available")
        
        # Select worker using load balancing
        worker = self._select_worker()
        
        # Submit task
        task_id = await worker.submit_task(url, **kwargs)
        
        logger.info(f"Submitted URL {url} to worker {worker.worker_id}")
        return task_id
    
    async def submit_urls(self, urls: List[str], **kwargs) -> List[str]:
        """Submit multiple URLs for crawling"""
        task_ids = []
        
        for url in urls:
            try:
                task_id = await self.submit_url(url, **kwargs)
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"Failed to submit URL {url}: {e}")
                task_ids.append(None)
        
        return task_ids
    
    def _select_worker(self) -> DistributedWorker:
        """Select worker using load balancing strategy"""
        if not self.workers:
            raise Exception("No workers available")
        
        if self.load_balancing == 'round_robin':
            worker_list = list(self.workers.values())
            worker = worker_list[self.worker_index % len(worker_list)]
            self.worker_index += 1
            return worker
        
        elif self.load_balancing == 'least_loaded':
            # Select worker with least current load
            return min(self.workers.values(), 
                      key=lambda w: w.worker.current_load / w.capacity)
        
        elif self.load_balancing == 'random':
            import random
            return random.choice(list(self.workers.values()))
        
        else:
            # Default to first worker
            return next(iter(self.workers.values()))
    
    async def _monitor_workers(self) -> None:
        """Monitor worker health and performance"""
        while self.is_running:
            try:
                # Check worker heartbeats
                current_time = time.time()
                stale_workers = []
                
                for worker_id, worker in self.workers.items():
                    # Check if worker is responsive
                    try:
                        stats = await worker.get_stats()
                        
                        # Update worker status based on heartbeat
                        worker_info = await self.task_queue.get_workers()
                        worker_data = next((w for w in worker_info if w.node_id == worker_id), None)
                        
                        if worker_data:
                            heartbeat_age = current_time - worker_data.last_heartbeat
                            if heartbeat_age > 120:  # 2 minutes
                                logger.warning(f"Worker {worker_id} heartbeat stale")
                                stale_workers.append(worker_id)
                    
                    except Exception as e:
                        logger.error(f"Error checking worker {worker_id}: {e}")
                        stale_workers.append(worker_id)
                
                # Remove stale workers
                for worker_id in stale_workers:
                    await self.remove_worker(worker_id)
                
                await asyncio.sleep(60)  # Check every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get overall crawler statistics"""
        worker_stats = []
        
        for worker_id, worker in self.workers.items():
            try:
                stats = await worker.get_stats()
                worker_stats.append(stats)
            except Exception as e:
                logger.error(f"Failed to get stats for worker {worker_id}: {e}")
        
        queue_stats = await self.task_queue.get_stats()
        
        return {
            'total_workers': len(self.workers),
            'active_workers': len([w for w in self.workers.values() if w.is_running]),
            'total_capacity': sum(w.capacity for w in self.workers.values()),
            'current_load': sum(w.worker.current_load for w in self.workers.values()),
            'queue_stats': queue_stats,
            'worker_stats': worker_stats
        }


# Factory function for easy initialization
def create_distributed_crawler(config: Optional[Dict[str, Any]] = None) -> DistributedCrawler:
    """Create and configure a distributed crawler"""
    if config is None:
        config = {
            'queue_type': 'redis',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
            'load_balancing': 'least_loaded',
            'heartbeat_interval': 30,
            'poll_interval': 1
        }
    
    return DistributedCrawler(config)


# CLI integration
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='axiom Distributed Worker')
    parser.add_argument('--worker-id', help='Worker ID')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--capacity', type=int, default=10, help='Worker capacity')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'worker_id': args.worker_id,
        'redis_host': args.redis_host,
        'redis_port': args.redis_port,
        'capacity': args.capacity,
        'queue_type': 'redis'
    }
    
    if args.config:
        import json
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Create and start worker
    worker = DistributedWorker(config)
    
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        print("\nShutting down worker...")
        asyncio.run(worker.stop())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)