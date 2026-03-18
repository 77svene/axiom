# axiom/monitoring/dashboard.py

import asyncio
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import threading
import weakref
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, generate_latest, 
        CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from axiom.core.custom_types import CrawlResult, CrawlStatus
from axiom.core.storage import StorageManager
from axiom.core.utils._utils import get_logger, Timer


logger = get_logger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    metric: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "acknowledged": self.acknowledged
        }


@dataclass
class CrawlMetrics:
    """Real-time metrics for a single crawl job"""
    job_id: str
    url: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Progress tracking
    total_requests: int = 0
    completed_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    
    # Performance metrics
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    response_time_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    
    # Status
    status: CrawlStatus = CrawlStatus.PENDING
    current_url: str = ""
    depth: int = 0
    
    # Rate limiting
    requests_per_second: float = 0.0
    last_request_time: Optional[datetime] = None
    
    def update_from_result(self, result: CrawlResult):
        """Update metrics from a crawl result"""
        self.completed_requests += 1
        
        if result.status == CrawlStatus.SUCCESS:
            self.successful_requests += 1
        elif result.status == CrawlStatus.BLOCKED:
            self.blocked_requests += 1
            self.failed_requests += 1
        else:
            self.failed_requests += 1
        
        if result.response_time:
            self.response_time_history.append(result.response_time)
            self.avg_response_time = sum(self.response_time_history) / len(self.response_time_history)
            self.min_response_time = min(self.min_response_time, result.response_time)
            self.max_response_time = max(self.max_response_time, result.response_time)
        
        self.current_url = result.url
        self.depth = result.depth
        self.last_request_time = datetime.utcnow()
        
        # Calculate requests per second
        if self.start_time:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            if elapsed > 0:
                self.requests_per_second = self.completed_requests / elapsed
    
    @property
    def success_rate(self) -> float:
        if self.completed_requests == 0:
            return 0.0
        return (self.successful_requests / self.completed_requests) * 100
    
    @property
    def block_rate(self) -> float:
        if self.completed_requests == 0:
            return 0.0
        return (self.blocked_requests / self.completed_requests) * 100
    
    @property
    def progress_percent(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.completed_requests / self.total_requests) * 100
    
    @property
    def elapsed_time(self) -> float:
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "url": self.url,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "blocked_requests": self.blocked_requests,
            "success_rate": self.success_rate,
            "block_rate": self.block_rate,
            "progress_percent": self.progress_percent,
            "avg_response_time": self.avg_response_time,
            "min_response_time": self.min_response_time if self.min_response_time != float('inf') else 0,
            "max_response_time": self.max_response_time,
            "requests_per_second": self.requests_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "status": self.status.value,
            "current_url": self.current_url,
            "depth": self.depth,
            "elapsed_time": self.elapsed_time
        }


@dataclass
class SystemMetrics:
    """System-wide resource metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    active_connections: int = 0
    active_crawlers: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates metrics from all active crawlers"""
    
    def __init__(self, max_history: int = 1000):
        self.crawl_metrics: Dict[str, CrawlMetrics] = {}
        self.system_metrics_history: deque = deque(maxlen=max_history)
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self._lock = threading.RLock()
        self._alert_id_counter = 0
        
        # Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        
        # Start system metrics collection
        self._collect_system_metrics()
    
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.registry = CollectorRegistry()
        
        # Crawl metrics
        self.crawl_requests_total = Counter(
            'axiom_crawl_requests_total',
            'Total crawl requests',
            ['job_id', 'status'],
            registry=self.registry
        )
        
        self.crawl_success_rate = Gauge(
            'axiom_crawl_success_rate',
            'Crawl success rate percentage',
            ['job_id'],
            registry=self.registry
        )
        
        self.crawl_response_time = Histogram(
            'axiom_crawl_response_time_seconds',
            'Crawl response time in seconds',
            ['job_id'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry
        )
        
        self.crawl_blocked_requests = Counter(
            'axiom_crawl_blocked_requests_total',
            'Total blocked requests',
            ['job_id'],
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'axiom_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'axiom_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.active_crawlers = Gauge(
            'axiom_active_crawlers',
            'Number of active crawlers',
            registry=self.registry
        )
    
    def _collect_system_metrics(self):
        """Collect system metrics using psutil"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                active_connections=len(self.crawl_metrics),
                active_crawlers=sum(1 for m in self.crawl_metrics.values() 
                                  if m.status == CrawlStatus.RUNNING)
            )
            
            with self._lock:
                self.system_metrics_history.append(metrics)
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self.system_cpu_usage.set(cpu_percent)
                self.system_memory_usage.set(memory.percent)
                self.active_crawlers.set(metrics.active_crawlers)
            
            # Check for anomalies and trigger alerts
            self._check_anomalies(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _check_anomalies(self, metrics: SystemMetrics):
        """Check for anomalies and trigger alerts"""
        # High CPU alert
        if metrics.cpu_percent > 90:
            self.create_alert(
                severity=AlertSeverity.WARNING,
                title="High CPU Usage",
                message=f"CPU usage is at {metrics.cpu_percent:.1f}%",
                metric="cpu_percent",
                value=metrics.cpu_percent,
                threshold=90.0
            )
        
        # High memory alert
        if metrics.memory_percent > 85:
            self.create_alert(
                severity=AlertSeverity.WARNING,
                title="High Memory Usage",
                message=f"Memory usage is at {metrics.memory_percent:.1f}%",
                metric="memory_percent",
                value=metrics.memory_percent,
                threshold=85.0
            )
        
        # Check crawl metrics for anomalies
        for job_id, crawl in self.crawl_metrics.items():
            if crawl.status == CrawlStatus.RUNNING:
                # High block rate alert
                if crawl.block_rate > 30 and crawl.completed_requests > 10:
                    self.create_alert(
                        severity=AlertSeverity.CRITICAL,
                        title="High Block Rate Detected",
                        message=f"Job {job_id} has {crawl.block_rate:.1f}% block rate",
                        metric="block_rate",
                        value=crawl.block_rate,
                        threshold=30.0
                    )
                
                # Slow response time alert
                if crawl.avg_response_time > 10.0 and crawl.completed_requests > 5:
                    self.create_alert(
                        severity=AlertSeverity.WARNING,
                        title="Slow Response Times",
                        message=f"Job {job_id} avg response time: {crawl.avg_response_time:.2f}s",
                        metric="avg_response_time",
                        value=crawl.avg_response_time,
                        threshold=10.0
                    )
    
    def create_alert(self, severity: AlertSeverity, title: str, message: str,
                    metric: Optional[str] = None, value: Optional[float] = None,
                    threshold: Optional[float] = None) -> Alert:
        """Create and store a new alert"""
        with self._lock:
            self._alert_id_counter += 1
            alert = Alert(
                id=f"alert_{self._alert_id_counter}",
                timestamp=datetime.utcnow(),
                severity=severity,
                title=title,
                message=message,
                metric=metric,
                value=value,
                threshold=threshold
            )
            self.alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            logger.warning(f"Alert created: {title} - {message}")
            return alert
    
    def register_alert_callback(self, callback: Callable):
        """Register a callback for new alerts"""
        self.alert_callbacks.append(callback)
    
    def register_crawl(self, job_id: str, url: str, total_requests: int = 0) -> CrawlMetrics:
        """Register a new crawl job"""
        with self._lock:
            metrics = CrawlMetrics(
                job_id=job_id,
                url=url,
                total_requests=total_requests,
                status=CrawlStatus.RUNNING
            )
            self.crawl_metrics[job_id] = metrics
            
            if PROMETHEUS_AVAILABLE:
                self.crawl_requests_total.labels(job_id=job_id, status="total").inc(0)
            
            return metrics
    
    def update_crawl_metrics(self, job_id: str, result: CrawlResult):
        """Update metrics for a crawl job with a new result"""
        with self._lock:
            if job_id not in self.crawl_metrics:
                logger.warning(f"Crawl job {job_id} not registered")
                return
            
            metrics = self.crawl_metrics[job_id]
            metrics.update_from_result(result)
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                status = result.status.value
                self.crawl_requests_total.labels(job_id=job_id, status=status).inc()
                self.crawl_success_rate.labels(job_id=job_id).set(metrics.success_rate)
                
                if result.response_time:
                    self.crawl_response_time.labels(job_id=job_id).observe(result.response_time)
                
                if result.status == CrawlStatus.BLOCKED:
                    self.crawl_blocked_requests.labels(job_id=job_id).inc()
    
    def complete_crawl(self, job_id: str, status: CrawlStatus = CrawlStatus.COMPLETED):
        """Mark a crawl job as completed"""
        with self._lock:
            if job_id in self.crawl_metrics:
                self.crawl_metrics[job_id].status = status
                self.crawl_metrics[job_id].end_time = datetime.utcnow()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data needed for the dashboard"""
        with self._lock:
            # Update system metrics
            self._collect_system_metrics()
            
            active_crawls = []
            completed_crawls = []
            
            for metrics in self.crawl_metrics.values():
                if metrics.status in [CrawlStatus.RUNNING, CrawlStatus.PAUSED]:
                    active_crawls.append(metrics.to_dict())
                else:
                    completed_crawls.append(metrics.to_dict())
            
            # Get latest system metrics
            latest_system = self.system_metrics_history[-1].to_dict() if self.system_metrics_history else {}
            
            # Get recent alerts
            recent_alerts = [alert.to_dict() for alert in self.alerts[-10:]]
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "active_crawls": active_crawls,
                "completed_crawls": completed_crawls[-20:],  # Last 20 completed
                "system_metrics": latest_system,
                "system_metrics_history": [m.to_dict() for m in list(self.system_metrics_history)[-60:]],  # Last 60 samples
                "alerts": recent_alerts,
                "summary": {
                    "total_active": len(active_crawls),
                    "total_completed": len(completed_crawls),
                    "total_requests": sum(m.total_requests for m in self.crawl_metrics.values()),
                    "total_successful": sum(m.successful_requests for m in self.crawl_metrics.values()),
                    "total_blocked": sum(m.blocked_requests for m in self.crawl_metrics.values()),
                    "avg_success_rate": self._calculate_avg_success_rate()
                }
            }
    
    def _calculate_avg_success_rate(self) -> float:
        """Calculate average success rate across all crawls"""
        completed = [m for m in self.crawl_metrics.values() 
                    if m.status in [CrawlStatus.COMPLETED, CrawlStatus.FAILED] 
                    and m.completed_requests > 0]
        
        if not completed:
            return 0.0
        
        total_rate = sum(m.success_rate for m in completed)
        return total_rate / len(completed)
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"
        
        return generate_latest(self.registry).decode('utf-8')


class AlertManager:
    """Manages alert notifications via Slack/email"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.slack_webhook_url = self.config.get("slack_webhook_url")
        self.email_config = self.config.get("email", {})
        self.enabled = self.config.get("enabled", True)
        
        if self.slack_webhook_url and not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, Slack notifications disabled")
            self.slack_webhook_url = None
    
    async def send_alert(self, alert: Alert):
        """Send alert notification"""
        if not self.enabled:
            return
        
        try:
            if self.slack_webhook_url:
                await self._send_slack_alert(alert)
            
            if self.email_config.get("enabled"):
                await self._send_email_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        if not AIOHTTP_AVAILABLE or not self.slack_webhook_url:
            return
        
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9800",
            AlertSeverity.CRITICAL: "#ff0000"
        }.get(alert.severity, "#808080")
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"🚨 {alert.title}",
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True}
                ],
                "footer": "axiom Monitoring",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        if alert.metric:
            payload["attachments"][0]["fields"].append({
                "title": "Metric",
                "value": f"{alert.metric}: {alert.value:.2f} (threshold: {alert.threshold:.2f})",
                "short": False
            })
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.slack_webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Slack notification failed: {response.status}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send alert via email (placeholder - implement with your email service)"""
        # Implementation depends on your email service
        # This is a placeholder for the interface
        logger.info(f"Email alert would be sent: {alert.title}")


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow()
        }
        logger.info(f"WebSocket connected: {self.connection_info[websocket]['client_id']}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_info = self.connection_info.pop(websocket, {})
            logger.info(f"WebSocket disconnected: {client_info.get('client_id', 'unknown')}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)


class MonitoringDashboard:
    """Main monitoring dashboard application"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.app = FastAPI(
            title="axiom Monitoring Dashboard",
            description="Real-time monitoring for web scraping operations",
            version="1.0.0"
        )
        
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.config.get("alerts", {}))
        self.connection_manager = ConnectionManager()
        
        # Register alert callback
        self.metrics_collector.register_alert_callback(self._on_alert)
        
        # Setup FastAPI routes
        self._setup_routes()
        
        # Setup CORS
        self._setup_cors()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        logger.info("Monitoring Dashboard initialized")
    
    def _setup_cors(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict this
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve dashboard HTML page"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/dashboard")
        async def get_dashboard_data():
            """Get dashboard data as JSON"""
            return self.metrics_collector.get_dashboard_data()
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get Prometheus metrics"""
            if not PROMETHEUS_AVAILABLE:
                raise HTTPException(status_code=501, detail="Prometheus client not available")
            
            from fastapi.responses import Response
            return Response(
                content=self.metrics_collector.get_prometheus_metrics(),
                media_type=CONTENT_TYPE_LATEST
            )
        
        @self.app.get("/api/alerts")
        async def get_alerts(limit: int = 50):
            """Get recent alerts"""
            with self.metrics_collector._lock:
                alerts = [alert.to_dict() for alert in self.metrics_collector.alerts[-limit:]]
            return {"alerts": alerts}
        
        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge an alert"""
            with self.metrics_collector._lock:
                for alert in self.metrics_collector.alerts:
                    if alert.id == alert_id:
                        alert.acknowledged = True
                        return {"status": "acknowledged"}
            
            raise HTTPException(status_code=404, detail="Alert not found")
        
        @self.app.websocket("/ws/dashboard")
        async def websocket_dashboard(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.connection_manager.connect(websocket)
            
            try:
                # Send initial data
                initial_data = self.metrics_collector.get_dashboard_data()
                await self.connection_manager.send_personal_message(
                    {"type": "initial", "data": initial_data},
                    websocket
                )
                
                # Keep connection alive and handle messages
                while True:
                    try:
                        # Wait for messages (like ping/pong or commands)
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        
                        if message.get("type") == "ping":
                            await self.connection_manager.send_personal_message(
                                {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                                websocket
                            )
                        
                        elif message.get("type") == "command":
                            await self._handle_websocket_command(message, websocket)
                            
                    except WebSocketDisconnect:
                        break
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON received from WebSocket client")
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                        break
                        
            finally:
                self.connection_manager.disconnect(websocket)
        
        @self.app.post("/api/crawl/register")
        async def register_crawl(job_id: str, url: str, total_requests: int = 0):
            """Register a new crawl job for monitoring"""
            metrics = self.metrics_collector.register_crawl(job_id, url, total_requests)
            return {"status": "registered", "metrics": metrics.to_dict()}
        
        @self.app.post("/api/crawl/{job_id}/update")
        async def update_crawl_metrics(job_id: str, result: Dict[str, Any]):
            """Update crawl metrics with a new result"""
            # Convert dict to CrawlResult (simplified - you might need proper validation)
            crawl_result = CrawlResult(
                url=result.get("url", ""),
                status=CrawlStatus(result.get("status", "failed")),
                response_time=result.get("response_time"),
                depth=result.get("depth", 0)
            )
            
            self.metrics_collector.update_crawl_metrics(job_id, crawl_result)
            return {"status": "updated"}
        
        @self.app.post("/api/crawl/{job_id}/complete")
        async def complete_crawl(job_id: str, status: str = "completed"):
            """Mark a crawl job as completed"""
            try:
                crawl_status = CrawlStatus(status)
            except ValueError:
                crawl_status = CrawlStatus.COMPLETED
            
            self.metrics_collector.complete_crawl(job_id, crawl_status)
            return {"status": "completed"}
    
    async def _handle_websocket_command(self, message: Dict[str, Any], websocket: WebSocket):
        """Handle commands from WebSocket clients"""
        command = message.get("command")
        
        if command == "get_metrics":
            data = self.metrics_collector.get_dashboard_data()
            await self.connection_manager.send_personal_message(
                {"type": "metrics", "data": data},
                websocket
            )
        
        elif command == "get_alerts":
            with self.metrics_collector._lock:
                alerts = [alert.to_dict() for alert in self.metrics_collector.alerts[-20:]]
            await self.connection_manager.send_personal_message(
                {"type": "alerts", "data": alerts},
                websocket
            )
    
    def _on_alert(self, alert: Alert):
        """Callback for new alerts"""
        # Send to WebSocket clients
        asyncio.create_task(
            self.connection_manager.broadcast({
                "type": "alert",
                "data": alert.to_dict()
            })
        )
        
        # Send notifications
        asyncio.create_task(self.alert_manager.send_alert(alert))
    
    async def _broadcast_updates(self):
        """Background task to broadcast updates to WebSocket clients"""
        while True:
            try:
                if self.connection_manager.active_connections:
                    data = self.metrics_collector.get_dashboard_data()
                    await self.connection_manager.broadcast({
                        "type": "update",
                        "data": data
                    })
                
                await asyncio.sleep(self.config.get("update_interval", 2))
                
            except Exception as e:
                logger.error(f"Error in broadcast task: {e}")
                await asyncio.sleep(5)
    
    async def _collect_metrics_periodically(self):
        """Background task to collect system metrics periodically"""
        while True:
            try:
                self.metrics_collector._collect_system_metrics()
                await asyncio.sleep(self.config.get("metrics_interval", 5))
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(10)
    
    def start_background_tasks(self):
        """Start background tasks"""
        self._background_tasks.append(
            asyncio.create_task(self._broadcast_updates())
        )
        self._background_tasks.append(
            asyncio.create_task(self._collect_metrics_periodically())
        )
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>axiom Monitoring Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    background: #0f172a;
                    color: #e2e8f0;
                    line-height: 1.6;
                }
                
                .container {
                    max-width: 1600px;
                    margin: 0 auto;
                    padding: 20px;
                }
                
                header {
                    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                    padding: 20px 0;
                    border-bottom: 1px solid #334155;
                    margin-bottom: 30px;
                }
                
                h1 {
                    font-size: 2.5rem;
                    background: linear-gradient(90deg, #38bdf8, #818cf8);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 10px;
                }
                
                .subtitle {
                    color: #94a3b8;
                    font-size: 1.1rem;
                }
                
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                
                .card {
                    background: #1e293b;
                    border-radius: 12px;
                    padding: 20px;
                    border: 1px solid #334155;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }
                
                .card-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #334155;
                }
                
                .card-title {
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: #f1f5f9;
                }
                
                .stat-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                }
                
                .stat {
                    text-align: center;
                    padding: 15px;
                    background: #0f172a;
                    border-radius: 8px;
                }
                
                .stat-value {
                    font-size: 2rem;
                    font-weight: 700;
                    color: #38bdf8;
                    margin-bottom: 5px;
                }
                
                .stat-label {
                    font-size: 0.9rem;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                
                .progress-bar {
                    height: 8px;
                    background: #334155;
                    border-radius: 4px;
                    overflow: hidden;
                    margin: 10px 0;
                }
                
                .progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #22c55e, #10b981);
                    transition: width 0.3s ease;
                }
                
                .alert-list {
                    max-height: 400px;
                    overflow-y: auto;
                }
                
                .alert-item {
                    padding: 12px;
                    margin-bottom: 8px;
                    border-radius: 6px;
                    border-left: 4px solid;
                }
                
                .alert-critical {
                    background: rgba(239, 68, 68, 0.1);
                    border-color: #ef4444;
                }
                
                .alert-warning {
                    background: rgba(245, 158, 11, 0.1);
                    border-color: #f59e0b;
                }
                
                .alert-info {
                    background: rgba(59, 130, 246, 0.1);
                    border-color: #3b82f6;
                }
                
                .alert-title {
                    font-weight: 600;
                    margin-bottom: 5px;
                }
                
                .alert-message {
                    font-size: 0.9rem;
                    color: #cbd5e1;
                }
                
                .alert-time {
                    font-size: 0.8rem;
                    color: #64748b;
                    margin-top: 5px;
                }
                
                .chart-container {
                    position: relative;
                    height: 300px;
                    margin-top: 20px;
                }
                
                .status-indicator {
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                
                .status-running {
                    background: #22c55e;
                    animation: pulse 2s infinite;
                }
                
                .status-paused {
                    background: #f59e0b;
                }
                
                .status-completed {
                    background: #3b82f6;
                }
                
                .status-failed {
                    background: #ef4444;
                }
                
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                
                .crawl-item {
                    background: #0f172a;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    border: 1px solid #334155;
                }
                
                .crawl-header {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                }
                
                .crawl-url {
                    font-weight: 600;
                    color: #f1f5f9;
                    word-break: break-all;
                }
                
                .crawl-stats {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 10px;
                    font-size: 0.9rem;
                }
                
                .connection-status {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    padding: 10px 15px;
                    background: #1e293b;
                    border-radius: 8px;
                    border: 1px solid #334155;
                    display: flex;
                    align-items: center;
                    font-size: 0.9rem;
                }
                
                .connected {
                    color: #22c55e;
                }
                
                .disconnected {
                    color: #ef4444;
                }
                
                @media (max-width: 768px) {
                    .grid {
                        grid-template-columns: 1fr;
                    }
                    
                    .stat-grid {
                        grid-template-columns: 1fr;
                    }
                    
                    .crawl-stats {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <header>
                <div class="container">
                    <h1>🕷️ axiom Monitoring Dashboard</h1>
                    <p class="subtitle">Real-time monitoring for web scraping operations</p>
                </div>
            </header>
            
            <div class="container">
                <div class="grid">
                    <!-- System Metrics -->
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">System Resources</h2>
                            <span id="last-updated" style="color: #64748b; font-size: 0.9rem;"></span>
                        </div>
                        <div class="stat-grid">
                            <div class="stat">
                                <div class="stat-value" id="cpu-usage">0%</div>
                                <div class="stat-label">CPU Usage</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="memory-usage">0%</div>
                                <div class="stat-label">Memory Usage</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="active-crawlers">0</div>
                                <div class="stat-label">Active Crawlers</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="active-connections">0</div>
                                <div class="stat-label">Connections</div>
                            </div>
                        </div>
                        <div class="chart-container">
                            <canvas id="system-chart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Crawl Summary -->
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Crawl Summary</h2>
                        </div>
                        <div class="stat-grid">
                            <div class="stat">
                                <div class="stat-value" id="total-requests">0</div>
                                <div class="stat-label">Total Requests</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="success-rate">0%</div>
                                <div class="stat-label">Success Rate</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="blocked-requests">0</div>
                                <div class="stat-label">Blocked</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="avg-response-time">0s</div>
                                <div class="stat-label">Avg Response</div>
                            </div>
                        </div>
                        <div class="chart-container">
                            <canvas id="crawl-chart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Active Crawls -->
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Active Crawls</h2>
                            <span id="active-crawl-count" style="color: #38bdf8;">0 active</span>
                        </div>
                        <div id="active-crawls-list">
                            <div style="text-align: center; color: #64748b; padding: 40px;">
                                No active crawls
                            </div>
                        </div>
                    </div>
                    
                    <!-- Alerts -->
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Recent Alerts</h2>
                            <span id="alert-count" style="color: #f59e0b;">0 alerts</span>
                        </div>
                        <div class="alert-list" id="alerts-list">
                            <div style="text-align: center; color: #64748b; padding: 40px;">
                                No recent alerts
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Activity -->
                <div class="card">
                    <div class="card-header">
                        <h2 class="card-title">Recent Activity</h2>
                    </div>
                    <div id="recent-activity" style="max-height: 400px; overflow-y: auto;">
                        <div style="text-align: center; color: #64748b; padding: 40px;">
                            No recent activity
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="connection-status">
                <span class="status-indicator" id="ws-status"></span>
                <span id="ws-status-text">Connecting...</span>
            </div>
            
            <script>
                // Dashboard JavaScript
                class Dashboard {
                    constructor() {
                        this.ws = null;
                        this.reconnectAttempts = 0;
                        this.maxReconnectAttempts = 5;
                        this.charts = {};
                        this.data = {
                            systemMetrics: [],
                            crawlMetrics: [],
                            alerts: []
                        };
                        
                        this.initCharts();
                        this.connect();
                        this.setupEventListeners();
                    }
                    
                    initCharts() {
                        // System metrics chart
                        const systemCtx = document.getElementById('system-chart').getContext('2d');
                        this.charts.system = new Chart(systemCtx, {
                            type: 'line',
                            data: {
                                labels: [],
                                datasets: [
                                    {
                                        label: 'CPU %',
                                        data: [],
                                        borderColor: '#38bdf8',
                                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                                        tension: 0.4
                                    },
                                    {
                                        label: 'Memory %',
                                        data: [],
                                        borderColor: '#818cf8',
                                        backgroundColor: 'rgba(129, 140, 248, 0.1)',
                                        tension: 0.4
                                    }
                                ]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: {
                                            color: '#e2e8f0'
                                        }
                                    }
                                },
                                scales: {
                                    x: {
                                        ticks: { color: '#94a3b8' },
                                        grid: { color: '#334155' }
                                    },
                                    y: {
                                        ticks: { color: '#94a3b8' },
                                        grid: { color: '#334155' },
                                        min: 0,
                                        max: 100
                                    }
                                }
                            }
                        });
                        
                        // Crawl metrics chart
                        const crawlCtx = document.getElementById('crawl-chart').getContext('2d');
                        this.charts.crawl = new Chart(crawlCtx, {
                            type: 'bar',
                            data: {
                                labels: ['Successful', 'Failed', 'Blocked'],
                                datasets: [{
                                    label: 'Requests',
                                    data: [0, 0, 0],
                                    backgroundColor: [
                                        'rgba(34, 197, 94, 0.8)',
                                        'rgba(239, 68, 68, 0.8)',
                                        'rgba(245, 158, 11, 0.8)'
                                    ],
                                    borderColor: [
                                        '#22c55e',
                                        '#ef4444',
                                        '#f59e0b'
                                    ],
                                    borderWidth: 1
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: {
                                            color: '#e2e8f0'
                                        }
                                    }
                                },
                                scales: {
                                    x: {
                                        ticks: { color: '#94a3b8' },
                                        grid: { color: '#334155' }
                                    },
                                    y: {
                                        ticks: { color: '#94a3b8' },
                                        grid: { color: '#334155' },
                                        beginAtZero: true
                                    }
                                }
                            }
                        });
                    }
                    
                    connect() {
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;
                        
                        this.ws = new WebSocket(wsUrl);
                        
                        this.ws.onopen = () => {
                            console.log('WebSocket connected');
                            this.reconnectAttempts = 0;
                            this.updateConnectionStatus(true);
                            
                            // Send initial ping
                            this.send({ type: 'ping' });
                        };
                        
                        this.ws.onmessage = (event) => {
                            try {
                                const message = JSON.parse(event.data);
                                this.handleMessage(message);
                            } catch (e) {
                                console.error('Error parsing message:', e);
                            }
                        };
                        
                        this.ws.onclose = () => {
                            console.log('WebSocket disconnected');
                            this.updateConnectionStatus(false);
                            this.scheduleReconnect();
                        };
                        
                        this.ws.onerror = (error) => {
                            console.error('WebSocket error:', error);
                        };
                    }
                    
                    scheduleReconnect() {
                        if (this.reconnectAttempts < this.maxReconnectAttempts) {
                            this.reconnectAttempts++;
                            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
                            console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
                            setTimeout(() => this.connect(), delay);
                        } else {
                            console.error('Max reconnection attempts reached');
                        }
                    }
                    
                    send(message) {
                        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                            this.ws.send(JSON.stringify(message));
                        }
                    }
                    
                    handleMessage(message) {
                        switch (message.type) {
                            case 'initial':
                            case 'update':
                                this.updateDashboard(message.data);
                                break;
                            case 'alert':
                                this.addAlert(message.data);
                                break;
                            case 'pong':
                                // Connection is alive
                                break;
                            default:
                                console.log('Unknown message type:', message.type);
                        }
                    }
                    
                    updateDashboard(data) {
                        // Update system metrics
                        if (data.system_metrics) {
                            this.updateSystemMetrics(data.system_metrics);
                        }
                        
                        // Update system chart
                        if (data.system_metrics_history) {
                            this.updateSystemChart(data.system_metrics_history);
                        }
                        
                        // Update summary
                        if (data.summary) {
                            this.updateSummary(data.summary);
                        }
                        
                        // Update active crawls
                        if (data.active_crawls) {
                            this.updateActiveCrawls(data.active_crawls);
                        }
                        
                        // Update alerts
                        if (data.alerts) {
                            this.updateAlerts(data.alerts);
                        }
                        
                        // Update last updated time
                        document.getElementById('last-updated').textContent = 
                            `Last updated: ${new Date().toLocaleTimeString()}`;
                    }
                    
                    updateSystemMetrics(metrics) {
                        document.getElementById('cpu-usage').textContent = 
                            `${metrics.cpu_percent?.toFixed(1) || 0}%`;
                        document.getElementById('memory-usage').textContent = 
                            `${metrics.memory_percent?.toFixed(1) || 0}%`;
                        document.getElementById('active-crawlers').textContent = 
                            metrics.active_crawlers || 0;
                        document.getElementById('active-connections').textContent = 
                            metrics.active_connections || 0;
                    }
                    
                    updateSystemChart(history) {
                        const labels = history.map(m => {
                            const date = new Date(m.timestamp);
                            return date.toLocaleTimeString();
                        }).slice(-30);
                        
                        const cpuData = history.map(m => m.cpu_percent).slice(-30);
                        const memoryData = history.map(m => m.memory_percent).slice(-30);
                        
                        this.charts.system.data.labels = labels;
                        this.charts.system.data.datasets[0].data = cpuData;
                        this.charts.system.data.datasets[1].data = memoryData;
                        this.charts.system.update('none');
                    }
                    
                    updateSummary(summary) {
                        document.getElementById('total-requests').textContent = 
                            summary.total_requests?.toLocaleString() || 0;
                        document.getElementById('success-rate').textContent = 
                            `${summary.avg_success_rate?.toFixed(1) || 0}%`;
                        document.getElementById('blocked-requests').textContent = 
                            summary.total_blocked?.toLocaleString() || 0;
                        
                        // Calculate average response time from active crawls
                        let avgResponse = 0;
                        if (this.data.crawlMetrics && this.data.crawlMetrics.length > 0) {
                            const totalAvg = this.data.crawlMetrics.reduce((sum, crawl) => 
                                sum + (crawl.avg_response_time || 0), 0);
                            avgResponse = totalAvg / this.data.crawlMetrics.length;
                        }
                        document.getElementById('avg-response-time').textContent = 
                            `${avgResponse.toFixed(2)}s`;
                        
                        // Update crawl chart
                        this.charts.crawl.data.datasets[0].data = [
                            summary.total_successful || 0,
                            (summary.total_requests - summary.total_successful - summary.total_blocked) || 0,
                            summary.total_blocked || 0
                        ];
                        this.charts.crawl.update('none');
                    }
                    
                    updateActiveCrawls(crawls) {
                        const container = document.getElementById('active-crawls-list');
                        const countElement = document.getElementById('active-crawl-count');
                        
                        countElement.textContent = `${crawls.length} active`;
                        
                        if (crawls.length === 0) {
                            container.innerHTML = `
                                <div style="text-align: center; color: #64748b; padding: 40px;">
                                    No active crawls
                                </div>
                            `;
                            return;
                        }
                        
                        container.innerHTML = crawls.map(crawl => `
                            <div class="crawl-item">
                                <div class="crawl-header">
                                    <div class="crawl-url">${this.truncateUrl(crawl.url)}</div>
                                    <div>
                                        <span class="status-indicator status-${crawl.status}"></span>
                                        ${crawl.status}
                                    </div>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${crawl.progress_percent}%"></div>
                                </div>
                                <div class="crawl-stats">
                                    <div>
                                        <strong>${crawl.completed_requests}</strong> / ${crawl.total_requests}
                                        <div style="font-size: 0.8rem; color: #64748b;">Requests</div>
                                    </div>
                                    <div>
                                        <strong>${crawl.success_rate?.toFixed(1)}%</strong>
                                        <div style="font-size: 0.8rem; color: #64748b;">Success</div>
                                    </div>
                                    <div>
                                        <strong>${crawl.requests_per_second?.toFixed(2)}/s</strong>
                                        <div style="font-size: 0.8rem; color: #64748b;">Speed</div>
                                    </div>
                                </div>
                            </div>
                        `).join('');
                    }
                    
                    updateAlerts(alerts) {
                        const container = document.getElementById('alerts-list');
                        const countElement = document.getElementById('alert-count');
                        
                        countElement.textContent = `${alerts.length} alerts`;
                        
                        if (alerts.length === 0) {
                            container.innerHTML = `
                                <div style="text-align: center; color: #64748b; padding: 40px;">
                                    No recent alerts
                                </div>
                            `;
                            return;
                        }
                        
                        container.innerHTML = alerts.map(alert => `
                            <div class="alert-item alert-${alert.severity}">
                                <div class="alert-title">${alert.title}</div>
                                <div class="alert-message">${alert.message}</div>
                                <div class="alert-time">
                                    ${new Date(alert.timestamp).toLocaleString()}
                                    ${alert.acknowledged ? ' • Acknowledged' : ''}
                                </div>
                            </div>
                        `).join('');
                    }
                    
                    addAlert(alert) {
                        this.data.alerts.unshift(alert);
                        if (this.data.alerts.length > 20) {
                            this.data.alerts = this.data.alerts.slice(0, 20);
                        }
                        this.updateAlerts(this.data.alerts);
                        
                        // Show browser notification if permitted
                        if (Notification.permission === 'granted') {
                            new Notification(`axiom Alert: ${alert.title}`, {
                                body: alert.message,
                                icon: '/favicon.ico'
                            });
                        }
                    }
                    
                    truncateUrl(url, maxLength = 50) {
                        if (url.length <= maxLength) return url;
                        return url.substring(0, maxLength) + '...';
                    }
                    
                    updateConnectionStatus(connected) {
                        const statusElement = document.getElementById('ws-status');
                        const textElement = document.getElementById('ws-status-text');
                        
                        if (connected) {
                            statusElement.className = 'status-indicator connected';
                            textElement.textContent = 'Connected';
                            textElement.className = 'connected';
                        } else {
                            statusElement.className = 'status-indicator disconnected';
                            textElement.textContent = 'Disconnected';
                            textElement.className = 'disconnected';
                        }
                    }
                    
                    setupEventListeners() {
                        // Request notification permission
                        if ('Notification' in window && Notification.permission === 'default') {
                            Notification.requestPermission();
                        }
                        
                        // Handle visibility change
                        document.addEventListener('visibilitychange', () => {
                            if (document.hidden) {
                                // Page is hidden, could reduce update frequency
                            } else {
                                // Page is visible, ensure we're connected
                                if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                                    this.connect();
                                }
                            }
                        });
                    }
                }
                
                // Initialize dashboard when page loads
                document.addEventListener('DOMContentLoaded', () => {
                    window.dashboard = new Dashboard();
                });
            </script>
        </body>
        </html>
        """
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the dashboard server"""
        self.start_background_tasks()
        
        try:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level="info",
                **kwargs
            )
        finally:
            self.stop_background_tasks()


# Integration with existing axiom components
class CrawlMonitor:
    """Wrapper to integrate monitoring with existing crawl sessions"""
    
    def __init__(self, dashboard_url: str = "http://localhost:8000"):
        self.dashboard_url = dashboard_url
        self.active_jobs: Dict[str, str] = {}  # job_id -> session_id
    
    async def register_job(self, job_id: str, url: str, total_requests: int = 0):
        """Register a crawl job with the monitoring dashboard"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.dashboard_url}/api/crawl/register",
                    params={
                        "job_id": job_id,
                        "url": url,
                        "total_requests": total_requests
                    }
                ) as response:
                    if response.status == 200:
                        self.active_jobs[job_id] = url
                        logger.info(f"Registered crawl job {job_id} with monitoring dashboard")
                    else:
                        logger.warning(f"Failed to register crawl job: {response.status}")
        except Exception as e:
            logger.error(f"Error registering with dashboard: {e}")
    
    async def report_result(self, job_id: str, result: CrawlResult):
        """Report a crawl result to the monitoring dashboard"""
        if job_id not in self.active_jobs:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.dashboard_url}/api/crawl/{job_id}/update",
                    json={
                        "url": result.url,
                        "status": result.status.value,
                        "response_time": result.response_time,
                        "depth": result.depth
                    }
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to report result: {response.status}")
        except Exception as e:
            logger.error(f"Error reporting to dashboard: {e}")
    
    async def complete_job(self, job_id: str, status: CrawlStatus = CrawlStatus.COMPLETED):
        """Mark a crawl job as completed in the monitoring dashboard"""
        if job_id not in self.active_jobs:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.dashboard_url}/api/crawl/{job_id}/complete",
                    params={"status": status.value}
                ) as response:
                    if response.status == 200:
                        del self.active_jobs[job_id]
                        logger.info(f"Completed crawl job {job_id} in monitoring dashboard")
        except Exception as e:
            logger.error(f"Error completing job in dashboard: {e}")


# Factory function for easy integration
def create_monitoring_dashboard(config: Optional[Dict[str, Any]] = None) -> MonitoringDashboard:
    """Create and configure a monitoring dashboard instance"""
    return MonitoringDashboard(config)


# Example configuration
DEFAULT_CONFIG = {
    "update_interval": 2,  # seconds between WebSocket updates
    "metrics_interval": 5,  # seconds between system metrics collection
    "alerts": {
        "enabled": True,
        "slack_webhook_url": None,  # Set via environment variable
        "email": {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": None,
            "sender_password": None,
            "recipients": []
        }
    }
}


if __name__ == "__main__":
    # Run dashboard standalone
    import argparse
    
    parser = argparse.ArgumentParser(description="axiom Monitoring Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    # Create and run dashboard
    dashboard = create_monitoring_dashboard(config)
    dashboard.run(host=args.host, port=args.port)