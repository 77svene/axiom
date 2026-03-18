import asyncio
import json
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from urllib.parse import urlparse

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from axiom.core.custom_types import CrawlResult
from axiom.core.utils._utils import get_logger

logger = get_logger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(Enum):
    CRAWL_SLOWDOWN = "crawl_slowdown"
    HIGH_FAILURE_RATE = "high_failure_rate"
    BLOCKED_REQUESTS = "blocked_requests"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    TARGET_UNREACHABLE = "target_unreachable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

@dataclass
class Alert:
    id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self):
        return {
            **asdict(self),
            "type": self.type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class CrawlMetrics:
    crawl_id: str
    start_time: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    total_response_time: float = 0.0
    total_bytes: int = 0
    urls_crawled: Set[str] = field(default_factory=set)
    urls_failed: Set[str] = field(default_factory=set)
    urls_blocked: Set[str] = field(default_factory=set)
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    @property
    def block_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.blocked_requests / self.total_requests) * 100
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def requests_per_second(self) -> float:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return 0.0
        return self.total_requests / elapsed
    
    def update_from_result(self, result: CrawlResult):
        self.total_requests += 1
        self.urls_crawled.add(result.url)
        
        if result.success:
            self.successful_requests += 1
            self.total_response_time += result.response_time or 0
            self.total_bytes += len(result.content or "")
            if result.response_time:
                self.response_times.append(result.response_time)
            if result.status_code:
                self.status_codes[result.status_code] += 1
        else:
            self.failed_requests += 1
            self.urls_failed.add(result.url)
            
            if "blocked" in str(result.error).lower() or result.status_code in [403, 429]:
                self.blocked_requests += 1
                self.urls_blocked.add(result.url)

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int = 0
    open_files: int = 0
    
    def to_dict(self):
        return asdict(self)

class ConnectionManager:
    """Manages WebSocket connections for the dashboard"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[str, Set[str]] = defaultdict(set)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Dashboard client connected: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_subscriptions:
            del self.connection_subscriptions[client_id]
        logger.info(f"Dashboard client disconnected: {client_id}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict, exclude: Optional[Set[str]] = None):
        disconnected = set()
        for client_id, connection in self.active_connections.items():
            if exclude and client_id in exclude:
                continue
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.add(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)
    
    def subscribe(self, client_id: str, topics: List[str]):
        self.connection_subscriptions[client_id].update(topics)
    
    def unsubscribe(self, client_id: str, topics: List[str]):
        self.connection_subscriptions[client_id].difference_update(topics)

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rules: List[Dict] = []
        self.notification_handlers: List[Callable] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default alert rules"""
        self.alert_rules = [
            {
                "type": AlertType.HIGH_FAILURE_RATE,
                "condition": lambda m: m.failure_rate > 20,
                "severity": AlertSeverity.WARNING,
                "message": lambda m: f"High failure rate detected: {m.failure_rate:.1f}%"
            },
            {
                "type": AlertType.BLOCKED_REQUESTS,
                "condition": lambda m: m.block_rate > 10,
                "severity": AlertSeverity.WARNING,
                "message": lambda m: f"High block rate detected: {m.block_rate:.1f}%"
            },
            {
                "type": AlertType.CRAWL_SLOWDOWN,
                "condition": lambda m: m.avg_response_time > 5.0,
                "severity": AlertSeverity.INFO,
                "message": lambda m: f"Crawl slowdown detected: {m.avg_response_time:.2f}s avg response time"
            }
        ]
    
    def add_rule(self, rule: Dict):
        """Add a custom alert rule"""
        self.alert_rules.append(rule)
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler (Slack, email, etc.)"""
        self.notification_handlers.append(handler)
    
    def check_metrics(self, crawl_metrics: CrawlMetrics, system_metrics: Optional[SystemMetrics] = None):
        """Check metrics against alert rules"""
        for rule in self.alert_rules:
            try:
                if rule["condition"](crawl_metrics):
                    alert_id = f"{rule['type'].value}_{crawl_metrics.crawl_id}_{int(time.time())}"
                    if alert_id not in self.alerts:
                        alert = Alert(
                            id=alert_id,
                            type=rule["type"],
                            severity=rule["severity"],
                            message=rule["message"](crawl_metrics),
                            timestamp=datetime.now(),
                            source=crawl_metrics.crawl_id,
                            metadata={
                                "crawl_id": crawl_metrics.crawl_id,
                                "metrics": {
                                    "success_rate": crawl_metrics.success_rate,
                                    "failure_rate": crawl_metrics.failure_rate,
                                    "block_rate": crawl_metrics.block_rate,
                                    "avg_response_time": crawl_metrics.avg_response_time
                                }
                            }
                        )
                        self.alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        self._trigger_notifications(alert)
            except Exception as e:
                logger.error(f"Error checking alert rule: {e}")
        
        if system_metrics:
            self._check_system_metrics(system_metrics)
    
    def _check_system_metrics(self, metrics: SystemMetrics):
        """Check system metrics for alerts"""
        if metrics.memory_percent > 90:
            alert_id = f"high_memory_{int(time.time())}"
            if alert_id not in self.alerts:
                alert = Alert(
                    id=alert_id,
                    type=AlertType.MEMORY_USAGE,
                    severity=AlertSeverity.CRITICAL,
                    message=f"High memory usage: {metrics.memory_percent:.1f}%",
                    timestamp=datetime.now(),
                    source="system",
                    metadata={"memory_percent": metrics.memory_percent}
                )
                self.alerts[alert_id] = alert
                self.alert_history.append(alert)
                self._trigger_notifications(alert)
        
        if metrics.cpu_percent > 80:
            alert_id = f"high_cpu_{int(time.time())}"
            if alert_id not in self.alerts:
                alert = Alert(
                    id=alert_id,
                    type=AlertType.CPU_USAGE,
                    severity=AlertSeverity.WARNING,
                    message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    timestamp=datetime.now(),
                    source="system",
                    metadata={"cpu_percent": metrics.cpu_percent}
                )
                self.alerts[alert_id] = alert
                self.alert_history.append(alert)
                self._trigger_notifications(alert)
    
    def _trigger_notifications(self, alert: Alert):
        """Trigger all notification handlers"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str = "system"):
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            self.alerts[alert_id].metadata["acknowledged_by"] = user
            self.alerts[alert_id].metadata["acknowledged_at"] = datetime.now().isoformat()
    
    def resolve_alert(self, alert_id: str, user: str = "system"):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].metadata["resolved_by"] = user
            self.alerts[alert_id].metadata["resolved_at"] = datetime.now().isoformat()
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff]

class MetricsCollector:
    """Collects and aggregates metrics from crawls"""
    
    def __init__(self):
        self.crawl_metrics: Dict[str, CrawlMetrics] = {}
        self.system_metrics_history: deque = deque(maxlen=3600)  # Keep 1 hour of system metrics
        self.global_metrics = {
            "total_crawls": 0,
            "total_requests": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_blocked": 0,
            "total_bytes": 0,
            "start_time": datetime.now()
        }
        self._lock = threading.RLock()
    
    def start_crawl(self, crawl_id: str) -> CrawlMetrics:
        """Start tracking a new crawl"""
        with self._lock:
            metrics = CrawlMetrics(
                crawl_id=crawl_id,
                start_time=datetime.now()
            )
            self.crawl_metrics[crawl_id] = metrics
            self.global_metrics["total_crawls"] += 1
            return metrics
    
    def update_crawl(self, crawl_id: str, result: CrawlResult):
        """Update crawl metrics with a new result"""
        with self._lock:
            if crawl_id in self.crawl_metrics:
                self.crawl_metrics[crawl_id].update_from_result(result)
                
                # Update global metrics
                self.global_metrics["total_requests"] += 1
                if result.success:
                    self.global_metrics["total_successful"] += 1
                    self.global_metrics["total_bytes"] += len(result.content or "")
                else:
                    self.global_metrics["total_failed"] += 1
                    if "blocked" in str(result.error).lower() or result.status_code in [403, 429]:
                        self.global_metrics["total_blocked"] += 1
    
    def end_crawl(self, crawl_id: str):
        """End tracking for a crawl"""
        with self._lock:
            if crawl_id in self.crawl_metrics:
                # Keep metrics for historical data
                pass
    
    def get_crawl_metrics(self, crawl_id: str) -> Optional[CrawlMetrics]:
        """Get metrics for a specific crawl"""
        return self.crawl_metrics.get(crawl_id)
    
    def get_all_crawl_metrics(self) -> Dict[str, CrawlMetrics]:
        """Get metrics for all crawls"""
        return self.crawl_metrics.copy()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        if not PSUTIL_AVAILABLE:
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_sent_mb=net_io.bytes_sent / (1024 * 1024),
                network_recv_mb=net_io.bytes_recv / (1024 * 1024),
                active_connections=len(psutil.net_connections()),
                open_files=len(psutil.Process().open_files())
            )
            
            self.system_metrics_history.append(metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
    
    def get_system_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get system metrics history"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.system_metrics_history if m.timestamp > cutoff]
    
    def get_global_summary(self) -> Dict:
        """Get global metrics summary"""
        with self._lock:
            uptime = (datetime.now() - self.global_metrics["start_time"]).total_seconds()
            return {
                **self.global_metrics,
                "uptime_seconds": uptime,
                "success_rate": (
                    self.global_metrics["total_successful"] / 
                    max(self.global_metrics["total_requests"], 1) * 100
                ),
                "active_crawls": len(self.crawl_metrics),
                "requests_per_second": (
                    self.global_metrics["total_requests"] / max(uptime, 1)
                )
            }

class GrafanaIntegration:
    """Integration with Grafana/Prometheus"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.prometheus_metrics = {}
        self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics format"""
        # This would typically use prometheus_client library
        # For now, we'll create a simple metrics endpoint
        pass
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics format"""
        lines = []
        
        # Global metrics
        global_summary = self.metrics_collector.get_global_summary()
        lines.append(f'# HELP axiom_requests_total Total number of requests')
        lines.append(f'# TYPE axiom_requests_total counter')
        lines.append(f'axiom_requests_total {global_summary["total_requests"]}')
        
        lines.append(f'# HELP axiom_requests_successful_total Total successful requests')
        lines.append(f'# TYPE axiom_requests_successful_total counter')
        lines.append(f'axiom_requests_successful_total {global_summary["total_successful"]}')
        
        lines.append(f'# HELP axiom_requests_failed_total Total failed requests')
        lines.append(f'# TYPE axiom_requests_failed_total counter')
        lines.append(f'axiom_requests_failed_total {global_summary["total_failed"]}')
        
        lines.append(f'# HELP axiom_requests_blocked_total Total blocked requests')
        lines.append(f'# TYPE axiom_requests_blocked_total counter')
        lines.append(f'axiom_requests_blocked_total {global_summary["total_blocked"]}')
        
        lines.append(f'# HELP axiom_bytes_total Total bytes downloaded')
        lines.append(f'# TYPE axiom_bytes_total counter')
        lines.append(f'axiom_bytes_total {global_summary["total_bytes"]}')
        
        # System metrics
        if PSUTIL_AVAILABLE:
            system_metrics = self.metrics_collector.collect_system_metrics()
            lines.append(f'# HELP axiom_cpu_percent CPU usage percentage')
            lines.append(f'# TYPE axiom_cpu_percent gauge')
            lines.append(f'axiom_cpu_percent {system_metrics.cpu_percent}')
            
            lines.append(f'# HELP axiom_memory_percent Memory usage percentage')
            lines.append(f'# TYPE axiom_memory_percent gauge')
            lines.append(f'axiom_memory_percent {system_metrics.memory_percent}')
        
        return '\n'.join(lines)
    
    def get_grafana_dashboard_json(self) -> Dict:
        """Generate Grafana dashboard JSON"""
        return {
            "dashboard": {
                "title": "axiom Monitoring Dashboard",
                "panels": [
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(axiom_requests_total[5m])",
                            "legendFormat": "Requests/sec"
                        }]
                    },
                    {
                        "title": "Success Rate",
                        "type": "gauge",
                        "targets": [{
                            "expr": "axiom_requests_successful_total / axiom_requests_total * 100",
                            "legendFormat": "Success Rate %"
                        }]
                    },
                    {
                        "title": "Blocked Requests",
                        "type": "stat",
                        "targets": [{
                            "expr": "axiom_requests_blocked_total",
                            "legendFormat": "Blocked"
                        }]
                    },
                    {
                        "title": "Response Times",
                        "type": "heatmap",
                        "targets": [{
                            "expr": "axiom_response_time_seconds",
                            "legendFormat": "Response Time"
                        }]
                    }
                ]
            }
        }

class MonitoringDashboard:
    """Main monitoring dashboard class"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for the monitoring dashboard. Install with: pip install fastapi uvicorn")
        
        self.host = host
        self.port = port
        self.app = FastAPI(title="axiom Monitoring Dashboard")
        self.connection_manager = ConnectionManager()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.grafana_integration = GrafanaIntegration(self.metrics_collector)
        
        self._setup_routes()
        self._setup_background_tasks()
        
        # Add default notification handlers
        self.alert_manager.add_notification_handler(self._handle_alert_notification)
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def dashboard():
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>axiom Monitoring Dashboard</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                    .card { background: #f5f5f5; padding: 20px; border-radius: 8px; }
                    .metric { margin: 10px 0; }
                    .metric-value { font-size: 24px; font-weight: bold; }
                    .metric-label { color: #666; }
                    .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
                    .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; }
                    .alert-critical { background: #f8d7da; border: 1px solid #f5c6cb; }
                </style>
            </head>
            <body>
                <h1>axiom Monitoring Dashboard</h1>
                <div class="dashboard">
                    <div class="card">
                        <h3>Global Metrics</h3>
                        <div id="global-metrics"></div>
                    </div>
                    <div class="card">
                        <h3>Active Crawls</h3>
                        <div id="active-crawls"></div>
                    </div>
                    <div class="card">
                        <h3>System Resources</h3>
                        <div id="system-metrics"></div>
                    </div>
                    <div class="card">
                        <h3>Active Alerts</h3>
                        <div id="alerts"></div>
                    </div>
                </div>
                <div class="card" style="margin-top: 20px;">
                    <h3>Response Time Chart</h3>
                    <canvas id="responseChart" width="400" height="200"></canvas>
                </div>
                <script>
                    const ws = new WebSocket(`ws://${window.location.host}/ws/dashboard`);
                    const responseTimes = [];
                    const responseChart = new Chart(document.getElementById('responseChart'), {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Response Time (s)',
                                data: [],
                                borderColor: 'rgb(75, 192, 192)',
                                tension: 0.1
                            }]
                        }
                    });
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'metrics_update') {
                            updateDashboard(data.payload);
                        } else if (data.type === 'alert') {
                            showAlert(data.payload);
                        }
                    };
                    
                    function updateDashboard(metrics) {
                        // Update global metrics
                        const globalHtml = `
                            <div class="metric">
                                <div class="metric-value">${metrics.global.total_requests}</div>
                                <div class="metric-label">Total Requests</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${metrics.global.success_rate.toFixed(1)}%</div>
                                <div class="metric-label">Success Rate</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${metrics.global.requests_per_second.toFixed(2)}</div>
                                <div class="metric-label">Requests/sec</div>
                            </div>
                        `;
                        document.getElementById('global-metrics').innerHTML = globalHtml;
                        
                        // Update system metrics
                        if (metrics.system) {
                            const systemHtml = `
                                <div class="metric">
                                    <div class="metric-value">${metrics.system.cpu_percent.toFixed(1)}%</div>
                                    <div class="metric-label">CPU Usage</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">${metrics.system.memory_percent.toFixed(1)}%</div>
                                    <div class="metric-label">Memory Usage</div>
                                </div>
                            `;
                            document.getElementById('system-metrics').innerHTML = systemHtml;
                        }
                        
                        // Update chart
                        if (metrics.crawls) {
                            Object.values(metrics.crawls).forEach(crawl => {
                                if (crawl.avg_response_time > 0) {
                                    responseTimes.push(crawl.avg_response_time);
                                    if (responseTimes.length > 20) responseTimes.shift();
                                    
                                    responseChart.data.labels = Array(responseTimes.length).fill('');
                                    responseChart.data.datasets[0].data = responseTimes;
                                    responseChart.update();
                                }
                            });
                        }
                    }
                    
                    function showAlert(alert) {
                        const alertsDiv = document.getElementById('alerts');
                        const alertClass = alert.severity === 'critical' ? 'alert-critical' : 'alert-warning';
                        const alertHtml = `
                            <div class="alert ${alertClass}">
                                <strong>${alert.type}:</strong> ${alert.message}
                                <small>${new Date(alert.timestamp).toLocaleTimeString()}</small>
                            </div>
                        `;
                        alertsDiv.innerHTML = alertHtml + alertsDiv.innerHTML;
                    }
                    
                    // Request initial data
                    ws.onopen = function() {
                        ws.send(JSON.stringify({type: 'subscribe', topics: ['metrics', 'alerts']}));
                    };
                </script>
            </body>
            </html>
            """)
        
        @self.app.websocket("/ws/dashboard")
        async def websocket_endpoint(websocket: WebSocket):
            client_id = f"client_{int(time.time())}_{id(websocket)}"
            await self.connection_manager.connect(websocket, client_id)
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message.get("type") == "subscribe":
                        topics = message.get("topics", [])
                        self.connection_manager.subscribe(client_id, topics)
                        
                        # Send initial data
                        initial_data = {
                            "type": "initial_data",
                            "payload": {
                                "global": self.metrics_collector.get_global_summary(),
                                "system": asdict(self.metrics_collector.collect_system_metrics()),
                                "alerts": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()]
                            }
                        }
                        await self.connection_manager.send_personal_message(initial_data, client_id)
                    
                    elif message.get("type") == "unsubscribe":
                        topics = message.get("topics", [])
                        self.connection_manager.unsubscribe(client_id, topics)
                    
                    elif message.get("type") == "acknowledge_alert":
                        alert_id = message.get("alert_id")
                        self.alert_manager.acknowledge_alert(alert_id, client_id)
                    
                    elif message.get("type") == "resolve_alert":
                        alert_id = message.get("alert_id")
                        self.alert_manager.resolve_alert(alert_id, client_id)
                        
            except WebSocketDisconnect:
                self.connection_manager.disconnect(client_id)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(client_id)
        
        @self.app.get("/metrics")
        async def prometheus_metrics():
            """Prometheus metrics endpoint"""
            return self.grafana_integration.get_prometheus_metrics()
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get all metrics as JSON"""
            return {
                "global": self.metrics_collector.get_global_summary(),
                "crawls": {
                    crawl_id: asdict(metrics) 
                    for crawl_id, metrics in self.metrics_collector.get_all_crawl_metrics().items()
                },
                "system": asdict(self.metrics_collector.collect_system_metrics()),
                "system_history": [
                    asdict(m) for m in self.metrics_collector.get_system_metrics_history(60)
                ]
            }
        
        @self.app.get("/api/alerts")
        async def get_alerts(active_only: bool = True):
            """Get alerts"""
            if active_only:
                alerts = self.alert_manager.get_active_alerts()
            else:
                alerts = self.alert_manager.get_alert_history(24)
            return [alert.to_dict() for alert in alerts]
        
        @self.app.get("/api/grafana/dashboard")
        async def get_grafana_dashboard():
            """Get Grafana dashboard JSON"""
            return self.grafana_integration.get_grafana_dashboard_json()
        
        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str, user: str = "api"):
            """Acknowledge an alert"""
            self.alert_manager.acknowledge_alert(alert_id, user)
            return {"status": "acknowledged"}
        
        @self.app.post("/api/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str, user: str = "api"):
            """Resolve an alert"""
            self.alert_manager.resolve_alert(alert_id, user)
            return {"status": "resolved"}
    
    def _setup_background_tasks(self):
        """Setup background tasks for metrics collection and alert checking"""
        
        @self.app.on_event("startup")
        async def startup_event():
            # Start metrics broadcasting task
            asyncio.create_task(self._broadcast_metrics())
            # Start alert checking task
            asyncio.create_task(self._check_alerts())
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to all connected clients"""
        while True:
            try:
                metrics_data = {
                    "type": "metrics_update",
                    "payload": {
                        "global": self.metrics_collector.get_global_summary(),
                        "crawls": {
                            crawl_id: asdict(metrics)
                            for crawl_id, metrics in self.metrics_collector.get_all_crawl_metrics().items()
                        },
                        "system": asdict(self.metrics_collector.collect_system_metrics()),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                await self.connection_manager.broadcast(metrics_data)
            except Exception as e:
                logger.error(f"Error broadcasting metrics: {e}")
            
            await asyncio.sleep(2)  # Broadcast every 2 seconds
    
    async def _check_alerts(self):
        """Check for alerts periodically"""
        while True:
            try:
                # Check crawl metrics for alerts
                for crawl_id, metrics in self.metrics_collector.get_all_crawl_metrics().items():
                    self.alert_manager.check_metrics(metrics)
                
                # Check system metrics for alerts
                system_metrics = self.metrics_collector.collect_system_metrics()
                self.alert_manager.check_metrics(
                    CrawlMetrics(crawl_id="system", start_time=datetime.now()),
                    system_metrics
                )
                
                # Broadcast any new alerts
                active_alerts = self.alert_manager.get_active_alerts()
                if active_alerts:
                    alert_data = {
                        "type": "alerts_update",
                        "payload": [alert.to_dict() for alert in active_alerts]
                    }
                    await self.connection_manager.broadcast(alert_data)
                    
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    def _handle_alert_notification(self, alert: Alert):
        """Handle alert notifications (can be extended for Slack/email)"""
        logger.warning(f"ALERT: {alert.severity.value.upper()} - {alert.message}")
        
        # Broadcast alert to connected clients
        alert_data = {
            "type": "alert",
            "payload": alert.to_dict()
        }
        asyncio.create_task(self.connection_manager.broadcast(alert_data))
    
    def record_crawl_start(self, crawl_id: str) -> CrawlMetrics:
        """Record the start of a crawl"""
        return self.metrics_collector.start_crawl(crawl_id)
    
    def record_crawl_result(self, crawl_id: str, result: CrawlResult):
        """Record a crawl result"""
        self.metrics_collector.update_crawl(crawl_id, result)
    
    def record_crawl_end(self, crawl_id: str):
        """Record the end of a crawl"""
        self.metrics_collector.end_crawl(crawl_id)
    
    def add_alert_rule(self, rule: Dict):
        """Add a custom alert rule"""
        self.alert_manager.add_rule(rule)
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler"""
        self.alert_manager.add_notification_handler(handler)
    
    def start(self):
        """Start the monitoring dashboard"""
        logger.info(f"Starting monitoring dashboard on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
    
    def start_background(self):
        """Start the dashboard in a background thread"""
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread

# Global instance for easy access
_monitoring_dashboard = None

def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get or create the global monitoring dashboard instance"""
    global _monitoring_dashboard
    if _monitoring_dashboard is None:
        _monitoring_dashboard = MonitoringDashboard()
    return _monitoring_dashboard

def setup_monitoring(
    host: str = "0.0.0.0",
    port: int = 8765,
    enable_dashboard: bool = True,
    enable_alerts: bool = True,
    alert_handlers: Optional[List[Callable]] = None
) -> MonitoringDashboard:
    """Setup monitoring for axiom"""
    dashboard = get_monitoring_dashboard()
    
    if enable_dashboard:
        dashboard.start_background()
    
    if alert_handlers:
        for handler in alert_handlers:
            dashboard.add_notification_handler(handler)
    
    return dashboard

# Example notification handlers
def slack_notification_handler(alert: Alert, webhook_url: str):
    """Send alert to Slack"""
    if not AIOHTTP_AVAILABLE:
        logger.warning("aiohttp not available for Slack notifications")
        return
    
    async def send_slack():
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": f"🚨 *{alert.severity.value.upper()} Alert*\n"
                            f"*Type:* {alert.type.value}\n"
                            f"*Message:* {alert.message}\n"
                            f"*Source:* {alert.source}\n"
                            f"*Time:* {alert.timestamp.isoformat()}"
                }
                async with session.post(webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send Slack notification: {response.status}")
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    asyncio.create_task(send_slack())

def email_notification_handler(alert: Alert, smtp_config: Dict):
    """Send alert via email"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['from_email']
        msg['To'] = smtp_config['to_email']
        msg['Subject'] = f"axiom Alert: {alert.severity.value.upper()} - {alert.type.value}"
        
        body = f"""
        Alert Details:
        --------------
        Type: {alert.type.value}
        Severity: {alert.severity.value}
        Message: {alert.message}
        Source: {alert.source}
        Time: {alert.timestamp.isoformat()}
        
        Metadata:
        {json.dumps(alert.metadata, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config.get('smtp_port', 587))
        server.starttls()
        server.login(smtp_config['username'], smtp_config['password'])
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email alert sent: {alert.id}")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")

# Integration with existing axiom code
class MonitoringMixin:
    """Mixin to add monitoring capabilities to existing axiom classes"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitoring_dashboard = get_monitoring_dashboard()
        self.crawl_id = kwargs.get('crawl_id', f"crawl_{int(time.time())}")
    
    def _record_result(self, result: CrawlResult):
        """Record a crawl result to monitoring"""
        if hasattr(self, 'monitoring_dashboard'):
            self.monitoring_dashboard.record_crawl_result(self.crawl_id, result)
        
        # Call parent method if it exists
        if hasattr(super(), '_record_result'):
            super()._record_result(result)
    
    def start_crawl(self, *args, **kwargs):
        """Start crawl with monitoring"""
        if hasattr(self, 'monitoring_dashboard'):
            self.monitoring_dashboard.record_crawl_start(self.crawl_id)
        
        # Call parent method
        return super().start_crawl(*args, **kwargs)
    
    def end_crawl(self, *args, **kwargs):
        """End crawl with monitoring"""
        if hasattr(self, 'monitoring_dashboard'):
            self.monitoring_dashboard.record_crawl_end(self.crawl_id)
        
        # Call parent method
        if hasattr(super(), 'end_crawl'):
            return super().end_crawl(*args, **kwargs)

# Example usage
if __name__ == "__main__":
    # Setup monitoring with Slack alerts
    dashboard = setup_monitoring(
        host="0.0.0.0",
        port=8765,
        enable_dashboard=True,
        enable_alerts=True,
        alert_handlers=[
            lambda alert: slack_notification_handler(
                alert, 
                webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
            )
        ]
    )
    
    # Example: Record some test metrics
    test_crawl = dashboard.record_crawl_start("test_crawl_1")
    
    # Simulate some crawl results
    from axiom.core.custom_types import CrawlResult
    
    for i in range(10):
        result = CrawlResult(
            url=f"https://example.com/page{i}",
            status_code=200,
            content=f"Page content {i}",
            success=True,
            response_time=0.5 + (i * 0.1)
        )
        dashboard.record_crawl_result("test_crawl_1", result)
    
    # Keep the dashboard running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down monitoring dashboard...")