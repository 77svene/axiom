"""
Real-time Monitoring Dashboard for axiom
WebSocket-based live dashboard with Grafana/Prometheus integration
"""

import asyncio
import json
import time
import threading
import os
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import deque
import psutil
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Prometheus client for metrics export
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, generate_latest, 
    CONTENT_TYPE_LATEST, REGISTRY, start_http_server
)

# Import from existing axiom modules
from axiom.core.custom_types import CrawlResult, SessionConfig
from axiom.core.utils._utils import get_logger

logger = get_logger(__name__)


# ============== Data Models ==============

@dataclass
class CrawlMetrics:
    """Real-time crawl metrics"""
    crawl_id: str
    start_time: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    avg_response_time: float = 0.0
    current_rps: float = 0.0
    pages_crawled: int = 0
    data_extracted_mb: float = 0.0
    active_sessions: int = 0
    proxy_usage: Dict[str, int] = field(default_factory=dict)
    status_codes: Dict[int, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    disk_usage_percent: float = 0.0
    active_threads: int = 0
    open_files: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class AlertConfig(BaseModel):
    """Alert configuration"""
    success_rate_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    response_time_threshold_ms: float = Field(default=5000.0, ge=0.0)
    blocked_rate_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    cpu_threshold_percent: float = Field(default=80.0, ge=0.0, le=100.0)
    memory_threshold_percent: float = Field(default=85.0, ge=0.0, le=100.0)
    check_interval_seconds: int = Field(default=30, ge=1)


class DashboardConfig(BaseModel):
    """Dashboard configuration"""
    refresh_interval_ms: int = Field(default=1000, ge=100)
    history_size: int = Field(default=3600, ge=60)  # 1 hour of history
    enable_alerts: bool = True
    alert_config: AlertConfig = Field(default_factory=AlertConfig)
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    grafana_url: Optional[str] = None


# ============== Metrics Collector ==============

class MetricsCollector:
    """Collects and aggregates metrics from all scraping operations"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.crawl_metrics: Dict[str, CrawlMetrics] = {}
        self.system_metrics_history: deque = deque(maxlen=config.history_size)
        self.crawl_history: Dict[str, deque] = {}
        self.active_websockets: Set[WebSocket] = set()
        self._lock = threading.RLock()
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Start background collection
        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collect_system_metrics_loop,
            daemon=True
        )
        self._collection_thread.start()
        
        # Alert state
        self._alert_state = {
            "last_alert_time": {},
            "active_alerts": set()
        }
        
        logger.info("Metrics collector initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Request metrics
        self.request_counter = Counter(
            'axiom_requests_total',
            'Total number of requests',
            ['crawl_id', 'status', 'proxy_type']
        )
        
        self.response_time_histogram = Histogram(
            'axiom_response_time_seconds',
            'Response time in seconds',
            ['crawl_id'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.blocked_counter = Counter(
            'axiom_blocked_requests_total',
            'Total blocked requests',
            ['crawl_id', 'block_type']
        )
        
        # Crawl metrics
        self.active_crawls_gauge = Gauge(
            'axiom_active_crawls',
            'Number of active crawls'
        )
        
        self.pages_crawled_gauge = Gauge(
            'axiom_pages_crawled',
            'Total pages crawled',
            ['crawl_id']
        )
        
        # System metrics
        self.cpu_usage_gauge = Gauge(
            'axiom_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage_gauge = Gauge(
            'axiom_memory_usage_percent',
            'Memory usage percentage'
        )
        
        # Success rate gauge
        self.success_rate_gauge = Gauge(
            'axiom_success_rate',
            'Success rate of requests',
            ['crawl_id']
        )
    
    def _collect_system_metrics_loop(self):
        """Background thread to collect system metrics"""
        while self._running:
            try:
                metrics = self._collect_system_metrics()
                with self._lock:
                    self.system_metrics_history.append(metrics)
                    
                    # Update Prometheus gauges
                    self.cpu_usage_gauge.set(metrics.cpu_percent)
                    self.memory_usage_gauge.set(metrics.memory_percent)
                    
                    # Check for alerts
                    if self.config.enable_alerts:
                        self._check_alerts(metrics)
                        
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(self.config.alert_config.check_interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        process = psutil.Process()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            network_sent_mb=net_io.bytes_sent / (1024 * 1024),
            network_recv_mb=net_io.bytes_recv / (1024 * 1024),
            disk_usage_percent=disk.percent,
            active_threads=threading.active_count(),
            open_files=len(process.open_files())
        )
    
    def start_crawl(self, crawl_id: str, config: Optional[Dict] = None):
        """Start tracking a new crawl"""
        with self._lock:
            if crawl_id not in self.crawl_metrics:
                self.crawl_metrics[crawl_id] = CrawlMetrics(
                    crawl_id=crawl_id,
                    start_time=datetime.now()
                )
                self.crawl_history[crawl_id] = deque(maxlen=self.config.history_size)
                self.active_crawls_gauge.inc()
                logger.info(f"Started tracking crawl: {crawl_id}")
    
    def record_request(self, crawl_id: str, result: CrawlResult):
        """Record a request result"""
        with self._lock:
            if crawl_id not in self.crawl_metrics:
                self.start_crawl(crawl_id)
            
            metrics = self.crawl_metrics[crawl_id]
            metrics.total_requests += 1
            metrics.last_updated = datetime.now()
            
            # Update status
            if result.success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                if result.blocked:
                    metrics.blocked_requests += 1
            
            # Update response time
            if result.response_time:
                total_time = metrics.avg_response_time * (metrics.total_requests - 1)
                metrics.avg_response_time = (total_time + result.response_time) / metrics.total_requests
            
            # Update status codes
            if result.status_code:
                metrics.status_codes[result.status_code] = metrics.status_codes.get(result.status_code, 0) + 1
            
            # Update error types
            if result.error:
                error_type = type(result.error).__name__
                metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
            
            # Update proxy usage
            if result.proxy_used:
                metrics.proxy_usage[result.proxy_used] = metrics.proxy_usage.get(result.proxy_used, 0) + 1
            
            # Update data extracted
            if result.data:
                data_size = len(str(result.data).encode('utf-8')) / (1024 * 1024)  # MB
                metrics.data_extracted_mb += data_size
            
            # Update pages crawled
            if result.success:
                metrics.pages_crawled += 1
            
            # Calculate current RPS
            elapsed = (datetime.now() - metrics.start_time).total_seconds()
            if elapsed > 0:
                metrics.current_rps = metrics.total_requests / elapsed
            
            # Update Prometheus metrics
            status = "success" if result.success else "failed"
            self.request_counter.labels(
                crawl_id=crawl_id,
                status=status,
                proxy_type=result.proxy_used or "none"
            ).inc()
            
            if result.response_time:
                self.response_time_histogram.labels(crawl_id=crawl_id).observe(result.response_time)
            
            if result.blocked:
                block_type = "captcha" if "captcha" in str(result.error).lower() else "other"
                self.blocked_counter.labels(crawl_id=crawl_id, block_type=block_type).inc()
            
            # Update success rate
            if metrics.total_requests > 0:
                success_rate = metrics.successful_requests / metrics.total_requests
                self.success_rate_gauge.labels(crawl_id=crawl_id).set(success_rate)
            
            self.pages_crawled_gauge.labels(crawl_id=crawl_id).set(metrics.pages_crawled)
            
            # Store in history
            self.crawl_history[crawl_id].append(asdict(metrics))
            
            # Broadcast to WebSocket clients
            asyncio.create_task(self._broadcast_update(crawl_id, metrics))
    
    def update_session_count(self, crawl_id: str, count: int):
        """Update active session count for a crawl"""
        with self._lock:
            if crawl_id in self.crawl_metrics:
                self.crawl_metrics[crawl_id].active_sessions = count
    
    def end_crawl(self, crawl_id: str):
        """End tracking for a crawl"""
        with self._lock:
            if crawl_id in self.crawl_metrics:
                self.active_crawls_gauge.dec()
                logger.info(f"Ended tracking crawl: {crawl_id}")
    
    def get_crawl_metrics(self, crawl_id: Optional[str] = None) -> Dict:
        """Get metrics for a specific crawl or all crawls"""
        with self._lock:
            if crawl_id:
                if crawl_id in self.crawl_metrics:
                    return asdict(self.crawl_metrics[crawl_id])
                return {}
            else:
                return {cid: asdict(m) for cid, m in self.crawl_metrics.items()}
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        with self._lock:
            if self.system_metrics_history:
                return asdict(self.system_metrics_history[-1])
            return {}
    
    def get_metrics_history(self, crawl_id: Optional[str] = None, 
                          minutes: int = 60) -> List[Dict]:
        """Get historical metrics"""
        with self._lock:
            if crawl_id and crawl_id in self.crawl_history:
                history = list(self.crawl_history[crawl_id])
                # Filter by time
                cutoff = datetime.now() - timedelta(minutes=minutes)
                return [h for h in history if datetime.fromisoformat(h['last_updated']) > cutoff]
            else:
                # Return system metrics history
                history = list(self.system_metrics_history)
                cutoff = datetime.now() - timedelta(minutes=minutes)
                return [asdict(m) for m in history if m.timestamp > cutoff]
    
    async def _broadcast_update(self, crawl_id: str, metrics: CrawlMetrics):
        """Broadcast metrics update to all connected WebSocket clients"""
        if not self.active_websockets:
            return
        
        message = {
            "type": "metrics_update",
            "crawl_id": crawl_id,
            "metrics": asdict(metrics),
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = set()
        for websocket in self.active_websockets:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        self.active_websockets -= disconnected
    
    def _check_alerts(self, system_metrics: SystemMetrics):
        """Check for alert conditions"""
        config = self.config.alert_config
        now = datetime.now()
        
        # System alerts
        if system_metrics.cpu_percent > config.cpu_threshold_percent:
            self._trigger_alert("high_cpu", f"CPU usage at {system_metrics.cpu_percent:.1f}%")
        
        if system_metrics.memory_percent > config.memory_threshold_percent:
            self._trigger_alert("high_memory", f"Memory usage at {system_metrics.memory_percent:.1f}%")
        
        # Crawl-specific alerts
        for crawl_id, metrics in self.crawl_metrics.items():
            if metrics.total_requests > 10:  # Only check after some requests
                success_rate = metrics.successful_requests / metrics.total_requests
                if success_rate < config.success_rate_threshold:
                    self._trigger_alert(
                        f"low_success_rate_{crawl_id}",
                        f"Crawl {crawl_id} success rate at {success_rate:.1%}"
                    )
                
                blocked_rate = metrics.blocked_requests / metrics.total_requests
                if blocked_rate > config.blocked_rate_threshold:
                    self._trigger_alert(
                        f"high_blocked_rate_{crawl_id}",
                        f"Crawl {crawl_id} blocked rate at {blocked_rate:.1%}"
                    )
                
                if metrics.avg_response_time > config.response_time_threshold_ms / 1000:
                    self._trigger_alert(
                        f"high_response_time_{crawl_id}",
                        f"Crawl {crawl_id} avg response time {metrics.avg_response_time:.1f}s"
                    )
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert"""
        now = datetime.now()
        last_alert = self._alert_state["last_alert_time"].get(alert_type)
        
        # Rate limit alerts (max once per 5 minutes per type)
        if last_alert and (now - last_alert).total_seconds() < 300:
            return
        
        self._alert_state["last_alert_time"][alert_type] = now
        self._alert_state["active_alerts"].add(alert_type)
        
        logger.warning(f"ALERT: {alert_type} - {message}")
        
        # Send alert via configured channels
        asyncio.create_task(self._send_alert_notification(alert_type, message))
    
    async def _send_alert_notification(self, alert_type: str, message: str):
        """Send alert notification via Slack/email"""
        # Slack webhook
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook:
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "text": f"🚨 axiom Alert: {alert_type}\n{message}",
                        "username": "axiom Monitor",
                        "icon_emoji": ":warning:"
                    }
                    await session.post(slack_webhook, json=payload)
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")
        
        # Email (simplified - would need proper SMTP setup)
        email_recipient = os.getenv("ALERT_EMAIL")
        if email_recipient:
            logger.info(f"Email alert would be sent to {email_recipient}: {message}")
    
    def stop(self):
        """Stop the metrics collector"""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Metrics collector stopped")


# ============== FastAPI Application ==============

class MonitoringDashboard:
    """FastAPI-based monitoring dashboard"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.metrics_collector = MetricsCollector(self.config)
        self.app = self._create_app()
        
        # Start Prometheus server
        if self.config.prometheus_port:
            start_http_server(self.config.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Monitoring dashboard starting up")
            yield
            # Shutdown
            self.metrics_collector.stop()
            logger.info("Monitoring dashboard shutting down")
        
        app = FastAPI(
            title="axiom Monitoring Dashboard",
            description="Real-time monitoring for web scraping operations",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: FastAPI):
        """Register API routes"""
        
        @app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve the main dashboard HTML"""
            return self._get_dashboard_html()
        
        @app.get("/api/metrics")
        async def get_metrics(crawl_id: Optional[str] = None):
            """Get current metrics"""
            return self.metrics_collector.get_crawl_metrics(crawl_id)
        
        @app.get("/api/metrics/system")
        async def get_system_metrics():
            """Get system metrics"""
            return self.metrics_collector.get_system_metrics()
        
        @app.get("/api/metrics/history")
        async def get_metrics_history(
            crawl_id: Optional[str] = None,
            minutes: int = 60
        ):
            """Get historical metrics"""
            return self.metrics_collector.get_metrics_history(crawl_id, minutes)
        
        @app.post("/api/crawl/{crawl_id}/start")
        async def start_crawl_tracking(crawl_id: str, config: Optional[Dict] = None):
            """Start tracking a new crawl"""
            self.metrics_collector.start_crawl(crawl_id, config)
            return {"status": "started", "crawl_id": crawl_id}
        
        @app.post("/api/crawl/{crawl_id}/end")
        async def end_crawl_tracking(crawl_id: str):
            """End tracking for a crawl"""
            self.metrics_collector.end_crawl(crawl_id)
            return {"status": "ended", "crawl_id": crawl_id}
        
        @app.post("/api/crawl/{crawl_id}/record")
        async def record_crawl_result(crawl_id: str, result: Dict):
            """Record a crawl result"""
            # Convert dict to CrawlResult (simplified)
            crawl_result = CrawlResult(**result)
            self.metrics_collector.record_request(crawl_id, crawl_result)
            return {"status": "recorded"}
        
        @app.get("/api/alerts")
        async def get_active_alerts():
            """Get active alerts"""
            return {
                "active_alerts": list(self.metrics_collector._alert_state["active_alerts"]),
                "last_alert_times": {
                    k: v.isoformat() 
                    for k, v in self.metrics_collector._alert_state["last_alert_time"].items()
                }
            }
        
        @app.get("/api/config")
        async def get_config():
            """Get current configuration"""
            return self.config.dict()
        
        @app.post("/api/config")
        async def update_config(config: DashboardConfig):
            """Update configuration"""
            self.config = config
            self.metrics_collector.config = config
            return {"status": "updated", "config": self.config.dict()}
        
        @app.get("/metrics")
        async def prometheus_metrics():
            """Prometheus metrics endpoint"""
            return Response(
                generate_latest(REGISTRY),
                media_type=CONTENT_TYPE_LATEST
            )
        
        @app.websocket("/ws/metrics")
        async def websocket_metrics(websocket: WebSocket):
            """WebSocket endpoint for real-time metrics"""
            await websocket.accept()
            self.metrics_collector.active_websockets.add(websocket)
            
            try:
                # Send initial data
                initial_data = {
                    "type": "initial",
                    "metrics": self.metrics_collector.get_crawl_metrics(),
                    "system": self.metrics_collector.get_system_metrics(),
                    "config": self.config.dict()
                }
                await websocket.send_json(initial_data)
                
                # Keep connection alive and handle messages
                while True:
                    data = await websocket.receive_text()
                    # Handle client messages if needed
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        
            except WebSocketDisconnect:
                self.metrics_collector.active_websockets.discard(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.metrics_collector.active_websockets.discard(websocket)
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_crawls": len(self.metrics_collector.crawl_metrics),
                "active_websockets": len(self.metrics_collector.active_websockets)
            }
    
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
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.3.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            color: #333;
            font-size: 28px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .metric-card h3 {
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .metric-change {
            font-size: 14px;
            color: #10b981;
        }
        
        .metric-change.negative {
            color: #ef4444;
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        .chart-container h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .crawl-list {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .crawl-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            border-bottom: 1px solid #eee;
        }
        
        .crawl-item:last-child {
            border-bottom: none;
        }
        
        .crawl-id {
            font-weight: bold;
            color: #333;
        }
        
        .crawl-stats {
            display: flex;
            gap: 20px;
        }
        
        .crawl-stat {
            text-align: center;
        }
        
        .crawl-stat-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        
        .crawl-stat-label {
            font-size: 12px;
            color: #666;
        }
        
        .alerts-panel {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .alerts-panel h3 {
            color: #ef4444;
            margin-bottom: 15px;
        }
        
        .alert-item {
            background: white;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            border-left: 4px solid #ef4444;
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #666;
        }
        
        .connection-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .connection-dot.connected {
            background: #10b981;
        }
        
        .connection-dot.disconnected {
            background: #ef4444;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>
                <div class="status-indicator"></div>
                axiom Monitoring Dashboard
            </h1>
            <div class="connection-status">
                <div class="connection-dot" id="connectionDot"></div>
                <span id="connectionStatus">Connecting...</span>
            </div>
        </div>
        
        <div id="alertsContainer"></div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Requests</h3>
                <div class="metric-value" id="totalRequests">0</div>
                <div class="metric-change" id="requestsChange">+0%</div>
            </div>
            
            <div class="metric-card">
                <h3>Success Rate</h3>
                <div class="metric-value" id="successRate">0%</div>
                <div class="metric-change" id="successChange">+0%</div>
            </div>
            
            <div class="metric-card">
                <h3>Avg Response Time</h3>
                <div class="metric-value" id="avgResponseTime">0ms</div>
                <div class="metric-change" id="responseChange">+0%</div>
            </div>
            
            <div class="metric-card">
                <h3>Blocked Requests</h3>
                <div class="metric-value" id="blockedRequests">0</div>
                <div class="metric-change negative" id="blockedChange">+0%</div>
            </div>
            
            <div class="metric-card">
                <h3>Active Sessions</h3>
                <div class="metric-value" id="activeSessions">0</div>
                <div class="metric-change" id="sessionsChange">+0</div>
            </div>
            
            <div class="metric-card">
                <h3>Pages Crawled</h3>
                <div class="metric-value" id="pagesCrawled">0</div>
                <div class="metric-change" id="pagesChange">+0</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Response Time Trend</h3>
            <canvas id="responseTimeChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Success Rate Over Time</h3>
            <canvas id="successRateChart"></canvas>
        </div>
        
        <div class="crawl-list">
            <h3>Active Crawls</h3>
            <div id="crawlList"></div>
        </div>
    </div>
    
    <script>
        class axiomDashboard {
            constructor() {
                this.ws = null;
                this.charts = {};
                this.metricsHistory = {
                    responseTimes: [],
                    successRates: [],
                    timestamps: []
                };
                this.previousMetrics = {};
                
                this.initWebSocket();
                this.initCharts();
                this.loadInitialData();
            }
            
            initWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    document.getElementById('connectionDot').className = 'connection-dot connected';
                    document.getElementById('connectionStatus').textContent = 'Connected';
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    document.getElementById('connectionDot').className = 'connection-dot disconnected';
                    document.getElementById('connectionStatus').textContent = 'Disconnected';
                    
                    // Try to reconnect after 5 seconds
                    setTimeout(() => this.initWebSocket(), 5000);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }
            
            handleMessage(data) {
                switch (data.type) {
                    case 'initial':
                        this.updateDashboard(data.metrics, data.system);
                        break;
                    case 'metrics_update':
                        this.updateCrawlMetrics(data.crawl_id, data.metrics);
                        break;
                }
            }
            
            async loadInitialData() {
                try {
                    const [metricsRes, systemRes, alertsRes] = await Promise.all([
                        fetch('/api/metrics'),
                        fetch('/api/metrics/system'),
                        fetch('/api/alerts')
                    ]);
                    
                    const metrics = await metricsRes.json();
                    const system = await systemRes.json();
                    const alerts = await alertsRes.json();
                    
                    this.updateDashboard(metrics, system);
                    this.updateAlerts(alerts.active_alerts);
                } catch (error) {
                    console.error('Failed to load initial data:', error);
                }
            }
            
            updateDashboard(crawlMetrics, systemMetrics) {
                // Calculate aggregate metrics
                let totalRequests = 0;
                let successfulRequests = 0;
                let blockedRequests = 0;
                let totalResponseTime = 0;
                let responseTimeCount = 0;
                let activeSessions = 0;
                let pagesCrawled = 0;
                
                for (const crawlId in crawlMetrics) {
                    const metrics = crawlMetrics[crawlId];
                    totalRequests += metrics.total_requests;
                    successfulRequests += metrics.successful_requests;
                    blockedRequests += metrics.blocked_requests;
                    activeSessions += metrics.active_sessions;
                    pagesCrawled += metrics.pages_crawled;
                    
                    if (metrics.avg_response_time > 0) {
                        totalResponseTime += metrics.avg_response_time * metrics.total_requests;
                        responseTimeCount += metrics.total_requests;
                    }
                }
                
                const successRate = totalRequests > 0 ? (successfulRequests / totalRequests) * 100 : 0;
                const avgResponseTime = responseTimeCount > 0 ? totalResponseTime / responseTimeCount : 0;
                
                // Update UI
                document.getElementById('totalRequests').textContent = totalRequests.toLocaleString();
                document.getElementById('successRate').textContent = successRate.toFixed(1) + '%';
                document.getElementById('avgResponseTime').textContent = avgResponseTime.toFixed(0) + 'ms';
                document.getElementById('blockedRequests').textContent = blockedRequests.toLocaleString();
                document.getElementById('activeSessions').textContent = activeSessions;
                document.getElementById('pagesCrawled').textContent = pagesCrawled.toLocaleString();
                
                // Update charts
                this.updateCharts(avgResponseTime, successRate);
                
                // Update crawl list
                this.updateCrawlList(crawlMetrics);
                
                // Store for comparison
                this.previousMetrics = {
                    totalRequests,
                    successRate,
                    avgResponseTime,
                    blockedRequests,
                    activeSessions,
                    pagesCrawled
                };
            }
            
            updateCrawlMetrics(crawlId, metrics) {
                // Update specific crawl in the list
                const crawlItem = document.querySelector(`[data-crawl-id="${crawlId}"]`);
                if (crawlItem) {
                    const successRate = metrics.total_requests > 0 
                        ? (metrics.successful_requests / metrics.total_requests) * 100 
                        : 0;
                    
                    crawlItem.querySelector('.success-rate').textContent = successRate.toFixed(1) + '%';
                    crawlItem.querySelector('.total-requests').textContent = metrics.total_requests;
                    crawlItem.querySelector('.response-time').textContent = metrics.avg_response_time.toFixed(0) + 'ms';
                }
                
                // Refresh full dashboard
                this.loadInitialData();
            }
            
            updateCrawlList(crawlMetrics) {
                const container = document.getElementById('crawlList');
                container.innerHTML = '';
                
                for (const crawlId in crawlMetrics) {
                    const metrics = crawlMetrics[crawlId];
                    const successRate = metrics.total_requests > 0 
                        ? (metrics.successful_requests / metrics.total_requests) * 100 
                        : 0;
                    
                    const item = document.createElement('div');
                    item.className = 'crawl-item';
                    item.dataset.crawlId = crawlId;
                    item.innerHTML = `
                        <div class="crawl-id">${crawlId}</div>
                        <div class="crawl-stats">
                            <div class="crawl-stat">
                                <div class="crawl-stat-value success-rate">${successRate.toFixed(1)}%</div>
                                <div class="crawl-stat-label">Success</div>
                            </div>
                            <div class="crawl-stat">
                                <div class="crawl-stat-value total-requests">${metrics.total_requests}</div>
                                <div class="crawl-stat-label">Requests</div>
                            </div>
                            <div class="crawl-stat">
                                <div class="crawl-stat-value response-time">${metrics.avg_response_time.toFixed(0)}ms</div>
                                <div class="crawl-stat-label">Avg Time</div>
                            </div>
                            <div class="crawl-stat">
                                <div class="crawl-stat-value">${metrics.active_sessions}</div>
                                <div class="crawl-stat-label">Sessions</div>
                            </div>
                        </div>
                    `;
                    container.appendChild(item);
                }
            }
            
            updateAlerts(activeAlerts) {
                const container = document.getElementById('alertsContainer');
                
                if (activeAlerts.length === 0) {
                    container.innerHTML = '';
                    return;
                }
                
                container.innerHTML = `
                    <div class="alerts-panel">
                        <h3>Active Alerts (${activeAlerts.length})</h3>
                        ${activeAlerts.map(alert => `
                            <div class="alert-item">
                                <strong>${alert.replace(/_/g, ' ').toUpperCase()}</strong>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            initCharts() {
                // Response Time Chart
                const responseCtx = document.getElementById('responseTimeChart').getContext('2d');
                this.charts.responseTime = new Chart(responseCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Response Time (ms)',
                            data: [],
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                display: false
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
                
                // Success Rate Chart
                const successCtx = document.getElementById('successRateChart').getContext('2d');
                this.charts.successRate = new Chart(successCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Success Rate (%)',
                            data: [],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                display: false
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100
                            }
                        }
                    }
                });
            }
            
            updateCharts(responseTime, successRate) {
                const now = new Date();
                const timeLabel = now.toLocaleTimeString();
                
                // Update response time chart
                this.metricsHistory.responseTimes.push(responseTime);
                this.metricsHistory.timestamps.push(timeLabel);
                
                // Keep only last 20 data points
                if (this.metricsHistory.responseTimes.length > 20) {
                    this.metricsHistory.responseTimes.shift();
                    this.metricsHistory.timestamps.shift();
                }
                
                this.charts.responseTime.data.labels = this.metricsHistory.timestamps;
                this.charts.responseTime.data.datasets[0].data = this.metricsHistory.responseTimes;
                this.charts.responseTime.update();
                
                // Update success rate chart
                this.metricsHistory.successRates.push(successRate);
                if (this.metricsHistory.successRates.length > 20) {
                    this.metricsHistory.successRates.shift();
                }
                
                this.charts.successRate.data.labels = this.metricsHistory.timestamps;
                this.charts.successRate.data.datasets[0].data = this.metricsHistory.successRates;
                this.charts.successRate.update();
            }
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new axiomDashboard();
        });
    </script>
</body>
</html>
        """
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the monitoring dashboard"""
        logger.info(f"Starting monitoring dashboard on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# ============== Integration with Existing axiom Code ==============

class axiomMonitor:
    """Integration layer between axiom and the monitoring dashboard"""
    
    _instance = None
    _dashboard = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(axiomMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._dashboard = None
            self._crawl_sessions = {}
            logger.info("axiom monitor initialized")
    
    def start_dashboard(self, config: Optional[DashboardConfig] = None, 
                       host: str = "0.0.0.0", port: int = 8000):
        """Start the monitoring dashboard"""
        if self._dashboard is None:
            self._dashboard = MonitoringDashboard(config)
            
            # Run in a separate thread
            thread = threading.Thread(
                target=self._dashboard.run,
                kwargs={"host": host, "port": port},
                daemon=True
            )
            thread.start()
            
            logger.info(f"Monitoring dashboard started at http://{host}:{port}")
            return f"http://{host}:{port}"
        else:
            logger.warning("Dashboard already running")
            return None
    
    def register_crawl_session(self, crawl_id: str, session):
        """Register a crawl session for monitoring"""
        if self._dashboard:
            self._dashboard.metrics_collector.start_crawl(crawl_id)
            self._crawl_sessions[crawl_id] = session
            
            # Monkey patch the session to record metrics
            original_fetch = session.fetch if hasattr(session, 'fetch') else None
            
            if original_fetch:
                async def monitored_fetch(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = await original_fetch(*args, **kwargs)
                        response_time = time.time() - start_time
                        
                        # Record metrics
                        crawl_result = CrawlResult(
                            url=kwargs.get('url', args[0] if args else ''),
                            success=True,
                            response_time=response_time,
                            status_code=getattr(result, 'status_code', None),
                            data=getattr(result, 'text', None),
                            blocked=False
                        )
                        
                        if self._dashboard:
                            self._dashboard.metrics_collector.record_request(crawl_id, crawl_result)
                        
                        return result
                    except Exception as e:
                        response_time = time.time() - start_time
                        
                        # Record failed request
                        crawl_result = CrawlResult(
                            url=kwargs.get('url', args[0] if args else ''),
                            success=False,
                            response_time=response_time,
                            error=e,
                            blocked='blocked' in str(e).lower()
                        )
                        
                        if self._dashboard:
                            self._dashboard.metrics_collector.record_request(crawl_id, crawl_result)
                        
                        raise
                
                session.fetch = monitored_fetch
    
    def record_result(self, crawl_id: str, result: CrawlResult):
        """Manually record a crawl result"""
        if self._dashboard:
            self._dashboard.metrics_collector.record_request(crawl_id, result)
    
    def update_session_count(self, crawl_id: str, count: int):
        """Update active session count"""
        if self._dashboard:
            self._dashboard.metrics_collector.update_session_count(crawl_id, count)
    
    def end_crawl(self, crawl_id: str):
        """End monitoring for a crawl"""
        if self._dashboard:
            self._dashboard.metrics_collector.end_crawl(crawl_id)
            if crawl_id in self._crawl_sessions:
                del self._crawl_sessions[crawl_id]
    
    def get_metrics(self, crawl_id: Optional[str] = None) -> Dict:
        """Get current metrics"""
        if self._dashboard:
            return self._dashboard.metrics_collector.get_crawl_metrics(crawl_id)
        return {}


# ============== Convenience Functions ==============

# Global monitor instance
_monitor = axiomMonitor()


def start_monitoring(config: Optional[DashboardConfig] = None, 
                    host: str = "0.0.0.0", port: int = 8000) -> str:
    """Start the monitoring dashboard"""
    return _monitor.start_dashboard(config, host, port)


def register_crawl(crawl_id: str, session):
    """Register a crawl session for monitoring"""
    _monitor.register_crawl_session(crawl_id, session)


def record_crawl_result(crawl_id: str, result: CrawlResult):
    """Record a crawl result"""
    _monitor.record_result(crawl_id, result)


def update_active_sessions(crawl_id: str, count: int):
    """Update active session count"""
    _monitor.update_session_count(crawl_id, count)


def end_crawl_monitoring(crawl_id: str):
    """End monitoring for a crawl"""
    _monitor.end_crawl(crawl_id)


def get_monitoring_metrics(crawl_id: Optional[str] = None) -> Dict:
    """Get current monitoring metrics"""
    return _monitor.get_metrics(crawl_id)


# ============== CLI Integration ==============

def add_monitoring_cli_commands(cli_app):
    """Add monitoring commands to the axiom CLI"""
    import argparse
    
    parser = cli_app.add_parser(
        'monitor',
        help='Start the monitoring dashboard'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind the dashboard to'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run the dashboard on'
    )
    parser.add_argument(
        '--prometheus-port',
        type=int,
        default=9090,
        help='Port for Prometheus metrics'
    )
    parser.add_argument(
        '--no-alerts',
        action='store_true',
        help='Disable alerting'
    )
    
    def monitor_command(args):
        config = DashboardConfig(
            prometheus_port=args.prometheus_port,
            enable_alerts=not args.no_alerts
        )
        
        url = start_monitoring(config, args.host, args.port)
        if url:
            print(f"Monitoring dashboard started at {url}")
            print(f"Prometheus metrics available at http://{args.host}:{args.prometheus_port}/metrics")
            print("Press Ctrl+C to stop")
            
            try:
                # Keep the main thread alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down monitoring dashboard...")
        else:
            print("Failed to start monitoring dashboard")
    
    parser.set_defaults(func=monitor_command)


# ============== Example Usage ==============

if __name__ == "__main__":
    # Example: Start monitoring dashboard
    config = DashboardConfig(
        refresh_interval_ms=1000,
        enable_alerts=True,
        alert_config=AlertConfig(
            success_rate_threshold=0.7,
            response_time_threshold_ms=3000,
            blocked_rate_threshold=0.4
        )
    )
    
    # Start the dashboard
    dashboard_url = start_monitoring(config, "0.0.0.0", 8000)
    print(f"Dashboard available at: {dashboard_url}")
    
    # Simulate some crawl activity
    import random
    import uuid
    
    crawl_id = f"test-crawl-{uuid.uuid4().hex[:8]}"
    register_crawl(crawl_id, None)  # In real usage, pass actual session
    
    try:
        for i in range(100):
            # Simulate request
            result = CrawlResult(
                url=f"https://example.com/page/{i}",
                success=random.random() > 0.2,
                response_time=random.uniform(0.1, 2.0),
                status_code=random.choice([200, 404, 403, 500]),
                blocked=random.random() > 0.9
            )
            
            record_crawl_result(crawl_id, result)
            update_active_sessions(crawl_id, random.randint(1, 10))
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        end_crawl_monitoring(crawl_id)
        print("Monitoring example completed")