"""
axiom Production Deployment Toolkit - Auto-scaling and deployment automation
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import tempfile
import yaml
import hashlib
from enum import Enum

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from kubernetes import client, config, watch
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DIGITALOCEAN = "digitalocean"


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions"""
    queue_depth: int
    requests_per_second: float
    cpu_utilization: float
    memory_utilization: float
    response_time_p95: float
    error_rate: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ResourceRequirements:
    """Resource requirements for containers"""
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    gpu_request: int = 0
    
    def to_k8s_resources(self) -> Dict[str, Any]:
        return {
            'requests': {
                'cpu': self.cpu_request,
                'memory': self.memory_request,
                **({'nvidia.com/gpu': str(self.gpu_request)} if self.gpu_request > 0 else {})
            },
            'limits': {
                'cpu': self.cpu_limit,
                'memory': self.memory_limit,
                **({'nvidia.com/gpu': str(self.gpu_request)} if self.gpu_request > 0 else {})
            }
        }


class QueueMonitor:
    """Monitor queue depth for scaling decisions"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not installed, using in-memory queue simulation")
            self._queue_simulation = []
            return
        
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("Connected to Redis for queue monitoring")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def get_queue_depth(self, queue_name: str = "axiom:tasks") -> int:
        """Get current queue depth"""
        if self.redis_client:
            try:
                return self.redis_client.llen(queue_name)
            except Exception as e:
                logger.error(f"Error getting queue depth: {e}")
                return 0
        else:
            # Simulation mode
            return len(getattr(self, '_queue_simulation', []))
    
    def add_to_queue(self, queue_name: str = "axiom:tasks", data: Dict = None):
        """Add item to queue (for testing)"""
        if self.redis_client:
            try:
                self.redis_client.rpush(queue_name, json.dumps(data or {}))
            except Exception as e:
                logger.error(f"Error adding to queue: {e}")
        else:
            if not hasattr(self, '_queue_simulation'):
                self._queue_simulation = []
            self._queue_simulation.append(data or {})
    
    def get_metrics(self, queue_name: str = "axiom:tasks") -> ScalingMetrics:
        """Get current scaling metrics"""
        queue_depth = self.get_queue_depth(queue_name)
        
        # In a real implementation, these would come from monitoring systems
        return ScalingMetrics(
            queue_depth=queue_depth,
            requests_per_second=0.0,
            cpu_utilization=0.0,
            memory_utilization=0.0,
            response_time_p95=0.0,
            error_rate=0.0,
            timestamp=datetime.utcnow()
        )


class CostOptimizer:
    """Optimize cloud resource costs"""
    
    def __init__(self, provider: CloudProvider = CloudProvider.AWS):
        self.provider = provider
        self.pricing_data = self._load_pricing_data()
    
    def _load_pricing_data(self) -> Dict[str, Any]:
        """Load cloud pricing data"""
        # In production, this would fetch from cloud provider APIs
        pricing_file = Path(__file__).parent / "pricing" / f"{self.provider.value}.json"
        
        if pricing_file.exists():
            with open(pricing_file) as f:
                return json.load(f)
        
        # Default pricing (simplified)
        return {
            "instance_types": {
                "t3.micro": {"cpu": 2, "memory": 1, "price_per_hour": 0.0104},
                "t3.small": {"cpu": 2, "memory": 2, "price_per_hour": 0.0208},
                "t3.medium": {"cpu": 2, "memory": 4, "price_per_hour": 0.0416},
                "t3.large": {"cpu": 2, "memory": 8, "price_per_hour": 0.0832},
            },
            "regions": {
                "us-east-1": 1.0,
                "us-west-2": 1.05,
                "eu-west-1": 1.12,
            }
        }
    
    def recommend_instance_type(self, 
                               cpu_required: float, 
                               memory_required: float,
                               workload_type: str = "balanced") -> str:
        """Recommend instance type based on requirements"""
        instance_types = self.pricing_data.get("instance_types", {})
        
        suitable_instances = []
        for instance_type, specs in instance_types.items():
            if specs["cpu"] >= cpu_required and specs["memory"] >= memory_required:
                suitable_instances.append((instance_type, specs["price_per_hour"]))
        
        if not suitable_instances:
            logger.warning("No suitable instance found, returning largest available")
            return max(instance_types.keys(), 
                      key=lambda x: instance_types[x]["price_per_hour"])
        
        # Sort by price (ascending)
        suitable_instances.sort(key=lambda x: x[1])
        
        # For cost optimization, choose the cheapest suitable instance
        return suitable_instances[0][0]
    
    def estimate_monthly_cost(self, 
                             instance_type: str, 
                             hours_per_day: float = 24,
                             region: str = "us-east-1") -> float:
        """Estimate monthly cost for an instance"""
        instance_data = self.pricing_data.get("instance_types", {}).get(instance_type, {})
        region_multiplier = self.pricing_data.get("regions", {}).get(region, 1.0)
        
        if not instance_data:
            return 0.0
        
        hourly_cost = instance_data["price_per_hour"] * region_multiplier
        monthly_hours = hours_per_day * 30
        
        return hourly_cost * monthly_hours
    
    def optimize_scaling_schedule(self, 
                                 current_load: float, 
                                 forecast_load: List[float]) -> Dict[str, Any]:
        """Optimize scaling schedule based on load forecast"""
        # Simple optimization: scale based on forecast
        max_load = max(forecast_load) if forecast_load else current_load
        avg_load = sum(forecast_load) / len(forecast_load) if forecast_load else current_load
        
        return {
            "recommended_min_instances": max(1, int(avg_load / 100)),
            "recommended_max_instances": max(2, int(max_load / 100) + 1),
            "scale_up_threshold": 70,  # CPU percentage
            "scale_down_threshold": 30,
            "cooldown_period_seconds": 300,
            "forecast_based_scaling": True
        }


class KubernetesScaler:
    """Auto-scaling for Kubernetes deployments"""
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.k8s_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Kubernetes cluster"""
        if not K8S_AVAILABLE:
            logger.warning("Kubernetes client not installed")
            return
        
        try:
            # Try in-cluster config first, then local
            try:
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes config")
            except config.ConfigException:
                config.load_kube_config()
                logger.info("Loaded local Kubernetes config")
            
            self.k8s_client = client.AppsV1Api()
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes: {e}")
    
    def scale_deployment(self, 
                        deployment_name: str, 
                        replicas: int,
                        wait: bool = True,
                        timeout: int = 300) -> bool:
        """Scale a Kubernetes deployment"""
        if not self.k8s_client:
            logger.error("Kubernetes client not available")
            return False
        
        try:
            # Get current deployment
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update replica count
            deployment.spec.replicas = replicas
            
            # Apply update
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
            
            if wait:
                return self._wait_for_scaling(deployment_name, replicas, timeout)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False
    
    def _wait_for_scaling(self, 
                         deployment_name: str, 
                         target_replicas: int,
                         timeout: int) -> bool:
        """Wait for scaling operation to complete"""
        if not K8S_AVAILABLE:
            return False
        
        w = watch.Watch()
        start_time = time.time()
        
        try:
            for event in w.stream(
                self.k8s_client.list_namespaced_deployment,
                namespace=self.namespace,
                field_selector=f"metadata.name={deployment_name}",
                timeout_seconds=timeout
            ):
                deployment = event['object']
                
                if deployment.status.ready_replicas == target_replicas:
                    logger.info(f"Deployment {deployment_name} scaled to {target_replicas} replicas")
                    w.stop()
                    return True
                
                if time.time() - start_time > timeout:
                    logger.warning(f"Timeout waiting for scaling of {deployment_name}")
                    w.stop()
                    return False
                    
        except Exception as e:
            logger.error(f"Error waiting for scaling: {e}")
            return False
        
        return False
    
    def get_current_replicas(self, deployment_name: str) -> Optional[int]:
        """Get current replica count for a deployment"""
        if not self.k8s_client:
            return None
        
        try:
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            return deployment.spec.replicas
        except Exception as e:
            logger.error(f"Failed to get deployment replicas: {e}")
            return None


class AutoScaler:
    """Main auto-scaling controller"""
    
    def __init__(self,
                 deployment_name: str,
                 min_replicas: int = 1,
                 max_replicas: int = 10,
                 queue_name: str = "axiom:tasks",
                 scaling_interval: int = 60,
                 metrics_window: int = 300):
        
        self.deployment_name = deployment_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.queue_name = queue_name
        self.scaling_interval = scaling_interval
        self.metrics_window = metrics_window
        
        self.queue_monitor = QueueMonitor()
        self.k8s_scaler = KubernetesScaler()
        self.cost_optimizer = CostOptimizer()
        
        self.scaling_history: List[Dict[str, Any]] = []
        self.last_scale_time: Optional[datetime] = None
        self.cooldown_period = 300  # 5 minutes
        
        # Load scaling policies
        self.scaling_policies = self._load_scaling_policies()
    
    def _load_scaling_policies(self) -> Dict[str, Any]:
        """Load scaling policies from configuration"""
        config_path = os.getenv('SCALING_CONFIG_PATH', 
                               str(Path(__file__).parent / "scaling_config.json"))
        
        if Path(config_path).exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load scaling config: {e}")
        
        # Default policies
        return {
            "scale_up": {
                "queue_depth_threshold": 100,
                "cpu_threshold": 70,
                "memory_threshold": 80,
                "step_size": 2,
                "aggressive": False
            },
            "scale_down": {
                "queue_depth_threshold": 10,
                "cpu_threshold": 30,
                "memory_threshold": 40,
                "step_size": 1,
                "conservative": True
            },
            "predictive_scaling": {
                "enabled": True,
                "forecast_hours": 24,
                "confidence_threshold": 0.8
            }
        }
    
    def _in_cooldown(self) -> bool:
        """Check if we're in scaling cooldown period"""
        if not self.last_scale_time:
            return False
        
        time_since_last_scale = (datetime.utcnow() - self.last_scale_time).total_seconds()
        return time_since_last_scale < self.cooldown_period
    
    def _calculate_desired_replicas(self, metrics: ScalingMetrics) -> int:
        """Calculate desired replica count based on metrics"""
        current_replicas = self.k8s_scaler.get_current_replicas(self.deployment_name) or 1
        
        # Check if we should scale up
        scale_up = False
        scale_down = False
        
        # Queue-based scaling
        if metrics.queue_depth > self.scaling_policies["scale_up"]["queue_depth_threshold"]:
            scale_up = True
        elif metrics.queue_depth < self.scaling_policies["scale_down"]["queue_depth_threshold"]:
            scale_down = True
        
        # Resource-based scaling
        if metrics.cpu_utilization > self.scaling_policies["scale_up"]["cpu_threshold"]:
            scale_up = True
        elif metrics.cpu_utilization < self.scaling_policies["scale_down"]["cpu_threshold"]:
            scale_down = True
        
        if scale_up and not scale_down:
            # Calculate how many replicas to add
            if self.scaling_policies["scale_up"]["aggressive"]:
                # Aggressive scaling: double the replicas
                desired = min(current_replicas * 2, self.max_replicas)
            else:
                # Step scaling
                step = self.scaling_policies["scale_up"]["step_size"]
                desired = min(current_replicas + step, self.max_replicas)
            
            logger.info(f"Scaling up from {current_replicas} to {desired} replicas")
            return desired
            
        elif scale_down and not scale_up:
            # Calculate how many replicas to remove
            if self.scaling_policies["scale_down"]["conservative"]:
                # Conservative: remove one at a time
                desired = max(current_replicas - 1, self.min_replicas)
            else:
                step = self.scaling_policies["scale_down"]["step_size"]
                desired = max(current_replicas - step, self.min_replicas)
            
            logger.info(f"Scaling down from {current_replicas} to {desired} replicas")
            return desired
        
        return current_replicas
    
    def _record_scaling_event(self, 
                             from_replicas: int, 
                             to_replicas: int,
                             metrics: ScalingMetrics,
                             reason: str):
        """Record scaling event for analysis"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "deployment": self.deployment_name,
            "from_replicas": from_replicas,
            "to_replicas": to_replicas,
            "metrics": metrics.to_dict(),
            "reason": reason,
            "queue_depth": metrics.queue_depth
        }
        
        self.scaling_history.append(event)
        
        # Keep only recent history
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-1000:]
        
        # Save to file for persistence
        history_file = Path(__file__).parent / "scaling_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.scaling_history[-100:], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scaling history: {e}")
    
    async def run_scaling_loop(self):
        """Main scaling loop"""
        logger.info(f"Starting auto-scaler for deployment {self.deployment_name}")
        
        while True:
            try:
                # Check cooldown
                if self._in_cooldown():
                    logger.debug("In cooldown period, skipping scaling check")
                    await asyncio.sleep(self.scaling_interval)
                    continue
                
                # Get current metrics
                metrics = self.queue_monitor.get_metrics(self.queue_name)
                
                # Calculate desired replicas
                current_replicas = self.k8s_scaler.get_current_replicas(self.deployment_name) or 1
                desired_replicas = self._calculate_desired_replicas(metrics)
                
                # Apply scaling if needed
                if desired_replicas != current_replicas:
                    success = self.k8s_scaler.scale_deployment(
                        self.deployment_name,
                        desired_replicas,
                        wait=False
                    )
                    
                    if success:
                        self._record_scaling_event(
                            current_replicas,
                            desired_replicas,
                            metrics,
                            "auto_scaling"
                        )
                        self.last_scale_time = datetime.utcnow()
                
                # Wait for next iteration
                await asyncio.sleep(self.scaling_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(self.scaling_interval)
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations based on historical data"""
        if not self.scaling_history:
            return {"status": "no_data"}
        
        # Analyze scaling patterns
        scale_ups = [e for e in self.scaling_history if e["to_replicas"] > e["from_replicas"]]
        scale_downs = [e for e in self.scaling_history if e["to_replicas"] < e["from_replicas"]]
        
        # Calculate average queue depth at scaling events
        avg_scale_up_depth = (sum(e["queue_depth"] for e in scale_ups) / len(scale_ups)) if scale_ups else 0
        avg_scale_down_depth = (sum(e["queue_depth"] for e in scale_downs) / len(scale_downs)) if scale_downs else 0
        
        # Get cost optimization recommendations
        current_replicas = self.k8s_scaler.get_current_replicas(self.deployment_name) or 1
        cost_recommendation = self.cost_optimizer.optimize_scaling_schedule(
            current_load=current_replicas * 100,  # Simplified
            forecast_load=[100] * 24  # Would come from forecasting model
        )
        
        return {
            "current_replicas": current_replicas,
            "total_scale_ups": len(scale_ups),
            "total_scale_downs": len(scale_downs),
            "avg_scale_up_queue_depth": avg_scale_up_depth,
            "avg_scale_down_queue_depth": avg_scale_down_depth,
            "cost_optimization": cost_recommendation,
            "recommendations": self._generate_recommendations(scale_ups, scale_downs)
        }
    
    def _generate_recommendations(self, 
                                 scale_ups: List[Dict], 
                                 scale_downs: List[Dict]) -> List[str]:
        """Generate human-readable recommendations"""
        recommendations = []
        
        if len(scale_ups) > len(scale_downs) * 2:
            recommendations.append(
                "Consider increasing min_replicas as you're scaling up frequently"
            )
        
        if len(scale_downs) > len(scale_ups) * 2:
            recommendations.append(
                "Consider decreasing max_replicas as you're scaling down frequently"
            )
        
        # Check for scaling oscillations
        if len(self.scaling_history) >= 10:
            recent_events = self.scaling_history[-10:]
            direction_changes = 0
            for i in range(1, len(recent_events)):
                if (recent_events[i]["to_replicas"] > recent_events[i]["from_replicas"]) != \
                   (recent_events[i-1]["to_replicas"] > recent_events[i-1]["from_replicas"]):
                    direction_changes += 1
            
            if direction_changes > 5:
                recommendations.append(
                    "Detected scaling oscillations. Consider increasing cooldown period or adjusting thresholds"
                )
        
        return recommendations


class DeploymentGenerator:
    """Generate deployment configurations"""
    
    def __init__(self, project_name: str = "axiom"):
        self.project_name = project_name
        self.templates_dir = Path(__file__).parent / "templates"
    
    def generate_dockerfile(self, 
                           base_image: str = "python:3.9-slim",
                           requirements_file: str = "requirements.txt",
                           app_module: str = "axiom.cli:main") -> str:
        """Generate Dockerfile for axiom application"""
        return f"""# axiom Production Dockerfile
# Auto-generated by axiom Deployment Toolkit

FROM {base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY {requirements_file} .
RUN pip install --no-cache-dir -r {requirements_file}

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "{app_module}"]
"""
    
    def generate_k8s_deployment(self,
                               image: str,
                               replicas: int = 2,
                               port: int = 8000,
                               resources: Optional[ResourceRequirements] = None,
                               env_vars: Optional[Dict[str, str]] = None) -> str:
        """Generate Kubernetes deployment manifest"""
        resources = resources or ResourceRequirements()
        env_vars = env_vars or {}
        
        env_section = ""
        if env_vars:
            env_lines = []
            for key, value in env_vars.items():
                env_lines.append(f'        - name: {key}\\n          value: "{value}"')
            env_section = "\\n      env:\\n" + "\\n".join(env_lines)
        
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {self.project_name}
  labels:
    app: {self.project_name}
    version: v1
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {self.project_name}
  template:
    metadata:
      labels:
        app: {self.project_name}
    spec:
      containers:
      - name: {self.project_name}
        image: {image}
        ports:
        - containerPort: {port}
        resources:
          requests:
            cpu: {resources.cpu_request}
            memory: {resources.memory_request}
          limits:
            cpu: {resources.cpu_limit}
            memory: {resources.memory_limit}
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5{env_section}
---
apiVersion: v1
kind: Service
metadata:
  name: {self.project_name}
spec:
  selector:
    app: {self.project_name}
  ports:
  - port: 80
    targetPort: {port}
  type: ClusterIP
"""
    
    def generate_helm_chart(self, output_dir: str = "./helm") -> Dict[str, str]:
        """Generate Helm chart for axiom"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        chart_yaml = f"""apiVersion: v2
name: {self.project_name}
description: A Helm chart for axiom web scraping framework
type: application
version: 0.1.0
appVersion: "1.0.0"
keywords:
  - scraping
  - web
  - automation
maintainers:
  - name: axiom Team
    email: team@axiom.dev
"""
        
        values_yaml = f"""# Default values for {self.project_name}
replicaCount: 2

image:
  repository: axiom/axiom
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: false
  className: ""
  annotations: {{}}
  hosts:
    - host: axiom.local
      paths:
        - path: /
          pathType: ImplementationSpecific
  tls: []

resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {{}}
tolerations: []
affinity: {{}}

env:
  REDIS_URL: "redis://redis-master:6379/0"
  LOG_LEVEL: "INFO"
  QUEUE_NAME: "axiom:tasks"
"""
        
        # Create directory structure
        templates_dir = output_path / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        # Write files
        files = {
            "Chart.yaml": chart_yaml,
            "values.yaml": values_yaml,
            "templates/deployment.yaml": self.generate_k8s_deployment(
                image="{{ .Values.image.repository }}:{{ .Values.image.tag }}",
                replicas="{{ .Values.replicaCount }}",
                resources=ResourceRequirements(
                    cpu_request="{{ .Values.resources.requests.cpu }}",
                    cpu_limit="{{ .Values.resources.limits.cpu }}",
                    memory_request="{{ .Values.resources.requests.memory }}",
                    memory_limit="{{ .Values.resources.limits.memory }}"
                )
            ),
            "templates/service.yaml": f"""apiVersion: v1
kind: Service
metadata:
  name: {{{{- include "{self.project_name}.fullname" . }}}}
  labels:
    {{{{- include "{self.project_name}.labels" . | nindent 4 }}}}
spec:
  type: {{{{ .Values.service.type }}}}
  ports:
    - port: {{{{ .Values.service.port }}}}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{{{- include "{self.project_name}.selectorLabels" . | nindent 4 }}}}
""",
            "templates/_helpers.tpl": f"""{{{{/*
Expand the name of the chart.
*/}}}}
{{{{- define "{self.project_name}.name" -}}}}
{{{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}}}
{{{{- end }}}}

{{{{/*
Create a default fully qualified app name.
*/}}}}
{{{{- define "{self.project_name}.fullname" -}}}}
{{{{- if .Values.fullnameOverride }}}}
{{{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}}}
{{{{- else }}}}
{{{{- $name := default .Chart.Name .Values.nameOverride }}}}
{{{{- if contains $name .Release.Name }}}}
{{{{- .Release.Name | trunc 63 | trimSuffix "-" }}}}
{{{{- else }}}}
{{{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}}}
{{{{- end }}}}
{{{{- end }}}}
{{{{- end }}}}

{{{{/*
Create chart name and version as used by the chart label.
*/}}}}
{{{{- define "{self.project_name}.chart" -}}}}
{{{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}}}
{{{{- end }}}}

{{{{/*
Common labels
*/}}}}
{{{{- define "{self.project_name}.labels" -}}}}
helm.sh/chart: {{{{- include "{self.project_name}.chart" . }}}}
{{{{ include "{self.project_name}.selectorLabels" . }}}}
{{{{- if .Chart.AppVersion }}}}
app.kubernetes.io/version: {{{{ .Chart.AppVersion | quote }}}}
{{{{- end }}}}
app.kubernetes.io/managed-by: {{{{ .Release.Service }}}}
{{{{- end }}}}

{{{{/*
Selector labels
*/}}}}
{{{{- define "{self.project_name}.selectorLabels" -}}}}
app.kubernetes.io/name: {{{{- include "{self.project_name}.name" . }}}}
app.kubernetes.io/instance: {{{{ .Release.Name }}}}
{{{{- end }}}}
"""
        }
        
        for filename, content in files.items():
            file_path = output_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
        
        logger.info(f"Generated Helm chart in {output_dir}")
        return {str(k): str(v) for k, v in files.items()}
    
    def generate_terraform_aws(self, output_dir: str = "./terraform-aws") -> Dict[str, str]:
        """Generate Terraform configuration for AWS"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        main_tf = f"""# axiom AWS Infrastructure
# Auto-generated by axiom Deployment Toolkit

terraform {{
  required_version = ">= 1.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

# EKS Cluster
module "eks" {{
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${{var.project_name}}-cluster"
  cluster_version = "1.27"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {{
    axiom = {{
      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.desired_nodes

      instance_types = var.instance_types
      capacity_type  = var.capacity_type

      labels = {{
        Environment = var.environment
        Application = var.project_name
      }}

      tags = {{
        ExtraTag = "${{var.project_name}}-workers"
      }}
    }}
  }}
}}

# VPC
module "vpc" {{
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${{var.project_name}}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_dns_hostnames = true

  tags = {{
    "kubernetes.io/cluster/${{var.project_name}}-cluster" = "shared"
  }}
}}

# ElastiCache Redis for queue
resource "aws_elasticache_cluster" "redis" {{
  cluster_id           = "${{var.project_name}}-redis"
  engine              = "redis"
  node_type           = var.redis_node_type
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  engine_version      = "7.0"
  port                = 6379

  subnet_group_name = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]
}}

resource "aws_elasticache_subnet_group" "redis" {{
  name       = "${{var.project_name}}-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}}

resource "aws_security_group" "redis" {{
  name_prefix = "${{var.project_name}}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {{
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }}

  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

# ECR Repository for container images
resource "aws_ecr_repository" "axiom" {{
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {{
    scan_on_push = true
  }}
}}

# Outputs
output "cluster_endpoint" {{
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}}

output "cluster_name" {{
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}}

output "redis_endpoint" {{
  description = "Redis endpoint for queue"
  value       = aws_elasticache_cluster.redis.cache_nodes[0].address
}}

output "ecr_repository_url" {{
  description = "ECR Repository URL"
  value       = aws_ecr_repository.axiom.repository_url
}}
"""
        
        variables_tf = f"""# axiom AWS Variables

variable "aws_region" {{
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}}

variable "project_name" {{
  description = "Project name"
  type        = string
  default     = "{self.project_name}"
}}

variable "environment" {{
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "dev"
}}

variable "vpc_cidr" {{
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}}

variable "availability_zones" {{
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}}

variable "private_subnets" {{
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}}

variable "public_subnets" {{
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}}

variable "min_nodes" {{
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}}

variable "max_nodes" {{
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}}

variable "desired_nodes" {{
  description = "Desired number of worker nodes"
  type        = number
  default     = 2
}}

variable "instance_types" {{
  description = "EC2 instance types for worker nodes"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}}

variable "capacity_type" {{
  description = "Capacity type (ON_DEMAND or SPOT)"
  type        = string
  default     = "SPOT"
}}

variable "redis_node_type" {{
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"
}}
"""
        
        outputs_tf = """# axiom AWS Outputs

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}
"""
        
        files = {
            "main.tf": main_tf,
            "variables.tf": variables_tf,
            "outputs.tf": outputs_tf,
            "terraform.tfvars.example": f"""# Example terraform.tfvars
aws_region      = "us-east-1"
project_name    = "{self.project_name}"
environment     = "dev"
min_nodes       = 1
max_nodes       = 5
desired_nodes   = 2
instance_types  = ["t3.medium"]
capacity_type   = "SPOT"
redis_node_type = "cache.t3.micro"
"""
        }
        
        for filename, content in files.items():
            file_path = output_path / filename
            with open(file_path, 'w') as f:
                f.write(content)
        
        logger.info(f"Generated Terraform AWS configuration in {output_dir}")
        return {str(k): str(v) for k, v in files.items()}
    
    def generate_cicd_pipeline(self, 
                              platform: str = "github",
                              output_dir: str = "./.github/workflows") -> str:
        """Generate CI/CD pipeline configuration"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if platform == "github":
            workflow = f"""name: axiom CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{{{ github.repository }}}}

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 axiom --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test with pytest
      run: |
        pytest --cov=axiom --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{{{version}}}}
          type=sha,prefix={{{{branch}}}}-
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: ${{{{ github.event_name != 'pull_request' }}}}
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}
    
    - name: Generate deployment artifacts
      run: |
        mkdir -p deployment
        python -c "
        from axiom.deploy.scaler import DeploymentGenerator
        gen = DeploymentGenerator('${{{{ github.event.repository.name }}}}')
        
        # Generate Dockerfile
        with open('deployment/Dockerfile', 'w') as f:
            f.write(gen.generate_dockerfile())
        
        # Generate Kubernetes manifests
        with open('deployment/deployment.yaml', 'w') as f:
            f.write(gen.generate_k8s_deployment(
                image='${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}:${{{{ github.sha }}}}'
            ))
        
        print('Generated deployment artifacts')
        "
    
    - name: Upload deployment artifacts
      uses: actions/upload-artifact@v3
      with:
        name: deployment-artifacts
        path: deployment/

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Download deployment artifacts
      uses: actions/download-artifact@v3
      with:
        name: deployment-artifacts
        path: deployment/
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
    
    - name: Deploy to staging
      run: |
        kubectl apply -f deployment/deployment.yaml
        kubectl rollout status deployment/${{{{ github.event.repository.name }}}} -n staging

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Download deployment artifacts
      uses: actions/download-artifact@v3
      with:
        name: deployment-artifacts
        path: deployment/
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
    
    - name: Deploy to production
      run: |
        kubectl apply -f deployment/deployment.yaml
        kubectl rollout status deployment/${{{{ github.event.repository.name }}}} -n production
"""
            
            workflow_file = output_path / "axiom-ci-cd.yml"
            with open(workflow_file, 'w') as f:
                f.write(workflow)
            
            logger.info(f"Generated GitHub Actions workflow at {workflow_file}")
            return str(workflow_file)
        
        elif platform == "gitlab":
            # GitLab CI configuration
            pass
        
        return ""


class DeploymentManager:
    """Main deployment management class"""
    
    def __init__(self, project_name: str = "axiom"):
        self.project_name = project_name
        self.generator = DeploymentGenerator(project_name)
        self.scaler = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for deployment"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('axiom-deployment.log'),
                logging.StreamHandler()
            ]
        )
    
    def initialize_deployment(self,
                             output_dir: str = "./deployment",
                             generate_all: bool = True) -> Dict[str, Any]:
        """Initialize complete deployment configuration"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        if generate_all:
            # Generate Dockerfile
            dockerfile = self.generator.generate_dockerfile()
            dockerfile_path = output_path / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile)
            generated_files['dockerfile'] = str(dockerfile_path)
            
            # Generate Kubernetes manifests
            k8s_manifest = self.generator.generate_k8s_deployment(
                image=f"{self.project_name}:latest",
                replicas=2
            )
            k8s_path = output_path / "kubernetes"
            k8s_path.mkdir(exist_ok=True)
            k8s_file = k8s_path / "deployment.yaml"
            with open(k8s_file, 'w') as f:
                f.write(k8s_manifest)
            generated_files['kubernetes'] = str(k8s_file)
            
            # Generate Helm chart
            helm_dir = output_path / "helm"
            helm_files = self.generator.generate_helm_chart(str(helm_dir))
            generated_files['helm'] = helm_files
            
            # Generate Terraform for AWS
            terraform_dir = output_path / "terraform-aws"
            terraform_files = self.generator.generate_terraform_aws(str(terraform_dir))
            generated_files['terraform_aws'] = terraform_files
            
            # Generate CI/CD pipeline
            cicd_file = self.generator.generate_cicd_pipeline(
                platform="github",
                output_dir=str(output_path / ".github/workflows")
            )
            generated_files['cicd'] = cicd_file
        
        # Generate configuration files
        config = {
            "project_name": self.project_name,
            "version": "1.0.0",
            "scaling": {
                "enabled": True,
                "min_replicas": 1,
                "max_replicas": 10,
                "queue_name": "axiom:tasks",
                "scaling_interval": 60
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9090,
                "health_endpoint": "/health"
            },
            "cost_optimization": {
                "enabled": True,
                "cloud_provider": "aws",
                "use_spot_instances": True,
                "right_sizing": True
            }
        }
        
        config_path = output_path / "deployment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        generated_files['config'] = str(config_path)
        
        # Generate README with deployment instructions
        readme = f"""# {self.project_name.title()} Deployment

## Quick Start

### Docker
```bash
docker build -t {self.project_name}:latest .
docker run -p 8000:8000 {self.project_name}:latest
```

### Kubernetes
```bash
kubectl apply -f kubernetes/deployment.yaml
```

### Helm
```bash
helm install {self.project_name} ./helm
```

### Terraform (AWS)
```bash
cd terraform-aws
terraform init
terraform plan
terraform apply
```

## Auto-scaling

The auto-scaler monitors queue depth and resource utilization to automatically scale your deployment.

### Configuration
Edit `deployment_config.json` to adjust scaling parameters.

### Starting the Auto-scaler
```python
from axiom.deploy.scaler import AutoScaler

scaler = AutoScaler(
    deployment_name="{self.project_name}",
    min_replicas=1,
    max_replicas=10
)

# Start the scaling loop
asyncio.run(scaler.run_scaling_loop())
```

## Monitoring

- Health endpoint: http://localhost:8000/health
- Metrics endpoint: http://localhost:9090/metrics
- Scaling history: `scaling_history.json`

## Cost Optimization

The deployment toolkit includes cost optimization features:
- Automatic instance right-sizing
- Spot instance support (AWS)
- Scheduled scaling based on load forecasting
- Resource utilization monitoring

## CI/CD

The GitHub Actions workflow automatically:
1. Runs tests
2. Builds Docker image
3. Pushes to container registry
4. Deploys to staging/production

## Support

For issues and feature requests, please open an issue on GitHub.
"""
        
        readme_path = output_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        generated_files['readme'] = str(readme_path)
        
        logger.info(f"Initialized deployment configuration in {output_dir}")
        return generated_files
    
    def start_auto_scaling(self,
                          deployment_name: Optional[str] = None,
                          config_file: Optional[str] = None):
        """Start the auto-scaling system"""
        deployment_name = deployment_name or self.project_name
        
        # Load configuration if provided
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                config = json.load(f)
                scaling_config = config.get('scaling', {})
        else:
            scaling_config = {}
        
        # Initialize auto-scaler
        self.scaler = AutoScaler(
            deployment_name=deployment_name,
            min_replicas=scaling_config.get('min_replicas', 1),
            max_replicas=scaling_config.get('max_replicas', 10),
            queue_name=scaling_config.get('queue_name', 'axiom:tasks'),
            scaling_interval=scaling_config.get('scaling_interval', 60)
        )
        
        logger.info(f"Starting auto-scaling for {deployment_name}")
        
        # Run the scaling loop
        try:
            asyncio.run(self.scaler.run_scaling_loop())
        except KeyboardInterrupt:
            logger.info("Auto-scaling stopped by user")
        except Exception as e:
            logger.error(f"Auto-scaling error: {e}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        if not self.scaler:
            return {"status": "not_running"}
        
        return {
            "deployment_name": self.scaler.deployment_name,
            "current_replicas": self.scaler.k8s_scaler.get_current_replicas(
                self.scaler.deployment_name
            ),
            "queue_depth": self.scaler.queue_monitor.get_queue_depth(
                self.scaler.queue_name
            ),
            "scaling_recommendations": self.scaler.get_scaling_recommendations(),
            "last_scale_time": self.scaler.last_scale_time.isoformat() 
                              if self.scaler.last_scale_time else None
        }


# CLI Integration
def main():
    """CLI entry point for deployment toolkit"""
    import argparse
    
    parser = argparse.ArgumentParser(description="axiom Deployment Toolkit")
    parser.add_argument("command", choices=["init", "scale", "status", "generate"],
                       help="Command to execute")
    parser.add_argument("--project-name", default="axiom",
                       help="Project name")
    parser.add_argument("--output-dir", default="./deployment",
                       help="Output directory for generated files")
    parser.add_argument("--deployment-name", 
                       help="Kubernetes deployment name")
    parser.add_argument("--config", 
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    manager = DeploymentManager(args.project_name)
    
    if args.command == "init":
        files = manager.initialize_deployment(args.output_dir)
        print(f"Generated {len(files)} deployment files:")
        for name, path in files.items():
            print(f"  {name}: {path}")
    
    elif args.command == "scale":
        deployment_name = args.deployment_name or args.project_name
        print(f"Starting auto-scaling for {deployment_name}")
        manager.start_auto_scaling(deployment_name, args.config)
    
    elif args.command == "status":
        status = manager.get_deployment_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == "generate":
        # Generate specific components
        generator = DeploymentGenerator(args.project_name)
        
        # Example: Generate just the Dockerfile
        dockerfile = generator.generate_dockerfile()
        dockerfile_path = Path(args.output_dir) / "Dockerfile"
        dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile)
        
        print(f"Generated Dockerfile at {dockerfile_path}")


if __name__ == "__main__":
    main()