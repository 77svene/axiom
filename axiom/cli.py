from pathlib import Path
from subprocess import check_output
from sys import executable as python_executable

from axiom.core.utils import log
from axiom.engines.toolbelt.custom import Response
from axiom.core.utils._shell import _CookieParser, _ParseHeaders
from axiom.core._types import List, Optional, Dict, Tuple, Any, Callable

from orjson import loads as json_loads, JSONDecodeError

try:
    from click import command, option, Choice, group, argument
except (ImportError, ModuleNotFoundError) as e:
    raise ModuleNotFoundError(
        "You need to install axiom with any of the extras to enable Shell commands. See: https://axiom.readthedocs.io/en/latest/#installation"
    ) from e

__OUTPUT_FILE_HELP__ = "The output file path can be an HTML file, a Markdown file of the HTML content, or the text content itself. Use file extensions (`.html`/`.md`/`.txt`) respectively."
__PACKAGE_DIR__ = Path(__file__).parent


def __Execute(cmd: List[str], help_line: str) -> None:  # pragma: no cover
    print(f"Installing {help_line}...")
    _ = check_output(cmd, shell=False)  # nosec B603
    # I meant to not use try except here


def __ParseJSONData(json_string: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Parse JSON string into a Python object"""
    if not json_string:
        return None

    try:
        return json_loads(json_string)
    except JSONDecodeError as err:  # pragma: no cover
        raise ValueError(f"Invalid JSON data '{json_string}': {err}")


def __Request_and_Save(
    fetcher_func: Callable[..., Response],
    url: str,
    output_file: str,
    css_selector: Optional[str] = None,
    **kwargs,
) -> None:
    """Make a request using the specified fetcher function and save the result"""
    from axiom.core.shell import Convertor

    # Handle relative paths - convert to an absolute path based on the current working directory
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_file

    response = fetcher_func(url, **kwargs)
    Convertor.write_content_to_file(response, str(output_path), css_selector)
    log.info(f"Content successfully saved to '{output_path}'")


def __ParseExtractArguments(
    headers: List[str], cookies: str, params: str, json: Optional[str] = None
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Optional[Dict[str, str]]]:
    """Parse arguments for extract command"""
    parsed_headers, parsed_cookies = _ParseHeaders(headers)
    if cookies:
        for key, value in _CookieParser(cookies):
            try:
                parsed_cookies[key] = value
            except Exception as err:
                raise ValueError(f"Could not parse cookies '{cookies}': {err}")

    parsed_json = __ParseJSONData(json)
    parsed_params = {}
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)
            parsed_params[key] = value

    return parsed_headers, parsed_cookies, parsed_params, parsed_json


def __BuildRequest(headers: List[str], cookies: str, params: str, json: Optional[str] = None, **kwargs) -> Dict:
    """Build a request object using the specified arguments"""
    # Parse parameters
    parsed_headers, parsed_cookies, parsed_params, parsed_json = __ParseExtractArguments(headers, cookies, params, json)
    # Build request arguments
    request_kwargs: Dict[str, Any] = {
        "headers": parsed_headers if parsed_headers else None,
        "cookies": parsed_cookies if parsed_cookies else None,
    }
    if parsed_json:
        request_kwargs["json"] = parsed_json
    if parsed_params:
        request_kwargs["params"] = parsed_params
    if "proxy" in kwargs:
        request_kwargs["proxy"] = kwargs.pop("proxy")

    # Parse impersonate parameter if it contains commas (for random selection)
    if "impersonate" in kwargs and "," in (kwargs.get("impersonate") or ""):
        kwargs["impersonate"] = [browser.strip() for browser in kwargs["impersonate"].split(",")]

    return {**request_kwargs, **kwargs}


@command(help="Install all axiom's Fetchers dependencies")
@option(
    "-f",
    "--force",
    "force",
    is_flag=True,
    default=False,
    type=bool,
    help="Force axiom to reinstall all Fetchers dependencies",
)
def install(force):  # pragma: no cover
    if force or not __PACKAGE_DIR__.joinpath(".axiom_dependencies_installed").exists():
        __Execute(
            [python_executable, "-m", "playwright", "install", "chromium"],
            "Playwright browsers",
        )
        __Execute(
            [
                python_executable,
                "-m",
                "playwright",
                "install-deps",
                "chromium",
            ],
            "Playwright dependencies",
        )
        from tld.utils import update_tld_names

        update_tld_names(fail_silently=True)
        # if no errors raised by the above commands, then we add the below file
        __PACKAGE_DIR__.joinpath(".axiom_dependencies_installed").touch()
    else:
        print("The dependencies are already installed")


@command(help="Run axiom's MCP server (Check the docs for more info).")
@option(
    "--http",
    is_flag=True,
    default=False,
    help="Whether to run the MCP server in streamable-http transport or leave it as stdio (Default: False)",
)
@option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="The host to use if streamable-http transport is enabled (Default: '0.0.0.0')",
)
@option(
    "--port", type=int, default=8000, help="The port to use if streamable-http transport is enabled (Default: 8000)"
)
def mcp(http, host, port):
    from axiom.core.ai import axiomMCPServer

    server = axiomMCPServer()
    server.serve(http, host, port)


@command(help="Interactive scraping console")
@option(
    "-c",
    "--code",
    "code",
    is_flag=False,
    default="",
    type=str,
    help="Evaluate the code in the shell, print the result and exit",
)
@option(
    "-L",
    "--loglevel",
    "level",
    is_flag=False,
    default="debug",
    type=Choice(["debug", "info", "warning", "error", "critical", "fatal"], case_sensitive=False),
    help="Log level (default: DEBUG)",
)
def shell(code, level):
    from axiom.core.shell import CustomShell

    console = CustomShell(code=code, log_level=level)
    console.start()


@group(
    help="Fetch web pages using various fetchers and extract full/selected HTML content as HTML, Markdown, or extract text content."
)
def extract():
    """Extract content from web pages and save to files"""
    pass


@extract.command(help=f"Perform a GET request and save the content to a file.\n\n{__OUTPUT_FILE_HELP__}")
@argument("url", required=True)
@argument("output_file", required=True)
@option(
    "--headers",
    "-H",
    multiple=True,
    help='HTTP headers in format "Key: Value" (can be used multiple times)',
)
@option("--cookies", help='Cookies string in format "name1=value1; name2=value2"')
@option("--timeout", type=int, default=30, help="Request timeout in seconds (default: 30)")
@option("--proxy", help='Proxy URL in format "http://username:password@host:port"')
@option(
    "--css-selector",
    "-s",
    help="CSS selector to extract specific content from the page. It returns all matches.",
)
@option(
    "--params",
    "-p",
    multiple=True,
    help='Query parameters in format "key=value" (can be used multiple times)',
)
@option(
    "--follow-redirects/--no-follow-redirects",
    default=True,
    help="Whether to follow redirects (default: True)",
)
@option(
    "--verify/--no-verify",
    default=True,
    help="Whether to verify SSL certificates (default: True)",
)
@option(
    "--impersonate",
    help="Browser to impersonate. Can be a single browser (e.g., chrome) or comma-separated list for random selection (e.g., chrome,firefox,safari).",
)
@option(
    "--stealthy-headers/--no-stealthy-headers",
    default=True,
    help="Use stealthy headers to avoid detection (default: True)",
)
def get(url, output_file, headers, cookies, timeout, proxy, css_selector, params, follow_redirects, verify, impersonate, stealthy_headers):
    from axiom.engines import Fetcher
    
    request_kwargs = __BuildRequest(
        headers=list(headers),
        cookies=cookies,
        params=list(params),
        timeout=timeout,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        impersonate=impersonate,
        stealthy_headers=stealthy_headers,
    )
    
    __Request_and_Save(
        Fetcher,
        url,
        output_file,
        css_selector=css_selector,
        **request_kwargs,
    )


@extract.command(help=f"Perform a POST request and save the content to a file.\n\n{__OUTPUT_FILE_HELP__}")
@argument("url", required=True)
@argument("output_file", required=True)
@option(
    "--headers",
    "-H",
    multiple=True,
    help='HTTP headers in format "Key: Value" (can be used multiple times)',
)
@option("--cookies", help='Cookies string in format "name1=value1; name2=value2"')
@option("--data", help='Form data to send (e.g., "key1=value1&key2=value2")')
@option("--json", help='JSON data to send (e.g., \'{"key": "value"}\')')
@option("--timeout", type=int, default=30, help="Request timeout in seconds (default: 30)")
@option("--proxy", help='Proxy URL in format "http://username:password@host:port"')
@option(
    "--css-selector",
    "-s",
    help="CSS selector to extract specific content from the page. It returns all matches.",
)
@option(
    "--params",
    "-p",
    multiple=True,
    help='Query parameters in format "key=value" (can be used multiple times)',
)
@option(
    "--follow-redirects/--no-follow-redirects",
    default=True,
    help="Whether to follow redirects (default: True)",
)
@option(
    "--verify/--no-verify",
    default=True,
    help="Whether to verify SSL certificates (default: True)",
)
@option(
    "--impersonate",
    help="Browser to impersonate. Can be a single browser (e.g., chrome) or comma-separated list for random selection (e.g., chrome,firefox,safari).",
)
@option(
    "--stealthy-headers/--no-stealthy-headers",
    default=True,
    help="Use stealthy headers to avoid detection (default: True)",
)
def post(url, output_file, headers, cookies, data, json, timeout, proxy, css_selector, params, follow_redirects, verify, impersonate, stealthy_headers):
    from axiom.engines import Fetcher
    
    request_kwargs = __BuildRequest(
        headers=list(headers),
        cookies=cookies,
        params=list(params),
        json=json,
        timeout=timeout,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        impersonate=impersonate,
        stealthy_headers=stealthy_headers,
    )
    
    if data:
        request_kwargs["data"] = data
    
    __Request_and_Save(
        Fetcher,
        url,
        output_file,
        css_selector=css_selector,
        **request_kwargs,
    )


@group(help="Production Deployment Toolkit - Generate deployment configurations for Docker, Kubernetes, and cloud providers")
def deploy():
    """Generate production deployment configurations"""
    pass


@deploy.command(help="Generate Docker deployment files (Dockerfile, docker-compose.yml)")
@option("--output-dir", "-o", default=".", help="Output directory for generated files")
@option("--base-image", default="python:3.11-slim", help="Base Docker image")
@option("--port", type=int, default=8000, help="Port to expose in the container")
@option("--with-nginx", is_flag=True, help="Include nginx reverse proxy configuration")
@option("--with-redis", is_flag=True, help="Include Redis for queue management")
def docker(output_dir, base_image, port, with_nginx, with_redis):
    """Generate Docker deployment files"""
    import os
    from string import Template
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate Dockerfile
    dockerfile_template = Template("""# axiom Production Deployment
FROM $base_image

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libxml2-dev \\
    libxslt-dev \\
    python3-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 axiom && chown -R axiom:axiom /app
USER axiom

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:$port/health || exit 1

EXPOSE $port

CMD ["python", "-m", "axiom.cli", "serve", "--port", "$port"]
""")
    
    dockerfile_content = dockerfile_template.substitute(
        base_image=base_image,
        port=port
    )
    
    with open(output_path / "Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Generate docker-compose.yml
    compose_services = {
        "axiom": {
            "build": ".",
            "ports": [f"{port}:{port}"],
            "environment": [
                "PYTHONUNBUFFERED=1",
                f"PORT={port}"
            ],
            "volumes": ["./data:/app/data"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD", "curl", "-f", f"http://localhost:{port}/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "40s"
            }
        }
    }
    
    if with_redis:
        compose_services["redis"] = {
            "image": "redis:7-alpine",
            "ports": ["6379:6379"],
            "volumes": ["redis_data:/data"],
            "restart": "unless-stopped",
            "command": "redis-server --appendonly yes"
        }
        compose_services["axiom"]["depends_on"] = ["redis"]
        compose_services["axiom"]["environment"].append("REDIS_URL=redis://redis:6379/0")
    
    if with_nginx:
        compose_services["nginx"] = {
            "image": "nginx:alpine",
            "ports": ["80:80", "443:443"],
            "volumes": [
                "./nginx.conf:/etc/nginx/nginx.conf:ro",
                "./certs:/etc/nginx/certs:ro"
            ],
            "depends_on": ["axiom"],
            "restart": "unless-stopped"
        }
    
    compose_config = {
        "version": "3.8",
        "services": compose_services,
        "volumes": {"redis_data": {}} if with_redis else {}
    }
    
    import yaml
    with open(output_path / "docker-compose.yml", "w") as f:
        yaml.dump(compose_config, f, default_flow_style=False)
    
    # Generate nginx.conf if requested
    if with_nginx:
        nginx_config = f"""events {{
    worker_connections 1024;
}}

http {{
    upstream axiom {{
        server axiom:{port};
    }}
    
    server {{
        listen 80;
        server_name _;
        
        location / {{
            proxy_pass http://axiom;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }}
        
        location /health {{
            access_log off;
            proxy_pass http://axiom/health;
        }}
    }}
}}"""
        with open(output_path / "nginx.conf", "w") as f:
            f.write(nginx_config)
    
    # Generate .dockerignore
    dockerignore = """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
*.sqlite3
*.db
data/
certs/
"""
    with open(output_path / ".dockerignore", "w") as f:
        f.write(dockerignore)
    
    print(f"✅ Docker deployment files generated in {output_path}")
    print("Files created:")
    print("  - Dockerfile")
    print("  - docker-compose.yml")
    if with_nginx:
        print("  - nginx.conf")
    print("  - .dockerignore")


@deploy.command(help="Generate Kubernetes Helm chart with auto-scaling")
@option("--output-dir", "-o", default=".", help="Output directory for generated files")
@option("--app-name", default="axiom", help="Application name")
@option("--namespace", default="default", help="Kubernetes namespace")
@option("--image-repository", required=True, help="Docker image repository (e.g., docker.io/username/axiom)")
@option("--image-tag", default="latest", help="Docker image tag")
@option("--replicas", type=int, default=3, help="Number of replicas")
@option("--port", type=int, default=8000, help="Port the container listens on")
@option("--service-type", type=Choice(["ClusterIP", "NodePort", "LoadBalancer"]), default="ClusterIP", help="Kubernetes service type")
@option("--enable-autoscaling", is_flag=True, help="Enable horizontal pod autoscaling")
@option("--min-replicas", type=int, default=2, help="Minimum number of replicas for autoscaling")
@option("--max-replicas", type=int, default=10, help="Maximum number of replicas for autoscaling")
@option("--cpu-percent", type=int, default=80, help="Target CPU utilization percentage for autoscaling")
@option("--memory-percent", type=int, default=80, help="Target memory utilization percentage for autoscaling")
@option("--queue-metric", default="axiom_queue_depth", help="Custom metric name for queue-based scaling")
@option("--queue-threshold", type=int, default=100, help="Queue depth threshold for scaling")
def helm(output_dir, app_name, namespace, image_repository, image_tag, replicas, port, service_type, enable_autoscaling, min_replicas, max_replicas, cpu_percent, memory_percent, queue_metric, queue_threshold):
    """Generate Kubernetes Helm chart"""
    import os
    import yaml
    
    output_path = Path(output_dir) / app_name
    output_path.mkdir(parents=True, exist_ok=True)
    templates_dir = output_path / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Generate Chart.yaml
    chart_yaml = {
        "apiVersion": "v2",
        "name": app_name,
        "description": "axiom Production Deployment",
        "type": "application",
        "version": "0.1.0",
        "appVersion": "1.0.0",
        "maintainers": [
            {"name": "axiom Team", "email": "team@axiom.dev"}
        ]
    }
    
    with open(output_path / "Chart.yaml", "w") as f:
        yaml.dump(chart_yaml, f, default_flow_style=False)
    
    # Generate values.yaml
    values = {
        "replicaCount": replicas,
        "image": {
            "repository": image_repository,
            "tag": image_tag,
            "pullPolicy": "IfNotPresent"
        },
        "service": {
            "type": service_type,
            "port": port,
            "targetPort": port
        },
        "resources": {
            "limits": {
                "cpu": "1000m",
                "memory": "1Gi"
            },
            "requests": {
                "cpu": "500m",
                "memory": "512Mi"
            }
        },
        "autoscaling": {
            "enabled": enable_autoscaling,
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "targetCPUUtilizationPercentage": cpu_percent,
            "targetMemoryUtilizationPercentage": memory_percent,
            "customMetrics": [
                {
                    "type": "Pods",
                    "pods": {
                        "metric": {
                            "name": queue_metric
                        },
                        "target": {
                            "type": "AverageValue",
                            "averageValue": queue_threshold
                        }
                    }
                }
            ] if enable_autoscaling else []
        },
        "ingress": {
            "enabled": False,
            "annotations": {},
            "hosts": [
                {
                    "host": "axiom.local",
                    "paths": [{"path": "/", "pathType": "Prefix"}]
                }
            ],
            "tls": []
        },
        "nodeSelector": {},
        "tolerations": [],
        "affinity": {},
        "env": [
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "PORT", "value": str(port)},
            {"name": "ENVIRONMENT", "value": "production"}
        ],
        "livenessProbe": {
            "httpGet": {
                "path": "/health",
                "port": port
            },
            "initialDelaySeconds": 30,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 3
        },
        "readinessProbe": {
            "httpGet": {
                "path": "/health",
                "port": port
            },
            "initialDelaySeconds": 5,
            "periodSeconds": 10
        }
    }
    
    with open(output_path / "values.yaml", "w") as f:
        yaml.dump(values, f, default_flow_style=False)
    
    # Generate deployment template
    deployment_template = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "axiom.fullname" . }}
  labels:
    {{- include "axiom.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "axiom.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "axiom.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
          livenessProbe:
            {{- toYaml .Values.livenessProbe | nindent 12 }}
          readinessProbe:
            {{- toYaml .Values.readinessProbe | nindent 12 }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            {{- toYaml .Values.env | nindent 12 }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
"""
    
    with open(templates_dir / "deployment.yaml", "w") as f:
        f.write(deployment_template)
    
    # Generate service template
    service_template = """apiVersion: v1
kind: Service
metadata:
  name: {{ include "axiom.fullname" . }}
  labels:
    {{- include "axiom.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{- include "axiom.selectorLabels" . | nindent 4 }}
"""
    
    with open(templates_dir / "service.yaml", "w") as f:
        f.write(service_template)
    
    # Generate HPA template if autoscaling is enabled
    if enable_autoscaling:
        hpa_template = """{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "axiom.fullname" . }}
  labels:
    {{- include "axiom.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "axiom.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
    {{- range .Values.autoscaling.customMetrics }}
    - {{- toYaml . | nindent 6 }}
    {{- end }}
{{- end }}
"""
        with open(templates_dir / "hpa.yaml", "w") as f:
            f.write(hpa_template)
    
    # Generate helpers template
    helpers_template = """{{/*
Expand the name of the chart.
*/}}
{{- define "axiom.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "axiom.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "axiom.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "axiom.labels" -}}
helm.sh/chart: {{ include "axiom.chart" . }}
{{ include "axiom.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "axiom.selectorLabels" -}}
app.kubernetes.io/name: {{ include "axiom.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
"""
    
    with open(templates_dir / "_helpers.tpl", "w") as f:
        f.write(helpers_template)
    
    print(f"✅ Helm chart generated in {output_path}")
    print("Files created:")
    print("  - Chart.yaml")
    print("  - values.yaml")
    print("  - templates/deployment.yaml")
    print("  - templates/service.yaml")
    if enable_autoscaling:
        print("  - templates/hpa.yaml")
    print("  - templates/_helpers.tpl")


@deploy.command(help="Generate Terraform modules for cloud providers")
@option("--output-dir", "-o", default=".", help="Output directory for generated files")
@option("--cloud-provider", type=Choice(["aws", "gcp", "azure"]), required=True, help="Cloud provider")
@option("--region", required=True, help="Cloud region")
@option("--app-name", default="axiom", help="Application name")
@option("--instance-type", help="Instance type (default: t3.micro for AWS, e2-micro for GCP, Standard_B1s for Azure)")
@option("--min-instances", type=int, default=1, help="Minimum number of instances for auto-scaling")
@option("--max-instances", type=int, default=5, help="Maximum number of instances for auto-scaling")
@option("--desired-instances", type=int, default=2, help="Desired number of instances")
@option("--enable-cost-optimization", is_flag=True, help="Enable cost optimization (spot instances, preemptible VMs)")
@option("--queue-based-scaling", is_flag=True, help="Enable queue-based auto-scaling")
@option("--vpc-cidr", default="10.0.0.0/16", help="VPC CIDR block")
def terraform(output_dir, cloud_provider, region, app_name, instance_type, min_instances, max_instances, desired_instances, enable_cost_optimization, queue_based_scaling, vpc_cidr):
    """Generate Terraform modules for cloud providers"""
    import os
    
    output_path = Path(output_dir) / f"terraform-{cloud_provider}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set default instance types based on provider
    if not instance_type:
        instance_type = {
            "aws": "t3.micro",
            "gcp": "e2-micro",
            "azure": "Standard_B1s"
        }[cloud_provider]
    
    # Generate main.tf based on cloud provider
    if cloud_provider == "aws":
        main_tf = f"""# axiom AWS Deployment
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
  required_version = ">= 1.0"
}}

provider "aws" {{
  region = "{region}"
}}

# VPC Configuration
resource "aws_vpc" "axiom_vpc" {{
  cidr_block           = "{vpc_cidr}"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {{
    Name = "{app_name}-vpc"
  }}
}}

resource "aws_subnet" "public_subnet" {{
  count             = 2
  vpc_id            = aws_vpc.axiom_vpc.id
  cidr_block        = cidrsubnet(aws_vpc.axiom_vpc.cidr_block, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {{
    Name = "{app_name}-public-subnet-${{count.index}}"
  }}
}}

resource "aws_internet_gateway" "igw" {{
  vpc_id = aws_vpc.axiom_vpc.id
  
  tags = {{
    Name = "{app_name}-igw"
  }}
}}

resource "aws_route_table" "public_rt" {{
  vpc_id = aws_vpc.axiom_vpc.id
  
  route {{
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }}
  
  tags = {{
    Name = "{app_name}-public-rt"
  }}
}}

resource "aws_route_table_association" "public_rta" {{
  count          = 2
  subnet_id      = aws_subnet.public_subnet[count.index].id
  route_table_id = aws_route_table.public_rt.id
}}

# Security Group
resource "aws_security_group" "axiom_sg" {{
  name        = "{app_name}-sg"
  description = "Security group for axiom application"
  vpc_id      = aws_vpc.axiom_vpc.id
  
  ingress {{
    description = "HTTP from VPC"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  ingress {{
    description = "HTTPS from VPC"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  ingress {{
    description = "Application port"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  tags = {{
    Name = "{app_name}-sg"
  }}
}}

# Launch Template
resource "aws_launch_template" "axiom_lt" {{
  name_prefix   = "{app_name}-lt-"
  image_id      = data.aws_ami.amazon_linux_2.id
  instance_type = "{instance_type}"
  
  network_interfaces {{
    associate_public_ip_address = true
    security_groups             = [aws_security_group.axiom_sg.id]
  }}
  
  user_data = base64encode(<<-EOF
    #!/bin/bash
    yum update -y
    amazon-linux-extras install docker -y
    service docker start
    usermod -a -G docker ec2-user
    docker run -d -p 8000:8000 \\
      -e ENVIRONMENT=production \\
      --name axiom \\
      {app_name}:latest
  EOF
  )
  
  tag_specifications {{
    resource_type = "instance"
    
    tags = {{
      Name = "{app_name}-instance"
    }}
  }}
  
  {"instance_market_options { market_type = \"spot\" }" if enable_cost_optimization else ""}
}}

# Auto Scaling Group
resource "aws_autoscaling_group" "axiom_asg" {{
  name                = "{app_name}-asg"
  desired_capacity    = {desired_instances}
  max_size            = {max_instances}
  min_size            = {min_instances}
  vpc_zone_identifier = aws_subnet.public_subnet[*].id
  
  launch_template {{
    id      = aws_launch_template.axiom_lt.id
    version = "$Latest"
  }}
  
  tag {{
    key                 = "Name"
    value               = "{app_name}-asg-instance"
    propagate_at_launch = true
  }}
}}

# Application Load Balancer
resource "aws_lb" "axiom_lb" {{
  name               = "{app_name}-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.axiom_sg.id]
  subnets            = aws_subnet.public_subnet[*].id
  
  tags = {{
    Name = "{app_name}-lb"
  }}
}}

resource "aws_lb_target_group" "axiom_tg" {{
  name     = "{app_name}-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.axiom_vpc.id
  
  health_check {{
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    timeout             = 5
    unhealthy_threshold = 2
  }}
}}

resource "aws_lb_listener" "http" {{
  load_balancer_arn = aws_lb.axiom_lb.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {{
    type             = "forward"
    target_group_arn = aws_lb_target_group.axiom_tg.arn
  }}
}}

resource "aws_autoscaling_attachment" "asg_attachment" {{
  autoscaling_group_name = aws_autoscaling_group.axiom_asg.id
  lb_target_group_arn    = aws_lb_target_group.axiom_tg.arn
}}

# CloudWatch Alarms for Scaling
resource "aws_cloudwatch_metric_alarm" "cpu_high" {{
  alarm_name          = "{app_name}-cpu-high"
  comparison_operator = "GreaterThanEvaluationPeriods"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 120
  statistic           = "Average"
  threshold           = 80
  
  dimensions = {{
    AutoScalingGroupName = aws_autoscaling_group.axiom_asg.name
  }}
  
  alarm_description = "Scale up when CPU exceeds 80%"
  alarm_actions     = [aws_autoscaling_policy.scale_up.arn]
}}

resource "aws_autoscaling_policy" "scale_up" {{
  name                   = "{app_name}-scale-up"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.axiom_asg.name
}}

# Data sources
data "aws_availability_zones" "available" {{
  state = "available"
}}

data "aws_ami" "amazon_linux_2" {{
  most_recent = true
  owners      = ["amazon"]
  
  filter {{
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }}
}}

# Outputs
output "load_balancer_dns" {{
  value = aws_lb.axiom_lb.dns_name
}}

output "vpc_id" {{
  value = aws_vpc.axiom_vpc.id
}}
"""
    
    elif cloud_provider == "gcp":
        main_tf = f"""# axiom GCP Deployment
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
  }}
  required_version = ">= 1.0"
}}

provider "google" {{
  project = var.project_id
  region  = "{region}"
}}

variable "project_id" {{
  description = "GCP Project ID"
  type        = string
}}

# VPC Network
resource "google_compute_network" "axiom_vpc" {{
  name                    = "{app_name}-vpc"
  auto_create_subnetworks = false
}}

resource "google_compute_subnetwork" "axiom_subnet" {{
  name          = "{app_name}-subnet"
  ip_cidr_range = "{vpc_cidr}"
  region        = "{region}"
  network       = google_compute_network.axiom_vpc.id
}}

# Firewall Rules
resource "google_compute_firewall" "axiom_firewall" {{
  name    = "{app_name}-firewall"
  network = google_compute_network.axiom_vpc.name
  
  allow {{
    protocol = "tcp"
    ports    = ["80", "443", "8000"]
  }}
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["axiom"]
}}

# Instance Template
resource "google_compute_instance_template" "axiom_template" {{
  name_prefix  = "{app_name}-template-"
  machine_type = "{instance_type}"
  
  disk {{
    source_image = "cos-cloud/cos-stable"
    auto_delete  = true
    boot         = true
  }}
  
  network_interface {{
    network    = google_compute_network.axiom_vpc.id
    subnetwork = google_compute_subnetwork.axiom_subnet.id
    
    access_config {{
      // Ephemeral public IP
    }}
  }}
  
  metadata = {{
    startup-script = <<-EOF
      #!/bin/bash
      docker run -d -p 8000:8000 \\
        -e ENVIRONMENT=production \\
        --name axiom \\
        {app_name}:latest
    EOF
  }}
  
  tags = ["axiom"]
  
  scheduling {{
    preemptible        = {"true" if enable_cost_optimization else "false"}
    automatic_restart  = {"false" if enable_cost_optimization else "true"}
    on_host_maintenance = "TERMINATE"
  }}
  
  lifecycle {{
    create_before_destroy = true
  }}
}}

# Managed Instance Group
resource "google_compute_region_instance_group_manager" "axiom_mig" {{
  name               = "{app_name}-mig"
  base_instance_name = "{app_name}"
  region             = "{region}"
  
  version {{
    instance_template = google_compute_instance_template.axiom_template.id
  }}
  
  target_size = {desired_instances}
  
  named_port {{
    name = "http"
    port = 8000
  }}
  
  auto_healing_policies {{
    health_check      = google_compute_health_check.axiom_health_check.id
    initial_delay_sec = 300
  }}
}}

# Auto-scaler
resource "google_compute_region_autoscaler" "axiom_autoscaler" {{
  name   = "{app_name}-autoscaler"
  region = "{region}"
  target = google_compute_region_instance_group_manager.axiom_mig.id
  
  autoscaling_policy {{
    min_replicas    = {min_instances}
    max_replicas    = {max_instances}
    cooldown_period = 60
    
    cpu_utilization {{
      target = 0.8
    }}
    
    metric {{
      name   = "compute.googleapis.com/instance/cpu/utilization"
      target = 0.8
      type   = "GAUGE"
    }}
  }}
}}

# Health Check
resource "google_compute_health_check" "axiom_health_check" {{
  name                = "{app_name}-health-check"
  check_interval_sec  = 30
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3
  
  http_health_check {{
    port         = 8000
    request_path = "/health"
  }}
}}

# Load Balancer
resource "google_compute_global_address" "axiom_ip" {{
  name = "{app_name}-ip"
}}

resource "google_compute_backend_service" "axiom_backend" {{
  name                  = "{app_name}-backend"
  protocol              = "HTTP"
  port_name             = "http"
  load_balancing_scheme = "EXTERNAL"
  timeout_sec           = 30
  health_checks         = [google_compute_health_check.axiom_health_check.id]
  
  backend {{
    group = google_compute_region_instance_group_manager.axiom_mig.instance_group
  }}
}}

resource "google_compute_url_map" "axiom_url_map" {{
  name            = "{app_name}-url-map"
  default_service = google_compute_backend_service.axiom_backend.id
}}

resource "google_compute_target_http_proxy" "axiom_proxy" {{
  name    = "{app_name}-proxy"
  url_map = google_compute_url_map.axiom_url_map.id
}}

resource "google_compute_global_forwarding_rule" "axiom_forwarding_rule" {{
  name       = "{app_name}-forwarding-rule"
  target     = google_compute_target_http_proxy.axiom_proxy.id
  port_range = "80"
  ip_address = google_compute_global_address.axiom_ip.address
}}

# Outputs
output "load_balancer_ip" {{
  value = google_compute_global_address.axiom_ip.address
}}

output "instance_group" {{
  value = google_compute_region_instance_group_manager.axiom_mig.instance_group
}}
"""
    
    elif cloud_provider == "azure":
        main_tf = f"""# axiom Azure Deployment
terraform {{
  required_providers {{
    azurerm = {{
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }}
  }}
  required_version = ">= 1.0"
}}

provider "azurerm" {{
  features {{}}
}}

# Resource Group
resource "azurerm_resource_group" "axiom_rg" {{
  name     = "{app_name}-rg"
  location = "{region}"
  
  tags = {{
    application = "axiom"
  }}
}}

# Virtual Network
resource "azurerm_virtual_network" "axiom_vnet" {{
  name                = "{app_name}-vnet"
  address_space       = ["{vpc_cidr}"]
  location            = azurerm_resource_group.axiom_rg.location
  resource_group_name = azurerm_resource_group.axiom_rg.name
}}

resource "azurerm_subnet" "axiom_subnet" {{
  name                 = "{app_name}-subnet"
  resource_group_name  = azurerm_resource_group.axiom_rg.name
  virtual_network_name = azurerm_virtual_network.axiom_vnet.name
  address_prefixes     = [cidrsubnet("{vpc_cidr}", 8, 1)]
}}

# Network Security Group
resource "azurerm_network_security_group" "axiom_nsg" {{
  name                = "{app_name}-nsg"
  location            = azurerm_resource_group.axiom_rg.location
  resource_group_name = azurerm_resource_group.axiom_rg.name
  
  security_rule {{
    name                       = "HTTP"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }}
  
  security_rule {{
    name                       = "HTTPS"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }}
  
  security_rule {{
    name                       = "AppPort"
    priority                   = 120
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }}
}}

# Public IP
resource "azurerm_public_ip" "axiom_pip" {{
  name                = "{app_name}-pip"
  location            = azurerm_resource_group.axiom_rg.location
  resource_group_name = azurerm_resource_group.axiom_rg.name
  allocation_method   = "Static"
  sku                 = "Standard"
  
  tags = {{
    application = "axiom"
  }}
}}

# Load Balancer
resource "azurerm_lb" "axiom_lb" {{
  name                = "{app_name}-lb"
  location            = azurerm_resource_group.axiom_rg.location
  resource_group_name = azurerm_resource_group.axiom_rg.name
  sku                 = "Standard"
  
  frontend_ip_configuration {{
    name                 = "PublicIPAddress"
    public_ip_address_id = azurerm_public_ip.axiom_pip.id
  }}
}}

resource "azurerm_lb_backend_address_pool" "axiom_pool" {{
  loadbalancer_id = azurerm_lb.axiom_lb.id
  name            = "BackEndAddressPool"
}}

resource "azurerm_lb_probe" "axiom_probe" {{
  loadbalancer_id = azurerm_lb.axiom_lb.id
  name            = "http-probe"
  protocol        = "Http"
  request_path    = "/health"
  port            = 8000
}}

resource "azurerm_lb_rule" "axiom_rule" {{
  loadbalancer_id                = azurerm_lb.axiom_lb.id
  name                           = "HTTP"
  protocol                       = "Tcp"
  frontend_port                  = 80
  backend_port                   = 8000
  frontend_ip_configuration_name = "PublicIPAddress"
  backend_address_pool_ids       = [azurerm_lb_backend_address_pool.axiom_pool.id]
  probe_id                       = azurerm_lb_probe.axiom_probe.id
}}

# Virtual Machine Scale Set
resource "azurerm_linux_virtual_machine_scale_set" "axiom_vmss" {{
  name                = "{app_name}-vmss"
  resource_group_name = azurerm_resource_group.axiom_rg.name
  location            = azurerm_resource_group.axiom_rg.location
  sku                 = "{instance_type}"
  instances           = {desired_instances}
  admin_username      = "adminuser"
  
  admin_ssh_key {{
    username   = "adminuser"
    public_key = file("~/.ssh/id_rsa.pub")
  }}
  
  source_image_reference {{
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }}
  
  os_disk {{
    storage_account_type = "Standard_LRS"
    caching              = "ReadWrite"
  }}
  
  network_interface {{
    name    = "axiom-nic"
    primary = true
    
    ip_configuration {{
      name                                   = "internal"
      primary                                = true
      subnet_id                              = azurerm_subnet.axiom_subnet.id
      load_balancer_backend_address_pool_ids = [azurerm_lb_backend_address_pool.axiom_pool.id]
    }}
  }}
  
  custom_data = base64encode(<<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
    docker run -d -p 8000:8000 \\
      -e ENVIRONMENT=production \\
      --name axiom \\
      {app_name}:latest
  EOF
  )
  
  upgrade_mode = "Automatic"
  
  automatic_os_upgrade_policy {{
    disable_automatic_rollback  = false
    enable_automatic_os_upgrade = true
  }}
  
  rolling_upgrade_policy {{
    max_batch_instance_percent              = 20
    max_unhealthy_instance_percent          = 20
    max_unhealthy_upgraded_instance_percent = 5
    pause_time_between_batches              = "PT0S"
  }}
  
  tags = {{
    application = "axiom"
  }}
}}

# Auto-scaling
resource "azurerm_monitor_autoscale_setting" "axiom_autoscale" {{
  name                = "{app_name}-autoscale"
  resource_group_name = azurerm_resource_group.axiom_rg.name
  location            = azurerm_resource_group.axiom_rg.location
  target_resource_id  = azurerm_linux_virtual_machine_scale_set.axiom_vmss.id
  
  profile {{
    name = "defaultProfile"
    
    capacity {{
      default = {desired_instances}
      minimum = {min_instances}
      maximum = {max_instances}
    }}
    
    rule {{
      metric_trigger {{
        metric_name        = "Percentage CPU"
        metric_resource_id = azurerm_linux_virtual_machine_scale_set.axiom_vmss.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "GreaterThan"
        threshold          = 80
      }}
      
      scale_action {{
        direction = "Increase"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT5M"
      }}
    }}
    
    rule {{
      metric_trigger {{
        metric_name        = "Percentage CPU"
        metric_resource_id = azurerm_linux_virtual_machine_scale_set.axiom_vmss.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "LessThan"
        threshold          = 20
      }}
      
      scale_action {{
        direction = "Decrease"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT5M"
      }}
    }}
  }}
  
  notification {{
    email {{
      send_to_subscription_administrator    = true
      send_to_subscription_co_administrator = true
    }}
  }}
}}

# Outputs
output "load_balancer_ip" {{
  value = azurerm_public_ip.axiom_pip.ip_address
}}

output "resource_group_name" {{
  value = azurerm_resource_group.axiom_rg.name
}}
"""
    
    # Write main.tf
    with open(output_path / "main.tf", "w") as f:
        f.write(main_tf)
    
    # Generate variables.tf
    variables_tf = f"""variable "project_id" {{
  description = "Cloud provider project ID"
  type        = string
  default     = ""
}}

variable "environment" {{
  description = "Deployment environment"
  type        = string
  default     = "production"
}}

variable "app_name" {{
  description = "Application name"
  type        = string
  default     = "{app_name}"
}}

variable "region" {{
  description = "Cloud region"
  type        = string
  default     = "{region}"
}}
"""
    
    with open(output_path / "variables.tf", "w") as f:
        f.write(variables_tf)
    
    # Generate outputs.tf
    outputs_tf = """output "load_balancer_ip" {
  description = "Load balancer IP address"
  value       = var.cloud_provider == "aws" ? aws_lb.axiom_lb.dns_name : (var.cloud_provider == "gcp" ? google_compute_global_address.axiom_ip.address : azurerm_public_ip.axiom_pip.ip_address)
}

output "instance_group" {
  description = "Instance group or scale set"
  value       = var.cloud_provider == "aws" ? aws_autoscaling_group.axiom_asg.name : (var.cloud_provider == "gcp" ? google_compute_region_instance_group_manager.axiom_mig.instance_group : azurerm_linux_virtual_machine_scale_set.axiom_vmss.name)
}
"""
    
    with open(output_path / "outputs.tf", "w") as f:
        f.write(outputs_tf)
    
    # Generate terraform.tfvars
    tfvars = f"""project_id    = "your-project-id"
environment   = "production"
app_name      = "{app_name}"
region        = "{region}"
"""
    
    with open(output_path / "terraform.tfvars", "w") as f:
        f.write(tfvars)
    
    print(f"✅ Terraform module for {cloud_provider.upper()} generated in {output_path}")
    print("Files created:")
    print("  - main.tf")
    print("  - variables.tf")
    print("  - outputs.tf")
    print("  - terraform.tfvars")


@deploy.command(help="Generate CI/CD pipeline configurations")
@option("--output-dir", "-o", default=".", help="Output directory for generated files")
@option("--ci-provider", type=Choice(["github", "gitlab", "jenkins", "azure"]), required=True, help="CI/CD provider")
@option("--docker-registry", required=True, help="Docker registry (e.g., docker.io/username)")
@option("--app-name", default="axiom", help="Application name")
@option("--branch", default="main", help="Main branch for deployments")
@option("--enable-tests", is_flag=True, help="Include test stage in pipeline")
@option("--enable-security-scan", is_flag=True, help="Include security scanning")
@option("--enable-deploy", is_flag=True, help="Include deployment stage")
def cicd(output_dir, ci_provider, docker_registry, app_name, branch, enable_tests, enable_security_scan, enable_deploy):
    """Generate CI/CD pipeline configurations"""
    import os
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if ci_provider == "github":
        workflows_dir = output_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate main workflow
        workflow = f"""name: axiom CI/CD Pipeline

on:
  push:
    branches: [ {branch} ]
  pull_request:
    branches: [ {branch} ]

env:
  DOCKER_REGISTRY: {docker_registry}
  APP_NAME: {app_name}
  PYTHON_VERSION: '3.11'

jobs:
  test:
    runs-on: ubuntu-latest
    {"if: always()" if enable_tests else "if: false"}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{{{ env.PYTHON_VERSION }}}}
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ env.PYTHON_VERSION }}}}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=axiom --cov-report=xml
    
    - name: Upload coverage report
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
  
  security-scan:
    runs-on: ubuntu-latest
    {"needs: [test]" if enable_tests else ""}
    {"if: always()" if enable_security_scan else "if: false"}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'table'
        exit-code: '1'
        ignore-unfixed: true
        vuln-type: 'os,library'
        severity: 'CRITICAL,HIGH'
    
    - name: Run Bandit security linter
      uses: py-actions/bandit@v2
      with:
        path: "."
        format: "txt"
        exit: true
  
  build:
    runs-on: ubuntu-latest
    needs: ${{{{ needs.test.result == 'success' || !{enable_tests} }}}}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Docker Registry
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.DOCKER_REGISTRY }}}}
        username: ${{{{ secrets.DOCKER_USERNAME }}}}
        password: ${{{{ secrets.DOCKER_PASSWORD }}}}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{{{ env.DOCKER_REGISTRY }}}}/${{{{ env.APP_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix=${{{{ github.ref_name }}}}-
          type=raw,value=latest,enable=${{{{ github.ref == format('refs/heads/{branch}') }}}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}
        cache-from: type=gha
        cache-to: type=gha,mode=max
  
  deploy:
    runs-on: ubuntu-latest
    needs: [build]
    {"if: github.ref == 'refs/heads/" + branch + "' && github.event_name == 'push'" if enable_deploy else "if: false"}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production
      run: |
        echo "Deploying ${{{{ env.APP_NAME }}}} to production..."
        # Add deployment commands here
        # Example: kubectl set image deployment/${{{{ env.APP_NAME }}}}} ${{{{ env.APP_NAME }}}}}=${{{{ env.DOCKER_REGISTRY }}}}/${{{{ env.APP_NAME }}}}}:latest
"""
        
        with open(workflows_dir / "ci-cd.yml", "w") as f:
            f.write(workflow)
        
        # Generate dependabot configuration
        dependabot = """version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "team-lead"
    labels:
      - "dependencies"
  
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "docker"
  
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "ci"
"""
        
        with open(output_path / ".github" / "dependabot.yml", "w") as f:
            f.write(dependabot)
        
        print(f"✅ GitHub Actions workflow generated in {workflows_dir}")
        print("Files created:")
        print("  - .github/workflows/ci-cd.yml")
        print("  - .github/dependabot.yml")
    
    elif ci_provider == "gitlab":
        gitlab_ci = f"""image: python:3.11-slim

variables:
  DOCKER_REGISTRY: {docker_registry}
  APP_NAME: {app_name}
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"

cache:
  paths:
    - .pip-cache/
    - .venv/

stages:
  - test
  - security
  - build
  - deploy

before_script:
  - python -V
  - pip install --upgrade pip
  - pip install -r requirements.txt

test:
  stage: test
  {"rules:" if enable_tests else "rules:"}
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "{branch}"
  script:
    - pip install pytest pytest-cov
    - pytest tests/ --cov=axiom --cov-report=xml
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    expire_in: 1 week
  coverage: '/TOTAL.*?(\d+(?:\.\d+)?%)/'

security_scan:
  stage: security
  {"needs: [test]" if enable_tests else ""}
  {"rules:" if enable_security_scan else "rules:"}
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "{branch}"
  script:
    - pip install bandit safety
    - bandit -r . -f json -o bandit-report.json || true
    - safety check --json --output safety-report.json || true
  artifacts:
    paths:
      - bandit-report.json
      - safety-report.json
    expire_in: 1 week

build:
  stage: build
  image: docker:24.0.5
  services:
    - docker:24.0.5-dind
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $DOCKER_REGISTRY/$APP_NAME:$CI_COMMIT_SHA .
    - docker tag $DOCKER_REGISTRY/$APP_NAME:$CI_COMMIT_SHA $DOCKER_REGISTRY/$APP_NAME:latest
    - docker push $DOCKER_REGISTRY/$APP_NAME:$CI_COMMIT_SHA
    - docker push $DOCKER_REGISTRY/$APP_NAME:latest
  rules:
    - if: $CI_COMMIT_BRANCH == "{branch}"

deploy_production:
  stage: deploy
  {"needs: [build]" if enable_deploy else ""}
  {"rules:" if enable_deploy else "rules:"}
    - if: $CI_COMMIT_BRANCH == "{branch}"
      when: manual
  script:
    - echo "Deploying $APP_NAME to production..."
    # Add deployment commands here
    # Example: kubectl set image deployment/$APP_NAME $APP_NAME=$DOCKER_REGISTRY/$APP_NAME:$CI_COMMIT_SHA
  environment:
    name: production
    url: https://$APP_NAME.example.com
"""
        
        with open(output_path / ".gitlab-ci.yml", "w") as f:
            f.write(gitlab_ci)
        
        print(f"✅ GitLab CI configuration generated in {output_path}")
        print("Files created:")
        print("  - .gitlab-ci.yml")
    
    elif ci_provider == "jenkins":
        jenkinsfile = f"""pipeline {{
    agent any
    
    environment {{
        DOCKER_REGISTRY = '{docker_registry}'
        APP_NAME = '{app_name}'
        PYTHON_VERSION = '3.11'
    }}
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}
        
        stage('Setup') {{
            steps {{
                sh 'python${{PYTHON_VERSION}} -m venv .venv'
                sh '. .venv/bin/activate && pip install --upgrade pip'
                sh '. .venv/bin/activate && pip install -r requirements.txt'
            }}
        }}
        
        stage('Test') {{
            {"when { expression { return true } }" if enable_tests else "when { expression { return false } }"}
            steps {{
                sh '. .venv/bin/activate && pip install pytest pytest-cov'
                sh '. .venv/bin/activate && pytest tests/ --cov=axiom --cov-report=xml'
            }}
            post {{
                always {{
                    junit 'test-results/*.xml'
                    cobertura coberturaReportFile: 'coverage.xml'
                }}
            }}
        }}
        
        stage('Security Scan') {{
            {"when { expression { return true } }" if enable_security_scan else "when { expression { return false } }"}
            steps {{
                sh '. .venv/bin/activate && pip install bandit safety'
                sh '. .venv/bin/activate && bandit -r . -f json -o bandit-report.json || true'
                sh '. .venv/bin/activate && safety check --json --output safety-report.json || true'
            }}
            post {{
                always {{
                    archiveArtifacts artifacts: '*-report.json', allowEmptyArchive: true
                }}
            }}
        }}
        
        stage('Build Docker Image') {{
            steps {{
                script {{
                    docker.build("${{DOCKER_REGISTRY}}/${{APP_NAME}}:${{env.BUILD_NUMBER}}")
                    docker.build("${{DOCKER_REGISTRY}}/${{APP_NAME}}:latest")
                }}
            }}
        }}
        
        stage('Push Docker Image') {{
            steps {{
                script {{
                    docker.withRegistry('https://${{DOCKER_REGISTRY}}', 'docker-credentials') {{
                        docker.image("${{DOCKER_REGISTRY}}/${{APP_NAME}}:${{env.BUILD_NUMBER}}").push()
                        docker.image("${{DOCKER_REGISTRY}}/${{APP_NAME}}:latest").push()
                    }}
                }}
            }}
        }}
        
        stage('Deploy to Production') {{
            {"when { branch '{branch}' }" if enable_deploy else "when { expression { return false } }"}
            steps {{
                echo "Deploying ${{APP_NAME}} to production..."
                // Add deployment commands here
                // Example: sh 'kubectl set image deployment/${{APP_NAME}} ${{APP_NAME}}=${{DOCKER_REGISTRY}}/${{APP_NAME}}:${{env.BUILD_NUMBER}}'
            }}
        }}
    }}
    
    post {{
        always {{
            cleanWs()
        }}
        success {{
            slackSend channel: '#deployments',
                      color: 'good',
                      message: "The pipeline ${{currentBuild.fullDisplayName}} completed successfully."
        }}
        failure {{
            slackSend channel: '#deployments',
                      color: 'danger',
                      message: "The pipeline ${{currentBuild.fullDisplayName}} failed."
        }}
    }}
}}
"""
        
        with open(output_path / "Jenkinsfile", "w") as f:
            f.write(jenkinsfile)
        
        print(f"✅ Jenkinsfile generated in {output_path}")
        print("Files created:")
        print("  - Jenkinsfile")
    
    elif ci_provider == "azure":
        azure_pipeline = f"""trigger:
  branches:
    include:
    - {branch}

pool:
  vmImage: 'ubuntu-latest'

variables:
  dockerRegistry: '{docker_registry}'
  appName: '{app_name}'
  pythonVersion: '3.11'

stages:
- stage: Test
  {"displayName: 'Test Stage'" if enable_tests else "displayName: 'Test Stage' condition: always()"}
  jobs:
  - job: TestJob
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
      displayName: 'Install dependencies'
    
    - script: |
        pytest tests/ --cov=axiom --cov-report=xml --junitxml=test-results.xml
      displayName: 'Run tests'
    
    - task: PublishTestResults@2
      inputs:
        testResultsFormat: 'JUnit'
        testResultsFiles: '**/test-results.xml'
        failTaskOnFailedTests: true
    
    - task: PublishCodeCoverageResults@1
      inputs:
        codeCoverageTool: 'Cobertura'
        summaryFileLocation: '$(System.DefaultWorkingDirectory)/coverage.xml'

- stage: SecurityScan
  {"displayName: 'Security Scan' dependsOn: Test" if enable_security_scan else "displayName: 'Security Scan' condition: always()"}
  jobs:
  - job: SecurityScanJob
    steps:
    - script: |
        pip install bandit safety
        bandit -r . -f json -o bandit-report.json || true
        safety check --json --output safety-report.json || true
      displayName: 'Run security scans'
    
    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: 'bandit-report.json'
        artifactName: 'security-reports'
    
    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: 'safety-report.json'
        artifactName: 'security-reports'

- stage: Build
  displayName: 'Build and Push Docker Image'
  dependsOn: ${{{{ if eq(true, {enable_tests}) }}}}Test{{{{ else }}}}\"\"{{{{ end }}}}
  jobs:
  - job: BuildJob
    steps:
    - task: Docker@2
      displayName: 'Build Docker image'
      inputs:
        containerRegistry: 'DockerHub'
        repository: '$(dockerRegistry)/$(appName)'
        command: 'build'
        Dockerfile: '**/Dockerfile'
        tags: |
          $(Build.BuildId)
          latest
    
    - task: Docker@2
      displayName: 'Push Docker image'
      inputs:
        containerRegistry: 'DockerHub'
        repository: '$(dockerRegistry)/$(appName)'
        command: 'push'
        tags: |
          $(Build.BuildId)
          latest

- stage: Deploy
  displayName: 'Deploy to Production'
  {"dependsOn: Build" if enable_deploy else "condition: always()"}
  jobs:
  - deployment: DeployProduction
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - script: |
              echo "Deploying $(appName) to production..."
              # Add deployment commands here
              # Example: kubectl set image deployment/$(appName) $(appName)=$(dockerRegistry)/$(appName):$(Build.BuildId)
            displayName: 'Deploy application'
"""
        
        with open(output_path / "azure-pipelines.yml", "w") as f:
            f.write(azure_pipeline)
        
        print(f"✅ Azure DevOps pipeline generated in {output_path}")
        print("Files created:")
        print("  - azure-pipelines.yml")


@deploy.command(help="Generate monitoring and observability configurations")
@option("--output-dir", "-o", default=".", help="Output directory for generated files")
@option("--app-name", default="axiom", help="Application name")
@option("--enable-prometheus", is_flag=True, help="Include Prometheus configuration")
@option("--enable-grafana", is_flag=True, help="Include Grafana dashboards")
@option("--enable-logging", is_flag=True, help="Include centralized logging configuration")
@option("--enable-alerts", is_flag=True, help="Include alerting rules")
def monitoring(output_dir, app_name, enable_prometheus, enable_grafana, enable_logging, enable_alerts):
    """Generate monitoring and observability configurations"""
    import os
    
    output_path = Path(output_dir) / "monitoring"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if enable_prometheus:
        prometheus_dir = output_path / "prometheus"
        prometheus_dir.mkdir(exist_ok=True)
        
        prometheus_config = f"""global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: '{app_name}'
    static_configs:
      - targets: ['{app_name}:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
  
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"""
        
        with open(prometheus_dir / "prometheus.yml", "w") as f:
            f.write(prometheus_config)
        
        if enable_alerts:
            alert_rules = f"""groups:
- name: {app_name}_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{{status=~"5.."}}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 10% for 5 minutes"
  
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is above 1 second"
  
  - alert: HighCPU
    expr: process_cpu_seconds_total > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is above 80%"
  
  - alert: HighMemory
    expr: process_resident_memory_bytes / 1024 / 1024 > 512
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 512MB"
  
  - alert: QueueDepthHigh
    expr: axiom_queue_depth > 1000
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Queue depth is high"
      description: "Queue depth exceeds 1000 items"
"""
            
            with open(prometheus_dir / "alert_rules.yml", "w") as f:
                f.write(alert_rules)
        
        print(f"✅ Prometheus configuration generated in {prometheus_dir}")
    
    if enable_grafana:
        grafana_dir = output_path / "grafana"
        grafana_dir.mkdir(exist_ok=True)
        
        dashboard = {
            "dashboard": {
                "title": f"{app_name} Dashboard",
                "panels": [
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [{"expr": f"rate(http_requests_total{{job='{app_name}'}}[5m])"}]
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [{"expr": f"rate(http_requests_total{{job='{app_name}', status=~'5..'}}[5m])"}]
                    },
                    {
                        "title": "Latency (95th percentile)",
                        "type": "graph",
                        "targets": [{"expr": f"histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{job='{app_name}'}}[5m]))"}]
                    },
                    {
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [{"expr": f"rate(process_cpu_seconds_total{{job='{app_name}'}}[5m])"}]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [{"expr": f"process_resident_memory_bytes{{job='{app_name}'}} / 1024 / 1024"}]
                    },
                    {
                        "title": "Queue Depth",
                        "type": "graph",
                        "targets": [{"expr": f"axiom_queue_depth{{job='{app_name}'}}"}]
                    }
                ],
                "refresh": "10s",
                "time": {"from": "now-1h", "to": "now"}
            }
        }
        
        import json
        with open(grafana_dir / "dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=2)
        
        print(f"✅ Grafana dashboard generated in {grafana_dir}")
    
    if enable_logging:
        logging_dir = output_path / "logging"
        logging_dir.mkdir(exist_ok=True)
        
        filebeat_config = f"""filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
    - add_docker_metadata:
        host: "unix:///var/run/docker.sock"

processors:
  - add_kubernetes_metadata:
      host: ${HOSTNAME}
      matchers:
      - logs_path:
          logs_path: "/var/log/containers/"

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  indices:
    - index: "filebeat-{app_name}-%{{[agent.version]}}-%{{+yyyy.MM.dd}}"

setup.kibana:
  host: "kibana:5601"

logging.level: info
"""
        
        with open(logging_dir / "filebeat.yml", "w") as f:
            f.write(filebeat_config)
        
        print(f"✅ Logging configuration generated in {logging_dir}")
    
    print(f"\n📊 Monitoring configurations generated in {output_path}")
    print("Next steps:")
    print("1. Deploy Prometheus and Grafana using Helm:")
    print("   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts")
    print("   helm install prometheus prometheus-community/kube-prometheus-stack")
    print("2. Import the Grafana dashboard")
    print("3. Configure alert notification channels in Alertmanager")


@deploy.command(help="Generate cost optimization reports and recommendations")
@option("--output-dir", "-o", default=".", help="Output directory for generated files")
@option("--cloud-provider", type=Choice(["aws", "gcp", "azure", "all"]), default="all", help="Cloud provider to analyze")
@option("--app-name", default="axiom", help="Application name")
@option("--current-cost", type=float, default=1000.0, help="Current monthly cost in USD")
@option("--generate-report", is_flag=True, help="Generate detailed cost optimization report")
def cost(output_dir, cloud_provider, app_name, current_cost, generate_report):
    """Generate cost optimization recommendations"""
    import os
    
    output_path = Path(output_dir) / "cost-optimization"
    output_path.mkdir(parents=True, exist_ok=True)
    
    recommendations = {
        "general": [
            "Implement auto-scaling to match demand and reduce idle resources",
            "Use spot/preemptible instances for non-critical workloads (save 60-90%)",
            "Right-size instances based on actual CPU/memory utilization",
            "Implement lifecycle policies for logs and temporary data",
            "Use reserved instances for predictable workloads (save 30-60%)",
            "Optimize container images to reduce storage and transfer costs",
            "Implement caching to reduce compute and database load",
            "Use serverless functions for intermittent workloads"
        ],
        "aws": [
            "Use AWS Savings Plans instead of Reserved Instances for flexibility",
            "Implement S3 Intelligent-Tiering for automatic storage class optimization",
            "Use AWS Graviton processors for better price-performance",
            "Enable AWS Cost Explorer and set up budget alerts",
            "Use AWS Lambda for lightweight scraping tasks",
            "Implement S3 lifecycle policies to move old data to Glacier"
        ],
        "gcp": [
            "Use GCP Committed Use Discounts for predictable workloads",
            "Implement GCP Preemptible VMs for fault-tolerant workloads",
            "Use GCP Cloud Functions for event-driven scraping",
            "Enable GCP Recommender for personalized cost optimization tips",
            "Use GCP Cloud Storage classes (Standard, Nearline, Coldline, Archive)",
            "Implement GCP BigQuery for cost-effective data analysis"
        ],
        "azure": [
            "Use Azure Reserved VM Instances for predictable workloads",
            "Implement Azure Spot VMs for flexible workloads",
            "Use Azure Functions for serverless scraping",
            "Enable Azure Cost Management + Billing for monitoring",
            "Use Azure Blob Storage tiers (Hot, Cool, Archive)",
            "Implement Azure Advisor for cost recommendations"
        ]
    }
    
    # Calculate potential savings
    savings_potential = {
        "auto_scaling": current_cost * 0.3,  # 30% savings
        "spot_instances": current_cost * 0.7,  # 70% savings
        "right_sizing": current_cost * 0.2,  # 20% savings
        "reserved_instances": current_cost * 0.4,  # 40% savings
    }
    
    total_potential_savings = sum(savings_potential.values())
    
    # Generate report
    report = f"""# Cost Optimization Report for {app_name}
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
Current Monthly Cost: ${current_cost:,.2f}
Potential Monthly Savings: ${total_potential_savings:,.2f} ({(total_potential_savings/current_cost)*100:.1f}%)

## Recommendations

### General Recommendations (All Cloud Providers)
{chr(10).join(f'- {rec}' for rec in recommendations['general'])}

### Provider-Specific Recommendations
"""
    
    if cloud_provider in ['aws', 'all']:
        report += f"""
#### AWS Recommendations
{chr(10).join(f'- {rec}' for rec in recommendations['aws'])}
"""
    
    if cloud_provider in ['gcp', 'all']:
        report += f"""
#### GCP Recommendations
{chr(10).join(f'- {rec}' for rec in recommendations['gcp'])}
"""
    
    if cloud_provider in ['azure', 'all']:
        report += f"""
#### Azure Recommendations
{chr(10).join(f'- {rec}' for rec in recommendations['azure'])}
"""
    
    report += f"""
## Savings Breakdown

| Optimization Strategy | Potential Savings | Percentage |
|----------------------|-------------------|------------|
| Auto-scaling         | ${savings_potential['auto_scaling']:,.2f} | 30%        |
| Spot/Preemptible Instances | ${savings_potential['spot_instances']:,.2f} | 70%        |
| Right-sizing         | ${savings_potential['right_sizing']:,.2f} | 20%        |
| Reserved Instances   | ${savings_potential['reserved_instances']:,.2f} | 40%        |
| **Total**            | **${total_potential_savings:,.2f}** | **{(total_potential_savings/current_cost)*100:.1f}%** |

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1-2)
1. Implement auto-scaling based on queue depth
2. Enable cost monitoring and alerts
3. Right-size current instances based on utilization metrics

### Phase 2: Medium-term (Month 1-2)
1. Migrate to spot/preemptible instances for non-critical workloads
2. Implement caching layer to reduce compute load
3. Optimize container images and storage

### Phase 3: Long-term (Month 3-6)
1. Purchase reserved instances for predictable workloads
2. Implement serverless architecture for intermittent tasks
3. Optimize data storage and lifecycle policies

## Monitoring Metrics to Track
- Cost per request
- Resource utilization (CPU, memory, network)
- Queue depth and processing time
- Storage usage and growth rate
- Auto-scaling efficiency

## Tools and Resources
- AWS Cost Explorer / GCP Cost Management / Azure Cost Management
- Kubernetes Metrics Server for auto-scaling
- Prometheus for custom metrics
- Terraform for infrastructure cost tracking

## Next Steps
1. Review and prioritize recommendations
2. Implement monitoring for cost metrics
3. Start with Phase 1 quick wins
4. Schedule monthly cost review meetings
"""
    
    with open(output_path / "cost-optimization-report.md", "w") as f:
        f.write(report)
    
    # Generate Terraform variables for cost optimization
    cost_vars = f"""# Cost Optimization Variables
variable "enable_cost_optimization" {{
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}}

variable "use_spot_instances" {{
  description = "Use spot/preemptible instances"
  type        = bool
  default     = true
}}

variable "spot_instance_max_price" {{
  description = "Maximum price for spot instances (USD per hour)"
  type        = string
  default     = "0.05"
}}

variable "enable_auto_scaling" {{
  description = "Enable auto-scaling"
  type        = bool
  default     = true
}}

variable "min_scaling_replicas" {{
  description = "Minimum number of replicas for auto-scaling"
  type        = number
  default     = 1
}}

variable "max_scaling_replicas" {{
  description = "Maximum number of replicas for auto-scaling"
  type        = number
  default     = 10
}}

variable "scaling_cpu_threshold" {{
  description = "CPU threshold for scaling (percentage)"
  type        = number
  default     = 70
}}

variable "enable_lifecycle_policies" {{
  description = "Enable storage lifecycle policies"
  type        = bool
  default     = true
}}

variable "log_retention_days" {{
  description = "Number of days to retain logs"
  type        = number
  default     = 30
}}
"""
    
    with open(output_path / "cost-optimization.tfvars", "w") as f:
        f.write(cost_vars)
    
    # Generate monitoring dashboard for cost
    cost_dashboard = {
        "dashboard": {
            "title": f"{app_name} Cost Dashboard",
            "panels": [
                {
                    "title": "Monthly Cost Trend",
                    "type": "graph",
                    "targets": [{"expr": "sum(cloud_cost_total) by (month)"}]
                },
                {
                    "title": "Cost by Service",
                    "type": "pie",
                    "targets": [{"expr": "sum(cloud_cost_total) by (service)"}]
                },
                {
                    "title": "Cost vs Budget",
                    "type": "gauge",
                    "targets": [{"expr": "cloud_cost_total / cloud_budget_total * 100"}]
                },
                {
                    "title": "Savings Achieved",
                    "type": "stat",
                    "targets": [{"expr": "cloud_savings_total"}]
                }
            ]
        }
    }
    
    import json
    with open(output_path / "cost-dashboard.json", "w") as f:
        json.dump(cost_dashboard, f, indent=2)
    
    print(f"✅ Cost optimization report generated in {output_path}")
    print("Files created:")
    print("  - cost-optimization-report.md")
    print("  - cost-optimization.tfvars")
    print("  - cost-dashboard.json")
    print(f"\n💰 Potential monthly savings: ${total_potential_savings:,.2f} ({(total_potential_savings/current_cost)*100:.1f}%)")


# Add the deploy group to the main CLI
# Note: In a real implementation, you would add this to the main CLI group
# For this example, we're showing it as a standalone command group
if __name__ == "__main__":
    # This allows testing the deploy commands directly
    deploy()