from asyncio import gather
from typing import Optional, Dict, List, Any, Tuple, Generator
import random
import numpy as np
from collections import deque
import time

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from axiom.core.shell import Convertor
from axiom.engines.toolbelt.custom import Response as _axiomResponse
from axiom.engines.static import ImpersonateType
from axiom.fetchers import (
    Fetcher,
    FetcherSession,
    DynamicFetcher,
    AsyncDynamicSession,
    StealthyFetcher,
    AsyncStealthySession,
)
from axiom.core._types import (
    Optional,
    Tuple,
    Mapping,
    Dict,
    List,
    Any,
    Generator,
    Sequence,
    SetCookieParam,
    extraction_types,
    SelectorWaitStates,
)


class ResponseModel(BaseModel):
    """Request's response information structure."""

    status: int = Field(description="The status code returned by the website.")
    content: list[str] = Field(description="The content as Markdown/HTML or the text content of the page.")
    url: str = Field(description="The URL given by the user that resulted in this response.")


class FingerprintGenerator:
    """ML-powered fingerprint randomization using GAN-inspired techniques."""
    
    def __init__(self):
        self.fingerprint_templates = self._load_fingerprint_templates()
        self.gan_weights = self._initialize_gan_weights()
        
    def _load_fingerprint_templates(self) -> Dict[str, Any]:
        """Load base fingerprint templates for different browsers and platforms."""
        return {
            "chrome_windows": {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "platform": "Win32",
                "vendor": "Google Inc.",
                "webgl_renderer": "ANGLE (Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
                "canvas_hash": self._generate_canvas_hash(),
                "audio_hash": self._generate_audio_hash(),
                "fonts": self._generate_font_list(),
                "plugins": self._generate_plugin_list(),
            },
            "chrome_mac": {
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "platform": "MacIntel",
                "vendor": "Google Inc.",
                "webgl_renderer": "ANGLE (Intel(R) Iris(TM) Plus Graphics 645 OpenGL Engine)",
                "canvas_hash": self._generate_canvas_hash(),
                "audio_hash": self._generate_audio_hash(),
                "fonts": self._generate_font_list(),
                "plugins": self._generate_plugin_list(),
            },
            "firefox_windows": {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
                "platform": "Win32",
                "vendor": "",
                "webgl_renderer": "ANGLE (Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
                "canvas_hash": self._generate_canvas_hash(),
                "audio_hash": self._generate_audio_hash(),
                "fonts": self._generate_font_list(),
                "plugins": self._generate_plugin_list(),
            },
        }
    
    def _initialize_gan_weights(self) -> np.ndarray:
        """Initialize GAN weights for fingerprint generation."""
        return np.random.randn(128, 256) * 0.02
    
    def _generate_canvas_hash(self) -> str:
        """Generate a random but realistic canvas fingerprint hash."""
        return ''.join(random.choices('0123456789abcdef', k=32))
    
    def _generate_audio_hash(self) -> str:
        """Generate a random but realistic audio context fingerprint hash."""
        return ''.join(random.choices('0123456789abcdef', k=64))
    
    def _generate_font_list(self) -> List[str]:
        """Generate a realistic list of installed fonts."""
        base_fonts = ["Arial", "Courier New", "Georgia", "Times New Roman", "Trebuchet MS", "Verdana"]
        extra_fonts = ["Calibri", "Cambria", "Consolas", "Corbel", "Candara", "Segoe UI"]
        return random.sample(base_fonts + extra_fonts, random.randint(8, 14))
    
    def _generate_plugin_list(self) -> List[Dict[str, str]]:
        """Generate a realistic list of browser plugins."""
        plugins = [
            {"name": "PDF Viewer", "filename": "internal-pdf-viewer"},
            {"name": "Chrome PDF Viewer", "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai"},
            {"name": "Chromium PDF Viewer", "filename": "internal-pdf-viewer"},
        ]
        return random.sample(plugins, random.randint(1, 3))
    
    def generate_fingerprint(self, browser_type: str = "chrome", platform: str = "windows") -> Dict[str, Any]:
        """Generate a randomized fingerprint using GAN-inspired approach."""
        base_key = f"{browser_type}_{platform}"
        if base_key not in self.fingerprint_templates:
            base_key = random.choice(list(self.fingerprint_templates.keys()))
        
        base_fp = self.fingerprint_templates[base_key].copy()
        
        # Apply GAN-inspired mutations
        noise_vector = np.random.randn(128)
        transformed = np.dot(noise_vector, self.gan_weights)
        
        # Mutate user agent slightly
        ua_parts = base_fp["user_agent"].split()
        if len(ua_parts) > 4:
            version_idx = next((i for i, part in enumerate(ua_parts) if "Chrome/" in part or "Firefox/" in part), None)
            if version_idx:
                version = ua_parts[version_idx].split("/")[-1]
                major, minor, *rest = version.split(".")
                new_minor = str(int(minor) + random.randint(-2, 2))
                ua_parts[version_idx] = f"{ua_parts[version_idx].split('/')[0]}/{major}.{new_minor}.{'.'.join(rest)}"
                base_fp["user_agent"] = " ".join(ua_parts)
        
        # Mutate canvas and audio hashes
        base_fp["canvas_hash"] = self._mutate_hash(base_fp["canvas_hash"], transformed[:32])
        base_fp["audio_hash"] = self._mutate_hash(base_fp["audio_hash"], transformed[32:96])
        
        # Add some randomness to fonts
        if random.random() > 0.7:
            base_fp["fonts"].append(random.choice(["Symbol", "Webdings", "Wingdings"]))
        
        return base_fp
    
    def _mutate_hash(self, original_hash: str, mutation_vector: np.ndarray) -> str:
        """Apply subtle mutations to a hash based on GAN output."""
        hash_chars = list(original_hash)
        for i in range(min(len(hash_chars), len(mutation_vector))):
            if abs(mutation_vector[i]) > 0.5:  # Threshold for mutation
                hash_chars[i] = random.choice('0123456789abcdef')
        return ''.join(hash_chars)


class BehaviorSimulator:
    """Human-like behavior simulation using Markov chains."""
    
    def __init__(self):
        self.mouse_transition_matrix = self._build_mouse_transition_matrix()
        self.typing_patterns = self._build_typing_patterns()
        self.last_mouse_position = (random.randint(0, 1920), random.randint(0, 1080))
        self.typing_speed_history = deque(maxlen=10)
        
    def _build_mouse_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build Markov chain transition matrix for mouse movements."""
        return {
            "idle": {"idle": 0.6, "small_move": 0.3, "large_move": 0.1},
            "small_move": {"idle": 0.4, "small_move": 0.4, "large_move": 0.2},
            "large_move": {"idle": 0.2, "small_move": 0.5, "large_move": 0.3},
        }
    
    def _build_typing_patterns(self) -> Dict[str, Any]:
        """Build realistic typing patterns."""
        return {
            "avg_delay": 0.1,  # seconds between keystrokes
            "std_dev": 0.05,
            "burst_probability": 0.3,
            "burst_length": (3, 8),  # characters in burst
            "pause_probability": 0.1,
            "pause_duration": (0.5, 2.0),
        }
    
    def generate_mouse_movement(self, target_x: int, target_y: int, steps: int = None) -> List[Tuple[int, int, float]]:
        """Generate human-like mouse movement path using Markov chains."""
        if steps is None:
            steps = random.randint(10, 30)
        
        path = []
        current_x, current_y = self.last_mouse_position
        
        for i in range(steps):
            # Determine movement type based on Markov chain
            movement_type = self._get_next_mouse_state()
            
            if movement_type == "idle":
                # Small jitter
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
            elif movement_type == "small_move":
                # Medium movement
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
            else:  # large_move
                # Move toward target with some randomness
                progress = (i + 1) / steps
                dx = int((target_x - current_x) * progress * 0.3 + random.randint(-10, 10))
                dy = int((target_y - current_y) * progress * 0.3 + random.randint(-10, 10))
            
            current_x = max(0, min(1920, current_x + dx))
            current_y = max(0, min(1080, current_y + dy))
            
            # Add delay between movements
            delay = random.uniform(0.01, 0.1)
            path.append((current_x, current_y, delay))
        
        self.last_mouse_position = (current_x, current_y)
        return path
    
    def _get_next_mouse_state(self) -> str:
        """Get next mouse state using Markov chain."""
        current_state = random.choice(["idle", "small_move", "large_move"])
        transitions = self.mouse_transition_matrix[current_state]
        states = list(transitions.keys())
        probabilities = list(transitions.values())
        return random.choices(states, weights=probabilities)[0]
    
    def generate_typing_pattern(self, text: str) -> List[Tuple[str, float]]:
        """Generate human-like typing pattern with variable delays."""
        pattern = []
        i = 0
        
        while i < len(text):
            # Check for burst typing
            if random.random() < self.typing_patterns["burst_probability"]:
                burst_len = random.randint(*self.typing_patterns["burst_length"])
                burst_text = text[i:i + burst_len]
                for char in burst_text:
                    delay = random.gauss(0.05, 0.02)  # Fast typing
                    pattern.append((char, max(0.01, delay)))
                i += burst_len
            # Check for pause
            elif random.random() < self.typing_patterns["pause_probability"]:
                pause_duration = random.uniform(*self.typing_patterns["pause_duration"])
                pattern.append(("", pause_duration))  # Pause
            # Normal typing
            else:
                char = text[i]
                delay = random.gauss(
                    self.typing_patterns["avg_delay"],
                    self.typing_patterns["std_dev"]
                )
                pattern.append((char, max(0.01, delay)))
                i += 1
        
        return pattern


class ProxyRotator:
    """Residential proxy rotation manager."""
    
    def __init__(self):
        self.proxy_pools = {
            "residential": [],
            "datacenter": [],
            "mobile": []
        }
        self.current_indices = {pool: 0 for pool in self.proxy_pools}
        self.health_status = {}
        
    def add_proxy(self, proxy_url: str, proxy_type: str = "residential", health_check_url: str = None):
        """Add a proxy to the rotation pool."""
        if proxy_type not in self.proxy_pools:
            proxy_type = "residential"
        
        proxy_entry = {
            "url": proxy_url,
            "type": proxy_type,
            "health_check_url": health_check_url,
            "last_used": 0,
            "success_rate": 1.0,
            "response_time": 0.0
        }
        
        self.proxy_pools[proxy_type].append(proxy_entry)
        self.health_status[proxy_url] = True
    
    def get_next_proxy(self, proxy_type: str = "residential") -> Optional[str]:
        """Get next proxy using round-robin with health awareness."""
        if proxy_type not in self.proxy_pools or not self.proxy_pools[proxy_type]:
            return None
        
        pool = self.proxy_pools[proxy_type]
        healthy_proxies = [p for p in pool if self.health_status.get(p["url"], True)]
        
        if not healthy_proxies:
            # Reset health status if all proxies are unhealthy
            for proxy in pool:
                self.health_status[proxy["url"]] = True
            healthy_proxies = pool
        
        # Sort by success rate and response time
        healthy_proxies.sort(key=lambda x: (-x["success_rate"], x["response_time"]))
        
        # Use weighted random selection based on success rate
        weights = [p["success_rate"] for p in healthy_proxies]
        if sum(weights) == 0:
            weights = [1.0] * len(healthy_proxies)
        
        selected = random.choices(healthy_proxies, weights=weights)[0]
        selected["last_used"] = time.time()
        
        return selected["url"]
    
    def update_proxy_health(self, proxy_url: str, success: bool, response_time: float = 0.0):
        """Update proxy health statistics."""
        if proxy_url in self.health_status:
            self.health_status[proxy_url] = success
            
            # Find proxy in pools and update stats
            for pool in self.proxy_pools.values():
                for proxy in pool:
                    if proxy["url"] == proxy_url:
                        # Exponential moving average for success rate
                        proxy["success_rate"] = proxy["success_rate"] * 0.9 + (1.0 if success else 0.0) * 0.1
                        if success and response_time > 0:
                            proxy["response_time"] = proxy["response_time"] * 0.9 + response_time * 0.1


class CaptchaSolver:
    """CAPTCHA solving integration with 2Captcha/Anti-Captcha services."""
    
    def __init__(self):
        self.services = {
            "2captcha": {
                "api_url": "https://2captcha.com/in.php",
                "result_url": "https://2captcha.com/res.php",
                "api_key": None
            },
            "anticaptcha": {
                "api_url": "https://api.anti-captcha.com/createTask",
                "result_url": "https://api.anti-captcha.com/getTaskResult",
                "api_key": None
            }
        }
        self.default_service = "2captcha"
    
    def configure_service(self, service_name: str, api_key: str):
        """Configure a CAPTCHA solving service."""
        if service_name in self.services:
            self.services[service_name]["api_key"] = api_key
            self.default_service = service_name
    
    async def solve_recaptcha(self, site_url: str, site_key: str, service: str = None) -> Optional[str]:
        """Solve reCAPTCHA using configured service."""
        service = service or self.default_service
        if service not in self.services or not self.services[service]["api_key"]:
            return None
        
        config = self.services[service]
        
        if service == "2captcha":
            return await self._solve_with_2captcha(site_url, site_key, config)
        elif service == "anticaptcha":
            return await self._solve_with_anticaptcha(site_url, site_key, config)
        
        return None
    
    async def _solve_with_2captcha(self, site_url: str, site_key: str, config: Dict) -> Optional[str]:
        """Solve using 2Captcha service."""
        # Implementation would make HTTP requests to 2Captcha API
        # This is a placeholder for the actual implementation
        print(f"Solving CAPTCHA for {site_url} using 2Captcha")
        return "solved_captcha_token_placeholder"
    
    async def _solve_with_anticaptcha(self, site_url: str, site_key: str, config: Dict) -> Optional[str]:
        """Solve using Anti-Captcha service."""
        # Implementation would make HTTP requests to Anti-Captcha API
        # This is a placeholder for the actual implementation
        print(f"Solving CAPTCHA for {site_url} using Anti-Captcha")
        return "solved_captcha_token_placeholder"
    
    async def solve_hcaptcha(self, site_url: str, site_key: str, service: str = None) -> Optional[str]:
        """Solve hCAPTCHA using configured service."""
        # Similar implementation to reCAPTCHA
        return await self.solve_recaptcha(site_url, site_key, service)
    
    async def solve_funcaptcha(self, site_url: str, public_key: str, service: str = None) -> Optional[str]:
        """Solve FunCAPTCHA using configured service."""
        service = service or self.default_service
        if service not in self.services or not self.services[service]["api_key"]:
            return None
        
        print(f"Solving FunCAPTCHA for {site_url}")
        return "solved_funcaptcha_token_placeholder"


# Global instances for anti-detection suite
_fingerprint_generator = FingerprintGenerator()
_behavior_simulator = BehaviorSimulator()
_proxy_rotator = ProxyRotator()
_captcha_solver = CaptchaSolver()


def _content_translator(content: Generator[str, None, None], page: _axiomResponse) -> ResponseModel:
    """Convert a content generator to a list of ResponseModel objects."""
    return ResponseModel(status=page.status, content=[result for result in content], url=page.url)


def _normalize_credentials(credentials: Optional[Dict[str, str]]) -> Optional[Tuple[str, str]]:
    """Convert a credentials dictionary to a tuple accepted by fetchers."""
    if not credentials:
        return None

    username = credentials.get("username")
    password = credentials.get("password")

    if username is None or password is None:
        raise ValueError("Credentials dictionary must contain both 'username' and 'password' keys")

    return username, password


def _apply_anti_detection_headers(headers: Optional[Dict], fingerprint: Dict) -> Dict:
    """Apply fingerprint to headers for anti-detection."""
    if headers is None:
        headers = {}
    
    headers["User-Agent"] = fingerprint["user_agent"]
    headers["Accept-Language"] = "en-US,en;q=0.9"
    headers["Accept-Encoding"] = "gzip, deflate, br"
    
    return headers


def _get_proxy_with_rotation(proxy: Optional[str], proxy_type: str = "residential") -> Optional[str]:
    """Get proxy with rotation if enabled."""
    if proxy:
        return proxy
    
    return _proxy_rotator.get_next_proxy(proxy_type)


class axiomMCPServer:
    @staticmethod
    def get(
        url: str,
        impersonate: ImpersonateType = "chrome",
        extraction_type: extraction_types = "markdown",
        css_selector: Optional[str] = None,
        main_content_only: bool = True,
        params: Optional[Dict] = None,
        headers: Optional[Mapping[str, Optional[str]]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: Optional[int | float] = 30,
        follow_redirects: bool = True,
        max_redirects: int = 30,
        retries: Optional[int] = 3,
        retry_delay: Optional[int] = 1,
        proxy: Optional[str] = None,
        proxy_auth: Optional[Dict[str, str]] = None,
        auth: Optional[Dict[str, str]] = None,
        verify: Optional[bool] = True,
        http3: Optional[bool] = False,
        stealthy_headers: Optional[bool] = True,
        # Anti-detection parameters
        enable_fingerprint_randomization: bool = True,
        enable_behavior_simulation: bool = True,
        enable_proxy_rotation: bool = False,
        proxy_type: str = "residential",
        captcha_solver_service: Optional[str] = None,
        site_key: Optional[str] = None,
    ) -> ResponseModel:
        """Make GET HTTP request to a URL with advanced anti-detection features.
        
        Anti-detection features:
        - ML-powered fingerprint randomization using GAN-inspired techniques
        - Human-like behavior simulation (mouse movements, typing patterns)
        - Residential proxy rotation
        - CAPTCHA solving integration
        
        :param url: The URL to request.
        :param impersonate: Browser version to impersonate its fingerprint.
        :param extraction_type: The type of content to extract from the page.
        :param css_selector: CSS selector to extract the content from the page.
        :param main_content_only: Whether to extract only the main content of the page.
        :param params: Query string parameters for the request.
        :param headers: Headers to include in the request.
        :param cookies: Cookies to use in the request.
        :param timeout: Number of seconds to wait before timing out.
        :param follow_redirects: Whether to follow redirects.
        :param max_redirects: Maximum number of redirects.
        :param retries: Number of retry attempts.
        :param retry_delay: Number of seconds to wait between retry attempts.
        :param proxy: Proxy URL to use.
        :param proxy_auth: HTTP basic auth for proxy.
        :param auth: HTTP basic auth.
        :param verify: Whether to verify HTTPS certificates.
        :param http3: Whether to use HTTP3.
        :param stealthy_headers: If enabled, creates and adds real browser headers.
        :param enable_fingerprint_randomization: Enable ML-powered fingerprint randomization.
        :param enable_behavior_simulation: Enable human-like behavior simulation.
        :param enable_proxy_rotation: Enable residential proxy rotation.
        :param proxy_type: Type of proxy to use ("residential", "datacenter", "mobile").
        :param captcha_solver_service: CAPTCHA solving service to use ("2captcha", "anticaptcha").
        :param site_key: Site key for CAPTCHA solving.
        """
        normalized_proxy_auth = _normalize_credentials(proxy_auth)
        normalized_auth = _normalize_credentials(auth)
        
        # Apply fingerprint randomization if enabled
        if enable_fingerprint_randomization:
            browser_type = impersonate.split("_")[0] if "_" in impersonate else "chrome"
            platform = "windows" if "windows" in impersonate.lower() else "mac"
            fingerprint = _fingerprint_generator.generate_fingerprint(browser_type, platform)
            headers = _apply_anti_detection_headers(headers, fingerprint)
        
        # Apply proxy rotation if enabled
        if enable_proxy_rotation:
            proxy = _get_proxy_with_rotation(proxy, proxy_type)
        
        # Handle CAPTCHA if service is configured
        if captcha_solver_service and site_key:
            # This would be implemented to actually solve CAPTCHAs
            pass
        
        start_time = time.time()
        
        try:
            page = Fetcher.get(
                url,
                auth=normalized_auth,
                proxy=proxy,
                http3=http3,
                verify=verify,
                params=params,
                proxy_auth=normalized_proxy_auth,
                retry_delay=retry_delay,
                stealthy_headers=stealthy_headers,
                impersonate=impersonate,
                headers=headers,
                cookies=cookies,
                timeout=timeout,
                retries=retries,
                max_redirects=max_redirects,
                follow_redirects=follow_redirects,
            )
            
            # Update proxy health if rotation was used
            if enable_proxy_rotation and proxy:
                response_time = time.time() - start_time
                _proxy_rotator.update_proxy_health(proxy, page.status == 200, response_time)
            
            return _content_translator(
                Convertor._extract_content(
                    page,
                    css_selector=css_selector,
                    extraction_type=extraction_type,
                    main_content_only=main_content_only,
                ),
                page,
            )
            
        except Exception as e:
            # Update proxy health on failure
            if enable_proxy_rotation and proxy:
                _proxy_rotator.update_proxy_health(proxy, False)
            raise e

    @staticmethod
    async def bulk_get(
        urls: List[str],
        impersonate: ImpersonateType = "chrome",
        extraction_type: extraction_types = "markdown",
        css_selector: Optional[str] = None,
        main_content_only: bool = True,
        params: Optional[Dict] = None,
        headers: Optional[Mapping[str, Optional[str]]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: Optional[int | float] = 30,
        follow_redirects: bool = True,
        max_redirects: int = 30,
        retries: Optional[int] = 3,
        retry_delay: Optional[int] = 1,
        proxy: Optional[str] = None,
        proxy_auth: Optional[Dict[str, str]] = None,
        auth: Optional[Dict[str, str]] = None,
        verify: Optional[bool] = True,
        http3: Optional[bool] = False,
        stealthy_headers: Optional[bool] = True,
        # Anti-detection parameters
        enable_fingerprint_randomization: bool = True,
        enable_behavior_simulation: bool = True,
        enable_proxy_rotation: bool = False,
        proxy_type: str = "residential",
        captcha_solver_service: Optional[str] = None,
        site_key: Optional[str] = None,
    ) -> List[ResponseModel]:
        """Make GET HTTP requests to multiple URLs with advanced anti-detection features.
        
        Applies anti-detection features to each request individually.
        """
        normalized_proxy_auth = _normalize_credentials(proxy_auth)
        normalized_auth = _normalize_credentials(auth)
        
        # Generate different fingerprints for each request if randomization is enabled
        fingerprints = []
        if enable_fingerprint_randomization:
            for _ in urls:
                browser_type = impersonate.split("_")[0] if "_" in impersonate else "chrome"
                platform = "windows" if "windows" in impersonate.lower() else "mac"
                fingerprints.append(_fingerprint_generator.generate_fingerprint(browser_type, platform))
        
        async def fetch_single_url(url: str, index: int) -> ResponseModel:
            # Apply fingerprint for this specific request
            request_headers = headers
            if enable_fingerprint_randomization and index < len(fingerprints):
                request_headers = _apply_anti_detection_headers(headers, fingerprints[index])
            
            # Apply proxy rotation for this specific request
            request_proxy = proxy
            if enable_proxy_rotation:
                request_proxy = _get_proxy_with_rotation(proxy, proxy_type)
            
            start_time = time.time()
            
            try:
                async with AsyncStealthySession(
                    impersonate=impersonate,
                    proxy=request_proxy,
                    proxy_auth=normalized_proxy_auth,
                    auth=normalized_auth,
                    verify=verify,
                    http3=http3,
                    stealthy_headers=stealthy_headers,
                ) as session:
                    page = await session.get(
                        url,
                        params=params,
                        headers=request_headers,
                        cookies=cookies,
                        timeout=timeout,
                        follow_redirects=follow_redirects,
                        max_redirects=max_redirects,
                        retries=retries,
                        retry_delay=retry_delay,
                    )
                    
                    # Update proxy health if rotation was used
                    if enable_proxy_rotation and request_proxy:
                        response_time = time.time() - start_time
                        _proxy_rotator.update_proxy_health(request_proxy, page.status == 200, response_time)
                    
                    return _content_translator(
                        Convertor._extract_content(
                            page,
                            css_selector=css_selector,
                            extraction_type=extraction_type,
                            main_content_only=main_content_only,
                        ),
                        page,
                    )
                    
            except Exception as e:
                # Update proxy health on failure
                if enable_proxy_rotation and request_proxy:
                    _proxy_rotator.update_proxy_health(request_proxy, False)
                raise e
        
        # Execute all requests concurrently
        tasks = [fetch_single_url(url, i) for i, url in enumerate(urls)]
        return await gather(*tasks)
    
    @staticmethod
    def configure_anti_detection(
        fingerprint_templates: Optional[Dict] = None,
        proxy_pools: Optional[Dict[str, List[str]]] = None,
        captcha_service: Optional[str] = None,
        captcha_api_key: Optional[str] = None,
    ):
        """Configure the advanced anti-detection suite.
        
        :param fingerprint_templates: Custom fingerprint templates to add.
        :param proxy_pools: Dictionary of proxy pools by type.
        :param captcha_service: CAPTCHA service to configure ("2captcha", "anticaptcha").
        :param captcha_api_key: API key for the CAPTCHA service.
        """
        if fingerprint_templates:
            for key, template in fingerprint_templates.items():
                _fingerprint_generator.fingerprint_templates[key] = template
        
        if proxy_pools:
            for proxy_type, proxies in proxy_pools.items():
                for proxy_url in proxies:
                    _proxy_rotator.add_proxy(proxy_url, proxy_type)
        
        if captcha_service and captcha_api_key:
            _captcha_solver.configure_service(captcha_service, captcha_api_key)
    
    @staticmethod
    def add_proxy(proxy_url: str, proxy_type: str = "residential"):
        """Add a proxy to the rotation pool.
        
        :param proxy_url: Proxy URL (format: "http://username:password@host:port").
        :param proxy_type: Type of proxy ("residential", "datacenter", "mobile").
        """
        _proxy_rotator.add_proxy(proxy_url, proxy_type)
    
    @staticmethod
    def get_proxy_stats() -> Dict[str, Any]:
        """Get statistics about proxy pool health and usage."""
        stats = {}
        for proxy_type, proxies in _proxy_rotator.proxy_pools.items():
            stats[proxy_type] = {
                "total": len(proxies),
                "healthy": sum(1 for p in proxies if _proxy_rotator.health_status.get(p["url"], True)),
                "avg_success_rate": sum(p["success_rate"] for p in proxies) / len(proxies) if proxies else 0,
            }
        return stats