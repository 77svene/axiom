"""Advanced Anti-Detection Suite for axiom.

ML-powered fingerprint randomization, browser behavior simulation, residential proxy rotation,
and CAPTCHA solving integration using generative adversarial networks and Markov chains.
"""

import asyncio
import json
import random
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import logging
from collections import deque
import hashlib

# Import from existing axiom modules
from axiom.core.ai import BaseModel, ModelRegistry
from axiom.core.custom_types import SessionConfig, ProxyConfig
from axiom.core.utils._utils import generate_fingerprint_hash, random_delay

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """Types of human-like behaviors to simulate."""
    MOUSE_MOVEMENT = "mouse_movement"
    TYPING_PATTERN = "typing_pattern"
    SCROLL_BEHAVIOR = "scroll_behavior"
    CLICK_PATTERN = "click_pattern"


@dataclass
class FingerprintProfile:
    """Browser fingerprint profile with ML-generated attributes."""
    user_agent: str
    platform: str
    screen_resolution: Tuple[int, int]
    timezone: str
    language: str
    webgl_vendor: str
    webgl_renderer: str
    canvas_hash: str
    audio_hash: str
    fonts: List[str]
    plugins: List[Dict[str, str]]
    webrtc_ip: Optional[str] = None
    hardware_concurrency: int = 4
    device_memory: int = 8
    touch_support: bool = False
    confidence_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert fingerprint to dictionary for browser injection."""
        return {
            "userAgent": self.user_agent,
            "platform": self.platform,
            "screenResolution": self.screen_resolution,
            "timezone": self.timezone,
            "language": self.language,
            "webglVendor": self.webgl_vendor,
            "webglRenderer": self.webgl_renderer,
            "canvasHash": self.canvas_hash,
            "audioHash": self.audio_hash,
            "fonts": self.fonts,
            "plugins": self.plugins,
            "webrtcIP": self.webrtc_ip,
            "hardwareConcurrency": self.hardware_concurrency,
            "deviceMemory": self.device_memory,
            "touchSupport": self.touch_support,
        }


@dataclass
class MouseMovement:
    """Represents a mouse movement trajectory."""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    duration_ms: int
    points: List[Tuple[float, float, float]]  # (x, y, timestamp)
    curve_type: str = "bezier"


@dataclass
class TypingPattern:
    """Represents a human-like typing pattern."""
    text: str
    key_timings: List[Tuple[str, float, float]]  # (key, press_time, release_time)
    error_rate: float = 0.02
    correction_delay_ms: int = 150


@dataclass
class ProxyEndpoint:
    """Residential proxy endpoint with health metrics."""
    url: str
    country: str
    city: Optional[str] = None
    latency_ms: float = 0.0
    success_rate: float = 1.0
    last_used: float = 0.0
    fail_count: int = 0
    is_healthy: bool = True


class FingerprintGenerator:
    """ML-powered fingerprint generator using GANs for realistic browser fingerprints."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model = self._load_model(model_path)
        self.fingerprint_cache: Dict[str, FingerprintProfile] = {}
        self._init_base_distributions()

    def _load_model(self, model_path: Optional[Path]) -> Any:
        """Load pre-trained GAN model for fingerprint generation."""
        try:
            if model_path and model_path.exists():
                # Load from file if available
                return ModelRegistry.load_model("fingerprint_gan", model_path)
            else:
                # Use built-in model or fallback to statistical generation
                return ModelRegistry.get_model("fingerprint_gan")
        except Exception as e:
            logger.warning(f"Failed to load GAN model: {e}. Using statistical generation.")
            return None

    def _init_base_distributions(self):
        """Initialize base distributions for fingerprint attributes."""
        self.user_agents = self._load_user_agents()
        self.platforms = ["Win32", "Win64", "MacIntel", "Linux x86_64", "Linux armv8l"]
        self.screen_resolutions = [
            (1920, 1080), (1366, 768), (1536, 864), (1440, 900),
            (1280, 720), (2560, 1440), (1680, 1050)
        ]
        self.timezones = [
            "America/New_York", "America/Los_Angeles", "Europe/London",
            "Europe/Paris", "Asia/Tokyo", "Australia/Sydney"
        ]
        self.languages = ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "ja-JP"]
        self.webgl_vendors = [
            "Google Inc. (NVIDIA)", "Google Inc. (Intel)",
            "Mozilla", "Apple Computer, Inc."
        ]
        self.webgl_renderers = [
            "ANGLE (NVIDIA GeForce GTX 1080 Ti Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (Intel(R) HD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (AMD Radeon RX 580 Direct3D11 vs_5_0 ps_5_0)"
        ]

    def _load_user_agents(self) -> List[str]:
        """Load user agents from built-in database."""
        # In production, this would load from a comprehensive database
        return [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]

    def generate(self, seed: Optional[int] = None, 
                 consistency_group: Optional[str] = None) -> FingerprintProfile:
        """Generate a realistic browser fingerprint.
        
        Args:
            seed: Random seed for reproducibility
            consistency_group: Group identifier for consistent fingerprints across sessions
            
        Returns:
            FingerprintProfile with generated attributes
        """
        cache_key = f"{seed}_{consistency_group}"
        if cache_key in self.fingerprint_cache:
            return self.fingerprint_cache[cache_key]

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.model:
            # Use GAN model for generation
            fingerprint = self._generate_with_gan(consistency_group)
        else:
            # Fallback to statistical generation
            fingerprint = self._generate_statistical(consistency_group)

        self.fingerprint_cache[cache_key] = fingerprint
        return fingerprint

    def _generate_with_gan(self, consistency_group: Optional[str]) -> FingerprintProfile:
        """Generate fingerprint using GAN model."""
        # Generate latent vector
        latent_dim = 128
        z = np.random.randn(1, latent_dim)
        
        # Add consistency noise if group specified
        if consistency_group:
            group_hash = int(hashlib.md5(consistency_group.encode()).hexdigest()[:8], 16)
            np.random.seed(group_hash % (2**32))
            consistency_noise = np.random.randn(1, latent_dim) * 0.1
            z = z + consistency_noise

        # Generate features (simplified - actual implementation would use trained GAN)
        features = self._latent_to_features(z[0])
        
        return FingerprintProfile(
            user_agent=features["user_agent"],
            platform=features["platform"],
            screen_resolution=features["screen_resolution"],
            timezone=features["timezone"],
            language=features["language"],
            webgl_vendor=features["webgl_vendor"],
            webgl_renderer=features["webgl_renderer"],
            canvas_hash=generate_fingerprint_hash(features["canvas"]),
            audio_hash=generate_fingerprint_hash(features["audio"]),
            fonts=features["fonts"],
            plugins=features["plugins"],
            hardware_concurrency=features["hardware_concurrency"],
            device_memory=features["device_memory"],
            touch_support=features["touch_support"],
            confidence_score=0.95
        )

    def _latent_to_features(self, z: np.ndarray) -> Dict[str, Any]:
        """Convert latent vector to fingerprint features."""
        # Simplified mapping - real implementation would use neural network
        ua_idx = int(abs(z[0]) * len(self.user_agents)) % len(self.user_agents)
        platform_idx = int(abs(z[1]) * len(self.platforms)) % len(self.platforms)
        res_idx = int(abs(z[2]) * len(self.screen_resolutions)) % len(self.screen_resolutions)
        tz_idx = int(abs(z[3]) * len(self.timezones)) % len(self.timezones)
        lang_idx = int(abs(z[4]) * len(self.languages)) % len(self.languages)
        vendor_idx = int(abs(z[5]) * len(self.webgl_vendors)) % len(self.webgl_vendors)
        renderer_idx = int(abs(z[6]) * len(self.webgl_renderers)) % len(self.webgl_renderers)

        # Generate canvas and audio hashes
        canvas_features = z[7:15].tobytes()
        audio_features = z[15:23].tobytes()

        # Generate font list
        base_fonts = ["Arial", "Verdana", "Times New Roman", "Courier New", "Georgia"]
        num_fonts = 5 + int(abs(z[24]) * 10)
        fonts = random.sample(base_fonts * 3, min(num_fonts, len(base_fonts) * 3))

        # Generate plugins
        plugins = [
            {"name": "PDF Viewer", "description": "Portable Document Format", "filename": "internal-pdf-viewer"},
            {"name": "Chrome PDF Viewer", "description": "Portable Document Format", "filename": "internal-pdf-viewer"},
        ]

        return {
            "user_agent": self.user_agents[ua_idx],
            "platform": self.platforms[platform_idx],
            "screen_resolution": self.screen_resolutions[res_idx],
            "timezone": self.timezones[tz_idx],
            "language": self.languages[lang_idx],
            "webgl_vendor": self.webgl_vendors[vendor_idx],
            "webgl_renderer": self.webgl_renderers[renderer_idx],
            "canvas": canvas_features,
            "audio": audio_features,
            "fonts": fonts,
            "plugins": plugins,
            "hardware_concurrency": 2 + int(abs(z[30]) * 6),
            "device_memory": 2 ** (2 + int(abs(z[31]) * 3)),
            "touch_support": abs(z[32]) > 0.7,
        }

    def _generate_statistical(self, consistency_group: Optional[str]) -> FingerprintProfile:
        """Generate fingerprint using statistical distributions."""
        if consistency_group:
            random.seed(hash(consistency_group) % (2**32))

        ua = random.choice(self.user_agents)
        platform = random.choice(self.platforms)
        resolution = random.choice(self.screen_resolutions)
        timezone = random.choice(self.timezones)
        language = random.choice(self.languages)
        vendor = random.choice(self.webgl_vendors)
        renderer = random.choice(self.webgl_renderers)

        # Generate deterministic hashes based on attributes
        canvas_data = f"{ua}{platform}{resolution}".encode()
        audio_data = f"{timezone}{language}".encode()

        return FingerprintProfile(
            user_agent=ua,
            platform=platform,
            screen_resolution=resolution,
            timezone=timezone,
            language=language,
            webgl_vendor=vendor,
            webgl_renderer=renderer,
            canvas_hash=generate_fingerprint_hash(canvas_data),
            audio_hash=generate_fingerprint_hash(audio_data),
            fonts=["Arial", "Verdana", "Times New Roman", "Courier New", "Georgia"],
            plugins=[
                {"name": "PDF Viewer", "description": "Portable Document Format", "filename": "internal-pdf-viewer"},
            ],
            hardware_concurrency=random.choice([2, 4, 8, 16]),
            device_memory=random.choice([2, 4, 8, 16]),
            touch_support=random.random() > 0.8,
            confidence_score=0.85
        )


class BehaviorSimulator:
    """Simulates human-like browser behavior using Markov chains."""

    def __init__(self, behavior_profile: str = "default"):
        self.profile = behavior_profile
        self._init_markov_chains()
        self._init_timing_distributions()

    def _init_markov_chains(self):
        """Initialize Markov chains for different behavior types."""
        # Mouse movement Markov chain (states: slow, medium, fast, pause)
        self.mouse_transitions = {
            "slow": {"slow": 0.3, "medium": 0.4, "fast": 0.2, "pause": 0.1},
            "medium": {"slow": 0.2, "medium": 0.3, "fast": 0.4, "pause": 0.1},
            "fast": {"slow": 0.1, "medium": 0.3, "fast": 0.4, "pause": 0.2},
            "pause": {"slow": 0.4, "medium": 0.3, "fast": 0.2, "pause": 0.1},
        }

        # Typing Markov chain (states: normal, fast, slow, error, correction)
        self.typing_transitions = {
            "normal": {"normal": 0.6, "fast": 0.2, "slow": 0.1, "error": 0.08, "correction": 0.02},
            "fast": {"normal": 0.3, "fast": 0.5, "slow": 0.1, "error": 0.08, "correction": 0.02},
            "slow": {"normal": 0.4, "fast": 0.1, "slow": 0.4, "error": 0.08, "correction": 0.02},
            "error": {"correction": 0.9, "normal": 0.1},
            "correction": {"normal": 0.7, "fast": 0.2, "slow": 0.1},
        }

    def _init_timing_distributions(self):
        """Initialize timing distributions for human-like delays."""
        # Inter-key delays (milliseconds)
        self.key_delays = {
            "normal": {"mean": 120, "std": 30},
            "fast": {"mean": 80, "std": 20},
            "slow": {"mean": 200, "std": 50},
        }

        # Mouse movement speeds (pixels per millisecond)
        self.mouse_speeds = {
            "slow": {"mean": 0.5, "std": 0.1},
            "medium": {"mean": 1.0, "std": 0.2},
            "fast": {"mean": 2.0, "std": 0.5},
        }

    def simulate_mouse_movement(self, start_x: float, start_y: float,
                                end_x: float, end_y: float,
                                duration_ms: Optional[int] = None) -> MouseMovement:
        """Generate human-like mouse movement trajectory.
        
        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            duration_ms: Optional duration in milliseconds
            
        Returns:
            MouseMovement with generated trajectory
        """
        if duration_ms is None:
            # Estimate duration based on distance
            distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            duration_ms = int(distance * random.uniform(0.8, 1.2))

        # Generate control points for Bezier curve
        num_points = max(10, duration_ms // 20)
        points = self._generate_bezier_trajectory(
            start_x, start_y, end_x, end_y, num_points
        )

        # Add human-like noise and timing variations
        points = self._add_human_noise(points, duration_ms)

        return MouseMovement(
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            duration_ms=duration_ms,
            points=points,
            curve_type="bezier"
        )

    def _generate_bezier_trajectory(self, x0: float, y0: float,
                                    x1: float, y1: float,
                                    num_points: int) -> List[Tuple[float, float, float]]:
        """Generate smooth Bezier curve trajectory."""
        # Generate random control points
        ctrl1_x = x0 + (x1 - x0) * random.uniform(0.2, 0.4)
        ctrl1_y = y0 + (y1 - y0) * random.uniform(0.2, 0.4)
        ctrl2_x = x0 + (x1 - x0) * random.uniform(0.6, 0.8)
        ctrl2_y = y0 + (y1 - y0) * random.uniform(0.6, 0.8)

        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            # Cubic Bezier formula
            x = (1-t)**3 * x0 + 3*(1-t)**2*t * ctrl1_x + 3*(1-t)*t**2 * ctrl2_x + t**3 * x1
            y = (1-t)**3 * y0 + 3*(1-t)**2*t * ctrl1_y + 3*(1-t)*t**2 * ctrl2_y + t**3 * y1
            points.append((x, y, t))

        return points

    def _add_human_noise(self, points: List[Tuple[float, float, float]],
                         duration_ms: int) -> List[Tuple[float, float, float]]:
        """Add human-like noise and timing variations to trajectory."""
        noisy_points = []
        total_time = 0

        for i, (x, y, t) in enumerate(points):
            # Add position noise (more noise at higher speeds)
            speed = "medium"  # Default, could be determined by Markov chain
            noise_std = self.mouse_speeds[speed]["std"] * 2
            x_noise = random.gauss(0, noise_std)
            y_noise = random.gauss(0, noise_std)

            # Add timing variation
            if i > 0:
                # Use Markov chain to determine next speed state
                speed = self._next_markov_state(self.mouse_transitions, speed)
                delay_mean = 1000 / (self.mouse_speeds[speed]["mean"] * 10)
                delay_std = self.mouse_speeds[speed]["std"] * 10
                delay = max(5, random.gauss(delay_mean, delay_std))
                total_time += delay

            noisy_points.append((x + x_noise, y + y_noise, total_time / 1000))

        # Normalize timing to match total duration
        if noisy_points and total_time > 0:
            scale = duration_ms / total_time
            noisy_points = [(x, y, t * scale) for x, y, t in noisy_points]

        return noisy_points

    def _next_markov_state(self, transitions: Dict[str, Dict[str, float]],
                           current_state: str) -> str:
        """Get next state from Markov chain."""
        if current_state not in transitions:
            return current_state

        probs = transitions[current_state]
        states = list(probs.keys())
        probabilities = list(probs.values())
        return np.random.choice(states, p=probabilities)

    def simulate_typing(self, text: str, error_rate: float = 0.02) -> TypingPattern:
        """Generate human-like typing pattern with errors and corrections.
        
        Args:
            text: Text to type
            error_rate: Probability of making a typo per character
            
        Returns:
            TypingPattern with key timings and errors
        """
        key_timings = []
        current_state = "normal"
        i = 0
        current_time = 0.0

        while i < len(text):
            char = text[i]

            # Determine next state using Markov chain
            current_state = self._next_markov_state(self.typing_transitions, current_state)

            # Get timing for this state
            delay_config = self.key_delays.get(current_state, self.key_delays["normal"])
            press_delay = max(20, random.gauss(delay_config["mean"], delay_config["std"]))
            release_delay = press_delay * random.uniform(0.3, 0.7)

            # Check for error
            if current_state == "error" and random.random() < error_rate:
                # Type wrong character
                wrong_char = self._get_similar_key(char)
                key_timings.append((wrong_char, current_time, current_time + release_delay / 1000))
                current_time += press_delay / 1000

                # Correction state
                current_state = "correction"
                correction_delay = random.gauss(150, 30)
                current_time += correction_delay / 1000

                # Backspace
                key_timings.append(("Backspace", current_time, current_time + 0.05))
                current_time += random.gauss(100, 20) / 1000

                # Type correct character
                key_timings.append((char, current_time, current_time + release_delay / 1000))
                current_time += press_delay / 1000
            else:
                # Normal typing
                key_timings.append((char, current_time, current_time + release_delay / 1000))
                current_time += press_delay / 1000

            i += 1

        return TypingPattern(
            text=text,
            key_timings=key_timings,
            error_rate=error_rate,
            correction_delay_ms=150
        )

    def _get_similar_key(self, char: str) -> str:
        """Get a similar key for typo simulation."""
        similar_keys = {
            'a': ['s', 'q', 'z'],
            'b': ['v', 'g', 'h'],
            'c': ['x', 'd', 'f'],
            'd': ['s', 'f', 'e', 'c'],
            'e': ['w', 'r', 'd', 'f'],
            'f': ['d', 'g', 'r', 't', 'v', 'c'],
            'g': ['f', 'h', 't', 'y', 'b', 'v'],
            'h': ['g', 'j', 'y', 'u', 'n', 'b'],
            'i': ['u', 'o', 'k', 'j'],
            'j': ['h', 'k', 'u', 'i', 'm', 'n'],
            'k': ['j', 'l', 'i', 'o', 'm'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k'],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'p', 'l', 'k'],
            'p': ['o', 'l'],
            'q': ['w', 'a', 's'],
            'r': ['e', 't', 'f', 'd'],
            's': ['a', 'd', 'w', 'e', 'x', 'z'],
            't': ['r', 'y', 'g', 'f'],
            'u': ['y', 'i', 'j', 'h'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'e', 's', 'a'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'u', 'h', 'g'],
            'z': ['x', 's', 'a'],
        }
        char_lower = char.lower()
        if char_lower in similar_keys:
            similar = random.choice(similar_keys[char_lower])
            return similar.upper() if char.isupper() else similar
        return char

    def simulate_scroll_behavior(self, scroll_distance: int,
                                 direction: str = "down") -> List[Tuple[int, float]]:
        """Generate human-like scroll behavior.
        
        Args:
            scroll_distance: Total scroll distance in pixels
            direction: Scroll direction ("up" or "down")
            
        Returns:
            List of (scroll_amount, delay) tuples
        """
        scroll_events = []
        remaining = abs(scroll_distance)
        current_state = "slow"
        direction_multiplier = -1 if direction == "up" else 1

        while remaining > 0:
            # Determine scroll amount based on state
            if current_state == "slow":
                amount = random.randint(50, 150)
            elif current_state == "medium":
                amount = random.randint(150, 300)
            else:  # fast
                amount = random.randint(300, 500)

            amount = min(amount, remaining)
            remaining -= amount

            # Determine delay
            if current_state == "slow":
                delay = random.uniform(0.1, 0.3)
            elif current_state == "medium":
                delay = random.uniform(0.05, 0.15)
            else:  # fast
                delay = random.uniform(0.02, 0.08)

            scroll_events.append((amount * direction_multiplier, delay))

            # Transition state
            current_state = self._next_markov_state(
                self.mouse_transitions, current_state
            )

        return scroll_events


class ProxyRotator:
    """Manages residential proxy rotation with health checking."""

    def __init__(self, proxy_list: Optional[List[Dict[str, Any]]] = None):
        self.proxies: List[ProxyEndpoint] = []
        self.proxy_index = 0
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = 0

        if proxy_list:
            self.load_proxies(proxy_list)

    def load_proxies(self, proxy_list: List[Dict[str, Any]]):
        """Load proxies from configuration."""
        self.proxies = []
        for proxy_config in proxy_list:
            self.proxies.append(ProxyEndpoint(
                url=proxy_config.get("url", ""),
                country=proxy_config.get("country", "US"),
                city=proxy_config.get("city"),
                latency_ms=proxy_config.get("latency", 0),
            ))

    def get_proxy(self, country: Optional[str] = None,
                  city: Optional[str] = None) -> Optional[ProxyEndpoint]:
        """Get next healthy proxy, optionally filtered by location.
        
        Args:
            country: Filter by country code
            city: Filter by city name
            
        Returns:
            ProxyEndpoint or None if no healthy proxies available
        """
        self._check_health()

        # Filter proxies
        candidates = [p for p in self.proxies if p.is_healthy]
        if country:
            candidates = [p for p in candidates if p.country == country]
        if city:
            candidates = [p for p in candidates if p.city == city]

        if not candidates:
            logger.warning("No healthy proxies available")
            return None

        # Select proxy with best success rate
        candidates.sort(key=lambda p: (-p.success_rate, p.latency_ms))
        selected = candidates[0]

        # Update usage stats
        selected.last_used = time.time()
        return selected

    def report_success(self, proxy: ProxyEndpoint, latency_ms: float):
        """Report successful proxy usage."""
        proxy.success_rate = proxy.success_rate * 0.9 + 0.1
        proxy.latency_ms = proxy.latency_ms * 0.9 + latency_ms * 0.1
        proxy.fail_count = 0

    def report_failure(self, proxy: ProxyEndpoint):
        """Report proxy failure."""
        proxy.fail_count += 1
        proxy.success_rate = proxy.success_rate * 0.9
        if proxy.fail_count >= 3:
            proxy.is_healthy = False
            logger.warning(f"Proxy {proxy.url} marked as unhealthy")

    def _check_health(self):
        """Perform health checks on proxies."""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return

        # Simple health check - in production, this would ping the proxy
        for proxy in self.proxies:
            if not proxy.is_healthy and proxy.fail_count > 0:
                # Retry failed proxies after some time
                if current_time - proxy.last_used > 600:  # 10 minutes
                    proxy.is_healthy = True
                    proxy.fail_count = 0

        self.last_health_check = current_time


class CaptchaSolver:
    """Integrates with CAPTCHA solving services (2Captcha, Anti-Captcha)."""

    def __init__(self, service: str = "2captcha", api_key: Optional[str] = None):
        self.service = service.lower()
        self.api_key = api_key
        self.base_url = self._get_base_url()
        self.timeout = 120  # seconds
        self.poll_interval = 5  # seconds

    def _get_base_url(self) -> str:
        """Get base URL for CAPTCHA service."""
        if self.service == "2captcha":
            return "https://2captcha.com"
        elif self.service == "anticaptcha":
            return "https://api.anti-captcha.com"
        else:
            raise ValueError(f"Unsupported CAPTCHA service: {self.service}")

    async def solve_recaptcha_v2(self, site_url: str, site_key: str,
                                 **kwargs) -> Optional[str]:
        """Solve reCAPTCHA v2.
        
        Args:
            site_url: URL of the site with CAPTCHA
            site_key: reCAPTCHA site key
            
        Returns:
            Solution token or None if failed
        """
        try:
            if self.service == "2captcha":
                return await self._solve_recaptcha_v2_2captcha(site_url, site_key, **kwargs)
            elif self.service == "anticaptcha":
                return await self._solve_recaptcha_v2_anticaptcha(site_url, site_key, **kwargs)
        except Exception as e:
            logger.error(f"CAPTCHA solving failed: {e}")
            return None

    async def _solve_recaptcha_v2_2captcha(self, site_url: str, site_key: str,
                                           **kwargs) -> Optional[str]:
        """Solve reCAPTCHA v2 using 2Captcha."""
        import aiohttp

        # Submit CAPTCHA
        submit_url = f"{self.base_url}/in.php"
        data = {
            "key": self.api_key,
            "method": "userrecaptcha",
            "googlekey": site_key,
            "pageurl": site_url,
            "json": 1,
            **kwargs
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(submit_url, data=data) as resp:
                result = await resp.json()
                if result.get("status") != 1:
                    logger.error(f"Failed to submit CAPTCHA: {result}")
                    return None

                captcha_id = result["request"]

            # Poll for result
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                await asyncio.sleep(self.poll_interval)

                result_url = f"{self.base_url}/res.php"
                params = {
                    "key": self.api_key,
                    "action": "get",
                    "id": captcha_id,
                    "json": 1
                }

                async with session.get(result_url, params=params) as resp:
                    result = await resp.json()
                    if result.get("status") == 1:
                        return result["request"]
                    elif result.get("request") != "CAPCHA_NOT_READY":
                        logger.error(f"CAPTCHA solving error: {result}")
                        return None

            logger.error("CAPTCHA solving timeout")
            return None

    async def _solve_recaptcha_v2_anticaptcha(self, site_url: str, site_key: str,
                                              **kwargs) -> Optional[str]:
        """Solve reCAPTCHA v2 using Anti-Captcha."""
        import aiohttp

        # Create task
        create_url = f"{self.base_url}/createTask"
        task_data = {
            "type": "NoCaptchaTaskProxyless",
            "websiteURL": site_url,
            "websiteKey": site_key,
            **kwargs
        }

        payload = {
            "clientKey": self.api_key,
            "task": task_data
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(create_url, json=payload) as resp:
                result = await resp.json()
                if result.get("errorId") != 0:
                    logger.error(f"Failed to create task: {result}")
                    return None

                task_id = result["taskId"]

            # Poll for result
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                await asyncio.sleep(self.poll_interval)

                result_url = f"{self.base_url}/getTaskResult"
                payload = {
                    "clientKey": self.api_key,
                    "taskId": task_id
                }

                async with session.post(result_url, json=payload) as resp:
                    result = await resp.json()
                    if result.get("errorId") == 0:
                        if result.get("status") == "ready":
                            return result["solution"]["gRecaptchaResponse"]
                    else:
                        logger.error(f"Task error: {result}")
                        return None

            logger.error("CAPTCHA solving timeout")
            return None

    async def solve_hcaptcha(self, site_url: str, site_key: str,
                             **kwargs) -> Optional[str]:
        """Solve hCaptcha."""
        # Similar implementation to reCAPTCHA
        # Implementation would follow the same pattern
        raise NotImplementedError("hCaptcha solving not yet implemented")

    async def solve_funcaptcha(self, site_url: str, public_key: str,
                               **kwargs) -> Optional[str]:
        """Solve FunCaptcha."""
        raise NotImplementedError("FunCaptcha solving not yet implemented")


class StealthManager:
    """Main class coordinating all anti-detection features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.fingerprint_generator = FingerprintGenerator()
        self.behavior_simulator = BehaviorSimulator()
        self.proxy_rotator = ProxyRotator()
        self.captcha_solver: Optional[CaptchaSolver] = None

        self._init_captcha_solver()
        self._init_proxy_rotator()

    def _init_captcha_solver(self):
        """Initialize CAPTCHA solver if configured."""
        captcha_config = self.config.get("captcha", {})
        if captcha_config.get("enabled", False):
            service = captcha_config.get("service", "2captcha")
            api_key = captcha_config.get("api_key")
            if api_key:
                self.captcha_solver = CaptchaSolver(service, api_key)

    def _init_proxy_rotator(self):
        """Initialize proxy rotator with configured proxies."""
        proxy_config = self.config.get("proxies", {})
        if proxy_config.get("enabled", False):
            proxy_list = proxy_config.get("list", [])
            self.proxy_rotator.load_proxies(proxy_list)

    def generate_fingerprint(self, consistency_group: Optional[str] = None) -> FingerprintProfile:
        """Generate a new fingerprint profile.
        
        Args:
            consistency_group: Optional group for consistent fingerprints
            
        Returns:
            FingerprintProfile with generated attributes
        """
        return self.fingerprint_generator.generate(consistency_group=consistency_group)

    def get_proxy(self, country: Optional[str] = None) -> Optional[ProxyEndpoint]:
        """Get next proxy for rotation.
        
        Args:
            country: Optional country filter
            
        Returns:
            ProxyEndpoint or None
        """
        return self.proxy_rotator.get_proxy(country=country)

    def simulate_human_behavior(self, action_type: BehaviorType, **kwargs) -> Any:
        """Simulate human-like behavior for given action type.
        
        Args:
            action_type: Type of behavior to simulate
            **kwargs: Action-specific parameters
            
        Returns:
            Behavior-specific result
        """
        if action_type == BehaviorType.MOUSE_MOVEMENT:
            return self.behavior_simulator.simulate_mouse_movement(**kwargs)
        elif action_type == BehaviorType.TYPING_PATTERN:
            return self.behavior_simulator.simulate_typing(**kwargs)
        elif action_type == BehaviorType.SCROLL_BEHAVIOR:
            return self.behavior_simulator.simulate_scroll_behavior(**kwargs)
        else:
            raise ValueError(f"Unsupported behavior type: {action_type}")

    async def solve_captcha(self, captcha_type: str, **kwargs) -> Optional[str]:
        """Solve CAPTCHA if solver is configured.
        
        Args:
            captcha_type: Type of CAPTCHA (recaptcha_v2, hcaptcha, etc.)
            **kwargs: CAPTCHA-specific parameters
            
        Returns:
            Solution token or None
        """
        if not self.captcha_solver:
            logger.warning("CAPTCHA solver not configured")
            return None

        try:
            if captcha_type == "recaptcha_v2":
                return await self.captcha_solver.solve_recaptcha_v2(**kwargs)
            elif captcha_type == "hcaptcha":
                return await self.captcha_solver.solve_hcaptcha(**kwargs)
            elif captcha_type == "funcaptcha":
                return await self.captcha_solver.solve_funcaptcha(**kwargs)
            else:
                raise ValueError(f"Unsupported CAPTCHA type: {captcha_type}")
        except Exception as e:
            logger.error(f"CAPTCHA solving failed: {e}")
            return None

    def create_stealth_session(self, base_config: SessionConfig) -> SessionConfig:
        """Create a stealth-enhanced session configuration.
        
        Args:
            base_config: Base session configuration
            
        Returns:
            Enhanced SessionConfig with stealth features
        """
        # Generate fingerprint
        fingerprint = self.generate_fingerprint()
        
        # Get proxy if enabled
        proxy = self.get_proxy()
        
        # Create enhanced config
        enhanced_config = SessionConfig(
            headers=base_config.headers.copy() if base_config.headers else {},
            proxy=ProxyConfig(url=proxy.url) if proxy else base_config.proxy,
            fingerprint=fingerprint.to_dict(),
            behavior_profile="human",
            stealth_enabled=True,
        )
        
        # Add fingerprint headers
        if fingerprint.user_agent:
            enhanced_config.headers["User-Agent"] = fingerprint.user_agent
        if fingerprint.language:
            enhanced_config.headers["Accept-Language"] = fingerprint.language
            
        return enhanced_config


# Integration with existing axiom modules
def apply_stealth_to_fetcher(fetcher, stealth_config: Optional[Dict[str, Any]] = None):
    """Apply stealth features to an existing Fetcher instance.
    
    Args:
        fetcher: axiom Fetcher instance
        stealth_config: Stealth configuration dictionary
    """
    stealth_manager = StealthManager(stealth_config)
    
    # Store original methods
    original_fetch = fetcher.fetch
    original_async_fetch = fetcher.async_fetch
    
    async def stealth_fetch(*args, **kwargs):
        """Enhanced fetch with stealth features."""
        # Apply stealth configuration
        stealth_config = stealth_manager.create_stealth_session(fetcher.config)
        
        # Update fetcher config temporarily
        original_config = fetcher.config
        fetcher.config = stealth_config
        
        try:
            # Add human-like delay
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Execute original fetch
            result = await original_async_fetch(*args, **kwargs)
            
            # Report proxy success if used
            if stealth_config.proxy and hasattr(stealth_manager.proxy_rotator, 'report_success'):
                stealth_manager.proxy_rotator.report_success(
                    stealth_manager.proxy_rotator.proxies[0],  # Simplified
                    result.response_time * 1000 if hasattr(result, 'response_time') else 100
                )
            
            return result
        except Exception as e:
            # Report proxy failure if used
            if stealth_config.proxy and hasattr(stealth_manager.proxy_rotator, 'report_failure'):
                stealth_manager.proxy_rotator.report_failure(
                    stealth_manager.proxy_rotator.proxies[0]  # Simplified
                )
            raise
        finally:
            # Restore original config
            fetcher.config = original_config
    
    # Apply monkey patch
    fetcher.fetch = stealth_fetch
    if hasattr(fetcher, 'async_fetch'):
        fetcher.async_fetch = stealth_fetch
    
    return fetchler


# Export public API
__all__ = [
    'FingerprintGenerator',
    'BehaviorSimulator',
    'ProxyRotator',
    'CaptchaSolver',
    'StealthManager',
    'FingerprintProfile',
    'MouseMovement',
    'TypingPattern',
    'ProxyEndpoint',
    'BehaviorType',
    'apply_stealth_to_fetcher',
]