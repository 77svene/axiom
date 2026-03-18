"""
axiom/stealth/proxies.py

Advanced Anti-Detection Suite for axiom
ML-powered fingerprint randomization, behavior simulation, and proxy management
"""

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from axiom.core.ai import BaseNeuralModel
from axiom.core.custom_types import (
    ProxyConfig, FingerprintProfile, BehaviorPattern,
    CaptchaSolution, AntiDetectionConfig
)
from axiom.core.utils._utils import generate_user_agent, get_random_viewport


class ProxyType(Enum):
    """Types of proxies supported"""
    RESIDENTIAL = "residential"
    DATACENTER = "datacenter"
    MOBILE = "mobile"
    ISP = "isp"


class CaptchaService(Enum):
    """Supported CAPTCHA solving services"""
    TWOCAPTCHA = "2captcha"
    ANTICAPTCHA = "anticaptcha"
    CAPMONSTER = "capmonster"
    DEATHBYCAPTCHA = "deathbycaptcha"


@dataclass
class ProxyNode:
    """Represents a single proxy node with metadata"""
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    proxy_type: ProxyType = ProxyType.RESIDENTIAL
    country_code: Optional[str] = None
    city: Optional[str] = None
    asn: Optional[str] = None
    latency: float = 0.0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    fail_count: int = 0
    is_banned: bool = False
    session_id: Optional[str] = None
    
    @property
    def url(self) -> str:
        """Generate proxy URL"""
        if self.username and self.password:
            return f"{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.host}:{self.port}"
    
    @property
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary for requests"""
        return {
            "http": f"http://{self.url}",
            "https": f"http://{self.url}",
        }


class FingerprintGeneratorGAN(nn.Module if TORCH_AVAILABLE else object):
    """
    GAN-based fingerprint generator for realistic browser fingerprint creation
    Uses adversarial training to generate undetectable fingerprints
    """
    
    def __init__(self, latent_dim: int = 100, feature_dim: int = 512):
        """Initialize the GAN architecture"""
        if TORCH_AVAILABLE:
            super().__init__()
            self.latent_dim = latent_dim
            
            # Generator network
            self.generator = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512),
                nn.Linear(512, feature_dim),
                nn.Tanh()
            )
            
            # Discriminator network
            self.discriminator = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        else:
            self.generator = None
            self.discriminator = None
    
    def generate_fingerprint(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a realistic browser fingerprint"""
        if TORCH_AVAILABLE and self.generator:
            # Use GAN for generation
            if seed is not None:
                torch.manual_seed(seed)
            z = torch.randn(1, self.latent_dim)
            with torch.no_grad():
                features = self.generator(z).squeeze().numpy()
            
            # Convert features to fingerprint attributes
            return self._features_to_fingerprint(features)
        else:
            # Fallback to rule-based generation
            return self._generate_rule_based()
    
    def _features_to_fingerprint(self, features: np.ndarray) -> Dict[str, Any]:
        """Convert neural network features to fingerprint attributes"""
        # Normalize features to [0, 1]
        features = (features + 1) / 2
        
        # Map features to fingerprint attributes
        return {
            "user_agent": self._generate_user_agent_from_features(features[:10]),
            "viewport": self._generate_viewport_from_features(features[10:14]),
            "platform": self._select_platform(features[14]),
            "hardware_concurrency": int(features[15] * 16) + 1,
            "device_memory": int(features[16] * 8) + 1,
            "max_touch_points": int(features[17] * 5),
            "webgl_vendor": self._select_webgl_vendor(features[18]),
            "webgl_renderer": self._select_webgl_renderer(features[19]),
            "canvas_hash": f"{int(features[20] * 1000000):08x}",
            "audio_hash": f"{int(features[21] * 1000000):08x}",
            "fonts": self._select_fonts(features[22:42]),
            "plugins": self._select_plugins(features[42:52]),
            "timezone": self._select_timezone(features[52]),
            "languages": self._select_languages(features[53:58]),
            "web_rtc_ip": self._generate_webrtc_ip(features[58]),
            "battery_charging": features[59] > 0.5,
            "battery_level": float(features[60]),
            "connection_type": self._select_connection_type(features[61]),
        }
    
    def _generate_rule_based(self) -> Dict[str, Any]:
        """Fallback rule-based fingerprint generation"""
        return {
            "user_agent": generate_user_agent(),
            "viewport": get_random_viewport(),
            "platform": random.choice(["Win32", "Win64", "MacIntel", "Linux x86_64"]),
            "hardware_concurrency": random.randint(2, 16),
            "device_memory": random.choice([2, 4, 8, 16, 32]),
            "max_touch_points": random.randint(0, 10),
            "webgl_vendor": random.choice(["Google Inc.", "NVIDIA Corporation", "Intel Inc."]),
            "webgl_renderer": random.choice([
                "ANGLE (Intel(R) UHD Graphics 630 Direct3D11)",
                "ANGLE (NVIDIA GeForce GTX 1660 Ti Direct3D11)",
                "ANGLE (AMD Radeon RX 580 Direct3D11)"
            ]),
            "canvas_hash": f"{random.randint(0, 0xFFFFFFFF):08x}",
            "audio_hash": f"{random.randint(0, 0xFFFFFFFF):08x}",
            "fonts": random.sample([
                "Arial", "Verdana", "Helvetica", "Times New Roman", "Courier New",
                "Georgia", "Palatino", "Garamond", "Bookman", "Trebuchet MS"
            ], k=random.randint(5, 10)),
            "plugins": self._generate_plugins(),
            "timezone": random.choice([
                "America/New_York", "America/Los_Angeles", "Europe/London",
                "Europe/Paris", "Asia/Tokyo", "Australia/Sydney"
            ]),
            "languages": random.sample([
                "en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "ja-JP"
            ], k=random.randint(1, 3)),
            "web_rtc_ip": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "battery_charging": random.choice([True, False]),
            "battery_level": random.random(),
            "connection_type": random.choice(["wifi", "4g", "3g", "ethernet"]),
        }
    
    def _generate_plugins(self) -> List[Dict[str, str]]:
        """Generate realistic browser plugins"""
        plugins = []
        if random.random() > 0.3:
            plugins.append({
                "name": "PDF Viewer",
                "filename": "internal-pdf-viewer",
                "description": "Portable Document Format"
            })
        if random.random() > 0.5:
            plugins.append({
                "name": "Chrome PDF Viewer",
                "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                "description": "Portable Document Format"
            })
        if random.random() > 0.7:
            plugins.append({
                "name": "Native Client",
                "filename": "internal-nacl-plugin",
                "description": "Native Client Executable"
            })
        return plugins
    
    # Helper methods for feature-based generation
    def _generate_user_agent_from_features(self, features: np.ndarray) -> str:
        """Generate user agent from feature vector"""
        browsers = [
            ("Chrome", f"{int(features[0] * 30) + 80}.0.{int(features[1] * 5000)}.{int(features[2] * 200)}"),
            ("Firefox", f"{int(features[3] * 50) + 80}.0"),
            ("Safari", f"{int(features[4] * 10) + 13}.0.{int(features[5] * 5)}"),
            ("Edge", f"{int(features[6] * 20) + 80}.0.{int(features[7] * 500)}.{int(features[8] * 50)}"),
        ]
        browser_name, browser_version = random.choice(browsers)
        
        os_choices = [
            f"Windows NT {random.choice(['10.0', '6.1', '6.3'])}",
            f"Macintosh; Intel Mac OS X {random.choice(['10_15_7', '10_14_6', '10_13_6'])}",
            f"X11; Linux x86_{random.choice(['64', '64'])}",
        ]
        
        ua_templates = {
            "Chrome": f"Mozilla/5.0 ({random.choice(os_choices)}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{browser_version} Safari/537.36",
            "Firefox": f"Mozilla/5.0 ({random.choice(os_choices)}; rv:{browser_version}) Gecko/20100101 Firefox/{browser_version}",
            "Safari": f"Mozilla/5.0 ({random.choice(os_choices)}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{browser_version} Safari/605.1.15",
            "Edge": f"Mozilla/5.0 ({random.choice(os_choices)}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{browser_version} Safari/537.36 Edg/{browser_version}",
        }
        
        return ua_templates.get(browser_name, ua_templates["Chrome"])
    
    def _generate_viewport_from_features(self, features: np.ndarray) -> Dict[str, int]:
        """Generate viewport dimensions from features"""
        common_resolutions = [
            (1920, 1080), (1366, 768), (1440, 900), (1536, 864),
            (1600, 900), (1280, 720), (2560, 1440), (3840, 2160)
        ]
        width, height = random.choice(common_resolutions)
        # Add some variation
        width += int((features[0] - 0.5) * 20)
        height += int((features[1] - 0.5) * 20)
        return {"width": width, "height": height}
    
    def _select_platform(self, feature: float) -> str:
        """Select platform based on feature value"""
        platforms = ["Win32", "Win64", "MacIntel", "Linux x86_64", "Linux armv8l"]
        idx = int(feature * len(platforms))
        return platforms[min(idx, len(platforms) - 1)]
    
    def _select_webgl_vendor(self, feature: float) -> str:
        """Select WebGL vendor based on feature"""
        vendors = ["Google Inc.", "NVIDIA Corporation", "Intel Inc.", "AMD", "Apple Inc."]
        idx = int(feature * len(vendors))
        return vendors[min(idx, len(vendors) - 1)]
    
    def _select_webgl_renderer(self, feature: float) -> str:
        """Select WebGL renderer based on feature"""
        renderers = [
            "ANGLE (Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (NVIDIA GeForce GTX 1660 Ti Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (AMD Radeon RX 580 Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (Intel(R) Iris(R) Plus Graphics 640 Direct3D11)",
            "ANGLE (NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0)",
        ]
        idx = int(feature * len(renderers))
        return renderers[min(idx, len(renderers) - 1)]
    
    def _select_fonts(self, features: np.ndarray) -> List[str]:
        """Select fonts based on feature vector"""
        all_fonts = [
            "Arial", "Arial Black", "Arial Narrow", "Calibri", "Cambria",
            "Comic Sans MS", "Courier", "Courier New", "Georgia", "Helvetica",
            "Impact", "Lucida Console", "Lucida Sans Unicode", "Microsoft Sans Serif",
            "Palatino Linotype", "Segoe UI", "Tahoma", "Times New Roman",
            "Trebuchet MS", "Verdana"
        ]
        # Select fonts where feature > 0.5
        selected = [font for font, feat in zip(all_fonts, features) if feat > 0.5]
        # Ensure at least 3 fonts
        if len(selected) < 3:
            selected = random.sample(all_fonts, k=3)
        return selected
    
    def _select_plugins(self, features: np.ndarray) -> List[Dict[str, str]]:
        """Select plugins based on feature vector"""
        all_plugins = [
            {"name": "PDF Viewer", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
            {"name": "Chrome PDF Viewer", "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai", "description": "Portable Document Format"},
            {"name": "Chromium PDF Viewer", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
            {"name": "Microsoft Edge PDF Viewer", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
            {"name": "WebKit built-in PDF", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
        ]
        selected = [plugin for plugin, feat in zip(all_plugins, features) if feat > 0.5]
        return selected if selected else [all_plugins[0]]
    
    def _select_timezone(self, feature: float) -> str:
        """Select timezone based on feature"""
        timezones = [
            "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
            "America/Anchorage", "Pacific/Honolulu", "Europe/London", "Europe/Paris",
            "Europe/Berlin", "Asia/Tokyo", "Asia/Shanghai", "Asia/Singapore",
            "Australia/Sydney", "Australia/Melbourne"
        ]
        idx = int(feature * len(timezones))
        return timezones[min(idx, len(timezones) - 1)]
    
    def _select_languages(self, features: np.ndarray) -> List[str]:
        """Select languages based on feature vector"""
        all_languages = ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "ja-JP", "zh-CN", "ko-KR"]
        selected = [lang for lang, feat in zip(all_languages, features) if feat > 0.5]
        if not selected:
            selected = ["en-US"]
        return selected
    
    def _generate_webrtc_ip(self, feature: float) -> str:
        """Generate WebRTC IP leak simulation"""
        if feature > 0.7:
            # Simulate VPN/proxy IP leak
            return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        else:
            return f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
    
    def _select_connection_type(self, feature: float) -> str:
        """Select connection type based on feature"""
        types = ["wifi", "4g", "3g", "2g", "ethernet", "other"]
        idx = int(feature * len(types))
        return types[min(idx, len(types) - 1)]


class MarkovBehaviorSimulator:
    """
    Human-like behavior simulation using Markov chains
    Generates realistic mouse movements, typing patterns, and scrolling
    """
    
    def __init__(self, complexity: str = "medium"):
        """
        Initialize behavior simulator
        
        Args:
            complexity: "low", "medium", or "high" - affects simulation detail
        """
        self.complexity = complexity
        self._init_markov_chains()
        self._init_timing_patterns()
    
    def _init_markov_chains(self):
        """Initialize Markov chains for different behaviors"""
        # Mouse movement states: idle, moving, clicking, scrolling
        self.mouse_states = ["idle", "moving", "clicking", "scrolling"]
        self.mouse_transitions = {
            "idle": {"idle": 0.6, "moving": 0.3, "clicking": 0.05, "scrolling": 0.05},
            "moving": {"idle": 0.2, "moving": 0.6, "clicking": 0.15, "scrolling": 0.05},
            "clicking": {"idle": 0.4, "moving": 0.4, "clicking": 0.1, "scrolling": 0.1},
            "scrolling": {"idle": 0.3, "moving": 0.4, "clicking": 0.1, "scrolling": 0.2},
        }
        
        # Typing states: thinking, typing_fast, typing_slow, correcting
        self.typing_states = ["thinking", "typing_fast", "typing_slow", "correcting"]
        self.typing_transitions = {
            "thinking": {"thinking": 0.3, "typing_fast": 0.4, "typing_slow": 0.2, "correcting": 0.1},
            "typing_fast": {"thinking": 0.2, "typing_fast": 0.5, "typing_slow": 0.2, "correcting": 0.1},
            "typing_slow": {"thinking": 0.3, "typing_fast": 0.2, "typing_slow": 0.4, "correcting": 0.1},
            "correcting": {"thinking": 0.4, "typing_fast": 0.2, "typing_slow": 0.3, "correcting": 0.1},
        }
    
    def _init_timing_patterns(self):
        """Initialize timing patterns for human-like delays"""
        # Base delays in seconds
        self.base_delays = {
            "mouse_move": (0.05, 0.3),
            "mouse_click": (0.08, 0.25),
            "keypress": (0.05, 0.2),
            "scroll": (0.1, 0.5),
            "page_load_wait": (1.0, 5.0),
            "form_fill_pause": (0.5, 2.0),
        }
        
        # Complexity multipliers
        complexity_multipliers = {
            "low": 0.7,
            "medium": 1.0,
            "high": 1.3,
        }
        multiplier = complexity_multipliers.get(self.complexity, 1.0)
        
        # Apply multipliers
        for key in self.base_delays:
            min_val, max_val = self.base_delays[key]
            self.base_delays[key] = (min_val * multiplier, max_val * multiplier)
    
    def generate_mouse_movement(self, start_x: int, start_y: int, 
                               end_x: int, end_y: int) -> List[Dict[str, Any]]:
        """
        Generate human-like mouse movement path
        
        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            
        Returns:
            List of movement points with timestamps
        """
        movements = []
        current_x, current_y = start_x, start_y
        steps = random.randint(10, 30) if self.complexity == "high" else random.randint(5, 15)
        
        # Add bezier curve for natural movement
        control_points = self._generate_control_points(start_x, start_y, end_x, end_y)
        
        for i in range(steps + 1):
            t = i / steps
            # Quadratic bezier curve
            x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * control_points[0] + t ** 2 * end_x
            y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * control_points[1] + t ** 2 * end_y
            
            # Add some randomness
            if self.complexity != "low":
                x += random.gauss(0, 2)
                y += random.gauss(0, 2)
            
            # Calculate delay
            distance = ((x - current_x) ** 2 + (y - current_y) ** 2) ** 0.5
            delay = self._calculate_movement_delay(distance)
            
            movements.append({
                "x": int(x),
                "y": int(y),
                "timestamp": time.time() + sum(m["delay"] for m in movements),
                "delay": delay,
                "state": self._get_next_mouse_state(movements[-1]["state"] if movements else "idle")
            })
            
            current_x, current_y = x, y
        
        return movements
    
    def _generate_control_points(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float]:
        """Generate control points for bezier curve"""
        # Midpoint with some randomness
        mid_x = (x1 + x2) / 2 + random.uniform(-100, 100)
        mid_y = (y1 + y2) / 2 + random.uniform(-100, 100)
        return (mid_x, mid_y)
    
    def _calculate_movement_delay(self, distance: float) -> float:
        """Calculate delay based on movement distance"""
        min_delay, max_delay = self.base_delays["mouse_move"]
        # Longer distances take relatively less time per pixel (Fitts's Law approximation)
        base_time = min_delay + (max_delay - min_delay) * (distance / 1000)
        # Add human-like variation
        return base_time * random.uniform(0.8, 1.2)
    
    def _get_next_mouse_state(self, current_state: str) -> str:
        """Get next mouse state using Markov chain"""
        transitions = self.mouse_transitions.get(current_state, self.mouse_transitions["idle"])
        states = list(transitions.keys())
        probabilities = list(transitions.values())
        return np.random.choice(states, p=probabilities)
    
    def generate_typing_pattern(self, text: str) -> List[Dict[str, Any]]:
        """
        Generate human-like typing pattern
        
        Args:
            text: Text to type
            
        Returns:
            List of keypress events with delays
        """
        events = []
        current_state = "thinking"
        
        for i, char in enumerate(text):
            # Determine next state
            current_state = self._get_next_typing_state(current_state)
            
            # Calculate delay based on state and character
            delay = self._calculate_typing_delay(current_state, char, i, len(text))
            
            # Simulate occasional mistakes and corrections
            if current_state == "correcting" and random.random() < 0.3:
                # Simulate backspace and retyping
                events.extend(self._simulate_correction(char, delay))
            else:
                events.append({
                    "key": char,
                    "timestamp": time.time() + sum(e["delay"] for e in events),
                    "delay": delay,
                    "state": current_state
                })
            
            # Add pause at punctuation
            if char in ".!?,:;":
                events.append({
                    "key": "pause",
                    "timestamp": time.time() + sum(e["delay"] for e in events),
                    "delay": random.uniform(0.3, 0.8),
                    "state": "thinking"
                })
        
        return events
    
    def _get_next_typing_state(self, current_state: str) -> str:
        """Get next typing state using Markov chain"""
        transitions = self.typing_transitions.get(current_state, self.typing_transitions["thinking"])
        states = list(transitions.keys())
        probabilities = list(transitions.values())
        return np.random.choice(states, p=probabilities)
    
    def _calculate_typing_delay(self, state: str, char: str, position: int, total: int) -> float:
        """Calculate typing delay based on state and character"""
        min_delay, max_delay = self.base_delays["keypress"]
        
        # State-based multipliers
        state_multipliers = {
            "thinking": 2.0,
            "typing_fast": 0.7,
            "typing_slow": 1.5,
            "correcting": 1.2,
        }
        multiplier = state_multipliers.get(state, 1.0)
        
        # Character-based adjustments
        if char.isupper():
            multiplier *= 1.1  # Slight delay for shift key
        elif char in "qwertyuiopasdfghjklzxcvbnm":
            multiplier *= 1.0  # Normal letters
        elif char in "1234567890":
            multiplier *= 1.05  # Numbers
        else:
            multiplier *= 1.15  # Special characters
        
        # Position-based fatigue (slower at end)
        if position > total * 0.8:
            multiplier *= 1.1
        
        base_delay = min_delay + (max_delay - min_delay) * random.random()
        return base_delay * multiplier * random.uniform(0.9, 1.1)
    
    def _simulate_correction(self, correct_char: str, base_delay: float) -> List[Dict[str, Any]]:
        """Simulate typing mistake and correction"""
        events = []
        
        # Type wrong character
        wrong_char = self._get_similar_key(correct_char)
        events.append({
            "key": wrong_char,
            "timestamp": time.time() + sum(e["delay"] for e in events),
            "delay": base_delay * 0.8,
            "state": "typing_fast"
        })
        
        # Pause (realize mistake)
        events.append({
            "key": "pause",
            "timestamp": time.time() + sum(e["delay"] for e in events),
            "delay": random.uniform(0.2, 0.5),
            "state": "thinking"
        })
        
        # Backspace
        events.append({
            "key": "Backspace",
            "timestamp": time.time() + sum(e["delay"] for e in events),
            "delay": random.uniform(0.1, 0.2),
            "state": "correcting"
        })
        
        # Type correct character
        events.append({
            "key": correct_char,
            "timestamp": time.time() + sum(e["delay"] for e in events),
            "delay": base_delay * 1.2,
            "state": "typing_slow"
        })
        
        return events
    
    def _get_similar_key(self, char: str) -> str:
        """Get a similar key for simulating mistakes"""
        similar_keys = {
            'a': ['s', 'q', 'z'],
            'b': ['v', 'g', 'h'],
            'c': ['x', 'v', 'd'],
            'd': ['s', 'f', 'e'],
            'e': ['w', 'r', 'd'],
            'f': ['d', 'g', 'r'],
            'g': ['f', 'h', 't'],
            'h': ['g', 'j', 'y'],
            'i': ['u', 'o', 'k'],
            'j': ['h', 'k', 'u'],
            'k': ['j', 'l', 'i'],
            'l': ['k', 'p', 'o'],
            'm': ['n', 'j', 'k'],
            'n': ['m', 'h', 'j'],
            'o': ['i', 'p', 'l'],
            'p': ['o', 'l', '0'],
            'q': ['w', 'a', '1'],
            'r': ['e', 't', 'f'],
            's': ['a', 'd', 'w'],
            't': ['r', 'y', 'g'],
            'u': ['y', 'i', 'j'],
            'v': ['c', 'b', 'f'],
            'w': ['q', 'e', 's'],
            'x': ['z', 'c', 's'],
            'y': ['t', 'u', 'h'],
            'z': ['x', 'a', 's'],
        }
        return random.choice(similar_keys.get(char.lower(), [char]))


class ProxyRotator:
    """
    Manages a pool of residential proxies with intelligent rotation
    """
    
    def __init__(self, proxy_list: List[Dict[str, Any]], 
                 rotation_strategy: str = "round_robin",
                 max_fails: int = 3,
                 ban_duration: int = 300):
        """
        Initialize proxy rotator
        
        Args:
            proxy_list: List of proxy configurations
            rotation_strategy: "round_robin", "random", "smart", or "geographic"
            max_fails: Maximum failures before banning proxy
            ban_duration: Duration to ban proxy in seconds
        """
        self.proxies = [self._create_proxy_node(proxy) for proxy in proxy_list]
        self.rotation_strategy = rotation_strategy
        self.max_fails = max_fails
        self.ban_duration = ban_duration
        self._current_index = 0
        self._geo_distribution = self._analyze_geo_distribution()
    
    def _create_proxy_node(self, proxy_config: Dict[str, Any]) -> ProxyNode:
        """Create ProxyNode from configuration"""
        return ProxyNode(
            host=proxy_config["host"],
            port=proxy_config["port"],
            username=proxy_config.get("username"),
            password=proxy_config.get("password"),
            proxy_type=ProxyType(proxy_config.get("type", "residential")),
            country_code=proxy_config.get("country"),
            city=proxy_config.get("city"),
            asn=proxy_config.get("asn"),
        )
    
    def _analyze_geo_distribution(self) -> Dict[str, List[ProxyNode]]:
        """Analyze geographic distribution of proxies"""
        distribution = {}
        for proxy in self.proxies:
            country = proxy.country_code or "unknown"
            if country not in distribution:
                distribution[country] = []
            distribution[country].append(proxy)
        return distribution
    
    def get_proxy(self, target_country: Optional[str] = None,
                  target_city: Optional[str] = None,
                  require_https: bool = True) -> Optional[ProxyNode]:
        """
        Get next proxy based on rotation strategy
        
        Args:
            target_country: Preferred country code
            target_city: Preferred city
            require_https: Whether to require HTTPS support
            
        Returns:
            ProxyNode or None if no suitable proxy available
        """
        # Filter available proxies
        available = self._get_available_proxies()
        
        # Apply filters
        if target_country:
            available = [p for p in available if p.country_code == target_country]
        
        if target_city:
            available = [p for p in available if p.city == target_city]
        
        if not available:
            # Fallback to all available if no matches
            available = self._get_available_proxies()
        
        if not available:
            return None
        
        # Select based on strategy
        if self.rotation_strategy == "round_robin":
            proxy = self._round_robin_select(available)
        elif self.rotation_strategy == "random":
            proxy = random.choice(available)
        elif self.rotation_strategy == "smart":
            proxy = self._smart_select(available)
        elif self.rotation_strategy == "geographic":
            proxy = self._geographic_select(available, target_country)
        else:
            proxy = available[0]
        
        # Update usage stats
        proxy.last_used = datetime.now()
        proxy.session_id = str(uuid.uuid4())
        
        return proxy
    
    def _get_available_proxies(self) -> List[ProxyNode]:
        """Get list of available (not banned) proxies"""
        now = datetime.now()
        available = []
        
        for proxy in self.proxies:
            if proxy.is_banned:
                # Check if ban has expired
                if proxy.last_used and (now - proxy.last_used).seconds > self.ban_duration:
                    proxy.is_banned = False
                    proxy.fail_count = 0
                    available.append(proxy)
            else:
                available.append(proxy)
        
        return available
    
    def _round_robin_select(self, proxies: List[ProxyNode]) -> ProxyNode:
        """Select proxy using round-robin strategy"""
        if not proxies:
            raise ValueError("No proxies available")
        
        proxy = proxies[self._current_index % len(proxies)]
        self._current_index += 1
        return proxy
    
    def _smart_select(self, proxies: List[ProxyNode]) -> ProxyNode:
        """
        Smart selection based on success rate and latency
        Uses weighted random selection
        """
        if not proxies:
            raise ValueError("No proxies available")
        
        # Calculate weights based on success rate and inverse latency
        weights = []
        for proxy in proxies:
            # Normalize latency (lower is better)
            latency_score = 1.0 / (proxy.latency + 0.1)  # Avoid division by zero
            
            # Combined score
            score = proxy.success_rate * latency_score
            weights.append(score)
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(proxies)] * len(proxies)
        
        # Weighted random selection
        return np.random.choice(proxies, p=weights)
    
    def _geographic_select(self, proxies: List[ProxyNode], 
                          target_country: Optional[str]) -> ProxyNode:
        """Select proxy based on geographic distribution"""
        if target_country and target_country in self._geo_distribution:
            country_proxies = self._geo_distribution[target_country]
            available_in_country = [p for p in country_proxies if p in proxies]
            if available_in_country:
                return random.choice(available_in_country)
        
        # Fallback to random selection
        return random.choice(proxies)
    
    def report_success(self, proxy: ProxyNode, latency: float):
        """Report successful request with proxy"""
        proxy.latency = latency
        proxy.success_rate = min(1.0, proxy.success_rate + 0.1)
        proxy.fail_count = 0
    
    def report_failure(self, proxy: ProxyNode):
        """Report failed request with proxy"""
        proxy.fail_count += 1
        proxy.success_rate = max(0.0, proxy.success_rate - 0.2)
        
        if proxy.fail_count >= self.max_fails:
            proxy.is_banned = True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get proxy pool statistics"""
        total = len(self.proxies)
        available = len(self._get_available_proxies())
        banned = total - available
        
        avg_success_rate = np.mean([p.success_rate for p in self.proxies]) if self.proxies else 0
        avg_latency = np.mean([p.latency for p in self.proxies if p.latency > 0]) if any(p.latency > 0 for p in self.proxies) else 0
        
        return {
            "total_proxies": total,
            "available_proxies": available,
            "banned_proxies": banned,
            "average_success_rate": float(avg_success_rate),
            "average_latency": float(avg_latency),
            "countries": list(self._geo_distribution.keys()),
        }


class CaptchaSolver:
    """
    CAPTCHA solving integration with multiple service providers
    """
    
    def __init__(self, service: CaptchaService = CaptchaService.TWOCAPTCHA,
                 api_key: Optional[str] = None,
                 timeout: int = 120,
                 polling_interval: int = 5):
        """
        Initialize CAPTCHA solver
        
        Args:
            service: CAPTCHA solving service to use
            api_key: API key for the service
            timeout: Maximum time to wait for solution in seconds
            polling_interval: Interval to check for solution in seconds
        """
        self.service = service
        self.api_key = api_key
        self.timeout = timeout
        self.polling_interval = polling_interval
        self._session = None
    
    async def solve_recaptcha(self, site_url: str, site_key: str,
                             **kwargs) -> Optional[CaptchaSolution]:
        """
        Solve reCAPTCHA v2/v3
        
        Args:
            site_url: URL of the site with CAPTCHA
            site_key: reCAPTCHA site key
            **kwargs: Additional parameters for the service
            
        Returns:
            CaptchaSolution or None if failed
        """
        if not self.api_key:
            raise ValueError("API key required for CAPTCHA solving")
        
        # Prepare request based on service
        if self.service == CaptchaService.TWOCAPTCHA:
            return await self._solve_recaptcha_2captcha(site_url, site_key, **kwargs)
        elif self.service == CaptchaService.ANTICAPTCHA:
            return await self._solve_recaptcha_anticaptcha(site_url, site_key, **kwargs)
        else:
            raise NotImplementedError(f"Service {self.service} not implemented for reCAPTCHA")
    
    async def _solve_recaptcha_2captcha(self, site_url: str, site_key: str,
                                       **kwargs) -> Optional[CaptchaSolution]:
        """Solve reCAPTCHA using 2Captcha service"""
        import aiohttp
        
        # Submit CAPTCHA
        submit_url = "http://2captcha.com/in.php"
        params = {
            "key": self.api_key,
            "method": "userrecaptcha",
            "googlekey": site_key,
            "pageurl": site_url,
            "json": 1,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(submit_url, data=params) as response:
                    data = await response.json()
                    
                    if data.get("status") != 1:
                        return None
                    
                    captcha_id = data["request"]
                    
                    # Poll for solution
                    start_time = time.time()
                    while time.time() - start_time < self.timeout:
                        await asyncio.sleep(self.polling_interval)
                        
                        check_url = "http://2captcha.com/res.php"
                        check_params = {
                            "key": self.api_key,
                            "action": "get",
                            "id": captcha_id,
                            "json": 1
                        }
                        
                        async with session.get(check_url, params=check_params) as check_response:
                            check_data = await check_response.json()
                            
                            if check_data.get("status") == 1:
                                return CaptchaSolution(
                                    solution=check_data["request"],
                                    captcha_id=captcha_id,
                                    service=self.service.value,
                                    cost=0.003,  # Approximate cost
                                    solving_time=time.time() - start_time
                                )
                            elif check_data.get("request") != "CAPCHA_NOT_READY":
                                # Error occurred
                                return None
        
        except Exception as e:
            print(f"CAPTCHA solving error: {e}")
            return None
        
        return None
    
    async def _solve_recaptcha_anticaptcha(self, site_url: str, site_key: str,
                                          **kwargs) -> Optional[CaptchaSolution]:
        """Solve reCAPTCHA using Anti-Captcha service"""
        import aiohttp
        
        # Create task
        create_task_url = "https://api.anti-captcha.com/createTask"
        task_data = {
            "clientKey": self.api_key,
            "task": {
                "type": "NoCaptchaTaskProxyless",
                "websiteURL": site_url,
                "websiteKey": site_key,
                **kwargs
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(create_task_url, json=task_data) as response:
                    data = await response.json()
                    
                    if data.get("errorId") != 0:
                        return None
                    
                    task_id = data["taskId"]
                    
                    # Poll for solution
                    start_time = time.time()
                    while time.time() - start_time < self.timeout:
                        await asyncio.sleep(self.polling_interval)
                        
                        get_result_url = "https://api.anti-captcha.com/getTaskResult"
                        result_data = {
                            "clientKey": self.api_key,
                            "taskId": task_id
                        }
                        
                        async with session.post(get_result_url, json=result_data) as result_response:
                            result = await result_response.json()
                            
                            if result.get("errorId") != 0:
                                return None
                            
                            if result["status"] == "ready":
                                return CaptchaSolution(
                                    solution=result["solution"]["gRecaptchaResponse"],
                                    captcha_id=str(task_id),
                                    service=self.service.value,
                                    cost=result.get("cost", 0.002),
                                    solving_time=time.time() - start_time
                                )
        
        except Exception as e:
            print(f"CAPTCHA solving error: {e}")
            return None
        
        return None
    
    async def solve_hcaptcha(self, site_url: str, site_key: str,
                            **kwargs) -> Optional[CaptchaSolution]:
        """Solve hCaptcha"""
        if not self.api_key:
            raise ValueError("API key required for CAPTCHA solving")
        
        if self.service == CaptchaService.TWOCAPTCHA:
            return await self._solve_hcaptcha_2captcha(site_url, site_key, **kwargs)
        else:
            raise NotImplementedError(f"Service {self.service} not implemented for hCaptcha")
    
    async def _solve_hcaptcha_2captcha(self, site_url: str, site_key: str,
                                      **kwargs) -> Optional[CaptchaSolution]:
        """Solve hCaptcha using 2Captcha service"""
        import aiohttp
        
        submit_url = "http://2captcha.com/in.php"
        params = {
            "key": self.api_key,
            "method": "hcaptcha",
            "sitekey": site_key,
            "pageurl": site_url,
            "json": 1,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(submit_url, data=params) as response:
                    data = await response.json()
                    
                    if data.get("status") != 1:
                        return None
                    
                    captcha_id = data["request"]
                    
                    # Poll for solution
                    start_time = time.time()
                    while time.time() - start_time < self.timeout:
                        await asyncio.sleep(self.polling_interval)
                        
                        check_url = "http://2captcha.com/res.php"
                        check_params = {
                            "key": self.api_key,
                            "action": "get",
                            "id": captcha_id,
                            "json": 1
                        }
                        
                        async with session.get(check_url, params=check_params) as check_response:
                            check_data = await check_response.json()
                            
                            if check_data.get("status") == 1:
                                return CaptchaSolution(
                                    solution=check_data["request"],
                                    captcha_id=captcha_id,
                                    service=self.service.value,
                                    cost=0.003,
                                    solving_time=time.time() - start_time
                                )
                            elif check_data.get("request") != "CAPCHA_NOT_READY":
                                return None
        
        except Exception as e:
            print(f"CAPTCHA solving error: {e}")
            return None
        
        return None
    
    async def solve_funcaptcha(self, site_url: str, public_key: str,
                              **kwargs) -> Optional[CaptchaSolution]:
        """Solve FunCaptcha (Arkose Labs)"""
        if not self.api_key:
            raise ValueError("API key required for CAPTCHA solving")
        
        if self.service == CaptchaService.TWOCAPTCHA:
            return await self._solve_funcaptcha_2captcha(site_url, public_key, **kwargs)
        else:
            raise NotImplementedError(f"Service {self.service} not implemented for FunCaptcha")
    
    async def _solve_funcaptcha_2captcha(self, site_url: str, public_key: str,
                                        **kwargs) -> Optional[CaptchaSolution]:
        """Solve FunCaptcha using 2Captcha service"""
        import aiohttp
        
        submit_url = "http://2captcha.com/in.php"
        params = {
            "key": self.api_key,
            "method": "funcaptcha",
            "publickey": public_key,
            "pageurl": site_url,
            "json": 1,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(submit_url, data=params) as response:
                    data = await response.json()
                    
                    if data.get("status") != 1:
                        return None
                    
                    captcha_id = data["request"]
                    
                    # Poll for solution
                    start_time = time.time()
                    while time.time() - start_time < self.timeout:
                        await asyncio.sleep(self.polling_interval)
                        
                        check_url = "http://2captcha.com/res.php"
                        check_params = {
                            "key": self.api_key,
                            "action": "get",
                            "id": captcha_id,
                            "json": 1
                        }
                        
                        async with session.get(check_url, params=check_params) as check_response:
                            check_data = await check_response.json()
                            
                            if check_data.get("status") == 1:
                                return CaptchaSolution(
                                    solution=check_data["request"],
                                    captcha_id=captcha_id,
                                    service=self.service.value,
                                    cost=0.003,
                                    solving_time=time.time() - start_time
                                )
                            elif check_data.get("request") != "CAPCHA_NOT_READY":
                                return None
        
        except Exception as e:
            print(f"CAPTCHA solving error: {e}")
            return None
        
        return None


class StealthSession:
    """
    Advanced stealth session with ML-powered anti-detection
    Combines fingerprint randomization, behavior simulation, and proxy rotation
    """
    
    def __init__(self, config: Optional[AntiDetectionConfig] = None):
        """
        Initialize stealth session
        
        Args:
            config: Anti-detection configuration
        """
        self.config = config or AntiDetectionConfig()
        
        # Initialize components
        self.fingerprint_generator = FingerprintGeneratorGAN()
        self.behavior_simulator = MarkovBehaviorSimulator(
            complexity=self.config.behavior_complexity
        )
        
        # Initialize proxy rotator if proxies provided
        self.proxy_rotator = None
        if self.config.proxies:
            self.proxy_rotator = ProxyRotator(
                proxy_list=self.config.proxies,
                rotation_strategy=self.config.proxy_rotation_strategy,
                max_fails=self.config.max_proxy_fails,
                ban_duration=self.config.proxy_ban_duration
            )
        
        # Initialize CAPTCHA solver if configured
        self.captcha_solver = None
        if self.config.captcha_service and self.config.captcha_api_key:
            self.captcha_solver = CaptchaSolver(
                service=CaptchaService(self.config.captcha_service),
                api_key=self.config.captcha_api_key,
                timeout=self.config.captcha_timeout
            )
        
        # Session state
        self.current_fingerprint = None
        self.current_proxy = None
        self.session_id = str(uuid.uuid4())
        self.request_count = 0
        self.last_request_time = None
        
        # Behavior history for adaptive learning
        self.behavior_history = []
    
    async def initialize(self):
        """Initialize session with fingerprint and proxy"""
        # Generate initial fingerprint
        self.current_fingerprint = self.fingerprint_generator.generate_fingerprint()
        
        # Get initial proxy if available
        if self.proxy_rotator:
            self.current_proxy = self.proxy_rotator.get_proxy()
    
    async def get_fingerprint(self) -> FingerprintProfile:
        """
        Get current fingerprint or generate new one
        
        Returns:
            FingerprintProfile with current fingerprint data
        """
        if not self.current_fingerprint:
            await self.initialize()
        
        # Occasionally rotate fingerprint
        if (self.config.fingerprint_rotation_interval and 
            self.request_count % self.config.fingerprint_rotation_interval == 0):
            self.current_fingerprint = self.fingerprint_generator.generate_fingerprint()
        
        return FingerprintProfile(**self.current_fingerprint)
    
    async def get_proxy(self) -> Optional[ProxyNode]:
        """
        Get current proxy or rotate to new one
        
        Returns:
            ProxyNode or None if no proxies available
        """
        if not self.proxy_rotator:
            return None
        
        # Rotate proxy if needed
        if (self.config.proxy_rotation_interval and 
            self.request_count % self.config.proxy_rotation_interval == 0):
            self.current_proxy = self.proxy_rotator.get_proxy(
                target_country=self.config.target_country
            )
        
        return self.current_proxy
    
    async def simulate_human_delay(self, action_type: str = "page_load"):
        """
        Simulate human-like delay between actions
        
        Args:
            action_type: Type of action for appropriate delay
        """
        if not self.config.enable_behavior_simulation:
            return
        
        delay_ranges = {
            "page_load": (1.0, 3.0),
            "form_fill": (0.5, 2.0),
            "click": (0.3, 1.0),
            "scroll": (0.2, 0.8),
            "navigation": (0.8, 2.5),
        }
        
        min_delay, max_delay = delay_ranges.get(action_type, (0.5, 1.5))
        
        # Add randomness based on time of day (more human-like)
        hour = datetime.now().hour
        if 9 <= hour <= 17:  # Business hours - faster
            multiplier = 0.8
        elif 0 <= hour <= 6:  # Late night - slower
            multiplier = 1.3
        else:
            multiplier = 1.0
        
        delay = random.uniform(min_delay, max_delay) * multiplier
        
        # Add occasional long pauses (human distraction)
        if random.random() < 0.05:  # 5% chance
            delay += random.uniform(2.0, 5.0)
        
        await asyncio.sleep(delay)
    
    async def generate_mouse_movement(self, start_element: Dict[str, int],
                                     end_element: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Generate mouse movement between elements
        
        Args:
            start_element: Starting element coordinates
            end_element: Ending element coordinates
            
        Returns:
            List of movement points
        """
        if not self.config.enable_behavior_simulation:
            return []
        
        return self.behavior_simulator.generate_mouse_movement(
            start_x=start_element["x"],
            start_y=start_element["y"],
            end_x=end_element["x"],
            end_y=end_element["y"]
        )
    
    async def generate_typing_pattern(self, text: str) -> List[Dict[str, Any]]:
        """
        Generate typing pattern for text input
        
        Args:
            text: Text to type
            
        Returns:
            List of keypress events
        """
        if not self.config.enable_behavior_simulation:
            return []
        
        return self.behavior_simulator.generate_typing_pattern(text)
    
    async def solve_captcha(self, captcha_type: str, **kwargs) -> Optional[CaptchaSolution]:
        """
        Solve CAPTCHA if solver is available
        
        Args:
            captcha_type: Type of CAPTCHA ("recaptcha", "hcaptcha", "funcaptcha")
            **kwargs: CAPTCHA-specific parameters
            
        Returns:
            CaptchaSolution or None if solving failed
        """
        if not self.captcha_solver:
            return None
        
        try:
            if captcha_type == "recaptcha":
                return await self.captcha_solver.solve_recaptcha(**kwargs)
            elif captcha_type == "hcaptcha":
                return await self.captcha_solver.solve_hcaptcha(**kwargs)
            elif captcha_type == "funcaptcha":
                return await self.captcha_solver.solve_funcaptcha(**kwargs)
            else:
                return None
        except Exception as e:
            print(f"CAPTCHA solving failed: {e}")
            return None
    
    async def report_request_result(self, success: bool, latency: float = 0.0):
        """
        Report request result for adaptive learning
        
        Args:
            success: Whether request was successful
            latency: Request latency in seconds
        """
        self.request_count += 1
        self.last_request_time = datetime.now()
        
        # Update proxy stats
        if self.current_proxy and self.proxy_rotator:
            if success:
                self.proxy_rotator.report_success(self.current_proxy, latency)
            else:
                self.proxy_rotator.report_failure(self.current_proxy)
        
        # Record behavior for adaptive learning
        self.behavior_history.append({
            "timestamp": datetime.now(),
            "success": success,
            "latency": latency,
            "fingerprint_hash": hash(str(self.current_fingerprint)),
            "proxy": self.current_proxy.host if self.current_proxy else None,
        })
        
        # Keep history limited
        if len(self.behavior_history) > 1000:
            self.behavior_history = self.behavior_history[-500:]
    
    async def get_detection_risk_score(self) -> float:
        """
        Calculate current detection risk score (0-1, lower is better)
        
        Returns:
            Risk score between 0 and 1
        """
        if not self.behavior_history:
            return 0.0
        
        # Calculate based on recent failures
        recent_history = self.behavior_history[-100:] if len(self.behavior_history) > 100 else self.behavior_history
        if not recent_history:
            return 0.0
        
        failure_rate = sum(1 for h in recent_history if not h["success"]) / len(recent_history)
        
        # Adjust based on proxy stats
        proxy_risk = 0.0
        if self.proxy_rotator:
            stats = self.proxy_rotator.get_stats()
            proxy_risk = 1.0 - stats["average_success_rate"]
        
        # Combined risk score
        risk_score = (failure_rate * 0.7) + (proxy_risk * 0.3)
        
        return min(1.0, max(0.0, risk_score))
    
    async def optimize_configuration(self):
        """
        Optimize configuration based on historical performance
        """
        risk_score = await self.get_detection_risk_score()
        
        # Adjust behavior complexity based on risk
        if risk_score > 0.7:
            # High risk - increase stealth
            self.config.behavior_complexity = "high"
            self.config.proxy_rotation_interval = max(1, self.config.proxy_rotation_interval - 1)
        elif risk_score < 0.3:
            # Low risk - can optimize for speed
            self.config.behavior_complexity = "medium"
            self.config.proxy_rotation_interval = min(10, self.config.proxy_rotation_interval + 1)
        
        # Update behavior simulator
        self.behavior_simulator = MarkovBehaviorSimulator(
            complexity=self.config.behavior_complexity
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        stats = {
            "session_id": self.session_id,
            "request_count": self.request_count,
            "current_fingerprint": bool(self.current_fingerprint),
            "current_proxy": self.current_proxy.host if self.current_proxy else None,
            "risk_score": asyncio.run(self.get_detection_risk_score()) if self.behavior_history else 0.0,
            "config": {
                "behavior_complexity": self.config.behavior_complexity,
                "proxy_rotation_strategy": self.config.proxy_rotation_strategy,
                "fingerprint_rotation_interval": self.config.fingerprint_rotation_interval,
                "proxy_rotation_interval": self.config.proxy_rotation_interval,
            }
        }
        
        if self.proxy_rotator:
            stats["proxy_stats"] = self.proxy_rotator.get_stats()
        
        return stats


# Factory functions for easy instantiation
def create_stealth_session(
    proxies: Optional[List[Dict[str, Any]]] = None,
    captcha_service: Optional[str] = None,
    captcha_api_key: Optional[str] = None,
    behavior_complexity: str = "medium",
    target_country: Optional[str] = None,
    **kwargs
) -> StealthSession:
    """
    Factory function to create a stealth session with common configurations
    
    Args:
        proxies: List of proxy configurations
        captcha_service: CAPTCHA service name
        captcha_api_key: CAPTCHA service API key
        behavior_complexity: "low", "medium", or "high"
        target_country: Preferred country for proxies
        **kwargs: Additional configuration options
        
    Returns:
        Configured StealthSession instance
    """
    config = AntiDetectionConfig(
        proxies=proxies or [],
        captcha_service=captcha_service,
        captcha_api_key=captcha_api_key,
        behavior_complexity=behavior_complexity,
        target_country=target_country,
        **kwargs
    )
    
    return StealthSession(config)


def load_proxies_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load proxy configurations from JSON file
    
    Args:
        filepath: Path to JSON file with proxy configurations
        
    Returns:
        List of proxy configurations
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle different file formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "proxies" in data:
            return data["proxies"]
        else:
            raise ValueError("Invalid proxy file format")
    except Exception as e:
        print(f"Error loading proxies from {filepath}: {e}")
        return []


# Export main classes and functions
__all__ = [
    "StealthSession",
    "FingerprintGeneratorGAN",
    "MarkovBehaviorSimulator",
    "ProxyRotator",
    "CaptchaSolver",
    "ProxyNode",
    "ProxyType",
    "CaptchaService",
    "create_stealth_session",
    "load_proxies_from_file",
]