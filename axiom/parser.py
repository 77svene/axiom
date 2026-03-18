from pathlib import Path
from inspect import signature
from urllib.parse import urljoin
from difflib import SequenceMatcher
from re import Pattern as re_Pattern
import json
import hashlib
import re
import pickle
from collections import defaultdict
from datetime import datetime
from typing import get_type_hints
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from lxml.html import HtmlElement, HTMLParser
from cssselect import SelectorError, SelectorSyntaxError, parse as split_selectors
from lxml.etree import (
    XPath,
    tostring,
    fromstring,
    XPathError,
    XPathEvalError,
    _ElementUnicodeResult,
)

from axiom.core._types import (
    Any,
    Set,
    Dict,
    cast,
    List,
    Tuple,
    Union,
    TypeVar,
    Pattern,
    Callable,
    Literal,
    Optional,
    Iterable,
    overload,
    Generator,
    SupportsIndex,
    TYPE_CHECKING,
)
from axiom.core.custom_types import AttributesHandler, TextHandler, TextHandlers
from axiom.core.mixins import SelectorsGeneration
from axiom.core.storage import (
    SQLiteStorageSystem,
    StorageSystemMixin,
    _StorageTools,
)
from axiom.core.translator import css_to_xpath as _css_to_xpath
from axiom.core.utils import clean_spaces, flatten, html_forbidden, log

# New imports for data quality pipeline
try:
    from pydantic import BaseModel, ValidationError, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # Fallback for type hints

__DEFAULT_DB_FILE__ = str(Path(__file__).parent / "elements_storage.db")
# Attributes that are Python reserved words and can't be used directly
# Ex: find_all('a', class="blah") -> find_all('a', class_="blah")
# https://www.w3schools.com/python/python_ref_keywords.asp
_whitelisted = {
    "class_": "class",
    "for_": "for",
}
_T = TypeVar("_T")
# Pre-compiled selectors for efficiency
_find_all_elements = XPath(".//*")
_find_all_elements_with_spaces = XPath(
    ".//*[normalize-space(text())]"
)  # This selector gets all elements with text content
_find_all_text_nodes = XPath(".//text()")


class DOMTransformerModel:
    """Transformer-based model for DOM structure understanding and selector generation."""
    
    __slots__ = (
        '_model_path',
        '_vectorizer',
        '_selector_patterns',
        '_dom_embeddings',
        '_example_cache',
        '_similarity_threshold',
        '_max_selector_length',
    )
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        similarity_threshold: float = 0.75,
        max_selector_length: int = 100,
    ):
        """
        Initialize the DOM transformer model.
        
        :param model_path: Path to pre-trained model (if available)
        :param similarity_threshold: Threshold for selector similarity matching
        :param max_selector_length: Maximum length for generated selectors
        """
        self._model_path = model_path
        self._vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=10000
        )
        self._selector_patterns = defaultdict(list)
        self._dom_embeddings = {}
        self._example_cache = {}
        self._similarity_threshold = similarity_threshold
        self._max_selector_length = max_selector_length
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def extract_dom_features(self, element: HtmlElement) -> str:
        """Extract features from DOM element for vectorization."""
        features = []
        
        # Tag name
        features.append(f"tag:{element.tag}")
        
        # Attributes
        for attr, value in element.attrib.items():
            features.append(f"attr:{attr}={value}")
        
        # Text content (normalized)
        text = element.text_content().strip()
        if text:
            features.append(f"text:{text[:50]}")  # Limit text length
        
        # Position in tree
        parent = element.getparent()
        if parent is not None:
            features.append(f"parent:{parent.tag}")
            siblings = list(parent)
            if len(siblings) > 1:
                index = siblings.index(element)
                features.append(f"position:{index}/{len(siblings)}")
        
        # Class patterns
        classes = element.get('class', '')
        if classes:
            for cls in classes.split():
                features.append(f"class:{cls}")
        
        return ' '.join(features)
    
    def generate_selector_from_examples(
        self,
        target_elements: List[HtmlElement],
        examples: List[Tuple[HtmlElement, str]],
        context: Optional[HtmlElement] = None
    ) -> List[str]:
        """
        Generate selectors using few-shot learning from examples.
        
        :param target_elements: Elements to generate selectors for
        :param examples: List of (example_element, expected_selector) pairs
        :param context: Optional context element (parent) for scoping
        :return: List of generated selectors
        """
        if not examples:
            return self._generate_fallback_selectors(target_elements)
        
        # Extract features from examples
        example_features = []
        example_selectors = []
        
        for elem, selector in examples:
            features = self.extract_dom_features(elem)
            example_features.append(features)
            example_selectors.append(selector)
        
        # Fit vectorizer if not already fitted
        if not hasattr(self._vectorizer, 'vocabulary_'):
            self._vectorizer.fit(example_features)
        
        # Transform examples
        example_vectors = self._vectorizer.transform(example_features)
        
        # Generate selectors for target elements
        generated_selectors = []
        
        for target_elem in target_elements:
            target_features = self.extract_dom_features(target_elem)
            target_vector = self._vectorizer.transform([target_features])
            
            # Find most similar example
            similarities = cosine_similarity(target_vector, example_vectors)[0]
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= self._similarity_threshold:
                # Use the selector from most similar example
                base_selector = example_selectors[best_idx]
                
                # Adapt selector to current element
                adapted_selector = self._adapt_selector(
                    base_selector, 
                    target_elem, 
                    examples[best_idx][0]
                )
                
                if adapted_selector:
                    generated_selectors.append(adapted_selector)
                    continue
            
            # Fallback: generate new selector
            generated_selectors.append(
                self._generate_element_selector(target_elem, context)
            )
        
        return generated_selectors
    
    def _adapt_selector(
        self,
        base_selector: str,
        target_elem: HtmlElement,
        example_elem: HtmlElement
    ) -> Optional[str]:
        """Adapt a selector from example to target element."""
        try:
            # Parse the base selector
            parts = base_selector.split(' > ')
            adapted_parts = []
            
            for part in parts:
                # Extract tag, classes, and attributes
                tag_match = re.match(r'^(\w+)', part)
                if not tag_match:
                    adapted_parts.append(part)
                    continue
                
                tag = tag_match.group(1)
                
                # Check if target has same tag
                if target_elem.tag != tag:
                    # Try to find equivalent tag
                    if self._tags_compatible(tag, target_elem.tag):
                        tag = target_elem.tag
                    else:
                        return None
                
                # Extract and adapt classes
                class_matches = re.findall(r'\.([\w-]+)', part)
                adapted_classes = []
                
                for cls in class_matches:
                    if cls in target_elem.get('class', '').split():
                        adapted_classes.append(cls)
                    else:
                        # Find similar class
                        similar = self._find_similar_class(cls, target_elem)
                        if similar:
                            adapted_classes.append(similar)
                
                # Extract and adapt attributes
                attr_matches = re.findall(r'\[@?(\w+)=([\'"])(.*?)\2\]', part)
                adapted_attrs = []
                
                for attr, quote, value in attr_matches:
                    target_value = target_elem.get(attr)
                    if target_value == value:
                        adapted_attrs.append(f'[@{attr}={quote}{value}{quote}]')
                    elif target_value:
                        # Use target's value
                        adapted_attrs.append(f'[@{attr}={quote}{target_value}{quote}]')
                
                # Rebuild selector part
                new_part = tag
                if adapted_classes:
                    new_part += '.' + '.'.join(adapted_classes)
                if adapted_attrs:
                    new_part += ''.join(adapted_attrs)
                
                adapted_parts.append(new_part)
            
            return ' > '.join(adapted_parts)
            
        except Exception as e:
            log(f"Selector adaptation failed: {e}", level="debug")
            return None
    
    def _tags_compatible(self, tag1: str, tag2: str) -> bool:
        """Check if two tags are semantically compatible."""
        compatible_groups = [
            {'div', 'section', 'article', 'main', 'aside'},
            {'span', 'a', 'em', 'strong', 'b', 'i'},
            {'ul', 'ol', 'nav'},
            {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'},
        ]
        
        for group in compatible_groups:
            if tag1 in group and tag2 in group:
                return True
        
        return False
    
    def _find_similar_class(self, class_name: str, element: HtmlElement) -> Optional[str]:
        """Find similar class name in element's classes."""
        element_classes = element.get('class', '').split()
        
        # Exact match
        if class_name in element_classes:
            return class_name
        
        # Prefix match
        for cls in element_classes:
            if cls.startswith(class_name.split('-')[0]):
                return cls
        
        # Substring match
        for cls in element_classes:
            if class_name in cls or cls in class_name:
                return cls
        
        return None
    
    def _generate_element_selector(
        self,
        element: HtmlElement,
        context: Optional[HtmlElement] = None
    ) -> str:
        """Generate a selector for a single element."""
        selector_parts = []
        
        # Add tag
        selector_parts.append(element.tag)
        
        # Add ID if available
        elem_id = element.get('id')
        if elem_id:
            selector_parts.append(f'#{elem_id}')
            return ' > '.join(selector_parts)  # ID is unique, return early
        
        # Add classes
        classes = element.get('class', '').split()
        if classes:
            # Use most specific classes (longest or with most semantic meaning)
            classes.sort(key=len, reverse=True)
            selector_parts.append('.' + '.'.join(classes[:2]))  # Limit to 2 classes
        
        # Add position if needed for disambiguation
        if context is not None:
            siblings = [e for e in context if e.tag == element.tag]
            if len(siblings) > 1:
                index = siblings.index(element)
                selector_parts.append(f':nth-of-type({index + 1})')
        
        # Add data attributes if available
        for attr, value in element.attrib.items():
            if attr.startswith('data-') and value:
                selector_parts.append(f'[@{attr}="{value}"]')
                break  # Use first data attribute
        
        return ' > '.join(selector_parts)
    
    def _generate_fallback_selectors(self, elements: List[HtmlElement]) -> List[str]:
        """Generate fallback selectors when no examples available."""
        selectors = []
        
        for elem in elements:
            # Try to generate unique selector
            selector = self._generate_element_selector(elem)
            
            # Verify selector is unique in document
            try:
                if len(elem.getroottree().xpath(f'//{selector}')) == 1:
                    selectors.append(selector)
                    continue
            except:
                pass
            
            # Fallback to more specific selector
            parent = elem.getparent()
            if parent is not None:
                parent_selector = self._generate_element_selector(parent)
                child_selector = self._generate_element_selector(elem)
                selectors.append(f"{parent_selector} > {child_selector}")
            else:
                selectors.append(selector)
        
        return selectors
    
    def save_model(self, path: str):
        """Save model to disk."""
        model_data = {
            'vectorizer': self._vectorizer,
            'selector_patterns': dict(self._selector_patterns),
            'dom_embeddings': self._dom_embeddings,
            'example_cache': self._example_cache,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self._vectorizer = model_data['vectorizer']
        self._selector_patterns = defaultdict(list, model_data['selector_patterns'])
        self._dom_embeddings = model_data['dom_embeddings']
        self._example_cache = model_data['example_cache']


class SchemaInferenceEngine:
    """Automatic schema inference from extracted data."""
    
    __slots__ = (
        '_min_confidence',
        '_max_depth',
        '_type_patterns',
        '_field_clusters',
    )
    
    def __init__(
        self,
        min_confidence: float = 0.8,
        max_depth: int = 5,
    ):
        """
        Initialize schema inference engine.
        
        :param min_confidence: Minimum confidence for type inference
        :param max_depth: Maximum depth for nested schema inference
        """
        self._min_confidence = min_confidence
        self._max_depth = max_depth
        self._type_patterns = self._initialize_type_patterns()
        self._field_clusters = {}
    
    def _initialize_type_patterns(self) -> Dict[str, List[Pattern]]:
        """Initialize regex patterns for type detection."""
        return {
            'email': [re.compile(r'^[^@]+@[^@]+\.[^@]+$')],
            'url': [re.compile(r'^https?://[^\s]+$')],
            'phone': [re.compile(r'^\+?[\d\s\-()]{7,}$')],
            'date': [
                re.compile(r'^\d{4}-\d{2}-\d{2}$'),
                re.compile(r'^\d{2}/\d{2}/\d{4}$'),
                re.compile(r'^\d{2}\.\d{2}\.\d{4}$'),
            ],
            'integer': [re.compile(r'^-?\d+$')],
            'float': [re.compile(r'^-?\d+\.\d+$')],
            'boolean': [re.compile(r'^(true|false|yes|no|1|0)$', re.IGNORECASE)],
        }
    
    def infer_schema(
        self,
        data_samples: List[Dict[str, Any]],
        field_descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Infer JSON Schema from data samples.
        
        :param data_samples: List of data dictionaries
        :param field_descriptions: Optional descriptions for fields
        :return: Inferred JSON Schema
        """
        if not data_samples:
            return {"type": "object", "properties": {}}
        
        # Analyze all fields across samples
        field_analysis = {}
        
        for sample in data_samples:
            for field, value in sample.items():
                if field not in field_analysis:
                    field_analysis[field] = {
                        'values': [],
                        'types': set(),
                        'patterns': [],
                        'nullable': False,
                    }
                
                analysis = field_analysis[field]
                analysis['values'].append(value)
                
                if value is None:
                    analysis['nullable'] = True
                    continue
                
                # Detect type
                value_type = self._infer_type(value)
                analysis['types'].add(value_type)
                
                # Detect patterns
                if isinstance(value, str):
                    patterns = self._detect_patterns(value)
                    analysis['patterns'].extend(patterns)
        
        # Build schema
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        
        for field, analysis in field_analysis.items():
            field_schema = self._build_field_schema(analysis, field_descriptions)
            schema["properties"][field] = field_schema
            
            # Field is required if it's never null and appears in all samples
            if not analysis['nullable'] and len(analysis['values']) == len(data_samples):
                schema["required"].append(field)
        
        return schema
    
    def _infer_type(self, value: Any) -> str:
        """Infer JSON Schema type from value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"  # Default fallback
    
    def _detect_patterns(self, text: str) -> List[str]:
        """Detect patterns in string value."""
        patterns = []
        
        for pattern_name, pattern_list in self._type_patterns.items():
            for pattern in pattern_list:
                if pattern.match(text):
                    patterns.append(pattern_name)
                    break
        
        return patterns
    
    def _build_field_schema(
        self,
        analysis: Dict[str, Any],
        field_descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Build schema for a single field."""
        field_schema = {}
        
        # Determine primary type
        types = list(analysis['types'])
        if len(types) == 1:
            field_schema["type"] = types[0]
        elif len(types) == 2 and "null" in types:
            # Nullable type
            non_null_type = [t for t in types if t != "null"][0]
            field_schema["type"] = non_null_type
        else:
            # Multiple types - use anyOf
            field_schema["anyOf"] = [{"type": t} for t in types]
        
        # Add pattern if detected
        if analysis['patterns']:
            # Use most common pattern
            pattern_counts = {}
            for pattern in analysis['patterns']:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            most_common = max(pattern_counts.items(), key=lambda x: x[1])[0]
            
            if most_common == 'email':
                field_schema["format"] = "email"
            elif most_common == 'url':
                field_schema["format"] = "uri"
            elif most_common == 'date':
                field_schema["format"] = "date"
            elif most_common == 'phone':
                field_schema["pattern"] = r'^\+?[\d\s\-()]{7,}$'
        
        # Add description if available
        if field_descriptions and field in field_descriptions:
            field_schema["description"] = field_descriptions[field]
        
        # Add examples if available
        if analysis['values']:
            unique_values = list(set(str(v) for v in analysis['values'][:5]))
            if unique_values:
                field_schema["examples"] = unique_values
        
        return field_schema
    
    def cluster_fields(
        self,
        data_samples: List[Dict[str, Any]],
        min_samples: int = 2
    ) -> Dict[str, List[str]]:
        """
        Cluster similar fields together.
        
        :param data_samples: List of data dictionaries
        :param min_samples: Minimum samples for clustering
        :return: Dictionary of cluster names to field lists
        """
        if len(data_samples) < min_samples:
            return {}
        
        # Extract all field names and their value patterns
        field_vectors = {}
        
        for i, sample in enumerate(data_samples):
            for field, value in sample.items():
                if field not in field_vectors:
                    field_vectors[field] = []
                
                # Create feature vector for field
                features = self._extract_field_features(value)
                field_vectors[field].append(features)
        
        # Average feature vectors
        field_features = {}
        for field, vectors in field_vectors.items():
            if vectors:
                # Pad vectors to same length
                max_len = max(len(v) for v in vectors)
                padded_vectors = []
                for v in vectors:
                    if len(v) < max_len:
                        v.extend([0] * (max_len - len(v)))
                    padded_vectors.append(v)
                
                field_features[field] = np.mean(padded_vectors, axis=0)
        
        # Cluster fields
        if not field_features:
            return {}
        
        fields = list(field_features.keys())
        feature_matrix = np.array([field_features[f] for f in fields])
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.5, min_samples=min_samples).fit(feature_matrix)
        
        # Group fields by cluster
        clusters = defaultdict(list)
        for field, label in zip(fields, clustering.labels_):
            if label != -1:  # -1 means noise
                clusters[f"cluster_{label}"].append(field)
        
        self._field_clusters = dict(clusters)
        return self._field_clusters
    
    def _extract_field_features(self, value: Any) -> List[float]:
        """Extract feature vector from field value."""
        features = []
        
        if value is None:
            features.extend([0, 0, 0, 0, 0])
        elif isinstance(value, str):
            features.append(1)  # is_string
            features.append(len(value))  # length
            features.append(1 if '@' in value else 0)  # contains @
            features.append(1 if 'http' in value.lower() else 0)  # contains http
            features.append(len(value.split()))  # word count
        elif isinstance(value, (int, float)):
            features.extend([0, 1, 0, 0, 0])
            features.append(float(value))  # numeric value
        elif isinstance(value, bool):
            features.extend([0, 0, 1, 0, 0])
        elif isinstance(value, list):
            features.extend([0, 0, 0, 1, len(value)])
        elif isinstance(value, dict):
            features.extend([0, 0, 0, 0, len(value)])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return features


class SelfHealingSelector:
    """Self-healing selector that adapts to site changes."""
    
    __slots__ = (
        '_original_selector',
        '_element_signature',
        '_backup_selectors',
        '_adaptation_history',
        '_max_adaptations',
        '_signature_cache',
    )
    
    def __init__(
        self,
        original_selector: str,
        element: Optional[HtmlElement] = None,
        max_adaptations: int = 5,
    ):
        """
        Initialize self-healing selector.
        
        :param original_selector: Original CSS/XPath selector
        :param element: Optional element for signature generation
        :param max_adaptations: Maximum number of adaptations to store
        """
        self._original_selector = original_selector
        self._element_signature = self._generate_signature(element) if element else None
        self._backup_selectors = []
        self._adaptation_history = []
        self._max_adaptations = max_adaptations
        self._signature_cache = {}
    
    def _generate_signature(self, element: HtmlElement) -> Dict[str, Any]:
        """Generate signature for element identification."""
        if element is None:
            return {}
        
        signature = {
            'tag': element.tag,
            'text_hash': self._hash_text(element.text_content()),
            'attributes': {},
            'parent_tag': None,
            'sibling_count': 0,
            'position': 0,
        }
        
        # Attributes (excluding dynamic ones)
        for attr, value in element.attrib.items():
            if not self._is_dynamic_attribute(attr, value):
                signature['attributes'][attr] = value
        
        # Parent and position
        parent = element.getparent()
        if parent is not None:
            signature['parent_tag'] = parent.tag
            siblings = [e for e in parent if e.tag == element.tag]
            signature['sibling_count'] = len(siblings)
            if element in siblings:
                signature['position'] = siblings.index(element)
        
        return signature
    
    def _hash_text(self, text: str) -> str:
        """Create hash of text content."""
        if not text:
            return ""
        
        # Normalize text
        normalized = ' '.join(text.split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]
    
    def _is_dynamic_attribute(self, attr: str, value: str) -> bool:
        """Check if attribute value is likely dynamic."""
        dynamic_patterns = [
            r'^[a-f0-9]{8,}$',  # Hex IDs
            r'^\d{10,}$',  # Long numbers (timestamps)
            r'^[A-Za-z0-9+/]{20,}={0,2}$',  # Base64
            r'^session',  # Session-related
            r'^temp',  # Temporary
            r'^generated',  # Generated
        ]
        
        for pattern in dynamic_patterns:
            if re.match(pattern, value, re.IGNORECASE):
                return True
        
        return False
    
    def find_element(
        self,
        document: HtmlElement,
        transformer_model: Optional[DOMTransformerModel] = None
    ) -> Optional[HtmlElement]:
        """
        Find element using self-healing mechanism.
        
        :param document: Document to search in
        :param transformer_model: Optional transformer model for AI-powered repair
        :return: Found element or None
        """
        # Try original selector first
        try:
            elements = document.cssselect(self._original_selector)
            if elements:
                return elements[0]
        except (SelectorError, SelectorSyntaxError):
            pass
        
        # Try backup selectors
        for backup in self._backup_selectors:
            try:
                elements = document.cssselect(backup)
                if elements:
                    return elements[0]
            except (SelectorError, SelectorSyntaxError):
                continue
        
        # Try signature-based matching
        if self._element_signature:
            element = self._find_by_signature(document)
            if element:
                # Generate new selector for found element
                new_selector = self._generate_selector_for_element(element)
                self.add_backup_selector(new_selector)
                return element
        
        # Try transformer model if available
        if transformer_model:
            element = self._find_with_transformer(document, transformer_model)
            if element:
                new_selector = self._generate_selector_for_element(element)
                self.add_backup_selector(new_selector)
                return element
        
        return None
    
    def _find_by_signature(self, document: HtmlElement) -> Optional[HtmlElement]:
        """Find element by signature matching."""
        candidates = []
        
        # Find all elements with matching tag
        for element in document.iter():
            if element.tag != self._element_signature.get('tag'):
                continue
            
            # Check text hash
            element_text_hash = self._hash_text(element.text_content())
            if element_text_hash != self._element_signature.get('text_hash'):
                continue
            
            # Check attributes
            sig_attrs = self._element_signature.get('attributes', {})
            elem_attrs = element.attrib
            
            # Calculate attribute similarity
            common_attrs = set(sig_attrs.keys()) & set(elem_attrs.keys())
            if not common_attrs:
                continue
            
            matching_attrs = 0
            for attr in common_attrs:
                if sig_attrs[attr] == elem_attrs[attr]:
                    matching_attrs += 1
            
            similarity = matching_attrs / len(sig_attrs) if sig_attrs else 0
            
            if similarity >= 0.5:  # At least 50% attributes match
                candidates.append((element, similarity))
        
        if not candidates:
            return None
        
        # Return best match
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _find_with_transformer(
        self,
        document: HtmlElement,
        transformer_model: DOMTransformerModel
    ) -> Optional[HtmlElement]:
        """Find element using transformer model."""
        # Extract features from original element signature
        if not self._element_signature:
            return None
        
        # Create a pseudo-element from signature
        # This is a simplified approach - in practice, you'd need the original element
        try:
            # Try to find by original selector in any available document
            # This would require storing the original element's features
            pass
        except:
            pass
        
        return None
    
    def _generate_selector_for_element(self, element: HtmlElement) -> str:
        """Generate a robust selector for an element."""
        selector_parts = []
        
        # Start with tag
        selector_parts.append(element.tag)
        
        # Add ID if available and stable
        elem_id = element.get('id')
        if elem_id and not self._is_dynamic_attribute('id', elem_id):
            return f'#{elem_id}'  # ID selector is most reliable
        
        # Add classes (prioritize stable-looking ones)
        classes = element.get('class', '').split()
        stable_classes = [c for c in classes if not self._looks_dynamic(c)]
        
        if stable_classes:
            # Use up to 2 stable classes
            selector_parts.append('.' + '.'.join(stable_classes[:2]))
        
        # Add data attributes if available
        for attr, value in element.attrib.items():
            if attr.startswith('data-') and value:
                if not self._is_dynamic_attribute(attr, value):
                    selector_parts.append(f'[{attr}="{value}"]')
                    break
        
        # Add position relative to parent if needed
        parent = element.getparent()
        if parent is not None:
            siblings = [e for e in parent if e.tag == element.tag]
            if len(siblings) > 1:
                index = siblings.index(element)
                selector_parts.append(f':nth-of-type({index + 1})')
        
        return ' '.join(selector_parts)
    
    def _looks_dynamic(self, value: str) -> bool:
        """Check if a value looks dynamically generated."""
        if len(value) > 20:  # Long values are often dynamic
            return True
        
        # Check for random-looking patterns
        if re.search(r'[A-Za-z0-9]{8,}', value):
            # Contains 8+ alphanumeric chars
            consonants = sum(1 for c in value.lower() if c in 'bcdfghjklmnpqrstvwxyz')
            if consonants / len(value) > 0.6:  # High consonant ratio
                return True
        
        return False
    
    def add_backup_selector(self, selector: str):
        """Add a backup selector."""
        if selector not in self._backup_selectors:
            self._backup_selectors.insert(0, selector)  # Prioritize new selectors
            
            # Limit number of backups
            if len(self._backup_selectors) > self._max_adaptations:
                self._backup_selectors.pop()
    
    def record_adaptation(self, old_selector: str, new_selector: str, success: bool):
        """Record selector adaptation for learning."""
        adaptation = {
            'timestamp': datetime.now().isoformat(),
            'old_selector': old_selector,
            'new_selector': new_selector,
            'success': success,
            'element_signature': self._element_signature,
        }
        
        self._adaptation_history.append(adaptation)
        
        # Limit history size
        if len(self._adaptation_history) > 100:
            self._adaptation_history.pop(0)
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptations."""
        if not self._adaptation_history:
            return {"total": 0, "success_rate": 0.0}
        
        successful = sum(1 for a in self._adaptation_history if a['success'])
        total = len(self._adaptation_history)
        
        return {
            "total": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "last_adaptation": self._adaptation_history[-1] if self._adaptation_history else None,
        }


class AdaptiveExtractionEngine:
    """Main engine combining all adaptive extraction capabilities."""
    
    __slots__ = (
        '_transformer_model',
        '_schema_engine',
        '_healing_selectors',
        '_example_store',
        '_schema_cache',
        '_learning_enabled',
        '_storage',
    )
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        storage_path: Optional[str] = None,
        learning_enabled: bool = True,
    ):
        """
        Initialize adaptive extraction engine.
        
        :param model_path: Path to transformer model
        :param storage_path: Path to storage database
        :param learning_enabled: Whether to enable learning from examples
        """
        self._transformer_model = DOMTransformerModel(model_path)
        self._schema_engine = SchemaInferenceEngine()
        self._healing_selectors = {}
        self._example_store = defaultdict(list)
        self._schema_cache = {}
        self._learning_enabled = learning_enabled
        
        # Initialize storage
        if storage_path:
            self._storage = SQLiteStorageSystem(storage_path)
        else:
            self._storage = SQLiteStorageSystem(__DEFAULT_DB_FILE__)
        
        # Load existing data
        self._load_examples()
    
    def learn_from_examples(
        self,
        url_pattern: str,
        examples: List[Tuple[str, str, Dict[str, Any]]]
    ):
        """
        Learn from user-provided examples.
        
        :param url_pattern: URL pattern for these examples
        :param examples: List of (html_snippet, selector, extracted_data) tuples
        """
        if not self._learning_enabled:
            return
        
        for html_snippet, selector, data in examples:
            try:
                # Parse HTML snippet
                element = fromstring(html_snippet)
                
                # Store example
                self._example_store[url_pattern].append({
                    'element': element,
                    'selector': selector,
                    'data': data,
                    'timestamp': datetime.now().isoformat(),
                })
                
                # Update transformer model
                self._update_model_with_example(element, selector)
                
                # Infer schema from data
                self._update_schema(url_pattern, data)
                
            except Exception as e:
                log(f"Failed to learn from example: {e}", level="warning")
        
        # Save examples to storage
        self._save_examples(url_pattern)
    
    def _update_model_with_example(self, element: HtmlElement, selector: str):
        """Update transformer model with new example."""
        # Extract features
        features = self._transformer_model.extract_dom_features(element)
        
        # Store in model's cache
        if not hasattr(self._transformer_model, '_example_cache'):
            self._transformer_model._example_cache = {}
        
        cache_key = hashlib.md5(features.encode()).hexdigest()
        self._transformer_model._example_cache[cache_key] = {
            'features': features,
            'selector': selector,
            'element': element,
        }
    
    def _update_schema(self, url_pattern: str, data: Dict[str, Any]):
        """Update schema inference with new data."""
        if url_pattern not in self._schema_cache:
            self._schema_cache[url_pattern] = []
        
        self._schema_cache[url_pattern].append(data)
        
        # Limit cache size
        if len(self._schema_cache[url_pattern]) > 100:
            self._schema_cache[url_pattern].pop(0)
    
    def generate_selectors(
        self,
        document: HtmlElement,
        target_description: str,
        url_pattern: Optional[str] = None,
        num_selectors: int = 3
    ) -> List[str]:
        """
        Generate selectors for target elements.
        
        :param document: Document to extract from
        :param target_description: Description of what to extract
        :param url_pattern: Optional URL pattern for context
        :param num_selectors: Number of selectors to generate
        :return: List of generated selectors
        """
        # Find candidate elements
        candidates = self._find_candidate_elements(document, target_description)
        
        if not candidates:
            return []
        
        # Get examples for this URL pattern
        examples = []
        if url_pattern and url_pattern in self._example_store:
            for example in self._example_store[url_pattern]:
                examples.append((example['element'], example['selector']))
        
        # Generate selectors using transformer model
        selectors = self._transformer_model.generate_selector_from_examples(
            candidates[:10],  # Limit to top 10 candidates
            examples,
            document
        )
        
        # Deduplicate and rank selectors
        unique_selectors = []
        seen = set()
        
        for selector in selectors:
            if selector not in seen:
                seen.add(selector)
                
                # Test selector
                try:
                    elements = document.cssselect(selector)
                    if elements:
                        unique_selectors.append(selector)
                except:
                    pass
        
        return unique_selectors[:num_selectors]
    
    def _find_candidate_elements(
        self,
        document: HtmlElement,
        description: str
    ) -> List[HtmlElement]:
        """Find candidate elements based on description."""
        candidates = []
        
        # Simple keyword matching (in practice, you'd use NLP)
        description_lower = description.lower()
        
        for element in document.iter():
            # Check text content
            text = element.text_content().lower()
            if any(word in text for word in description_lower.split()):
                candidates.append(element)
                continue
            
            # Check attributes
            for attr, value in element.attrib.items():
                if any(word in value.lower() for word in description_lower.split()):
                    candidates.append(element)
                    break
        
        # Score and sort candidates
        scored_candidates = []
        for elem in candidates:
            score = self._score_candidate(elem, description)
            scored_candidates.append((elem, score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [elem for elem, _ in scored_candidates]
    
    def _score_candidate(self, element: HtmlElement, description: str) -> float:
        """Score a candidate element."""
        score = 0.0
        
        # Text relevance
        text = element.text_content().lower()
        description_words = description.lower().split()
        
        for word in description_words:
            if word in text:
                score += 1.0
        
        # Attribute relevance
        for attr, value in element.attrib.items():
            for word in description_words:
                if word in value.lower():
                    score += 0.5
        
        # Element specificity (prefer less generic elements)
        if element.tag not in ['div', 'span', 'p']:
            score += 0.5
        
        # Has meaningful content
        if text.strip():
            score += 0.5
        
        return score
    
    def infer_schema(
        self,
        url_pattern: str,
        data_samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Infer schema for extracted data.
        
        :param url_pattern: URL pattern for context
        :param data_samples: Optional data samples (uses cached if not provided)
        :return: Inferred JSON Schema
        """
        if data_samples is None:
            data_samples = self._schema_cache.get(url_pattern, [])
        
        if not data_samples:
            return {"type": "object", "properties": {}}
        
        # Infer schema
        schema = self._schema_engine.infer_schema(data_samples)
        
        # Add metadata
        schema["$metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "url_pattern": url_pattern,
            "sample_count": len(data_samples),
            "engine": "axiom Adaptive Extraction Engine",
        }
        
        return schema
    
    def create_healing_selector(
        self,
        selector: str,
        element: Optional[HtmlElement] = None
    ) -> SelfHealingSelector:
        """
        Create a self-healing selector.
        
        :param selector: Original selector
        :param element: Optional element for signature
        :return: Self-healing selector instance
        """
        healing_selector = SelfHealingSelector(selector, element)
        
        # Store for later reference
        selector_id = hashlib.md5(selector.encode()).hexdigest()
        self._healing_selectors[selector_id] = healing_selector
        
        return healing_selector
    
    def extract_with_healing(
        self,
        document: HtmlElement,
        selector: str,
        element_signature: Optional[Dict[str, Any]] = None
    ) -> Optional[HtmlElement]:
        """
        Extract element using self-healing mechanism.
        
        :param document: Document to extract from
        :param selector: CSS/XPath selector
        :param element_signature: Optional element signature for matching
        :return: Extracted element or None
        """
        # Try to find existing healing selector
        selector_id = hashlib.md5(selector.encode()).hexdigest()
        
        if selector_id in self._healing_selectors:
            healing_selector = self._healing_selectors[selector_id]
        else:
            healing_selector = self.create_healing_selector(selector)
        
        # Find element
        element = healing_selector.find_element(document, self._transformer_model)
        
        if element:
            # Record successful adaptation
            healing_selector.record_adaptation(
                selector,
                healing_selector._backup_selectors[0] if healing_selector._backup_selectors else selector,
                True
            )
        else:
            # Record failed adaptation
            healing_selector.record_adaptation(selector, "", False)
        
        return element
    
    def _load_examples(self):
        """Load examples from storage."""
        try:
            # In a real implementation, you'd load from the database
            pass
        except Exception as e:
            log(f"Failed to load examples: {e}", level="debug")
    
    def _save_examples(self, url_pattern: str):
        """Save examples to storage."""
        try:
            # In a real implementation, you'd save to the database
            pass
        except Exception as e:
            log(f"Failed to save examples: {e}", level="debug")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "learning_enabled": self._learning_enabled,
            "url_patterns_learned": len(self._example_store),
            "total_examples": sum(len(examples) for examples in self._example_store.values()),
            "schemas_inferred": len(self._schema_cache),
            "healing_selectors": len(self._healing_selectors),
        }
        
        # Add healing selector stats
        if self._healing_selectors:
            total_adaptations = 0
            successful_adaptations = 0
            
            for selector in self._healing_selectors.values():
                selector_stats = selector.get_adaptation_stats()
                total_adaptations += selector_stats['total']
                successful_adaptations += selector_stats['successful']
            
            stats["total_adaptations"] = total_adaptations
            stats["successful_adaptations"] = successful_adaptations
            stats["adaptation_success_rate"] = (
                successful_adaptations / total_adaptations 
                if total_adaptations > 0 else 0.0
            )
        
        return stats
    
    def save_model(self, path: str):
        """Save transformer model to disk."""
        self._transformer_model.save_model(path)
    
    def load_model(self, path: str):
        """Load transformer model from disk."""
        self._transformer_model.load_model(path)


class DataQualityPipeline:
    """Comprehensive data quality and validation pipeline for extracted data."""
    
    __slots__ = (
        '_schema',
        '_enrichment_sources',
        '_quality_threshold',
        '_duplicate_threshold',
        '_simhash_size',
        '_minhash_permutations',
        '_cache',
    )
    
    def __init__(
        self,
        schema: Optional[Union[Dict, BaseModel]] = None,
        enrichment_sources: Optional[List[Callable]] = None,
        quality_threshold: float = 0.7,
        duplicate_threshold: float = 0.85,
        simhash_size: int = 64,
        minhash_permutations: int = 128,
    ):
        """
        Initialize the data quality pipeline.
        
        :param schema: JSON Schema dict or Pydantic model for validation
        :param enrichment_sources: List of enrichment functions
        :param quality_threshold: Minimum quality score (0-1) to accept data
        :param duplicate_threshold: Similarity threshold for duplicate detection
        :param simhash_size: Bit size for SimHash fingerprints
        :param minhash_permutations: Number of permutations for MinHash
        """
        self._schema = schema
        self._enrichment_sources = enrichment_sources or []
        self._quality_threshold = quality_threshold
        self._duplicate_threshold = duplicate_threshold
        self._simhash_size = simhash_size
        self._minhash_permutations = minhash_permutations
        self._cache = {}
    
    def validate_with_json_schema(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate data against JSON Schema.
        
        :param data: Data dictionary to validate
        :return: Tuple of (is_valid, list_of_errors)
        """
        if not self._schema or not isinstance(self._schema, dict):
            return True, []
        
        errors = []
        
        # Basic JSON Schema validation
        try:
            # Check required fields
            required_fields = self._schema.get('required', [])
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            # Check properties
            properties = self._schema.get('properties', {})
            for field, field_schema in properties.items():
                if field in data:
                    value = data[field]
                    field_type = field_schema.get('type')
                    
                    # Type validation
                    if field_type and not self._validate_type(value, field_type):
                        errors.append(f"Field '{field}' has invalid type. Expected {field_type}")
                    
                    # Pattern validation
                    if 'pattern' in field_schema and isinstance(value, str):
                        if not re.match(field_schema['pattern'], value):
                            errors.append(f"Field '{field}' doesn't match pattern")
                    
                    # Enum validation
                    if 'enum' in field_schema:
                        if value not in field_schema['enum']:
                            errors.append(f"Field '{field}' not in allowed values")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    def validate_with_pydantic(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        validate data using Pydantic model.
        
        :param data: Data dictionary to validate
        :return: Tuple of (is_valid, list_of_errors)
        """
        if not PYDANTIC_AVAILABLE or not self._schema or not isinstance(self._schema, type) or not issubclass(self._schema, BaseModel):
            return True, []
        
        try:
            # Create instance of Pydantic model
            instance = self._schema(**data)
            return True, []
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field = ' -> '.join(str(loc) for loc in error['loc'])
                errors.append(f"{field}: {error['msg']}")
            return False, errors
        except Exception as e:
            return False, [f"Pydantic validation error: {str(e)}"]
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value against JSON Schema type."""
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None),
        }
        
        if expected_type in type_map:
            return isinstance(value, type_map[expected_type])
        return True
    
    def compute_simhash(self, text: str) -> int:
        """
        Compute SimHash fingerprint for text.
        
        :param text: Text to hash
        :return: SimHash fingerprint as integer
        """
        if not text:
            return 0
        
        # Tokenize and weight tokens
        tokens = self._tokenize_text(text)
        if not tokens:
            return 0
        
        # Initialize vector
        vector = [0] * self._simhash_size
        
        for token, weight in tokens.items():
            # Hash token
            token_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
            
            # Update vector
            for i in range(self._simhash_size):
                bit = (token_hash >> i) & 1
                if bit:
                    vector[i] += weight
                else:
                    vector[i] -= weight
        
        # Compute fingerprint
        fingerprint = 0
        for i in range(self._simhash_size):
            if vector[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def _tokenize_text(self, text: str) -> Dict[str, int]:
        """Tokenize text and count frequencies."""
        # Simple tokenization (in practice, you'd use better NLP)
        tokens = {}
        
        # Normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Split into words
        words = text.split()
        
        for word in words:
            if len(word) > 2:  # Skip very short words
                tokens[word] = tokens.get(word, 0) + 1
        
        return tokens
    
    def compute_minhash(self, text: str) -> List[int]:
        """
        Compute MinHash signature for text.
        
        :param text: Text to hash
        :return: MinHash signature as list of integers
        """
        if not text:
            return [0] * self._minhash_permutations
        
        # Tokenize
        tokens = set(self._tokenize_text(text).keys())
        if not tokens:
            return [0] * self._minhash_permutations
        
        # Generate hash functions (simulated)
        signature = []
        
        for i in range(self._minhash_permutations):
            min_hash = float('inf')
            
            for token in tokens:
                # Simulate different hash functions
                hash_val = int(hashlib.md5(f"{token}_{i}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, hash_val)
            
            signature.append(min_hash)
        
        return signature
    
    def detect_duplicates(
        self,
        data_items: List[Dict[str, Any]],
        text_fields: Optional[List[str]] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Detect duplicate items in dataset.
        
        :param data_items: List of data dictionaries
        :param text_fields: Fields to use for text similarity
        :return: List of (index1, index2, similarity) tuples
        """
        if not data_items:
            return []
        
        # Extract text for comparison
        texts = []
        for item in data_items:
            if text_fields:
                text = ' '.join(str(item.get(field, '')) for field in text_fields)
            else:
                text = ' '.join(str(v) for v in item.values())
            texts.append(text)
        
        # Compute SimHash fingerprints
        fingerprints = [self.compute_simhash(text) for text in texts]
        
        # Find similar pairs
        duplicates = []
        
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                # Calculate similarity
                similarity = self._simhash_similarity(fingerprints[i], fingerprints[j])
                
                if similarity >= self._duplicate_threshold:
                    duplicates.append((i, j, similarity))
        
        return duplicates
    
    def _simhash_similarity(self, hash1: int, hash2: int) -> float:
        """Calculate similarity between two SimHash fingerprints."""
        # Hamming distance
        xor = hash1 ^ hash2
        distance = bin(xor).count('1')
        
        # Convert to similarity (0-1)
        similarity = 1.0 - (distance / self._simhash_size)
        return similarity
    
    def enrich_data(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enrich data using configured enrichment sources.
        
        :param data: Data dictionary to enrich
        :param context: Optional context for enrichment
        :return: Enriched data dictionary
        """
        enriched_data = data.copy()
        
        for enrichment_func in self._enrichment_sources:
            try:
                enrichment = enrichment_func(data, context)
                if isinstance(enrichment, dict):
                    enriched_data.update(enrichment)
            except Exception as e:
                log(f"Enrichment failed: {e}", level="warning")
        
        return enriched_data
    
    def calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate quality score for data.
        
        :param data: Data dictionary to score
        :return: Quality score (0-1)
        """
        if not data:
            return 0.0
        
        scores = []
        
        # Completeness score
        total_fields = len(data)
        non_empty_fields = sum(1 for v in data.values() if v is not None and v != '')
        completeness = non_empty_fields / total_fields if total_fields > 0 else 0
        scores.append(completeness)
        
        # Consistency score (check for mixed types in similar fields)
        consistency = self._calculate_consistency_score(data)
        scores.append(consistency)
        
        # Validity score (basic type checking)
        validity = self._calculate_validity_score(data)
        scores.append(validity)
        
        # Average score
        return sum(scores) / len(scores)
    
    def _calculate_consistency_score(self, data: Dict[str, Any]) -> float:
        """Calculate consistency score for data."""
        # Group fields by naming pattern
        field_groups = defaultdict(list)
        
        for key in data.keys():
            # Extract base name (remove numbers, suffixes)
            base_name = re.sub(r'\d+$', '', key)
            base_name = re.sub(r'_(id|key|code)$', '', base_name)
            field_groups[base_name].append(key)
        
        # Check type consistency within groups
        consistent_groups = 0
        total_groups = 0
        
        for base_name, fields in field_groups.items():
            if len(fields) > 1:
                total_groups += 1
                types = [type(data[field]) for field in fields if field in data]
                
                if len(set(types)) == 1:  # All same type
                    consistent_groups += 1
        
        return consistent_groups / total_groups if total_groups > 0 else 1.0
    
    def _calculate_validity_score(self, data: Dict[str, Any]) -> float:
        """Calculate validity score for data."""
        valid_fields = 0
        total_fields = 0
        
        for key, value in data.items():
            total_fields += 1
            
            # Basic validity checks
            if value is None:
                valid_fields += 1  # Null is valid
            elif isinstance(value, (str, int, float, bool, list, dict)):
                valid_fields += 1
            else:
                # Invalid type
                pass
        
        return valid_fields / total_fields if total_fields > 0 else 1.0
    
    def process_pipeline(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run complete data quality pipeline.
        
        :param data: Input data dictionary
        :param context: Optional context for processing
        :return: Processed data with quality metadata
        """
        result = {
            'original_data': data.copy(),
            'processed_data': {},
            'quality_score': 0.0,
            'validation_errors': [],
            'is_valid': False,
            'enrichments_applied': [],
            'duplicates_detected': [],
        }
        
        # Step 1: Validate
        if self._schema:
            if isinstance(self._schema, dict):
                is_valid, errors = self.validate_with_json_schema(data)
            else:
                is_valid, errors = self.validate_with_pydantic(data)
            
            result['is_valid'] = is_valid
            result['validation_errors'] = errors
        
        # Step 2: Enrich
        enriched_data = self.enrich_data(data, context)
        result['processed_data'] = enriched_data
        
        # Track enrichments
        new_fields = set(enriched_data.keys()) - set(data.keys())
        result['enrichments_applied'] = list(new_fields)
        
        # Step 3: Calculate quality score
        quality_score = self.calculate_quality_score(enriched_data)
        result['quality_score'] = quality_score
        
        # Step 4: Check against threshold
        if quality_score < self._quality_threshold:
            result['quality_warning'] = f"Quality score {quality_score:.2f} below threshold {self._quality_threshold}"
        
        return result