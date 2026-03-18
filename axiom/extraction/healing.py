"""
Adaptive Extraction Engine — AI-powered selector generation, schema inference, and self-healing selectors.
"""

import logging
import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pickle
from collections import defaultdict
import difflib

import numpy as np
from bs4 import BeautifulSoup, Tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from axiom.core.ai import DOMTransformerModel
from axiom.core.custom_types import Selector, ExtractionResult
from axiom.core.utils._utils import load_json, save_json, generate_hash

logger = logging.getLogger(__name__)


class SelectorType(Enum):
    """Types of selectors the engine can generate."""
    CSS = "css"
    XPATH = "xpath"
    REGEX = "regex"
    COMBINED = "combined"


@dataclass
class ExtractionExample:
    """Example for few-shot learning."""
    html_snippet: str
    target_data: Dict[str, Any]
    page_url: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def fingerprint(self) -> str:
        """Generate unique fingerprint for this example."""
        content = f"{self.html_snippet}:{json.dumps(self.target_data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class HealingResult:
    """Result of selector healing operation."""
    original_selector: str
    healed_selector: str
    confidence: float
    selector_type: SelectorType
    similarity_score: float
    healed: bool = False
    explanation: str = ""


@dataclass 
class SchemaField:
    """Inferred schema field."""
    name: str
    data_type: str
    selector: str
    selector_type: SelectorType
    confidence: float
    examples: List[str] = field(default_factory=list)
    required: bool = True


@dataclass
class ExtractionSchema:
    """Inferred extraction schema."""
    fields: Dict[str, SchemaField]
    page_type: str
    confidence: float
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "fields": {k: asdict(v) for k, v in self.fields.items()},
            "page_type": self.page_type,
            "confidence": self.confidence,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtractionSchema':
        """Create from dictionary."""
        fields = {}
        for k, v in data.get("fields", {}).items():
            v["selector_type"] = SelectorType(v["selector_type"])
            fields[k] = SchemaField(**v)
        return cls(
            fields=fields,
            page_type=data.get("page_type", "unknown"),
            confidence=data.get("confidence", 0.0),
            version=data.get("version", "1.0")
        )


class DOMStructureEncoder:
    """Encodes DOM structure into feature vectors for similarity comparison."""
    
    def __init__(self, model_name: str = "dom-transformer-base"):
        """Initialize encoder with pre-trained model."""
        self.model = DOMTransformerModel(model_name)
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=1000
        )
        self._fitted = False
        
    def encode_dom(self, dom: Union[str, Tag]) -> np.ndarray:
        """Encode DOM structure into feature vector."""
        if isinstance(dom, str):
            soup = BeautifulSoup(dom, 'html.parser')
            dom = soup.find() or soup
            
        # Extract structural features
        features = self._extract_structural_features(dom)
        
        # Use transformer model for deep encoding
        dom_str = str(dom)
        transformer_embedding = self.model.encode(dom_str)
        
        # Combine with TF-IDF features
        if not self._fitted:
            self.vectorizer.fit([dom_str])
            self._fitted = True
            
        tfidf_features = self.vectorizer.transform([dom_str]).toarray()[0]
        
        # Concatenate all features
        combined = np.concatenate([
            transformer_embedding,
            tfidf_features,
            np.array(features)
        ])
        
        return combined / np.linalg.norm(combined)
    
    def _extract_structural_features(self, dom: Tag) -> List[float]:
        """Extract handcrafted structural features from DOM."""
        features = []
        
        # Tag distribution
        tag_counts = defaultdict(int)
        for tag in dom.find_all(True):
            tag_counts[tag.name] += 1
            
        # Common HTML tags
        common_tags = ['div', 'span', 'a', 'p', 'img', 'ul', 'li', 'table', 'tr', 'td']
        for tag in common_tags:
            features.append(tag_counts.get(tag, 0) / max(sum(tag_counts.values()), 1))
            
        # Depth statistics
        depths = []
        def get_depth(element, current=0):
            depths.append(current)
            for child in element.children:
                if hasattr(child, 'children'):
                    get_depth(child, current + 1)
                    
        get_depth(dom)
        features.extend([
            np.mean(depths) if depths else 0,
            np.std(depths) if depths else 0,
            max(depths) if depths else 0
        ])
        
        # Attribute statistics
        attr_counts = defaultdict(int)
        for tag in dom.find_all(True):
            for attr in tag.attrs:
                attr_counts[attr] += 1
                
        common_attrs = ['class', 'id', 'href', 'src', 'style', 'data-', 'aria-']
        for attr in common_attrs:
            count = sum(v for k, v in attr_counts.items() if attr in k)
            features.append(count / max(sum(attr_counts.values()), 1))
            
        return features


class SelectorGenerator:
    """Generates selectors from examples using multiple strategies."""
    
    def __init__(self, encoder: DOMStructureEncoder):
        self.encoder = encoder
        self.selector_cache: Dict[str, List[Selector]] = {}
        
    def generate_selectors(self, 
                          html: str, 
                          target_element: Tag,
                          selector_types: List[SelectorType] = None) -> List[Selector]:
        """Generate multiple selectors for target element."""
        if selector_types is None:
            selector_types = [SelectorType.CSS, SelectorType.XPATH]
            
        cache_key = generate_hash(f"{html}:{str(target_element)}")
        if cache_key in self.selector_cache:
            return self.selector_cache[cache_key]
            
        selectors = []
        
        for selector_type in selector_types:
            try:
                if selector_type == SelectorType.CSS:
                    selector = self._generate_css_selector(target_element)
                elif selector_type == SelectorType.XPATH:
                    selector = self._generate_xpath_selector(target_element)
                elif selector_type == SelectorType.REGEX:
                    selector = self._generate_regex_selector(html, target_element)
                else:
                    continue
                    
                if selector:
                    selectors.append(Selector(
                        value=selector,
                        selector_type=selector_type,
                        confidence=self._calculate_selector_confidence(selector, target_element, html)
                    ))
            except Exception as e:
                logger.warning(f"Failed to generate {selector_type} selector: {e}")
                
        # Sort by confidence
        selectors.sort(key=lambda x: x.confidence, reverse=True)
        self.selector_cache[cache_key] = selectors
        return selectors
    
    def _generate_css_selector(self, element: Tag) -> Optional[str]:
        """Generate robust CSS selector for element."""
        if not element or not hasattr(element, 'name'):
            return None
            
        # Strategy 1: Use ID if available and unique
        if element.get('id'):
            id_selector = f"#{element['id']}"
            if self._is_selector_unique(id_selector, element):
                return id_selector
                
        # Strategy 2: Build path with classes and attributes
        path = []
        current = element
        
        while current and current.name:
            selector_part = current.name
            
            # Add classes if they seem stable
            if current.get('class'):
                stable_classes = [c for c in current['class'] 
                                 if not re.match(r'^(active|selected|hover|focus)', c)]
                if stable_classes:
                    selector_part += '.' + '.'.join(stable_classes[:2])
                    
            # Add data attributes if present
            for attr, value in current.attrs.items():
                if attr.startswith('data-') and not re.search(r'\d+', str(value)):
                    selector_part += f'[{attr}="{value}"]'
                    break
                    
            path.insert(0, selector_part)
            
            # Check if current path is unique
            full_selector = ' > '.join(path)
            if self._is_selector_unique(full_selector, element):
                return full_selector
                
            current = current.parent
            
        # Fallback: Use nth-child if needed
        if element.parent:
            siblings = [child for child in element.parent.children 
                       if hasattr(child, 'name') and child.name == element.name]
            if len(siblings) > 1:
                index = siblings.index(element) + 1
                return f"{element.name}:nth-of-type({index})"
                
        return element.name
    
    def _generate_xpath_selector(self, element: Tag) -> Optional[str]:
        """Generate XPath selector for element."""
        components = []
        current = element
        
        while current and current.name:
            # Build position predicate
            siblings = []
            if current.parent:
                siblings = [s for s in current.parent.children 
                           if hasattr(s, 'name') and s.name == current.name]
            
            if len(siblings) > 1:
                index = siblings.index(current) + 1
                predicate = f"[{index}]"
            else:
                predicate = ""
                
            # Add attribute conditions
            attrs = []
            if current.get('id'):
                attrs.append(f"@id='{current['id']}'")
            elif current.get('class'):
                attrs.append(f"contains(@class, '{current['class'][0]}')")
                
            if attrs:
                predicate = f"[{' and '.join(attrs)}]" if not predicate else f"{predicate}[{' and '.join(attrs)}]"
                
            components.insert(0, f"{current.name}{predicate}")
            current = current.parent
            
        return '/' + '/'.join(components) if components else None
    
    def _generate_regex_selector(self, html: str, element: Tag) -> Optional[str]:
        """Generate regex pattern to extract element content."""
        element_html = str(element)
        
        # Escape regex special characters
        escaped = re.escape(element_html)
        
        # Find unique surrounding context
        start = html.find(element_html)
        if start == -1:
            return None
            
        # Get context before and after
        context_before = html[max(0, start-50):start]
        context_after = html[start+len(element_html):start+len(element_html)+50]
        
        # Build pattern with context
        pattern_parts = []
        
        if context_before:
            # Use last 20 chars of context before
            before_pattern = re.escape(context_before[-20:])
            pattern_parts.append(f"{before_pattern}\\s*")
            
        pattern_parts.append(f"({escaped})")
        
        if context_after:
            # Use first 20 chars of context after
            after_pattern = re.escape(context_after[:20])
            pattern_parts.append(f"\\s*{after_pattern}")
            
        return ''.join(pattern_parts)
    
    def _is_selector_unique(self, selector: str, element: Tag, html: str = None) -> bool:
        """Check if selector uniquely identifies element."""
        try:
            if html:
                soup = BeautifulSoup(html, 'html.parser')
            else:
                soup = element.find_parent() if element.parent else element
                
            if selector.startswith('/') or selector.startswith('//'):
                # XPath
                import lxml.html as lh
                from lxml import etree
                tree = etree.HTML(str(soup))
                results = tree.xpath(selector)
                return len(results) == 1
            else:
                # CSS
                results = soup.select(selector)
                return len(results) == 1
        except:
            return False
    
    def _calculate_selector_confidence(self, selector: str, element: Tag, html: str) -> float:
        """Calculate confidence score for selector."""
        confidence = 0.5  # Base confidence
        
        # Uniqueness bonus
        if self._is_selector_unique(selector, element, html):
            confidence += 0.3
            
        # Stability indicators
        if '#' in selector:  # ID selector
            confidence += 0.2
        if 'data-' in selector:  # Data attribute
            confidence += 0.1
        if ':nth' in selector:  # Positional selector
            confidence -= 0.1
            
        # Length penalty (shorter is usually better)
        if len(selector) < 50:
            confidence += 0.1
        elif len(selector) > 100:
            confidence -= 0.1
            
        return min(max(confidence, 0.0), 1.0)


class AdaptiveExtractor:
    """Main adaptive extraction engine with self-healing capabilities."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 cache_dir: str = ".axiom_cache"):
        """Initialize adaptive extractor.
        
        Args:
            model_path: Path to pre-trained transformer model
            cache_dir: Directory for caching learned patterns
        """
        self.encoder = DOMStructureEncoder(model_path or "dom-transformer-base")
        self.selector_generator = SelectorGenerator(self.encoder)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Learning state
        self.learned_patterns: Dict[str, List[ExtractionExample]] = defaultdict(list)
        self.selector_history: Dict[str, List[Dict]] = defaultdict(list)
        self.schema_cache: Dict[str, ExtractionSchema] = {}
        
        # Load existing knowledge
        self._load_knowledge_base()
        
    def learn_from_examples(self, 
                           examples: List[ExtractionExample],
                           page_type: str = "auto") -> ExtractionSchema:
        """Learn extraction patterns from examples.
        
        Args:
            examples: List of extraction examples
            page_type: Type of page (auto-inferred if "auto")
            
        Returns:
            Inferred extraction schema
        """
        if not examples:
            raise ValueError("No examples provided for learning")
            
        # Group examples by similarity
        grouped_examples = self._group_similar_examples(examples)
        
        # Infer schema from examples
        schema = self._infer_schema(grouped_examples, page_type)
        
        # Store patterns for future healing
        pattern_key = self._generate_pattern_key(schema)
        self.learned_patterns[pattern_key].extend(examples)
        
        # Cache schema
        self.schema_cache[pattern_key] = schema
        self._save_knowledge_base()
        
        logger.info(f"Learned schema with {len(schema.fields)} fields from {len(examples)} examples")
        return schema
    
    def extract_with_healing(self,
                            html: str,
                            schema: ExtractionSchema,
                            previous_selectors: Optional[Dict[str, str]] = None,
                            url: str = "") -> ExtractionResult:
        """Extract data with self-healing selectors.
        
        Args:
            html: HTML content to extract from
            schema: Extraction schema
            previous_selectors: Previously working selectors (for healing)
            url: Source URL for context
            
        Returns:
            Extraction results with healing information
        """
        results = {}
        healing_results = {}
        
        for field_name, field_schema in schema.fields.items():
            try:
                # Try original selector first
                if previous_selectors and field_name in previous_selectors:
                    selector = previous_selectors[field_name]
                    value = self._evaluate_selector(html, selector, field_schema.selector_type)
                    
                    if value is not None:
                        results[field_name] = value
                        continue
                        
                    # Selector broken, attempt healing
                    healed = self._heal_selector(
                        html=html,
                        broken_selector=selector,
                        field_schema=field_schema,
                        url=url
                    )
                    
                    if healed.healed:
                        results[field_name] = self._evaluate_selector(
                            html, healed.healed_selector, healed.selector_type
                        )
                        healing_results[field_name] = healed
                        logger.info(f"Healed selector for {field_name}: {healed.explanation}")
                    else:
                        results[field_name] = None
                        logger.warning(f"Failed to heal selector for {field_name}")
                else:
                    # Generate new selector
                    selectors = self._generate_selectors_for_field(html, field_schema)
                    if selectors:
                        best_selector = selectors[0]
                        results[field_name] = self._evaluate_selector(
                            html, best_selector.value, best_selector.selector_type
                        )
                        
                        # Store for future healing
                        self.selector_history[field_name].append({
                            "selector": best_selector.value,
                            "type": best_selector.selector_type.value,
                            "timestamp": time.time(),
                            "url": url
                        })
                    else:
                        results[field_name] = None
                        
            except Exception as e:
                logger.error(f"Error extracting {field_name}: {e}")
                results[field_name] = None
                
        return ExtractionResult(
            data=results,
            healing_results=healing_results,
            schema_version=schema.version,
            confidence=self._calculate_extraction_confidence(results, schema)
        )
    
    def _heal_selector(self,
                      html: str,
                      broken_selector: str,
                      field_schema: SchemaField,
                      url: str = "") -> HealingResult:
        """Attempt to heal a broken selector."""
        # Find similar elements in current DOM
        candidate_elements = self._find_candidate_elements(html, field_schema)
        
        if not candidate_elements:
            return HealingResult(
                original_selector=broken_selector,
                healed_selector="",
                confidence=0.0,
                selector_type=field_schema.selector_type,
                similarity_score=0.0,
                explanation="No candidate elements found"
            )
        
        # Encode broken selector's expected element (from examples)
        example_embeddings = []
        for example_text in field_schema.examples[:3]:  # Use first 3 examples
            soup = BeautifulSoup(example_text, 'html.parser')
            if soup.find():
                example_embeddings.append(self.encoder.encode_dom(soup.find()))
        
        if not example_embeddings:
            return HealingResult(
                original_selector=broken_selector,
                healed_selector="",
                confidence=0.0,
                selector_type=field_schema.selector_type,
                similarity_score=0.0,
                explanation="No example embeddings available"
            )
        
        # Average example embedding
        target_embedding = np.mean(example_embeddings, axis=0)
        
        # Find most similar element
        best_similarity = -1
        best_element = None
        
        for element in candidate_elements:
            element_embedding = self.encoder.encode_dom(element)
            similarity = cosine_similarity(
                target_embedding.reshape(1, -1),
                element_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_element = element
        
        if best_element is None or best_similarity < 0.3:  # Threshold
            return HealingResult(
                original_selector=broken_selector,
                healed_selector="",
                confidence=best_similarity,
                selector_type=field_schema.selector_type,
                similarity_score=best_similarity,
                explanation=f"Best similarity {best_similarity:.2f} below threshold"
            )
        
        # Generate new selector for best element
        new_selectors = self.selector_generator.generate_selectors(
            html, best_element, [field_schema.selector_type]
        )
        
        if not new_selectors:
            return HealingResult(
                original_selector=broken_selector,
                healed_selector="",
                confidence=best_similarity,
                selector_type=field_schema.selector_type,
                similarity_score=best_similarity,
                explanation="Failed to generate new selector"
            )
        
        healed_selector = new_selectors[0]
        
        # Calculate confidence
        confidence = (best_similarity + healed_selector.confidence) / 2
        
        return HealingResult(
            original_selector=broken_selector,
            healed_selector=healed_selector.value,
            confidence=confidence,
            selector_type=healed_selector.selector_type,
            similarity_score=best_similarity,
            healed=True,
            explanation=f"Healed with similarity {best_similarity:.2f}, confidence {confidence:.2f}"
        )
    
    def _find_candidate_elements(self, html: str, field_schema: SchemaField) -> List[Tag]:
        """Find candidate elements for a field."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Strategy 1: Use similar tags from examples
        candidate_tags = set()
        for example in field_schema.examples[:5]:
            example_soup = BeautifulSoup(example, 'html.parser')
            if example_soup.find():
                candidate_tags.add(example_soup.find().name)
        
        # Strategy 2: Look for elements with similar text patterns
        candidates = []
        
        # Search by tag names
        for tag in candidate_tags:
            candidates.extend(soup.find_all(tag))
        
        # If no candidates from tags, search broadly
        if not candidates:
            # Look for elements with text content
            for element in soup.find_all(True):
                if element.string and len(element.string.strip()) > 0:
                    candidates.append(element)
        
        # Filter by data type hints
        if field_schema.data_type == "link":
            candidates = [c for c in candidates if c.name == 'a' or c.find('a')]
        elif field_schema.data_type == "image":
            candidates = [c for c in candidates if c.name == 'img' or c.find('img')]
        elif field_schema.data_type == "number":
            # Look for elements with numeric content
            numeric_candidates = []
            for c in candidates:
                text = c.get_text()
                if re.search(r'\d+', text):
                    numeric_candidates.append(c)
            if numeric_candidates:
                candidates = numeric_candidates
        
        return candidates
    
    def _generate_selectors_for_field(self, 
                                     html: str, 
                                     field_schema: SchemaField) -> List[Selector]:
        """Generate selectors for a field based on schema."""
        candidates = self._find_candidate_elements(html, field_schema)
        
        if not candidates:
            return []
        
        # Generate selectors for each candidate
        all_selectors = []
        for element in candidates[:10]:  # Limit to top 10 candidates
            selectors = self.selector_generator.generate_selectors(
                html, element, [field_schema.selector_type]
            )
            all_selectors.extend(selectors)
        
        # Sort by confidence and return best
        all_selectors.sort(key=lambda x: x.confidence, reverse=True)
        return all_selectors[:5]  # Return top 5
    
    def _evaluate_selector(self, 
                          html: str, 
                          selector: str, 
                          selector_type: SelectorType) -> Optional[Any]:
        """Evaluate selector on HTML and extract content."""
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            if selector_type == SelectorType.CSS:
                elements = soup.select(selector)
            elif selector_type == SelectorType.XPATH:
                import lxml.html as lh
                from lxml import etree
                tree = etree.HTML(html)
                elements = tree.xpath(selector)
            elif selector_type == SelectorType.REGEX:
                match = re.search(selector, html, re.DOTALL)
                return match.group(1) if match else None
            else:
                return None
            
            if not elements:
                return None
            
            # Extract appropriate content based on element type
            element = elements[0] if isinstance(elements, list) else elements
            
            if hasattr(element, 'get'):
                # BeautifulSoup element
                if element.name == 'a' and element.get('href'):
                    return element['href']
                elif element.name == 'img' and element.get('src'):
                    return element['src']
                else:
                    return element.get_text(strip=True)
            else:
                # lxml element
                return element.text_content().strip() if hasattr(element, 'text_content') else str(element)
                
        except Exception as e:
            logger.debug(f"Selector evaluation failed: {e}")
            return None
    
    def _group_similar_examples(self, 
                               examples: List[ExtractionExample]) -> Dict[str, List[ExtractionExample]]:
        """Group similar examples together."""
        if len(examples) <= 1:
            return {"group_0": examples}
        
        # Extract text content for comparison
        texts = []
        for example in examples:
            soup = BeautifulSoup(example.html_snippet, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            texts.append(text)
        
        # Vectorize texts
        vectorizer = TfidfVectorizer(max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Cluster similar examples
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Simple clustering: group examples with similarity > 0.7
            groups = defaultdict(list)
            visited = set()
            
            for i in range(len(examples)):
                if i in visited:
                    continue
                    
                group_key = f"group_{len(groups)}"
                groups[group_key].append(examples[i])
                visited.add(i)
                
                for j in range(i + 1, len(examples)):
                    if j not in visited and similarity_matrix[i, j] > 0.7:
                        groups[group_key].append(examples[j])
                        visited.add(j)
            
            return dict(groups)
        except:
            # Fallback: return all in one group
            return {"group_0": examples}
    
    def _infer_schema(self, 
                     grouped_examples: Dict[str, List[ExtractionExample]],
                     page_type: str) -> ExtractionSchema:
        """Infer extraction schema from grouped examples."""
        # Analyze all target data keys
        all_keys = set()
        key_types = defaultdict(list)
        
        for group_examples in grouped_examples.values():
            for example in group_examples:
                all_keys.update(example.target_data.keys())
                for key, value in example.target_data.items():
                    key_types[key].append(type(value).__name__)
        
        # Determine most common type for each key
        field_schemas = {}
        
        for key in all_keys:
            # Determine data type
            type_counts = defaultdict(int)
            for type_name in key_types.get(key, []):
                type_counts[type_name] += 1
            
            if type_counts:
                data_type = max(type_counts.items(), key=lambda x: x[1])[0]
            else:
                data_type = "str"
            
            # Collect examples for this key
            examples = []
            for group_examples in grouped_examples.values():
                for example in group_examples:
                    if key in example.target_data:
                        examples.append(example.html_snippet)
            
            # Generate initial selector (will be refined during extraction)
            field_schemas[key] = SchemaField(
                name=key,
                data_type=data_type,
                selector="",  # Will be generated during extraction
                selector_type=SelectorType.CSS,  # Default
                confidence=0.5,
                examples=examples[:5],  # Store up to 5 examples
                required=True
            )
        
        # Determine page type if auto
        if page_type == "auto":
            page_type = self._infer_page_type(grouped_examples)
        
        return ExtractionSchema(
            fields=field_schemas,
            page_type=page_type,
            confidence=0.7  # Initial confidence
        )
    
    def _infer_page_type(self, grouped_examples: Dict[str, List[ExtractionExample]]) -> str:
        """Infer page type from examples."""
        # Analyze HTML structure patterns
        tag_patterns = defaultdict(int)
        
        for group_examples in grouped_examples.values():
            for example in group_examples:
                soup = BeautifulSoup(example.html_snippet, 'html.parser')
                for tag in soup.find_all(True):
                    tag_patterns[tag.name] += 1
        
        # Common page type patterns
        if tag_patterns.get('article', 0) > 2:
            return "article"
        elif tag_patterns.get('product', 0) > 2 or 'price' in str(tag_patterns):
            return "product"
        elif tag_patterns.get('table', 0) > 2:
            return "table"
        elif tag_patterns.get('form', 0) > 1:
            return "form"
        elif tag_patterns.get('li', 0) > 5:
            return "list"
        else:
            return "generic"
    
    def _generate_pattern_key(self, schema: ExtractionSchema) -> str:
        """Generate unique key for schema pattern."""
        field_names = sorted(schema.fields.keys())
        key_string = f"{schema.page_type}:{','.join(field_names)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _calculate_extraction_confidence(self, 
                                        results: Dict[str, Any], 
                                        schema: ExtractionSchema) -> float:
        """Calculate overall extraction confidence."""
        if not results:
            return 0.0
        
        filled_fields = sum(1 for v in results.values() if v is not None)
        total_fields = len(schema.fields)
        
        if total_fields == 0:
            return 0.0
        
        return filled_fields / total_fields
    
    def _load_knowledge_base(self):
        """Load learned patterns from cache."""
        try:
            patterns_file = self.cache_dir / "learned_patterns.pkl"
            if patterns_file.exists():
                with open(patterns_file, 'rb') as f:
                    self.learned_patterns = pickle.load(f)
                    
            schemas_file = self.cache_dir / "schema_cache.json"
            if schemas_file.exists():
                schemas_data = load_json(schemas_file)
                for key, schema_dict in schemas_data.items():
                    self.schema_cache[key] = ExtractionSchema.from_dict(schema_dict)
                    
            logger.info(f"Loaded {len(self.learned_patterns)} patterns and {len(self.schema_cache)} schemas")
        except Exception as e:
            logger.warning(f"Failed to load knowledge base: {e}")
    
    def _save_knowledge_base(self):
        """Save learned patterns to cache."""
        try:
            patterns_file = self.cache_dir / "learned_patterns.pkl"
            with open(patterns_file, 'wb') as f:
                pickle.dump(self.learned_patterns, f)
                
            schemas_file = self.cache_dir / "schema_cache.json"
            schemas_data = {k: v.to_dict() for k, v in self.schema_cache.items()}
            save_json(schemas_data, schemas_file)
            
            logger.info("Saved knowledge base to cache")
        except Exception as e:
            logger.warning(f"Failed to save knowledge base: {e}")
    
    def export_knowledge(self, export_path: str):
        """Export learned knowledge to file."""
        knowledge = {
            "learned_patterns": {
                k: [asdict(e) for e in examples]
                for k, examples in self.learned_patterns.items()
            },
            "schema_cache": {k: v.to_dict() for k, v in self.schema_cache.items()},
            "selector_history": dict(self.selector_history)
        }
        
        save_json(knowledge, export_path)
        logger.info(f"Exported knowledge to {export_path}")
    
    def import_knowledge(self, import_path: str):
        """Import knowledge from file."""
        knowledge = load_json(import_path)
        
        # Import patterns
        for key, examples_data in knowledge.get("learned_patterns", {}).items():
            examples = [ExtractionExample(**e) for e in examples_data]
            self.learned_patterns[key].extend(examples)
        
        # Import schemas
        for key, schema_dict in knowledge.get("schema_cache", {}).items():
            self.schema_cache[key] = ExtractionSchema.from_dict(schema_dict)
        
        # Import selector history
        self.selector_history.update(knowledge.get("selector_history", {}))
        
        self._save_knowledge_base()
        logger.info(f"Imported knowledge from {import_path}")


# Convenience functions for integration with existing axiom
def create_adaptive_extractor(**kwargs) -> AdaptiveExtractor:
    """Factory function to create adaptive extractor."""
    return AdaptiveExtractor(**kwargs)


def heal_extraction(html: str,
                   broken_selectors: Dict[str, str],
                   examples: List[ExtractionExample],
                   **kwargs) -> ExtractionResult:
    """Convenience function for self-healing extraction."""
    extractor = AdaptiveExtractor(**kwargs)
    schema = extractor.learn_from_examples(examples)
    return extractor.extract_with_healing(html, schema, broken_selectors)


# Import time at the end to avoid circular imports
import time