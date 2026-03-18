"""
axiom/extraction/learner.py
Adaptive Extraction Engine — AI-powered selector generation that learns from examples, automatic schema inference, and self-healing selectors that adapt to site changes
"""

import json
import hashlib
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from axiom.core.custom_types import (
    Selector, Element, Elements, Attributes, 
    DOMNode, DOMTree, ExtractionResult, SchemaDefinition
)
from axiom.core.ai import TransformerModel, EmbeddingModel
from axiom.core.utils._utils import (
    normalize_whitespace, extract_text_nodes, 
    get_element_signature, calculate_similarity
)
from axiom.core.storage import CacheStorage
from axiom.core.mixins import LoggerMixin, SerializableMixin


@dataclass
class ExtractionExample:
    """Example for few-shot learning of extraction patterns"""
    url: str
    dom_hash: str
    selectors: List[Selector]
    extracted_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    success_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def example_id(self) -> str:
        """Generate unique ID for this example"""
        content = f"{self.url}:{self.dom_hash}:{json.dumps(self.selectors)}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass 
class SelectorCandidate:
    """Candidate selector with confidence and metadata"""
    selector: Selector
    confidence: float
    specificity: int  # CSS specificity score
    stability_score: float  # How stable this selector is likely to be
    coverage: float  # What percentage of target elements it matches
    examples_used: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'selector': self.selector,
            'confidence': self.confidence,
            'specificity': self.specificity,
            'stability_score': self.stability_score,
            'coverage': self.coverage,
            'examples_used': self.examples_used,
            'generated_at': self.generated_at.isoformat()
        }


@dataclass
class SchemaField:
    """Field definition in inferred schema"""
    name: str
    field_type: str  # 'text', 'number', 'date', 'url', 'image', 'list', 'object'
    selectors: List[Selector]
    required: bool = True
    multiple: bool = False
    validation_regex: Optional[str] = None
    transformation: Optional[str] = None  # 'strip', 'lower', 'date_parse', etc.
    children: Optional[List['SchemaField']] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'name': self.name,
            'field_type': self.field_type,
            'selectors': self.selectors,
            'required': self.required,
            'multiple': self.multiple,
        }
        if self.validation_regex:
            result['validation_regex'] = self.validation_regex
        if self.transformation:
            result['transformation'] = self.transformation
        if self.children:
            result['children'] = [c.to_dict() for c in self.children]
        return result


class DOMAnalyzer(LoggerMixin):
    """Analyzes DOM structure for pattern recognition"""
    
    def __init__(self, transformer_model: Optional[TransformerModel] = None):
        self.transformer = transformer_model or TransformerModel()
        self.embedding_model = EmbeddingModel()
        self._node_cache = {}
        
    def extract_features(self, element: Element) -> Dict[str, Any]:
        """Extract features from DOM element for ML processing"""
        features = {
            'tag': element.tag,
            'attributes': dict(element.attributes),
            'text_length': len(element.text or ''),
            'child_count': len(element.children),
            'depth': self._calculate_depth(element),
            'siblings_count': len(element.siblings) if hasattr(element, 'siblings') else 0,
            'class_count': len(element.get('class', '').split()) if element.get('class') else 0,
            'id_present': 'id' in element.attributes,
            'signature': get_element_signature(element),
        }
        
        # Get contextual features
        if element.parent:
            features['parent_tag'] = element.parent.tag
            features['parent_class'] = element.parent.get('class', '')
        
        return features
    
    def _calculate_depth(self, element: Element, current: int = 0) -> int:
        """Calculate DOM depth of element"""
        if not element.parent:
            return current
        return self._calculate_depth(element.parent, current + 1)
    
    def get_element_embedding(self, element: Element) -> np.ndarray:
        """Get vector embedding for element"""
        cache_key = f"emb:{get_element_signature(element)}"
        if cache_key in self._node_cache:
            return self._node_cache[cache_key]
        
        features = self.extract_features(element)
        text_content = f"{element.tag} {' '.join(f'{k}={v}' for k, v in element.attributes.items())} {element.text or ''}"
        embedding = self.embedding_model.encode(text_content)
        
        self._node_cache[cache_key] = embedding
        return embedding
    
    def find_similar_elements(self, target: Element, dom: DOMTree, threshold: float = 0.7) -> List[Tuple[Element, float]]:
        """Find elements similar to target in DOM"""
        target_embedding = self.get_element_embedding(target)
        similar = []
        
        for element in dom.css('*'):
            if element == target:
                continue
            
            element_embedding = self.get_element_embedding(element)
            similarity = np.dot(target_embedding, element_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(element_embedding) + 1e-8
            )
            
            if similarity >= threshold:
                similar.append((element, float(similarity)))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)


class SelectorGenerator(LoggerMixin):
    """Generates CSS/XPath selectors using AI and heuristics"""
    
    def __init__(self, analyzer: Optional[DOMAnalyzer] = None):
        self.analyzer = analyzer or DOMAnalyzer()
        self.selector_cache = {}
        self._pattern_memory = defaultdict(list)
        
    def generate_selectors(self, 
                          target_elements: Elements, 
                          dom: DOMTree,
                          examples: Optional[List[ExtractionExample]] = None,
                          max_selectors: int = 5) -> List[SelectorCandidate]:
        """Generate multiple selector candidates for target elements"""
        if not target_elements:
            return []
        
        candidates = []
        
        # Strategy 1: Direct selector generation
        direct_selectors = self._generate_direct_selectors(target_elements)
        candidates.extend(direct_selectors)
        
        # Strategy 2: Contextual selectors using parent/sibling relationships
        contextual_selectors = self._generate_contextual_selectors(target_elements, dom)
        candidates.extend(contextual_selectors)
        
        # Strategy 3: AI-powered selector generation if examples provided
        if examples:
            ai_selectors = self._generate_ai_selectors(target_elements, dom, examples)
            candidates.extend(ai_selectors)
        
        # Strategy 4: Pattern-based selectors for repeated structures
        pattern_selectors = self._generate_pattern_selectors(target_elements, dom)
        candidates.extend(pattern_selectors)
        
        # Score and rank candidates
        scored_candidates = self._score_candidates(candidates, target_elements, dom)
        
        # Deduplicate and return top candidates
        return self._deduplicate_candidates(scored_candidates)[:max_selectors]
    
    def _generate_direct_selectors(self, elements: Elements) -> List[SelectorCandidate]:
        """Generate direct CSS selectors for elements"""
        candidates = []
        
        for element in elements:
            # Try ID selector
            if element.get('id'):
                selector = f"#{element['id']}"
                candidates.append(SelectorCandidate(
                    selector=selector,
                    confidence=0.9,
                    specificity=100,
                    stability_score=0.8,
                    coverage=1.0
                ))
            
            # Try class combinations
            classes = element.get('class', '').split()
            if classes:
                # Single class
                for cls in classes[:3]:  # Limit to first 3 classes
                    selector = f".{cls}"
                    candidates.append(SelectorCandidate(
                        selector=selector,
                        confidence=0.6,
                        specificity=10,
                        stability_score=0.5,
                        coverage=0.0  # Will be calculated later
                    ))
                
                # Multiple classes
                if len(classes) > 1:
                    selector = '.' + '.'.join(classes[:3])
                    candidates.append(SelectorCandidate(
                        selector=selector,
                        confidence=0.7,
                        specificity=10 * min(3, len(classes)),
                        stability_score=0.6,
                        coverage=0.0
                    ))
            
            # Tag with attributes
            attrs = []
            for attr, value in element.attributes.items():
                if attr not in ['class', 'id', 'style'] and value:
                    attrs.append(f'[{attr}="{value}"]')
            
            if attrs:
                selector = element.tag + ''.join(attrs[:2])  # Limit to 2 attributes
                candidates.append(SelectorCandidate(
                    selector=selector,
                    confidence=0.5,
                    specificity=20,
                    stability_score=0.4,
                    coverage=0.0
                ))
        
        return candidates
    
    def _generate_contextual_selectors(self, elements: Elements, dom: DOMTree) -> List[SelectorCandidate]:
        """Generate selectors using parent/sibling context"""
        candidates = []
        
        for element in elements:
            if not element.parent:
                continue
            
            parent = element.parent
            
            # Parent > Child
            if parent.get('class'):
                parent_class = parent.get('class').split()[0]
                selector = f".{parent_class} > {element.tag}"
                candidates.append(SelectorCandidate(
                    selector=selector,
                    confidence=0.65,
                    specificity=11,
                    stability_score=0.7,
                    coverage=0.0
                ))
            
            # Sibling-based selectors
            if hasattr(element, 'previous_sibling') and element.previous_sibling:
                prev = element.previous_sibling
                if prev.get('class'):
                    prev_class = prev.get('class').split()[0]
                    selector = f".{prev_class} + {element.tag}"
                    candidates.append(SelectorCandidate(
                        selector=selector,
                        confidence=0.6,
                        specificity=11,
                        stability_score=0.5,
                        coverage=0.0
                    ))
        
        return candidates
    
    def _generate_ai_selectors(self, 
                              elements: Elements, 
                              dom: DOMTree,
                              examples: List[ExtractionExample]) -> List[SelectorCandidate]:
        """Generate selectors using transformer model and examples"""
        candidates = []
        
        # Prepare context for transformer
        context_elements = []
        for element in elements[:3]:  # Use first 3 elements as context
            features = self.analyzer.extract_features(element)
            context_elements.append(features)
        
        # Find similar examples
        similar_examples = self._find_similar_examples(context_elements, examples)
        
        for example in similar_examples[:2]:  # Use top 2 similar examples
            for selector in example.selectors:
                # Adapt selector to current DOM
                adapted = self._adapt_selector(selector, dom, elements)
                if adapted:
                    candidates.append(SelectorCandidate(
                        selector=adapted,
                        confidence=0.75,
                        specificity=self._calculate_specificity(adapted),
                        stability_score=0.8,
                        coverage=0.0,
                        examples_used=[example.example_id]
                    ))
        
        return candidates
    
    def _generate_pattern_selectors(self, elements: Elements, dom: DOMTree) -> List[SelectorCandidate]:
        """Generate selectors for repeated patterns (lists, grids, etc.)"""
        candidates = []
        
        # Group elements by similar structure
        groups = self._group_similar_elements(elements)
        
        for group in groups:
            if len(group) < 2:
                continue
            
            # Find common parent
            common_parent = self._find_common_parent(group)
            if not common_parent:
                continue
            
            # Generate selector for the pattern
            if common_parent.get('class'):
                parent_class = common_parent.get('class').split()[0]
                child_tag = group[0].tag
                
                # nth-child pattern
                selector = f".{parent_class} > {child_tag}:nth-child(n)"
                candidates.append(SelectorCandidate(
                    selector=selector,
                    confidence=0.7,
                    specificity=11,
                    stability_score=0.6,
                    coverage=len(group) / len(dom.css(child_tag)) if dom.css(child_tag) else 0
                ))
        
        return candidates
    
    def _score_candidates(self, 
                         candidates: List[SelectorCandidate], 
                         target_elements: Elements,
                         dom: DOMTree) -> List[SelectorCandidate]:
        """Score and adjust candidate selectors"""
        for candidate in candidates:
            # Calculate actual coverage
            try:
                matched = dom.css(candidate.selector)
                target_set = set(id(e) for e in target_elements)
                matched_set = set(id(e) for e in matched)
                
                if target_set:
                    candidate.coverage = len(target_set & matched_set) / len(target_set)
                    
                    # Adjust confidence based on coverage
                    candidate.confidence *= candidate.coverage
                    
                    # Penalize if matches too many non-target elements
                    if len(matched_set) > len(target_set):
                        precision = len(target_set & matched_set) / len(matched_set)
                        candidate.confidence *= precision
                
                # Calculate stability score based on selector characteristics
                candidate.stability_score = self._calculate_stability_score(
                    candidate.selector, 
                    dom
                )
                
            except Exception as e:
                self.logger.warning(f"Error scoring selector {candidate.selector}: {e}")
                candidate.confidence *= 0.5
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def _calculate_stability_score(self, selector: str, dom: DOMTree) -> float:
        """Calculate how stable a selector is likely to be"""
        score = 1.0
        
        # Penalize overly specific selectors
        if selector.count(' > ') > 2:
            score *= 0.7
        if selector.count('.') > 3:
            score *= 0.8
        if ':nth-child' in selector:
            score *= 0.6
        
        # Reward semantic selectors
        semantic_tags = ['article', 'section', 'nav', 'header', 'footer', 'main']
        if any(tag in selector for tag in semantic_tags):
            score *= 1.2
        
        # Check selector consistency in DOM
        try:
            elements = dom.css(selector)
            if elements:
                # Check if selector matches similar structure across elements
                structures = [get_element_signature(e) for e in elements[:5]]
                if len(set(structures)) == 1:
                    score *= 1.1
        except:
            pass
        
        return min(1.0, score)
    
    def _calculate_specificity(self, selector: str) -> int:
        """Calculate CSS specificity score"""
        specificity = 0
        specificity += selector.count('#') * 100  # IDs
        specificity += selector.count('.') * 10   # Classes
        specificity += len(re.findall(r'\[\w+\]', selector)) * 10  # Attributes
        specificity += len(re.findall(r':\w+', selector)) * 10     # Pseudo-classes
        specificity += len(re.findall(r'\b\w+\b', selector)) * 1   # Elements
        return specificity
    
    def _deduplicate_candidates(self, candidates: List[SelectorCandidate]) -> List[SelectorCandidate]:
        """Remove duplicate or very similar selectors"""
        seen = set()
        unique = []
        
        for candidate in candidates:
            # Normalize selector for comparison
            normalized = re.sub(r'\s+', ' ', candidate.selector.strip())
            
            # Check for near-duplicates
            is_duplicate = False
            for seen_selector in seen:
                if SequenceMatcher(None, normalized, seen_selector).ratio() > 0.9:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.add(normalized)
                unique.append(candidate)
        
        return unique
    
    def _find_similar_examples(self, 
                              context: List[Dict[str, Any]], 
                              examples: List[ExtractionExample]) -> List[ExtractionExample]:
        """Find examples similar to current context"""
        # Simple similarity based on URL patterns and DOM structure
        scored_examples = []
        
        for example in examples:
            score = 0
            
            # URL similarity
            if context and example.url:
                current_domain = context[0].get('domain', '')
                example_domain = example.url.split('/')[2] if '/' in example.url else ''
                if current_domain == example_domain:
                    score += 0.5
            
            # DOM structure similarity (simplified)
            if example.dom_hash:
                score += 0.3
            
            # Success rate
            score += example.success_rate * 0.2
            
            scored_examples.append((example, score))
        
        return [ex for ex, _ in sorted(scored_examples, key=lambda x: x[1], reverse=True)]
    
    def _adapt_selector(self, selector: str, dom: DOMTree, target_elements: Elements) -> Optional[str]:
        """Adapt a selector from examples to current DOM"""
        try:
            # Try the selector as-is
            matches = dom.css(selector)
            if matches:
                # Check if it matches target elements
                target_ids = set(id(e) for e in target_elements)
                match_ids = set(id(e) for e in matches)
                
                if target_ids & match_ids:
                    return selector
            
            # Try to generalize the selector
            parts = selector.split(' > ')
            if len(parts) > 1:
                # Try removing the most specific part
                for i in range(len(parts) - 1, 0, -1):
                    test_selector = ' > '.join(parts[:i])
                    try:
                        test_matches = dom.css(test_selector)
                        if test_matches and any(id(e) in set(id(e) for e in target_elements) for e in test_matches):
                            return test_selector
                    except:
                        continue
            
            return None
        except:
            return None
    
    def _group_similar_elements(self, elements: Elements) -> List[Elements]:
        """Group elements with similar structure"""
        groups = []
        used = set()
        
        for i, elem in enumerate(elements):
            if i in used:
                continue
            
            group = [elem]
            used.add(i)
            
            for j, other in enumerate(elements[i+1:], i+1):
                if j in used:
                    continue
                
                if self._elements_similar(elem, other):
                    group.append(other)
                    used.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _elements_similar(self, elem1: Element, elem2: Element) -> bool:
        """Check if two elements have similar structure"""
        # Same tag
        if elem1.tag != elem2.tag:
            return False
        
        # Similar class names
        classes1 = set(elem1.get('class', '').split())
        classes2 = set(elem2.get('class', '').split())
        
        if classes1 and classes2:
            similarity = len(classes1 & classes2) / max(len(classes1), len(classes2))
            return similarity > 0.5
        
        # Similar attributes
        attrs1 = set(elem1.attributes.keys())
        attrs2 = set(elem2.attributes.keys())
        
        if attrs1 and attrs2:
            similarity = len(attrs1 & attrs2) / max(len(attrs1), len(attrs2))
            return similarity > 0.7
        
        return True
    
    def _find_common_parent(self, elements: Elements) -> Optional[Element]:
        """Find common parent for a group of elements"""
        if not elements:
            return None
        
        # Get all ancestors for first element
        first = elements[0]
        ancestors = []
        current = first.parent
        
        while current:
            ancestors.append(current)
            current = current.parent
        
        # Check which ancestor contains all elements
        for ancestor in ancestors:
            if all(ancestor.contains(e) for e in elements):
                return ancestor
        
        return None


class SchemaInferer(LoggerMixin):
    """Infers data schema from examples and DOM structure"""
    
    def __init__(self, analyzer: Optional[DOMAnalyzer] = None):
        self.analyzer = analyzer or DOMAnalyzer()
        self.type_patterns = {
            'url': r'https?://[^\s]+',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'date': r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}',
            'number': r'^\d+(\.\d+)?$',
            'phone': r'^\+?[\d\s-]{10,}$',
            'image': r'\.(jpg|jpeg|png|gif|webp|svg)$',
        }
    
    def infer_schema(self, 
                    examples: List[ExtractionExample],
                    dom: Optional[DOMTree] = None) -> SchemaDefinition:
        """Infer schema from extraction examples"""
        if not examples:
            return SchemaDefinition(fields=[])
        
        # Analyze all extracted data
        field_candidates = defaultdict(list)
        
        for example in examples:
            for key, value in example.extracted_data.items():
                field_candidates[key].append({
                    'value': value,
                    'selectors': example.selectors,
                    'example_id': example.example_id
                })
        
        # Create schema fields
        fields = []
        for field_name, candidates in field_candidates.items():
            field = self._analyze_field(field_name, candidates, dom)
            if field:
                fields.append(field)
        
        return SchemaDefinition(
            fields=fields,
            inferred_at=datetime.now(),
            confidence=self._calculate_schema_confidence(fields, examples)
        )
    
    def _analyze_field(self, 
                      name: str, 
                      candidates: List[Dict[str, Any]],
                      dom: Optional[DOMTree] = None) -> Optional[SchemaField]:
        """Analyze a field to determine its type and properties"""
        if not candidates:
            return None
        
        # Collect all values
        values = [c['value'] for c in candidates]
        
        # Determine field type
        field_type = self._infer_field_type(values)
        
        # Determine if field is required (appears in all examples)
        required = len(candidates) > 0  # Simplified
        
        # Determine if field can have multiple values
        multiple = any(isinstance(v, list) for v in values)
        
        # Find best selectors for this field
        selectors = self._find_best_selectors(candidates, dom)
        
        # Infer validation regex
        validation_regex = self._infer_validation_regex(values, field_type)
        
        # Infer transformation
        transformation = self._infer_transformation(values, field_type)
        
        return SchemaField(
            name=name,
            field_type=field_type,
            selectors=selectors,
            required=required,
            multiple=multiple,
            validation_regex=validation_regex,
            transformation=transformation
        )
    
    def _infer_field_type(self, values: List[Any]) -> str:
        """Infer the type of a field from its values"""
        if not values:
            return 'text'
        
        # Check for list type
        if any(isinstance(v, list) for v in values):
            return 'list'
        
        # Check for object type
        if any(isinstance(v, dict) for v in values):
            return 'object'
        
        # Check patterns for scalar types
        for value in values:
            if not isinstance(value, str):
                continue
            
            for type_name, pattern in self.type_patterns.items():
                if re.search(pattern, str(value), re.IGNORECASE):
                    return type_name
        
        # Check if all values are numbers
        if all(self._is_number(v) for v in values if v):
            return 'number'
        
        # Check if all values look like dates
        if all(self._is_date(v) for v in values if v):
            return 'date'
        
        return 'text'
    
    def _is_number(self, value: Any) -> bool:
        """Check if value is a number"""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value.replace(',', ''))
                return True
            except:
                return False
        return False
    
    def _is_date(self, value: Any) -> bool:
        """Check if value looks like a date"""
        if not isinstance(value, str):
            return False
        
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}\.\d{2}\.\d{4}',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',
        ]
        
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in date_patterns)
    
    def _find_best_selectors(self, 
                            candidates: List[Dict[str, Any]], 
                            dom: Optional[DOMTree] = None) -> List[Selector]:
        """Find the most reliable selectors for a field"""
        selector_scores = defaultdict(float)
        
        for candidate in candidates:
            for selector in candidate.get('selectors', []):
                # Score based on how many examples used this selector
                selector_scores[selector] += 1
        
        # Sort by score and return top selectors
        sorted_selectors = sorted(selector_scores.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in sorted_selectors[:3]]  # Return top 3
    
    def _infer_validation_regex(self, values: List[Any], field_type: str) -> Optional[str]:
        """Infer validation regex for a field"""
        if field_type == 'url':
            return r'^https?://[^\s]+$'
        elif field_type == 'email':
            return r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        elif field_type == 'number':
            return r'^-?\d+(\.\d+)?$'
        
        # Try to find common pattern in values
        str_values = [str(v) for v in values if v]
        if len(str_values) >= 3:
            # Find common prefix/suffix
            prefix = self._find_common_prefix(str_values)
            suffix = self._find_common_suffix(str_values)
            
            if prefix or suffix:
                pattern = f"^{re.escape(prefix)}.*{re.escape(suffix)}$"
                return pattern
        
        return None
    
    def _infer_transformation(self, values: List[Any], field_type: str) -> Optional[str]:
        """Infer data transformation for a field"""
        if field_type == 'text':
            # Check if values have extra whitespace
            str_values = [str(v) for v in values if v]
            if any('  ' in v or v != v.strip() for v in str_values):
                return 'strip'
        
        elif field_type == 'number':
            # Check if numbers have commas
            str_values = [str(v) for v in values if v]
            if any(',' in v for v in str_values):
                return 'remove_commas'
        
        elif field_type == 'date':
            return 'date_parse'
        
        return None
    
    def _find_common_prefix(self, strings: List[str]) -> str:
        """Find common prefix among strings"""
        if not strings:
            return ""
        
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        
        return prefix
    
    def _find_common_suffix(self, strings: List[str]) -> str:
        """Find common suffix among strings"""
        if not strings:
            return ""
        
        suffix = strings[0]
        for s in strings[1:]:
            while not s.endswith(suffix):
                suffix = suffix[1:]
                if not suffix:
                    return ""
        
        return suffix
    
    def _calculate_schema_confidence(self, 
                                   fields: List[SchemaField], 
                                   examples: List[ExtractionExample]) -> float:
        """Calculate confidence score for inferred schema"""
        if not fields or not examples:
            return 0.0
        
        # Base confidence on number of examples and consistency
        base_confidence = min(1.0, len(examples) / 10)  # Max confidence at 10 examples
        
        # Adjust based on field consistency
        field_consistency = 0
        for field in fields:
            # Check how consistently this field appears
            appearances = sum(1 for ex in examples if field.name in ex.extracted_data)
            consistency = appearances / len(examples)
            field_consistency += consistency
        
        field_consistency /= len(fields) if fields else 1
        
        return base_confidence * field_consistency


class SelectorRepairer(LoggerMixin):
    """Repairs broken selectors by finding similar elements"""
    
    def __init__(self, analyzer: Optional[DOMAnalyzer] = None):
        self.analyzer = analyzer or DOMAnalyzer()
        self.repair_history = []
    
    def repair_selector(self, 
                       broken_selector: Selector, 
                       original_dom: DOMTree,
                       current_dom: DOMTree,
                       target_description: Optional[str] = None) -> List[SelectorCandidate]:
        """Attempt to repair a broken selector"""
        self.logger.info(f"Attempting to repair selector: {broken_selector}")
        
        candidates = []
        
        # Strategy 1: Find elements that were matched by the original selector
        original_matches = self._safe_query(original_dom, broken_selector)
        if original_matches:
            # Find similar elements in current DOM
            for original_elem in original_matches[:3]:  # Use first 3 as reference
                similar = self.analyzer.find_similar_elements(original_elem, current_dom, threshold=0.6)
                
                for similar_elem, similarity in similar[:5]:
                    # Generate selector for similar element
                    new_selector = self._generate_selector_for_element(similar_elem, current_dom)
                    if new_selector:
                        candidates.append(SelectorCandidate(
                            selector=new_selector,
                            confidence=similarity * 0.8,
                            specificity=self._calculate_specificity(new_selector),
                            stability_score=0.7,
                            coverage=0.0
                        ))
        
        # Strategy 2: Parse and adapt the broken selector
        adapted_selectors = self._adapt_broken_selector(broken_selector, current_dom)
        candidates.extend(adapted_selectors)
        
        # Strategy 3: Use transformer model to predict new selector
        if target_description:
            ai_selectors = self._ai_repair_selector(broken_selector, target_description, current_dom)
            candidates.extend(ai_selectors)
        
        # Score and filter candidates
        scored = self._score_repair_candidates(candidates, current_dom)
        
        # Record repair attempt
        self.repair_history.append({
            'original_selector': broken_selector,
            'timestamp': datetime.now(),
            'candidates_found': len(scored),
            'best_candidate': scored[0] if scored else None
        })
        
        return scored
    
    def _safe_query(self, dom: DOMTree, selector: Selector) -> Elements:
        """Safely query DOM, returning empty list on error"""
        try:
            return dom.css(selector)
        except Exception as e:
            self.logger.debug(f"Selector query failed: {selector} - {e}")
            return []
    
    def _generate_selector_for_element(self, element: Element, dom: DOMTree) -> Optional[Selector]:
        """Generate a selector for a specific element"""
        try:
            # Try ID first
            if element.get('id'):
                return f"#{element['id']}"
            
            # Try unique class combination
            classes = element.get('class', '').split()
            if classes:
                for cls in classes:
                    selector = f".{cls}"
                    matches = dom.css(selector)
                    if len(matches) == 1:
                        return selector
                
                # Try combination of classes
                for i in range(2, min(4, len(classes) + 1)):
                    selector = '.' + '.'.join(classes[:i])
                    matches = dom.css(selector)
                    if len(matches) == 1:
                        return selector
            
            # Try tag with attributes
            attrs = []
            for attr, value in element.attributes.items():
                if attr not in ['class', 'id', 'style'] and value:
                    attrs.append(f'[{attr}="{value}"]')
            
            if attrs:
                selector = element.tag + ''.join(attrs[:2])
                matches = dom.css(selector)
                if len(matches) == 1:
                    return selector
            
            # Generate path-based selector
            return self._generate_path_selector(element, dom)
            
        except Exception as e:
            self.logger.warning(f"Failed to generate selector for element: {e}")
            return None
    
    def _generate_path_selector(self, element: Element, dom: DOMTree) -> Optional[Selector]:
        """Generate a path-based selector for an element"""
        path = []
        current = element
        
        while current and current.parent:
            # Build selector for current element
            selector_parts = [current.tag]
            
            if current.get('class'):
                classes = current.get('class').split()
                selector_parts.append(f".{classes[0]}")
            
            if current.get('id'):
                selector_parts.append(f"#{current['id']}")
            
            # Check if this selector is unique among siblings
            sibling_selector = ''.join(selector_parts)
            siblings = current.parent.css(sibling_selector) if current.parent else []
            
            if len(siblings) == 1:
                path.insert(0, sibling_selector)
            else:
                # Add nth-child if needed
                siblings_list = list(current.parent.children) if current.parent else []
                try:
                    index = siblings_list.index(current) + 1
                    path.insert(0, f"{current.tag}:nth-child({index})")
                except ValueError:
                    path.insert(0, current.tag)
            
            current = current.parent
            
            # Stop if path gets too long
            if len(path) > 4:
                break
        
        return ' > '.join(path) if path else None
    
    def _adapt_broken_selector(self, selector: Selector, dom: DOMTree) -> List[SelectorCandidate]:
        """Try to adapt a broken selector by modifying it"""
        candidates = []
        
        # Remove overly specific parts
        parts = selector.split(' > ')
        for i in range(len(parts) - 1, 0, -1):
            test_selector = ' > '.join(parts[:i])
            matches = self._safe_query(dom, test_selector)
            if matches:
                candidates.append(SelectorCandidate(
                    selector=test_selector,
                    confidence=0.6,
                    specificity=self._calculate_specificity(test_selector),
                    stability_score=0.5,
                    coverage=0.0
                ))
        
        # Try replacing classes with wildcards
        class_pattern = r'\.[a-zA-Z0-9_-]+'
        if re.search(class_pattern, selector):
            # Try removing one class at a time
            classes = re.findall(class_pattern, selector)
            for cls_to_remove in classes[:2]:  # Try removing first 2 classes
                new_selector = selector.replace(cls_to_remove, '')
                new_selector = re.sub(r'\s+', ' ', new_selector).strip()
                if new_selector and new_selector != selector:
                    matches = self._safe_query(dom, new_selector)
                    if matches:
                        candidates.append(SelectorCandidate(
                            selector=new_selector,
                            confidence=0.5,
                            specificity=self._calculate_specificity(new_selector),
                            stability_score=0.4,
                            coverage=0.0
                        ))
        
        return candidates
    
    def _ai_repair_selector(self, 
                           broken_selector: Selector, 
                           description: str, 
                           dom: DOMTree) -> List[SelectorCandidate]:
        """Use AI to predict a new selector based on description"""
        # This would use a trained model to predict selectors
        # For now, return empty list
        return []
    
    def _score_repair_candidates(self, 
                                candidates: List[SelectorCandidate], 
                                dom: DOMTree) -> List[SelectorCandidate]:
        """Score repair candidates"""
        for candidate in candidates:
            matches = self._safe_query(dom, candidate.selector)
            
            if matches:
                # Reward selectors that match elements
                candidate.confidence *= 1.2
                
                # Check selector stability
                candidate.stability_score = self._calculate_stability_score(
                    candidate.selector, 
                    dom
                )
            else:
                # Penalize selectors that don't match anything
                candidate.confidence *= 0.3
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def _calculate_stability_score(self, selector: str, dom: DOMTree) -> float:
        """Calculate stability score for a selector"""
        # Similar to SelectorGenerator's method
        score = 1.0
        
        # Penalize overly complex selectors
        complexity = selector.count(' > ') + selector.count('.') + selector.count('#')
        if complexity > 5:
            score *= 0.7
        
        # Reward semantic selectors
        semantic_tags = ['article', 'section', 'nav', 'header', 'footer', 'main']
        if any(tag in selector for tag in semantic_tags):
            score *= 1.1
        
        # Check if selector matches consistent structure
        try:
            elements = dom.css(selector)
            if elements:
                signatures = [get_element_signature(e) for e in elements[:3]]
                if len(set(signatures)) == 1:
                    score *= 1.1
        except:
            pass
        
        return min(1.0, score)
    
    def _calculate_specificity(self, selector: str) -> int:
        """Calculate CSS specificity score"""
        specificity = 0
        specificity += selector.count('#') * 100
        specificity += selector.count('.') * 10
        specificity += len(re.findall(r'\[\w+\]', selector)) * 10
        specificity += len(re.findall(r':\w+', selector)) * 10
        specificity += len(re.findall(r'\b\w+\b', selector)) * 1
        return specificity


class AdaptiveExtractor(LoggerMixin, SerializableMixin):
    """Main adaptive extraction engine"""
    
    def __init__(self, 
                 cache_dir: Optional[Union[str, Path]] = None,
                 transformer_model: Optional[TransformerModel] = None):
        self.cache_dir = Path(cache_dir or '.axiom_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.analyzer = DOMAnalyzer(transformer_model)
        self.selector_generator = SelectorGenerator(self.analyzer)
        self.schema_inferer = SchemaInferer(self.analyzer)
        self.selector_repairer = SelectorRepairer(self.analyzer)
        
        # Storage
        self.storage = CacheStorage(self.cache_dir)
        self.examples: List[ExtractionExample] = []
        self.schemas: Dict[str, SchemaDefinition] = {}
        
        # Load existing data
        self._load_cached_data()
        
        self.logger.info("AdaptiveExtractor initialized")
    
    def learn_from_example(self, 
                          url: str,
                          dom: DOMTree,
                          selectors: List[Selector],
                          extracted_data: Dict[str, Any],
                          success_rate: float = 1.0) -> str:
        """Learn from an extraction example"""
        # Create DOM hash for caching
        dom_hash = self._calculate_dom_hash(dom)
        
        example = ExtractionExample(
            url=url,
            dom_hash=dom_hash,
            selectors=selectors,
            extracted_data=extracted_data,
            success_rate=success_rate
        )
        
        # Add to examples
        self.examples.append(example)
        
        # Update schema if we have enough examples
        if len(self.examples) % 5 == 0:  # Update every 5 examples
            self._update_schema(url, dom)
        
        # Save to cache
        self._save_example(example)
        
        self.logger.info(f"Learned from example: {example.example_id}")
        return example.example_id
    
    def generate_extraction_schema(self, 
                                  url: str,
                                  dom: DOMTree,
                                  target_elements: Optional[Elements] = None,
                                  examples: Optional[List[ExtractionExample]] = None) -> SchemaDefinition:
        """Generate extraction schema for a URL"""
        # Use provided examples or find similar ones
        if examples is None:
            examples = self._find_similar_examples(url, dom)
        
        # Generate selectors if target elements provided
        if target_elements:
            selector_candidates = self.selector_generator.generate_selectors(
                target_elements, 
                dom, 
                examples
            )
            
            # Add selector info to examples
            for example in examples:
                if not example.selectors and selector_candidates:
                    example.selectors = [c.selector for c in selector_candidates[:3]]
        
        # Infer schema
        schema = self.schema_inferer.infer_schema(examples, dom)
        schema.url_pattern = self._extract_url_pattern(url)
        
        # Cache schema
        self.schemas[url] = schema
        self._save_schema(url, schema)
        
        return schema
    
    def extract_with_schema(self,
                           url: str,
                           dom: DOMTree,
                           schema: Optional[SchemaDefinition] = None) -> ExtractionResult:
        """Extract data using a schema"""
        if schema is None:
            schema = self.schemas.get(url)
            if schema is None:
                # Try to find similar schema
                schema = self._find_similar_schema(url)
        
        if schema is None:
            raise ValueError("No schema available for extraction")
        
        extracted_data = {}
        selector_performance = {}
        
        for field in schema.fields:
            field_data = self._extract_field(field, dom)
            extracted_data[field.name] = field_data
            
            # Track selector performance
            for selector in field.selectors:
                try:
                    matches = dom.css(selector)
                    selector_performance[selector] = {
                        'matches': len(matches),
                        'success': len(matches) > 0
                    }
                except:
                    selector_performance[selector] = {
                        'matches': 0,
                        'success': False
                    }
        
        # Create extraction result
        result = ExtractionResult(
            url=url,
            data=extracted_data,
            schema_used=schema,
            selectors_used=list(selector_performance.keys()),
            timestamp=datetime.now(),
            confidence=self._calculate_extraction_confidence(selector_performance)
        )
        
        # Learn from this extraction
        if result.confidence > 0.7:  # Only learn from successful extractions
            self.learn_from_example(
                url=url,
                dom=dom,
                selectors=result.selectors_used,
                extracted_data=extracted_data,
                success_rate=result.confidence
            )
        
        return result
    
    def _extract_field(self, field: SchemaField, dom: DOMTree) -> Any:
        """Extract a single field using its selectors"""
        values = []
        
        for selector in field.selectors:
            try:
                elements = dom.css(selector)
                
                for element in elements:
                    value = self._extract_value_from_element(element, field)
                    if value is not None:
                        values.append(value)
                
                if values:
                    break  # Use first successful selector
                    
            except Exception as e:
                self.logger.debug(f"Selector failed: {selector} - {e}")
                continue
        
        # Apply transformations
        if values and field.transformation:
            values = [self._apply_transformation(v, field.transformation) for v in values]
        
        # Return single value or list based on field configuration
        if field.multiple:
            return values
        elif values:
            return values[0]
        else:
            return None if field.required else None
    
    def _extract_value_from_element(self, element: Element, field: SchemaField) -> Any:
        """Extract value from element based on field type"""
        if field.field_type == 'url':
            # Try href, src, or data attributes
            for attr in ['href', 'src', 'data-src', 'data-url']:
                if element.get(attr):
                    return element[attr]
            return element.get('href')
        
        elif field.field_type == 'image':
            # Try src, data-src, or srcset
            for attr in ['src', 'data-src', 'srcset']:
                if element.get(attr):
                    return element[attr]
            return None
        
        elif field.field_type == 'text':
            # Get text content
            text = element.text or ''
            for child in element.children:
                if hasattr(child, 'text') and child.text:
                    text += ' ' + child.text
            return normalize_whitespace(text).strip()
        
        elif field.field_type == 'number':
            # Extract number from text
            text = element.text or ''
            numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
            if numbers:
                try:
                    return float(numbers[0])
                except:
                    return numbers[0]
            return None
        
        else:
            # Default to text extraction
            return element.text
    
    def _apply_transformation(self, value: Any, transformation: str) -> Any:
        """Apply data transformation"""
        if value is None:
            return None
        
        if transformation == 'strip':
            return str(value).strip()
        
        elif transformation == 'lower':
            return str(value).lower()
        
        elif transformation == 'upper':
            return str(value).upper()
        
        elif transformation == 'remove_commas':
            if isinstance(value, str):
                return value.replace(',', '')
            return value
        
        elif transformation == 'date_parse':
            # Simple date parsing - in production would use dateutil
            date_patterns = [
                (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
                (r'(\d{2})/(\d{2})/(\d{4})', '%m/%d/%Y'),
                (r'(\d{2})\.(\d{2})\.(\d{4})', '%d.%m.%Y'),
            ]
            
            for pattern, fmt in date_patterns:
                match = re.search(pattern, str(value))
                if match:
                    try:
                        from datetime import datetime
                        return datetime.strptime(match.group(0), fmt).date()
                    except:
                        pass
            
            return value
        
        return value
    
    def repair_extraction(self,
                         url: str,
                         dom: DOMTree,
                         broken_selectors: List[Selector],
                         target_descriptions: Optional[Dict[Selector, str]] = None) -> Dict[Selector, List[SelectorCandidate]]:
        """Attempt to repair broken selectors"""
        repairs = {}
        
        # Try to find original DOM for comparison
        original_dom = self._get_cached_dom(url)
        
        for selector in broken_selectors:
            description = target_descriptions.get(selector) if target_descriptions else None
            
            if original_dom:
                candidates = self.selector_repairer.repair_selector(
                    selector,
                    original_dom,
                    dom,
                    description
                )
            else:
                # No original DOM, try to generate new selector
                candidates = self.selector_generator.generate_selectors(
                    [],  # No target elements
                    dom,
                    self._find_similar_examples(url, dom)
                )
            
            if candidates:
                repairs[selector] = candidates
                
                # Update schema with repaired selectors
                self._update_schema_with_repairs(url, selector, candidates)
        
        return repairs
    
    def _find_similar_examples(self, url: str, dom: DOMTree) -> List[ExtractionExample]:
        """Find examples similar to current URL and DOM"""
        url_pattern = self._extract_url_pattern(url)
        dom_hash = self._calculate_dom_hash(dom)
        
        similar = []
        for example in self.examples:
            score = 0
            
            # URL pattern similarity
            example_pattern = self._extract_url_pattern(example.url)
            if url_pattern == example_pattern:
                score += 0.5
            
            # DOM structure similarity
            if example.dom_hash == dom_hash:
                score += 0.5
            
            if score > 0.3:
                similar.append((example, score))
        
        return [ex for ex, _ in sorted(similar, key=lambda x: x[1], reverse=True)]
    
    def _find_similar_schema(self, url: str) -> Optional[SchemaDefinition]:
        """Find similar schema for a URL"""
        url_pattern = self._extract_url_pattern(url)
        
        for schema_url, schema in self.schemas.items():
            schema_pattern = self._extract_url_pattern(schema_url)
            if url_pattern == schema_pattern:
                return schema
        
        return None
    
    def _update_schema(self, url: str, dom: DOMTree):
        """Update schema based on accumulated examples"""
        similar_examples = self._find_similar_examples(url, dom)
        
        if similar_examples:
            schema = self.schema_inferer.infer_schema(similar_examples, dom)
            schema.url_pattern = self._extract_url_pattern(url)
            
            self.schemas[url] = schema
            self._save_schema(url, schema)
    
    def _update_schema_with_repairs(self, 
                                   url: str, 
                                   broken_selector: Selector,
                                   repair_candidates: List[SelectorCandidate]):
        """Update schema with repaired selectors"""
        schema = self.schemas.get(url)
        if not schema:
            return
        
        # Find fields using the broken selector
        for field in schema.fields:
            if broken_selector in field.selectors:
                # Replace with best repair candidate
                if repair_candidates:
                    best_repair = repair_candidates[0]
                    idx = field.selectors.index(broken_selector)
                    field.selectors[idx] = best_repair.selector
                    
                    self.logger.info(f"Repaired selector for field {field.name}: {broken_selector} -> {best_repair.selector}")
        
        # Save updated schema
        self._save_schema(url, schema)
    
    def _calculate_dom_hash(self, dom: DOMTree) -> str:
        """Calculate hash for DOM structure"""
        # Simplified hash - in production would be more sophisticated
        try:
            structure = []
            for element in dom.css('*')[:100]:  # Limit to first 100 elements
                structure.append(f"{element.tag}:{len(element.children)}")
            
            content = ''.join(structure)
            return hashlib.md5(content.encode()).hexdigest()
        except:
            return hashlib.md5(str(id(dom)).encode()).hexdigest()
    
    def _extract_url_pattern(self, url: str) -> str:
        """Extract URL pattern for matching"""
        # Remove protocol and www
        pattern = re.sub(r'^https?://(www\.)?', '', url)
        
        # Remove query parameters and fragments
        pattern = re.sub(r'[?#].*$', '', pattern)
        
        # Normalize path separators
        pattern = pattern.rstrip('/')
        
        # Extract domain and first path segment
        parts = pattern.split('/')
        if len(parts) > 1:
            return f"{parts[0]}/{parts[1]}"
        return parts[0]
    
    def _calculate_extraction_confidence(self, selector_performance: Dict[Selector, Dict]) -> float:
        """Calculate confidence score for extraction"""
        if not selector_performance:
            return 0.0
        
        successful = sum(1 for p in selector_performance.values() if p['success'])
        total = len(selector_performance)
        
        return successful / total if total > 0 else 0.0
    
    def _load_cached_data(self):
        """Load cached examples and schemas"""
        try:
            # Load examples
            examples_data = self.storage.load('extraction_examples')
            if examples_data:
                self.examples = [ExtractionExample(**ex) for ex in examples_data]
                self.logger.info(f"Loaded {len(self.examples)} cached examples")
            
            # Load schemas
            schemas_data = self.storage.load('extraction_schemas')
            if schemas_data:
                for url, schema_dict in schemas_data.items():
                    self.schemas[url] = SchemaDefinition(**schema_dict)
                self.logger.info(f"Loaded {len(self.schemas)} cached schemas")
                
        except Exception as e:
            self.logger.warning(f"Failed to load cached data: {e}")
    
    def _save_example(self, example: ExtractionExample):
        """Save example to cache"""
        try:
            examples_data = [asdict(ex) for ex in self.examples[-1000:]]  # Keep last 1000
            self.storage.save('extraction_examples', examples_data)
        except Exception as e:
            self.logger.warning(f"Failed to save example: {e}")
    
    def _save_schema(self, url: str, schema: SchemaDefinition):
        """Save schema to cache"""
        try:
            schemas_data = {url: asdict(schema) for url, schema in self.schemas.items()}
            self.storage.save('extraction_schemas', schemas_data)
        except Exception as e:
            self.logger.warning(f"Failed to save schema: {e}")
    
    def _get_cached_dom(self, url: str) -> Optional[DOMTree]:
        """Get cached DOM for URL"""
        # This would retrieve previously cached DOM
        # For now, return None
        return None
    
    def export_knowledge(self, filepath: Union[str, Path]) -> None:
        """Export learned knowledge to file"""
        knowledge = {
            'examples': [asdict(ex) for ex in self.examples],
            'schemas': {url: asdict(schema) for url, schema in self.schemas.items()},
            'exported_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(knowledge, f, indent=2, default=str)
        
        self.logger.info(f"Exported knowledge to {filepath}")
    
    def import_knowledge(self, filepath: Union[str, Path]) -> None:
        """Import knowledge from file"""
        with open(filepath, 'r') as f:
            knowledge = json.load(f)
        
        # Import examples
        for ex_data in knowledge.get('examples', []):
            example = ExtractionExample(**ex_data)
            if not any(ex.example_id == example.example_id for ex in self.examples):
                self.examples.append(example)
        
        # Import schemas
        for url, schema_data in knowledge.get('schemas', {}).items():
            schema = SchemaDefinition(**schema_data)
            self.schemas[url] = schema
        
        self.logger.info(f"Imported knowledge from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about learned knowledge"""
        return {
            'total_examples': len(self.examples),
            'total_schemas': len(self.schemas),
            'unique_urls': len(set(ex.url for ex in self.examples)),
            'avg_success_rate': np.mean([ex.success_rate for ex in self.examples]) if self.examples else 0,
            'repair_history': len(self.selector_repairer.repair_history),
            'cache_size': sum(1 for _ in self.cache_dir.rglob('*') if _.is_file())
        }


# Convenience function for quick extraction
def extract_adaptive(url: str, 
                    html: str, 
                    target_description: Optional[str] = None,
                    examples: Optional[List[Dict[str, Any]]] = None) -> ExtractionResult:
    """
    Quick adaptive extraction from HTML
    
    Args:
        url: URL of the page
        html: HTML content
        target_description: Description of what to extract
        examples: Optional list of examples for learning
    
    Returns:
        ExtractionResult with extracted data
    """
    from axiom.core.shell import Shell
    
    # Parse HTML
    shell = Shell()
    dom = shell.parse(html)
    
    # Initialize extractor
    extractor = AdaptiveExtractor()
    
    # Learn from examples if provided
    if examples:
        for example in examples:
            extractor.learn_from_example(
                url=example.get('url', url),
                dom=dom,
                selectors=example.get('selectors', []),
                extracted_data=example.get('data', {})
            )
    
    # Generate schema and extract
    schema = extractor.generate_extraction_schema(url, dom)
    result = extractor.extract_with_schema(url, dom, schema)
    
    return result


# Integration with existing axiom sessions
class AdaptiveSessionMixin:
    """Mixin to add adaptive extraction capabilities to sessions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_extractor = AdaptiveExtractor()
    
    def extract_adaptive(self, 
                        selectors: Optional[List[Selector]] = None,
                        schema: Optional[SchemaDefinition] = None,
                        learn: bool = True) -> ExtractionResult:
        """
        Perform adaptive extraction on current page
        
        Args:
            selectors: Optional list of selectors to use
            schema: Optional schema to use
            learn: Whether to learn from this extraction
        
        Returns:
            ExtractionResult with extracted data
        """
        if not hasattr(self, 'current_url') or not hasattr(self, 'dom'):
            raise ValueError("No page loaded in session")
        
        # Find target elements if selectors provided
        target_elements = []
        if selectors:
            for selector in selectors:
                try:
                    elements = self.dom.css(selector)
                    target_elements.extend(elements)
                except:
                    continue
        
        # Generate or use schema
        if schema is None:
            schema = self.adaptive_extractor.generate_extraction_schema(
                self.current_url,
                self.dom,
                target_elements if target_elements else None
            )
        
        # Extract data
        result = self.adaptive_extractor.extract_with_schema(
            self.current_url,
            self.dom,
            schema
        )
        
        # Learn from extraction if enabled
        if learn and result.confidence > 0.7:
            self.adaptive_extractor.learn_from_example(
                url=self.current_url,
                dom=self.dom,
                selectors=result.selectors_used,
                extracted_data=result.data,
                success_rate=result.confidence
            )
        
        return result
    
    def repair_selectors(self, 
                        broken_selectors: List[Selector],
                        target_descriptions: Optional[Dict[Selector, str]] = None) -> Dict[Selector, List[SelectorCandidate]]:
        """
        Attempt to repair broken selectors
        
        Args:
            broken_selectors: List of selectors that no longer work
            target_descriptions: Optional descriptions of what each selector should match
        
        Returns:
            Dictionary mapping broken selectors to repair candidates
        """
        if not hasattr(self, 'current_url') or not hasattr(self, 'dom'):
            raise ValueError("No page loaded in session")
        
        return self.adaptive_extractor.repair_extraction(
            self.current_url,
            self.dom,
            broken_selectors,
            target_descriptions
        )


# Export main classes
__all__ = [
    'AdaptiveExtractor',
    'SelectorGenerator',
    'SchemaInferer', 
    'SelectorRepairer',
    'DOMAnalyzer',
    'ExtractionExample',
    'SelectorCandidate',
    'SchemaField',
    'AdaptiveSessionMixin',
    'extract_adaptive'
]