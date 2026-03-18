"""
axiom/extraction/schema.py

Adaptive Extraction Engine — AI-powered selector generation, automatic schema inference, and self-healing selectors.
"""

import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from lxml import html, etree

from axiom.core.ai import BaseAIModel, DOMFeatureExtractor
from axiom.core.custom_types import Selector, SelectorType
from axiom.core.utils._utils import normalize_whitespace, safe_json_serialize

logger = logging.getLogger(__name__)


@dataclass
class ExtractionExample:
    """Example for few-shot learning of extraction patterns."""
    dom: str
    target_data: Dict[str, Any]
    selectors: Optional[Dict[str, str]] = None
    schema: Optional[Dict[str, str]] = None
    weight: float = 1.0


@dataclass
class ExtractionSchema:
    """Schema definition for structured data extraction."""
    fields: Dict[str, str] = field(default_factory=dict)  # field_name -> css_selector
    field_types: Dict[str, str] = field(default_factory=dict)  # field_name -> data_type
    required_fields: Set[str] = field(default_factory=set)
    optional_fields: Set[str] = field(default_factory=set)
    transformations: Dict[str, str] = field(default_factory=dict)  # field_name -> transform_func_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            'fields': self.fields,
            'field_types': self.field_types,
            'required_fields': list(self.required_fields),
            'optional_fields': list(self.optional_fields),
            'transformations': self.transformations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionSchema':
        """Create schema from dictionary."""
        return cls(
            fields=data.get('fields', {}),
            field_types=data.get('field_types', {}),
            required_fields=set(data.get('required_fields', [])),
            optional_fields=set(data.get('optional_fields', [])),
            transformations=data.get('transformations', {})
        )


class SelectorRepairEngine:
    """Engine for repairing broken selectors using similarity metrics."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.dom_parser = html.HTMLParser()
        
    def calculate_dom_similarity(self, dom1: str, dom2: str) -> float:
        """Calculate structural similarity between two DOM trees."""
        try:
            tree1 = html.fromstring(dom1, parser=self.dom_parser)
            tree2 = html.fromstring(dom2, parser=self.dom_parser)
            
            # Extract structural features
            features1 = self._extract_structural_features(tree1)
            features2 = self._extract_structural_features(tree2)
            
            # Calculate Jaccard similarity of tag sets
            tags1 = set(features1['tags'])
            tags2 = set(features2['tags'])
            
            if not tags1 or not tags2:
                return 0.0
                
            intersection = len(tags1.intersection(tags2))
            union = len(tags1.union(tags2))
            
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error calculating DOM similarity: {e}")
            return 0.0
    
    def _extract_structural_features(self, element: html.HtmlElement) -> Dict[str, Any]:
        """Extract structural features from DOM element."""
        features = {
            'tags': [],
            'depth': 0,
            'child_count': 0,
            'attributes': set()
        }
        
        def traverse(el, depth=0):
            features['tags'].append(el.tag)
            features['depth'] = max(features['depth'], depth)
            features['child_count'] += len(el)
            
            for attr, value in el.attrib.items():
                features['attributes'].add(f"{attr}:{value[:50]}")
            
            for child in el:
                traverse(child, depth + 1)
        
        traverse(element)
        return features
    
    def repair_selector(self, broken_selector: str, old_dom: str, new_dom: str) -> Optional[str]:
        """Attempt to repair a broken selector by finding similar elements."""
        try:
            old_tree = html.fromstring(old_dom, parser=self.dom_parser)
            new_tree = html.fromstring(new_dom, parser=self.dom_parser)
            
            # Try to find elements matching broken selector in old DOM
            try:
                old_elements = old_tree.cssselect(broken_selector)
                if not old_elements:
                    return None
            except Exception:
                # Invalid selector, try XPath
                try:
                    old_elements = old_tree.xpath(broken_selector)
                    if not old_elements:
                        return None
                except Exception:
                    return None
            
            # Extract features from old elements
            old_features = [self._extract_element_features(el) for el in old_elements[:5]]
            
            # Find similar elements in new DOM
            candidates = self._find_similar_elements(new_tree, old_features)
            
            if not candidates:
                return None
            
            # Generate new selector from best candidate
            best_candidate = candidates[0][0]  # (element, similarity_score)
            new_selector = self._generate_selector(best_candidate, new_tree)
            
            return new_selector
            
        except Exception as e:
            logger.error(f"Error repairing selector: {e}")
            return None
    
    def _extract_element_features(self, element: html.HtmlElement) -> Dict[str, Any]:
        """Extract features from a single element for matching."""
        features = {
            'tag': element.tag,
            'classes': set(element.get('class', '').split()),
            'id': element.get('id'),
            'attributes': {k: v for k, v in element.attrib.items() 
                          if k not in ['class', 'id']},
            'text_content': normalize_whitespace(element.text_content())[:100],
            'parent_tag': element.getparent().tag if element.getparent() else None,
            'sibling_count': len(element.getparent()) if element.getparent() else 0,
            'position_among_siblings': list(element.getparent()).index(element) 
                                      if element.getparent() else 0
        }
        return features
    
    def _find_similar_elements(self, tree: html.HtmlElement, 
                               target_features: List[Dict]) -> List[Tuple[html.HtmlElement, float]]:
        """Find elements in tree similar to target features."""
        candidates = []
        
        for element in tree.iter():
            if element.tag == etree.Comment:
                continue
                
            elem_features = self._extract_element_features(element)
            
            # Calculate similarity for each target feature set
            max_similarity = 0.0
            for target_feat in target_features:
                similarity = self._calculate_element_similarity(elem_features, target_feat)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= self.similarity_threshold:
                candidates.append((element, max_similarity))
        
        # Sort by similarity score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _calculate_element_similarity(self, feat1: Dict, feat2: Dict) -> float:
        """Calculate similarity between two element feature sets."""
        scores = []
        
        # Tag similarity
        scores.append(1.0 if feat1['tag'] == feat2['tag'] else 0.0)
        
        # Class similarity
        classes1 = feat1['classes']
        classes2 = feat2['classes']
        if classes1 or classes2:
            intersection = len(classes1.intersection(classes2))
            union = len(classes1.union(classes2))
            scores.append(intersection / union if union > 0 else 0.0)
        
        # ID similarity
        if feat1['id'] and feat2['id']:
            scores.append(1.0 if feat1['id'] == feat2['id'] else 0.0)
        
        # Text similarity
        text1 = feat1['text_content']
        text2 = feat2['text_content']
        if text1 and text2:
            # Simple substring matching
            if text1 in text2 or text2 in text1:
                scores.append(1.0)
            else:
                # Word overlap
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                if words1 and words2:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    scores.append(intersection / union if union > 0 else 0.0)
        
        # Attribute similarity
        attrs1 = set(f"{k}:{v}" for k, v in feat1['attributes'].items())
        attrs2 = set(f"{k}:{v}" for k, v in feat2['attributes'].items())
        if attrs1 or attrs2:
            intersection = len(attrs1.intersection(attrs2))
            union = len(attrs1.union(attrs2))
            scores.append(intersection / union if union > 0 else 0.0)
        
        # Position similarity
        if (feat1['parent_tag'] == feat2['parent_tag'] and 
            feat1['sibling_count'] == feat2['sibling_count']):
            pos_diff = abs(feat1['position_among_siblings'] - feat2['position_among_siblings'])
            max_pos = max(feat1['position_among_siblings'], feat2['position_among_siblings'])
            if max_pos > 0:
                scores.append(1.0 - (pos_diff / max_pos))
        
        # Calculate weighted average
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def _generate_selector(self, element: html.HtmlElement, 
                          tree: html.HtmlElement) -> str:
        """Generate a CSS selector for the given element."""
        # Try ID selector first
        if element.get('id'):
            return f"#{element.get('id')}"
        
        # Try class selector
        classes = element.get('class', '').split()
        if classes:
            class_selector = '.' + '.'.join(classes)
            # Verify uniqueness
            matches = tree.cssselect(class_selector)
            if len(matches) == 1:
                return class_selector
        
        # Build path-based selector
        path = []
        current = element
        
        while current is not None and current.tag != 'html':
            tag = current.tag
            
            # Add position if needed
            siblings = [s for s in current.getparent() 
                       if s.tag == current.tag] if current.getparent() else []
            
            if len(siblings) > 1:
                index = siblings.index(current) + 1
                tag = f"{tag}:nth-of-type({index})"
            
            path.insert(0, tag)
            current = current.getparent()
        
        return ' > '.join(path) if path else element.tag


class SchemaInferenceEngine:
    """Engine for inferring extraction schemas from examples."""
    
    def __init__(self):
        self.type_patterns = {
            'integer': r'^-?\d+$',
            'float': r'^-?\d+\.\d+$',
            'boolean': r'^(true|false|yes|no|1|0)$',
            'date': r'^\d{4}-\d{2}-\d{2}',
            'email': r'^[^@]+@[^@]+\.[^@]+$',
            'url': r'^https?://',
            'phone': r'^[\d\s\-\+\(\)]{7,}$'
        }
    
    def infer_schema(self, examples: List[ExtractionExample]) -> ExtractionSchema:
        """Infer schema from extraction examples."""
        if not examples:
            return ExtractionSchema()
        
        # Collect all field names and their values
        field_values: Dict[str, List[Any]] = {}
        field_sources: Dict[str, List[str]] = {}  # Track where values came from
        
        for example in examples:
            for field_name, value in example.target_data.items():
                if field_name not in field_values:
                    field_values[field_name] = []
                    field_sources[field_name] = []
                
                field_values[field_name].append(value)
                if example.selectors and field_name in example.selectors:
                    field_sources[field_name].append(example.selectors[field_name])
        
        # Infer types for each field
        field_types = {}
        for field_name, values in field_values.items():
            field_types[field_name] = self._infer_field_type(values)
        
        # Determine required vs optional fields
        total_examples = len(examples)
        required_fields = set()
        optional_fields = set()
        
        for field_name in field_values.keys():
            # Count how many examples have this field
            count = sum(1 for ex in examples if field_name in ex.target_data)
            if count == total_examples:
                required_fields.add(field_name)
            else:
                optional_fields.add(field_name)
        
        # Infer selectors from examples
        selectors = self._infer_selectors(examples)
        
        return ExtractionSchema(
            fields=selectors,
            field_types=field_types,
            required_fields=required_fields,
            optional_fields=optional_fields,
            transformations=self._infer_transformations(field_values)
        )
    
    def _infer_field_type(self, values: List[Any]) -> str:
        """Infer data type from a list of values."""
        if not values:
            return 'string'
        
        # Check if all values match a pattern
        str_values = [str(v) for v in values if v is not None]
        if not str_values:
            return 'string'
        
        type_counts = {t: 0 for t in self.type_patterns.keys()}
        type_counts['string'] = 0
        
        for value in str_values:
            matched = False
            for type_name, pattern in self.type_patterns.items():
                if re.match(pattern, value, re.IGNORECASE):
                    type_counts[type_name] += 1
                    matched = True
                    break
            if not matched:
                type_counts['string'] += 1
        
        # Return most common type
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    def _infer_selectors(self, examples: List[ExtractionExample]) -> Dict[str, str]:
        """Infer CSS selectors from examples."""
        if not examples:
            return {}
        
        # Use the first example with selectors as base
        base_selectors = {}
        for example in examples:
            if example.selectors:
                base_selectors = example.selectors.copy()
                break
        
        # If no examples have selectors, we can't infer them
        if not base_selectors:
            return {}
        
        return base_selectors
    
    def _infer_transformations(self, field_values: Dict[str, List[Any]]) -> Dict[str, str]:
        """Infer data transformations needed for each field."""
        transformations = {}
        
        for field_name, values in field_values.items():
            str_values = [str(v) for v in values if v is not None]
            
            # Check for whitespace issues
            has_whitespace_issues = any(
                v != v.strip() or '  ' in v for v in str_values
            )
            
            if has_whitespace_issues:
                transformations[field_name] = 'normalize_whitespace'
            
            # Check for HTML entities
            has_html_entities = any(
                '&' in v and ';' in v for v in str_values
            )
            
            if has_html_entities:
                if field_name in transformations:
                    transformations[field_name] += ',unescape_html'
                else:
                    transformations[field_name] = 'unescape_html'
        
        return transformations


class DOMTransformerModel(BaseAIModel):
    """Transformer model for DOM structure understanding."""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.feature_extractor = DOMFeatureExtractor()
        self.model = None
        self.tokenizer = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model."""
        try:
            model_path = Path(model_path)
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data.get('model')
                    self.tokenizer = saved_data.get('tokenizer')
                logger.info(f"Loaded DOM transformer model from {model_path}")
            else:
                logger.warning(f"Model path {model_path} not found, using default")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def save_model(self, model_path: str) -> None:
        """Save model to file."""
        try:
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'tokenizer': self.tokenizer
                }, f)
            logger.info(f"Saved DOM transformer model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def generate_selector(self, dom: str, target_description: str) -> Optional[str]:
        """Generate selector for target described in natural language."""
        if not self.model:
            # Fallback to rule-based approach
            return self._rule_based_selector_generation(dom, target_description)
        
        try:
            # Extract features from DOM
            features = self.feature_extractor.extract_features(dom)
            
            # TODO: Implement actual transformer inference
            # This would involve:
            # 1. Tokenizing the DOM and target description
            # 2. Running through transformer model
            # 3. Decoding output to selector
            
            # Placeholder for now
            return self._rule_based_selector_generation(dom, target_description)
        except Exception as e:
            logger.error(f"Error in transformer selector generation: {e}")
            return self._rule_based_selector_generation(dom, target_description)
    
    def _rule_based_selector_generation(self, dom: str, target_description: str) -> Optional[str]:
        """Fallback rule-based selector generation."""
        try:
            tree = html.fromstring(dom)
            
            # Extract keywords from description
            keywords = set(re.findall(r'\w+', target_description.lower()))
            
            # Search for elements containing keywords
            candidates = []
            for element in tree.iter():
                if element.tag == etree.Comment:
                    continue
                
                text = normalize_whitespace(element.text_content()).lower()
                element_keywords = set(re.findall(r'\w+', text))
                
                # Calculate keyword overlap
                overlap = len(keywords.intersection(element_keywords))
                if overlap > 0:
                    candidates.append((element, overlap))
            
            if not candidates:
                return None
            
            # Sort by keyword overlap
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_element = candidates[0][0]
            
            # Generate selector for best element
            repair_engine = SelectorRepairEngine()
            return repair_engine._generate_selector(best_element, tree)
            
        except Exception as e:
            logger.error(f"Error in rule-based selector generation: {e}")
            return None
    
    def train_on_examples(self, examples: List[ExtractionExample]) -> None:
        """Train model on extraction examples."""
        if not examples:
            return
        
        logger.info(f"Training on {len(examples)} examples")
        
        # Prepare training data
        training_data = []
        for example in examples:
            if example.selectors:
                for field_name, selector in example.selectors.items():
                    training_data.append({
                        'dom': example.dom,
                        'target': f"Extract {field_name}",
                        'selector': selector
                    })
        
        # TODO: Implement actual training
        # This would involve:
        # 1. Tokenizing DOM and target descriptions
        # 2. Training transformer model
        # 3. Updating model weights
        
        logger.info("Training complete (placeholder)")


class AdaptiveExtractor:
    """Main adaptive extraction engine."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 repair_threshold: float = 0.7,
                 cache_dir: Optional[str] = None):
        """
        Initialize adaptive extractor.
        
        Args:
            model_path: Path to pre-trained transformer model
            repair_threshold: Threshold for selector repair similarity
            cache_dir: Directory for caching models and schemas
        """
        self.model = DOMTransformerModel(model_path)
        self.repair_engine = SelectorRepairEngine(repair_threshold)
        self.schema_engine = SchemaInferenceEngine()
        self.examples: List[ExtractionExample] = []
        self.schemas: Dict[str, ExtractionSchema] = {}
        
        # Setup cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.axiom' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cached data
        self._load_cache()
    
    def add_example(self, 
                    dom: str, 
                    target_data: Dict[str, Any],
                    selectors: Optional[Dict[str, str]] = None,
                    schema_name: str = "default",
                    weight: float = 1.0) -> None:
        """Add an extraction example for learning."""
        example = ExtractionExample(
            dom=dom,
            target_data=target_data,
            selectors=selectors,
            weight=weight
        )
        self.examples.append(example)
        
        # Update schema if we have selectors
        if selectors:
            if schema_name not in self.schemas:
                self.schemas[schema_name] = ExtractionSchema()
            
            schema = self.schemas[schema_name]
            for field_name, selector in selectors.items():
                schema.fields[field_name] = selector
        
        # Auto-save cache
        self._save_cache()
    
    def infer_schema(self, 
                     examples: Optional[List[ExtractionExample]] = None,
                     schema_name: str = "default") -> ExtractionSchema:
        """Infer schema from examples."""
        examples = examples or self.examples
        
        if not examples:
            logger.warning("No examples provided for schema inference")
            return ExtractionSchema()
        
        schema = self.schema_engine.infer_schema(examples)
        self.schemas[schema_name] = schema
        
        # Save to cache
        self._save_cache()
        
        return schema
    
    def extract(self, 
                dom: str, 
                schema: Optional[Union[ExtractionSchema, str]] = None,
                old_dom: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Extract data from DOM using adaptive extraction.
        
        Args:
            dom: HTML content to extract from
            schema: ExtractionSchema object or name of cached schema
            old_dom: Previous version of DOM for selector repair
            **kwargs: Additional extraction parameters
            
        Returns:
            Dictionary of extracted data
        """
        # Resolve schema
        if schema is None:
            schema = self.schemas.get("default", ExtractionSchema())
        elif isinstance(schema, str):
            schema = self.schemas.get(schema, ExtractionSchema())
        
        if not schema.fields:
            logger.warning("No schema defined for extraction")
            return {}
        
        # Parse DOM
        try:
            tree = html.fromstring(dom)
        except Exception as e:
            logger.error(f"Error parsing DOM: {e}")
            return {}
        
        # Extract data
        result = {}
        broken_selectors = []
        
        for field_name, selector in schema.fields.items():
            try:
                # Try original selector
                elements = tree.cssselect(selector)
                
                if not elements and old_dom:
                    # Selector might be broken, try repair
                    repaired_selector = self.repair_engine.repair_selector(
                        selector, old_dom, dom
                    )
                    
                    if repaired_selector:
                        logger.info(f"Repaired selector for {field_name}: {selector} -> {repaired_selector}")
                        elements = tree.cssselect(repaired_selector)
                        # Update schema with repaired selector
                        schema.fields[field_name] = repaired_selector
                        selector = repaired_selector
                    else:
                        broken_selectors.append(field_name)
                        continue
                
                if elements:
                    # Extract value based on field type
                    value = self._extract_field_value(
                        elements[0], 
                        schema.field_types.get(field_name, 'string'),
                        schema.transformations.get(field_name)
                    )
                    
                    # Only add if we have a value
                    if value is not None:
                        result[field_name] = value
                
            except Exception as e:
                logger.warning(f"Error extracting field {field_name}: {e}")
                broken_selectors.append(field_name)
        
        # Log broken selectors
        if broken_selectors:
            logger.warning(f"Failed to extract fields: {broken_selectors}")
        
        # Apply schema validation
        if schema.required_fields:
            missing = schema.required_fields - set(result.keys())
            if missing:
                logger.warning(f"Missing required fields: {missing}")
        
        return result
    
    def _extract_field_value(self, 
                            element: html.HtmlElement, 
                            field_type: str,
                            transformation: Optional[str] = None) -> Any:
        """Extract and transform field value from element."""
        # Get raw value
        if field_type == 'html':
            raw_value = etree.tostring(element, encoding='unicode', method='html')
        elif field_type == 'text':
            raw_value = normalize_whitespace(element.text_content())
        else:
            # For other types, get text content
            raw_value = normalize_whitespace(element.text_content())
        
        # Apply transformations
        if transformation:
            raw_value = self._apply_transformation(raw_value, transformation)
        
        # Type conversion
        return self._convert_type(raw_value, field_type)
    
    def _apply_transformation(self, value: str, transformation: str) -> str:
        """Apply transformation to value."""
        transforms = transformation.split(',')
        
        for transform in transforms:
            transform = transform.strip()
            
            if transform == 'normalize_whitespace':
                value = normalize_whitespace(value)
            elif transform == 'unescape_html':
                import html as html_module
                value = html_module.unescape(value)
            elif transform == 'strip':
                value = value.strip()
            elif transform == 'lower':
                value = value.lower()
            elif transform == 'upper':
                value = value.upper()
            elif transform == 'title':
                value = value.title()
        
        return value
    
    def _convert_type(self, value: str, field_type: str) -> Any:
        """Convert string value to specified type."""
        if not value:
            return None
        
        try:
            if field_type == 'integer':
                return int(value.replace(',', ''))
            elif field_type == 'float':
                return float(value.replace(',', ''))
            elif field_type == 'boolean':
                return value.lower() in ('true', 'yes', '1')
            elif field_type == 'date':
                # Simple date parsing
                from datetime import datetime
                # Try common formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(value, fmt).date()
                    except ValueError:
                        continue
                return value
            elif field_type == 'json':
                return json.loads(value)
            else:  # string
                return value
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting value '{value}' to {field_type}: {e}")
            return value
    
    def generate_selectors(self, 
                          dom: str, 
                          target_descriptions: Dict[str, str]) -> Dict[str, str]:
        """Generate selectors for targets using AI."""
        selectors = {}
        
        for field_name, description in target_descriptions.items():
            selector = self.model.generate_selector(dom, description)
            if selector:
                selectors[field_name] = selector
        
        return selectors
    
    def train(self, examples: Optional[List[ExtractionExample]] = None) -> None:
        """Train the adaptive extractor on examples."""
        examples = examples or self.examples
        
        if not examples:
            logger.warning("No examples provided for training")
            return
        
        # Train transformer model
        self.model.train_on_examples(examples)
        
        # Update schemas
        self.infer_schema(examples)
        
        # Save trained model
        model_path = self.cache_dir / 'dom_transformer.pkl'
        self.model.save_model(str(model_path))
        
        logger.info(f"Training complete on {len(examples)} examples")
    
    def save_schema(self, name: str, path: Optional[str] = None) -> None:
        """Save schema to file."""
        if name not in self.schemas:
            logger.warning(f"Schema '{name}' not found")
            return
        
        schema = self.schemas[name]
        path = path or str(self.cache_dir / f'schema_{name}.json')
        
        try:
            with open(path, 'w') as f:
                json.dump(schema.to_dict(), f, indent=2)
            logger.info(f"Saved schema '{name}' to {path}")
        except Exception as e:
            logger.error(f"Error saving schema: {e}")
    
    def load_schema(self, name: str, path: str) -> None:
        """Load schema from file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            schema = ExtractionSchema.from_dict(data)
            self.schemas[name] = schema
            logger.info(f"Loaded schema '{name}' from {path}")
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            cache_file = self.cache_dir / 'adaptive_extractor.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'examples': self.examples,
                    'schemas': self.schemas
                }, f)
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / 'adaptive_extractor.pkl'
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.examples = cache_data.get('examples', [])
                    self.schemas = cache_data.get('schemas', {})
                logger.info(f"Loaded cache with {len(self.examples)} examples")
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.examples = []
        self.schemas = {}
        self._save_cache()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the extractor."""
        return {
            'examples_count': len(self.examples),
            'schemas_count': len(self.schemas),
            'schema_names': list(self.schemas.keys()),
            'cache_dir': str(self.cache_dir),
            'model_loaded': self.model.model is not None
        }


# Convenience functions
def create_extractor_from_examples(examples: List[Dict[str, Any]], **kwargs) -> AdaptiveExtractor:
    """Create an extractor from a list of examples."""
    extractor = AdaptiveExtractor(**kwargs)
    
    for example in examples:
        extractor.add_example(
            dom=example['dom'],
            target_data=example['target_data'],
            selectors=example.get('selectors'),
            schema_name=example.get('schema_name', 'default'),
            weight=example.get('weight', 1.0)
        )
    
    # Train on all examples
    extractor.train()
    
    return extractor


def extract_with_adaptive_ai(dom: str, 
                            target_descriptions: Dict[str, str],
                            examples: Optional[List[Dict[str, Any]]] = None,
                            **kwargs) -> Dict[str, Any]:
    """
    One-shot adaptive extraction.
    
    Args:
        dom: HTML to extract from
        target_descriptions: Dict mapping field names to natural language descriptions
        examples: Optional list of examples for few-shot learning
        **kwargs: Additional parameters for AdaptiveExtractor
        
    Returns:
        Extracted data dictionary
    """
    extractor = AdaptiveExtractor(**kwargs)
    
    # Add examples if provided
    if examples:
        for example in examples:
            extractor.add_example(
                dom=example['dom'],
                target_data=example['target_data'],
                selectors=example.get('selectors')
            )
        extractor.train()
    
    # Generate selectors
    selectors = extractor.generate_selectors(dom, target_descriptions)
    
    # Create schema
    schema = ExtractionSchema(fields=selectors)
    
    # Extract data
    return extractor.extract(dom, schema=schema)