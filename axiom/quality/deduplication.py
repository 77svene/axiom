"""
axiom Data Quality & Validation Pipeline

This module provides automated data validation, duplicate detection,
data enrichment, and quality scoring for scraped data.

Features:
- JSON Schema validation with Pydantic models
- Near-duplicate detection using SimHash/MinHash
- Data enrichment from multiple sources
- Quality scoring system
- Integration with axiom's existing storage and core modules

Usage:
    from axiom.quality.deduplication import DataQualityPipeline
    
    pipeline = DataQualityPipeline()
    validated_data = pipeline.process(raw_data)
"""

import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from pydantic import BaseModel, Field, validator, ValidationError

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash = None
    MinHashLSH = None

from axiom.core.custom_types import SelectorType
from axiom.core.storage import StorageManager
from axiom.core.utils._utils import clean_text, generate_hash

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    """Quality levels for data scoring."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in data."""
    field: str
    message: str
    severity: ValidationSeverity
    code: str
    value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
            "value": str(self.value) if self.value is not None else None
        }


@dataclass
class QualityScore:
    """Quality score for a data item."""
    overall_score: float = 0.0  # 0.0 to 1.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    uniqueness_score: float = 0.0
    timeliness_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.INVALID
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_overall(self):
        """Calculate overall score from component scores."""
        weights = {
            "completeness": 0.25,
            "accuracy": 0.30,
            "consistency": 0.20,
            "uniqueness": 0.15,
            "timeliness": 0.10
        }
        
        self.overall_score = (
            self.completeness_score * weights["completeness"] +
            self.accuracy_score * weights["accuracy"] +
            self.consistency_score * weights["consistency"] +
            self.uniqueness_score * weights["uniqueness"] +
            self.timeliness_score * weights["timeliness"]
        )
        
        # Determine quality level
        if self.overall_score >= 0.9:
            self.quality_level = QualityLevel.EXCELLENT
        elif self.overall_score >= 0.7:
            self.quality_level = QualityLevel.GOOD
        elif self.overall_score >= 0.5:
            self.quality_level = QualityLevel.ACCEPTABLE
        elif self.overall_score >= 0.3:
            self.quality_level = QualityLevel.POOR
        else:
            self.quality_level = QualityLevel.INVALID
        
        return self.overall_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "completeness_score": self.completeness_score,
            "accuracy_score": self.accuracy_score,
            "consistency_score": self.consistency_score,
            "uniqueness_score": self.uniqueness_score,
            "timeliness_score": self.timeliness_score,
            "quality_level": self.quality_level.value,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata
        }


class DataSchema(BaseModel):
    """Base schema for data validation with common fields."""
    url: Optional[str] = Field(None, description="Source URL")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    source: Optional[str] = Field(None, description="Data source identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"  # Allow extra fields
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProductSchema(DataSchema):
    """Example schema for product data validation."""
    name: str = Field(..., min_length=1, max_length=500)
    price: Optional[float] = Field(None, ge=0)
    currency: Optional[str] = Field("USD", pattern=r"^[A-Z]{3}$")
    description: Optional[str] = Field(None, max_length=5000)
    category: Optional[str] = None
    brand: Optional[str] = None
    sku: Optional[str] = None
    availability: Optional[bool] = True
    rating: Optional[float] = Field(None, ge=0, le=5)
    review_count: Optional[int] = Field(None, ge=0)
    
    @validator('url')
    def validate_url(cls, v):
        if v:
            parsed = urlparse(v)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError("Invalid URL format")
        return v
    
    @validator('price')
    def validate_price(cls, v, values):
        if v is not None and 'currency' in values and not values['currency']:
            raise ValueError("Currency required when price is provided")
        return v


class ArticleSchema(DataSchema):
    """Example schema for article/news data validation."""
    title: str = Field(..., min_length=1, max_length=1000)
    content: str = Field(..., min_length=10)
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    word_count: Optional[int] = Field(None, ge=0)
    
    @validator('word_count', always=True)
    def calculate_word_count(cls, v, values):
        if 'content' in values:
            return len(values['content'].split())
        return v


class SimHashDeduplicator:
    """
    SimHash-based near-duplicate detection for text content.
    
    SimHash is efficient for detecting near-duplicates in large datasets
    by computing a fingerprint that preserves similarity.
    """
    
    def __init__(self, hash_bits: int = 64, threshold: float = 0.85):
        """
        Initialize SimHash deduplicator.
        
        Args:
            hash_bits: Number of bits for hash (64 is standard)
            threshold: Similarity threshold (0-1) for considering duplicates
        """
        self.hash_bits = hash_bits
        self.threshold = threshold
        self.seen_hashes: Dict[int, Set[str]] = defaultdict(set)
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for hashing."""
        # Simple tokenization - can be enhanced with NLP
        text = clean_text(text)
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _compute_simhash(self, tokens: List[str]) -> int:
        """Compute SimHash for a list of tokens."""
        if not tokens:
            return 0
            
        # Initialize vector
        vector = [0] * self.hash_bits
        
        for token in tokens:
            # Hash each token
            token_hash = int(generate_hash(token, 'sha256'), 16)
            
            # Update vector
            for i in range(self.hash_bits):
                bit = (token_hash >> i) & 1
                if bit:
                    vector[i] += 1
                else:
                    vector[i] -= 1
        
        # Compute final hash
        simhash = 0
        for i in range(self.hash_bits):
            if vector[i] > 0:
                simhash |= (1 << i)
        
        return simhash
    
    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """Calculate Hamming distance between two hashes."""
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1
        return distance
    
    def _similarity(self, hash1: int, hash2: int) -> float:
        """Calculate similarity between two hashes (0-1)."""
        distance = self._hamming_distance(hash1, hash2)
        return 1 - (distance / self.hash_bits)
    
    def is_duplicate(self, text: str, identifier: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if text is a near-duplicate of previously seen content.
        
        Args:
            text: Text content to check
            identifier: Optional identifier for the content
            
        Returns:
            Tuple of (is_duplicate, duplicate_identifier)
        """
        tokens = self._tokenize(text)
        if not tokens:
            return False, None
            
        content_hash = self._compute_simhash(tokens)
        
        # Check against existing hashes
        for existing_hash, identifiers in self.seen_hashes.items():
            similarity = self._similarity(content_hash, existing_hash)
            if similarity >= self.threshold:
                # Found duplicate
                duplicate_id = next(iter(identifiers)) if identifiers else str(existing_hash)
                return True, duplicate_id
        
        # Not a duplicate, store hash
        if identifier:
            self.seen_hashes[content_hash].add(identifier)
        else:
            self.seen_hashes[content_hash].add(str(content_hash))
        
        return False, None
    
    def add_to_index(self, text: str, identifier: str):
        """Add text to duplicate index without checking."""
        tokens = self._tokenize(text)
        if tokens:
            content_hash = self._compute_simhash(tokens)
            self.seen_hashes[content_hash].add(identifier)
    
    def clear_index(self):
        """Clear the duplicate index."""
        self.seen_hashes.clear()


class MinHashDeduplicator:
    """
    MinHash-based near-duplicate detection using datasketch.
    
    MinHash is more accurate than SimHash for set similarity,
    especially for shorter texts.
    """
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.7):
        """
        Initialize MinHash deduplicator.
        
        Args:
            num_perm: Number of permutations for MinHash
            threshold: Jaccard similarity threshold (0-1)
        """
        if MinHash is None:
            raise ImportError("datasketch package required for MinHash. Install with: pip install datasketch")
            
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes: Dict[str, MinHash] = {}
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into shingles."""
        text = clean_text(text)
        tokens = re.findall(r'\w+', text.lower())
        
        # Create shingles (n-grams)
        shingle_size = 3
        shingles = set()
        for i in range(len(tokens) - shingle_size + 1):
            shingle = ' '.join(tokens[i:i + shingle_size])
            shingles.add(shingle)
        
        return shingles
    
    def _compute_minhash(self, shingles: Set[str]) -> MinHash:
        """Compute MinHash for a set of shingles."""
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        return minhash
    
    def is_duplicate(self, text: str, identifier: str) -> Tuple[bool, List[str]]:
        """
        Check if text is a near-duplicate.
        
        Args:
            text: Text content to check
            identifier: Unique identifier for the content
            
        Returns:
            Tuple of (is_duplicate, list_of_duplicate_identifiers)
        """
        shingles = self._tokenize(text)
        if not shingles:
            return False, []
        
        minhash = self._compute_minhash(shingles)
        
        # Query for similar items
        result = self.lsh.query(minhash)
        
        if result:
            return True, result
        
        # Not a duplicate, add to index
        self.lsh.insert(identifier, minhash)
        self.minhashes[identifier] = minhash
        
        return False, []
    
    def add_to_index(self, text: str, identifier: str):
        """Add text to duplicate index without checking."""
        shingles = self._tokenize(text)
        if shingles:
            minhash = self._compute_minhash(shingles)
            self.lsh.insert(identifier, minhash)
            self.minhashes[identifier] = minhash
    
    def remove_from_index(self, identifier: str):
        """Remove item from duplicate index."""
        if identifier in self.minhashes:
            self.lsh.remove(identifier)
            del self.minhashes[identifier]
    
    def clear_index(self):
        """Clear the duplicate index."""
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.minhashes.clear()


class DataEnricher(ABC):
    """Abstract base class for data enrichment sources."""
    
    @abstractmethod
    def enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich data with additional information.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Enriched data dictionary
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if enrichment source is available."""
        pass


class GeolocationEnricher(DataEnricher):
    """Enrich data with geolocation information from URLs."""
    
    def __init__(self):
        self.geo_cache: Dict[str, Dict[str, Any]] = {}
    
    def enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and add geolocation data from URL."""
        enriched = data.copy()
        
        url = data.get('url')
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc
                
                # Simple geolocation inference from TLD
                tld = domain.split('.')[-1].lower() if '.' in domain else ''
                
                country_mapping = {
                    'uk': 'United Kingdom',
                    'de': 'Germany',
                    'fr': 'France',
                    'jp': 'Japan',
                    'cn': 'China',
                    'au': 'Australia',
                    'ca': 'Canada',
                    'in': 'India',
                    'br': 'Brazil',
                    'ru': 'Russia'
                }
                
                if tld in country_mapping:
                    enriched['geolocation'] = {
                        'country': country_mapping[tld],
                        'tld': tld,
                        'domain': domain
                    }
                    
                # Store in cache
                self.geo_cache[url] = enriched.get('geolocation', {})
                    
            except Exception as e:
                logger.warning(f"Failed to extract geolocation from URL {url}: {e}")
        
        return enriched
    
    def is_available(self) -> bool:
        return True


class SentimentEnricher(DataEnricher):
    """Simple sentiment analysis enricher."""
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'best', 'awesome', 'perfect', 'happy', 'pleased'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst',
            'hate', 'dislike', 'disappointing', 'broken', 'useless', 'angry'
        }
    
    def enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add sentiment analysis to text fields."""
        enriched = data.copy()
        
        # Analyze text fields
        text_fields = ['content', 'description', 'review', 'comment', 'text']
        
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                text = data[field].lower()
                words = set(re.findall(r'\w+', text))
                
                positive_count = len(words.intersection(self.positive_words))
                negative_count = len(words.intersection(self.negative_words))
                total = positive_count + negative_count
                
                if total > 0:
                    sentiment_score = (positive_count - negative_count) / total
                    enriched[f'{field}_sentiment'] = {
                        'score': sentiment_score,
                        'positive_words': positive_count,
                        'negative_words': negative_count,
                        'label': 'positive' if sentiment_score > 0.1 else 
                                'negative' if sentiment_score < -0.1 else 'neutral'
                    }
        
        return enriched
    
    def is_available(self) -> bool:
        return True


class QualityScorer:
    """Calculates quality scores for data items."""
    
    def __init__(self, schema: Optional[BaseModel] = None):
        """
        Initialize quality scorer.
        
        Args:
            schema: Optional Pydantic schema for validation
        """
        self.schema = schema or DataSchema
        self.required_fields = set()
        self.important_fields = set()
        
        # Extract field information from schema
        if hasattr(self.schema, '__fields__'):
            for field_name, field_info in self.schema.__fields__.items():
                if field_info.required:
                    self.required_fields.add(field_name)
                elif field_name in ['name', 'title', 'price', 'url', 'content']:
                    self.important_fields.add(field_name)
    
    def _calculate_completeness(self, data: Dict[str, Any]) -> Tuple[float, List[ValidationIssue]]:
        """Calculate completeness score based on field presence."""
        issues = []
        total_fields = len(data)
        
        if total_fields == 0:
            return 0.0, [ValidationIssue(
                field="__all__",
                message="No data fields present",
                severity=ValidationSeverity.ERROR,
                code="empty_data"
            )]
        
        # Check required fields
        missing_required = []
        for field in self.required_fields:
            if field not in data or data[field] is None:
                missing_required.append(field)
                issues.append(ValidationIssue(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    severity=ValidationSeverity.ERROR,
                    code="missing_required_field"
                ))
        
        # Check important fields
        missing_important = []
        for field in self.important_fields:
            if field not in data or data[field] is None:
                missing_important.append(field)
                issues.append(ValidationIssue(
                    field=field,
                    message=f"Important field '{field}' is missing",
                    severity=ValidationSeverity.WARNING,
                    code="missing_important_field"
                ))
        
        # Calculate score
        required_weight = 0.6
        important_weight = 0.3
        other_weight = 0.1
        
        required_score = 1.0 - (len(missing_required) / max(1, len(self.required_fields)))
        important_score = 1.0 - (len(missing_important) / max(1, len(self.important_fields)))
        
        # Count non-empty fields
        non_empty_fields = sum(1 for v in data.values() if v is not None and v != "")
        other_score = non_empty_fields / total_fields if total_fields > 0 else 0
        
        completeness = (
            required_score * required_weight +
            important_score * important_weight +
            other_score * other_weight
        )
        
        return min(1.0, max(0.0, completeness)), issues
    
    def _calculate_accuracy(self, data: Dict[str, Any]) -> Tuple[float, List[ValidationIssue]]:
        """Calculate accuracy score based on data validation."""
        issues = []
        accuracy_score = 1.0
        
        # Validate with schema if available
        try:
            if self.schema:
                validated = self.schema(**data)
                # Schema validation passed
        except ValidationError as e:
            accuracy_score *= 0.7  # Penalty for validation errors
            for error in e.errors():
                field = error.get('loc', ['unknown'])[0] if error.get('loc') else 'unknown'
                issues.append(ValidationIssue(
                    field=str(field),
                    message=error.get('msg', 'Validation error'),
                    severity=ValidationSeverity.ERROR,
                    code="validation_error",
                    value=error.get('input')
                ))
        
        # Check data types and formats
        for field, value in data.items():
            if value is None:
                continue
                
            # URL validation
            if 'url' in field.lower() and isinstance(value, str):
                if not value.startswith(('http://', 'https://')):
                    accuracy_score *= 0.9
                    issues.append(ValidationIssue(
                        field=field,
                        message="URL should start with http:// or https://",
                        severity=ValidationSeverity.WARNING,
                        code="invalid_url_format",
                        value=value
                    ))
            
            # Email validation
            elif 'email' in field.lower() and isinstance(value, str):
                if '@' not in value or '.' not in value:
                    accuracy_score *= 0.9
                    issues.append(ValidationIssue(
                        field=field,
                        message="Invalid email format",
                        severity=ValidationSeverity.WARNING,
                        code="invalid_email_format",
                        value=value
                    ))
            
            # Price validation
            elif 'price' in field.lower() and isinstance(value, (int, float)):
                if value < 0:
                    accuracy_score *= 0.8
                    issues.append(ValidationIssue(
                        field=field,
                        message="Price cannot be negative",
                        severity=ValidationSeverity.ERROR,
                        code="negative_price",
                        value=value
                    ))
        
        return min(1.0, max(0.0, accuracy_score)), issues
    
    def _calculate_consistency(self, data: Dict[str, Any]) -> Tuple[float, List[ValidationIssue]]:
        """Calculate consistency score based on internal data consistency."""
        issues = []
        consistency_score = 1.0
        
        # Check for consistent currency/price pairs
        price_fields = [k for k in data.keys() if 'price' in k.lower()]
        currency_fields = [k for k in data.keys() if 'currency' in k.lower()]
        
        if price_fields and currency_fields:
            # If we have prices, we should have currencies
            for price_field in price_fields:
                if data.get(price_field) is not None:
                    # Check if corresponding currency exists
                    currency_found = False
                    for currency_field in currency_fields:
                        if data.get(currency_field):
                            currency_found = True
                            break
                    
                    if not currency_found:
                        consistency_score *= 0.9
                        issues.append(ValidationIssue(
                            field=price_field,
                            message="Price provided without currency",
                            severity=ValidationSeverity.WARNING,
                            code="missing_currency"
                        ))
        
        # Check for date consistency
        date_fields = [k for k in data.keys() if 'date' in k.lower() or 'time' in k.lower()]
        for date_field in date_fields:
            value = data.get(date_field)
            if value and isinstance(value, str):
                # Try to parse date
                try:
                    # Simple date format check
                    if not re.match(r'\d{4}-\d{2}-\d{2}', value):
                        consistency_score *= 0.95
                        issues.append(ValidationIssue(
                            field=date_field,
                            message="Date should be in YYYY-MM-DD format",
                            severity=ValidationSeverity.INFO,
                            code="date_format_suggestion",
                            value=value
                        ))
                except:
                    pass
        
        return min(1.0, max(0.0, consistency_score)), issues
    
    def _calculate_uniqueness(self, data: Dict[str, Any], 
                            is_duplicate: bool = False) -> Tuple[float, List[ValidationIssue]]:
        """Calculate uniqueness score."""
        issues = []
        
        if is_duplicate:
            issues.append(ValidationIssue(
                field="__all__",
                message="Data is duplicate of existing record",
                severity=ValidationSeverity.WARNING,
                code="duplicate_data"
            ))
            return 0.3, issues
        
        # Check for unique identifiers
        unique_fields = ['id', 'sku', 'uuid', 'guid', 'url']
        has_unique_id = any(field in data for field in unique_fields)
        
        if not has_unique_id:
            issues.append(ValidationIssue(
                field="__all__",
                message="No unique identifier found",
                severity=ValidationSeverity.INFO,
                code="missing_unique_id"
            ))
            return 0.7, issues
        
        return 1.0, issues
    
    def _calculate_timeliness(self, data: Dict[str, Any]) -> Tuple[float, List[ValidationIssue]]:
        """Calculate timeliness score based on data freshness."""
        issues = []
        timeliness_score = 1.0
        
        # Check timestamp
        timestamp = data.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    # Try to parse timestamp
                    from dateutil.parser import parse
                    timestamp = parse(timestamp)
                
                if isinstance(timestamp, datetime):
                    age_days = (datetime.now() - timestamp).days
                    
                    # Score decreases with age
                    if age_days <= 1:
                        timeliness_score = 1.0
                    elif age_days <= 7:
                        timeliness_score = 0.9
                    elif age_days <= 30:
                        timeliness_score = 0.7
                    elif age_days <= 365:
                        timeliness_score = 0.5
                    else:
                        timeliness_score = 0.3
                        issues.append(ValidationIssue(
                            field="timestamp",
                            message="Data is more than 1 year old",
                            severity=ValidationSeverity.WARNING,
                            code="stale_data"
                        ))
            except Exception as e:
                logger.debug(f"Could not parse timestamp: {e}")
        
        return timeliness_score, issues
    
    def calculate_score(self, data: Dict[str, Any], 
                       is_duplicate: bool = False) -> QualityScore:
        """
        Calculate overall quality score for data.
        
        Args:
            data: Data dictionary to score
            is_duplicate: Whether this data is a duplicate
            
        Returns:
            QualityScore object
        """
        score = QualityScore()
        
        # Calculate component scores
        score.completeness_score, completeness_issues = self._calculate_completeness(data)
        score.accuracy_score, accuracy_issues = self._calculate_accuracy(data)
        score.consistency_score, consistency_issues = self._calculate_consistency(data)
        score.uniqueness_score, uniqueness_issues = self._calculate_uniqueness(data, is_duplicate)
        score.timeliness_score, timeliness_issues = self._calculate_timeliness(data)
        
        # Combine all issues
        score.issues = completeness_issues + accuracy_issues + consistency_issues + uniqueness_issues + timeliness_issues
        
        # Calculate overall score
        score.calculate_overall()
        
        # Add metadata
        score.metadata = {
            "field_count": len(data),
            "non_empty_fields": sum(1 for v in data.values() if v is not None and v != ""),
            "has_required_fields": all(field in data for field in self.required_fields),
            "calculation_time": datetime.now().isoformat()
        }
        
        return score


class DataQualityPipeline:
    """
    Main pipeline for data quality processing.
    
    Integrates validation, deduplication, enrichment, and scoring.
    """
    
    def __init__(self, 
                 schema: Optional[BaseModel] = None,
                 deduplication_method: str = "simhash",
                 deduplication_threshold: float = 0.85,
                 enrichers: Optional[List[DataEnricher]] = None,
                 storage_manager: Optional[StorageManager] = None):
        """
        Initialize data quality pipeline.
        
        Args:
            schema: Pydantic model for validation
            deduplication_method: 'simhash' or 'minhash'
            deduplication_threshold: Similarity threshold for duplicates
            enrichers: List of data enrichers
            storage_manager: Storage manager for persistence
        """
        self.schema = schema or DataSchema
        self.storage_manager = storage_manager
        
        # Initialize deduplicator
        if deduplication_method == "minhash":
            self.deduplicator = MinHashDeduplicator(threshold=deduplication_threshold)
        else:
            self.deduplicator = SimHashDeduplicator(threshold=deduplication_threshold)
        
        # Initialize enrichers
        self.enrichers = enrichers or [
            GeolocationEnricher(),
            SentimentEnricher()
        ]
        
        # Initialize quality scorer
        self.quality_scorer = QualityScorer(self.schema)
        
        # Statistics
        self.stats = {
            "processed": 0,
            "validated": 0,
            "duplicates_found": 0,
            "enriched": 0,
            "quality_scores": defaultdict(int)
        }
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Validate with Pydantic schema
            validated_data = self.schema(**data)
            return True, issues
        except ValidationError as e:
            for error in e.errors():
                field = error.get('loc', ['unknown'])[0] if error.get('loc') else 'unknown'
                issues.append(ValidationIssue(
                    field=str(field),
                    message=error.get('msg', 'Validation error'),
                    severity=ValidationSeverity.ERROR,
                    code="schema_validation_error",
                    value=error.get('input')
                ))
            return False, issues
        except Exception as e:
            issues.append(ValidationIssue(
                field="__all__",
                message=f"Validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="validation_exception"
            ))
            return False, issues
    
    def check_duplicate(self, data: Dict[str, Any], 
                       content_field: str = "content") -> Tuple[bool, Optional[str]]:
        """
        Check if data is a duplicate.
        
        Args:
            data: Data to check
            content_field: Field containing text content for comparison
            
        Returns:
            Tuple of (is_duplicate, duplicate_identifier)
        """
        # Extract content for deduplication
        content = data.get(content_field) or data.get("description") or data.get("text") or ""
        
        if not content:
            # No content to deduplicate on
            return False, None
        
        # Generate identifier for this content
        identifier = data.get("url") or data.get("id") or generate_hash(json.dumps(data, sort_keys=True))
        
        # Check for duplicates
        if isinstance(self.deduplicator, SimHashDeduplicator):
            return self.deduplicator.is_duplicate(content, identifier)
        elif isinstance(self.deduplicator, MinHashDeduplicator):
            is_dup, dup_ids = self.deduplicator.is_duplicate(content, identifier)
            return is_dup, dup_ids[0] if dup_ids else None
        
        return False, None
    
    def enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich data using available enrichers.
        
        Args:
            data: Data to enrich
            
        Returns:
            Enriched data dictionary
        """
        enriched_data = data.copy()
        
        for enricher in self.enrichers:
            if enricher.is_available():
                try:
                    enriched_data = enricher.enrich(enriched_data)
                except Exception as e:
                    logger.warning(f"Enrichment failed with {enricher.__class__.__name__}: {e}")
        
        return enriched_data
    
    def calculate_quality(self, data: Dict[str, Any], 
                         is_duplicate: bool = False) -> QualityScore:
        """
        Calculate quality score for data.
        
        Args:
            data: Data to score
            is_duplicate: Whether data is duplicate
            
        Returns:
            QualityScore object
        """
        return self.quality_scorer.calculate_score(data, is_duplicate)
    
    def process(self, data: Dict[str, Any], 
                enrich: bool = True,
                check_duplicates: bool = True,
                validate: bool = True) -> Dict[str, Any]:
        """
        Process data through the complete quality pipeline.
        
        Args:
            data: Raw data dictionary
            enrich: Whether to enrich data
            check_duplicates: Whether to check for duplicates
            validate: Whether to validate data
            
        Returns:
            Processed data with quality metadata
        """
        self.stats["processed"] += 1
        result = data.copy()
        
        # 1. Validation
        validation_issues = []
        if validate:
            is_valid, validation_issues = self.validate(data)
            result["_validation"] = {
                "is_valid": is_valid,
                "issues": [issue.to_dict() for issue in validation_issues],
                "validated_at": datetime.now().isoformat()
            }
            if is_valid:
                self.stats["validated"] += 1
        
        # 2. Deduplication
        is_duplicate = False
        duplicate_of = None
        if check_duplicates:
            is_duplicate, duplicate_of = self.check_duplicate(data)
            result["_deduplication"] = {
                "is_duplicate": is_duplicate,
                "duplicate_of": duplicate_of,
                "checked_at": datetime.now().isoformat()
            }
            if is_duplicate:
                self.stats["duplicates_found"] += 1
        
        # 3. Enrichment
        if enrich:
            result = self.enrich(result)
            result["_enrichment"] = {
                "enriched": True,
                "enrichers_used": [e.__class__.__name__ for e in self.enrichers if e.is_available()],
                "enriched_at": datetime.now().isoformat()
            }
            self.stats["enriched"] += 1
        
        # 4. Quality Scoring
        quality_score = self.calculate_quality(result, is_duplicate)
        result["_quality"] = quality_score.to_dict()
        self.stats["quality_scores"][quality_score.quality_level.value] += 1
        
        # Add pipeline metadata
        result["_pipeline"] = {
            "processed_at": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
            "processing_time": time.time()
        }
        
        # Store if storage manager available
        if self.storage_manager and not is_duplicate:
            try:
                # Generate storage key
                storage_key = data.get("url") or data.get("id") or generate_hash(json.dumps(data, sort_keys=True))
                self.storage_manager.set(
                    f"quality:{storage_key}",
                    result,
                    expire=86400 * 30  # 30 days
                )
            except Exception as e:
                logger.warning(f"Failed to store processed data: {e}")
        
        return result
    
    def process_batch(self, data_list: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Process a batch of data items.
        
        Args:
            data_list: List of data dictionaries
            **kwargs: Additional arguments for process()
            
        Returns:
            List of processed data dictionaries
        """
        results = []
        for data in data_list:
            try:
                processed = self.process(data, **kwargs)
                results.append(processed)
            except Exception as e:
                logger.error(f"Failed to process data item: {e}")
                # Add error information to result
                error_result = data.copy()
                error_result["_error"] = {
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        return {
            **self.stats,
            "quality_scores": dict(self.stats["quality_scores"]),
            "duplicate_rate": self.stats["duplicates_found"] / max(1, self.stats["processed"]),
            "validation_rate": self.stats["validated"] / max(1, self.stats["processed"]),
            "enrichment_rate": self.stats["enriched"] / max(1, self.stats["processed"])
        }
    
    def clear_deduplication_index(self):
        """Clear the deduplication index."""
        if isinstance(self.deduplicator, SimHashDeduplicator):
            self.deduplicator.clear_index()
        elif isinstance(self.deduplicator, MinHashDeduplicator):
            self.deduplicator.clear_index()
    
    def export_quality_report(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a quality report for a list of processed data items.
        
        Args:
            data_list: List of processed data items
            
        Returns:
            Quality report dictionary
        """
        if not data_list:
            return {"error": "No data to report on"}
        
        # Collect quality scores
        quality_scores = []
        validation_issues = defaultdict(list)
        duplicate_count = 0
        
        for item in data_list:
            if "_quality" in item:
                quality_scores.append(item["_quality"]["overall_score"])
            
            if "_validation" in item and item["_validation"]["issues"]:
                for issue in item["_validation"]["issues"]:
                    validation_issues[issue["code"]].append(issue)
            
            if "_deduplication" in item and item["_deduplication"]["is_duplicate"]:
                duplicate_count += 1
        
        # Calculate statistics
        if quality_scores:
            avg_score = sum(quality_scores) / len(quality_scores)
            min_score = min(quality_scores)
            max_score = max(quality_scores)
        else:
            avg_score = min_score = max_score = 0
        
        # Generate report
        report = {
            "summary": {
                "total_items": len(data_list),
                "average_quality_score": round(avg_score, 3),
                "min_quality_score": round(min_score, 3),
                "max_quality_score": round(max_score, 3),
                "duplicate_count": duplicate_count,
                "duplicate_rate": duplicate_count / len(data_list),
                "report_generated_at": datetime.now().isoformat()
            },
            "quality_distribution": {},
            "common_issues": {},
            "recommendations": []
        }
        
        # Quality distribution
        quality_levels = [QualityLevel.EXCELLENT, QualityLevel.GOOD, 
                         QualityLevel.ACCEPTABLE, QualityLevel.POOR, QualityLevel.INVALID]
        
        for level in quality_levels:
            count = sum(1 for item in data_list 
                       if "_quality" in item and item["_quality"]["quality_level"] == level.value)
            report["quality_distribution"][level.value] = {
                "count": count,
                "percentage": round(count / len(data_list) * 100, 2)
            }
        
        # Common issues
        for issue_code, issues in validation_issues.items():
            report["common_issues"][issue_code] = {
                "count": len(issues),
                "sample_message": issues[0]["message"] if issues else "",
                "fields_affected": list(set(issue["field"] for issue in issues))
            }
        
        # Generate recommendations
        if avg_score < 0.5:
            report["recommendations"].append(
                "Overall data quality is low. Consider improving data collection methods."
            )
        
        if duplicate_count / len(data_list) > 0.1:
            report["recommendations"].append(
                "High duplicate rate detected. Review deduplication settings."
            )
        
        if "missing_required_field" in validation_issues:
            report["recommendations"].append(
                "Missing required fields found. Validate data sources for completeness."
            )
        
        return report


# Example schemas for common use cases
class EcommerceProductSchema(DataSchema):
    """Schema for e-commerce product data."""
    name: str = Field(..., min_length=1, max_length=500)
    price: float = Field(..., ge=0)
    currency: str = Field("USD", pattern=r"^[A-Z]{3}$")
    description: Optional[str] = Field(None, max_length=5000)
    category: Optional[str] = None
    brand: Optional[str] = None
    sku: Optional[str] = None
    availability: bool = True
    rating: Optional[float] = Field(None, ge=0, le=5)
    review_count: Optional[int] = Field(None, ge=0)
    images: List[str] = Field(default_factory=list)
    
    @validator('images', each_item=True)
    def validate_image_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("Image URL must start with http:// or https://")
        return v


class NewsArticleSchema(DataSchema):
    """Schema for news article data."""
    title: str = Field(..., min_length=1, max_length=1000)
    content: str = Field(..., min_length=10)
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    source: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    word_count: Optional[int] = Field(None, ge=0)
    
    @validator('word_count', always=True)
    def calculate_word_count(cls, v, values):
        if 'content' in values:
            return len(values['content'].split())
        return v


# Utility functions
def create_quality_pipeline(schema_name: str = "default", **kwargs) -> DataQualityPipeline:
    """
    Factory function to create quality pipeline with predefined schema.
    
    Args:
        schema_name: Name of schema ('default', 'ecommerce', 'news')
        **kwargs: Additional arguments for DataQualityPipeline
        
    Returns:
        Configured DataQualityPipeline instance
    """
    schemas = {
        "default": DataSchema,
        "ecommerce": EcommerceProductSchema,
        "news": NewsArticleSchema,
        "product": ProductSchema,
        "article": ArticleSchema
    }
    
    schema = schemas.get(schema_name, DataSchema)
    return DataQualityPipeline(schema=schema, **kwargs)


def validate_with_schema(data: Dict[str, Any], 
                        schema: BaseModel) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate data against a Pydantic schema.
    
    Args:
        data: Data to validate
        schema: Pydantic model class
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    try:
        schema(**data)
        return True, []
    except ValidationError as e:
        errors = []
        for error in e.errors():
            errors.append({
                "field": error.get('loc', ['unknown'])[0] if error.get('loc') else 'unknown',
                "message": error.get('msg', 'Validation error'),
                "type": error.get('type', 'unknown'),
                "input": error.get('input')
            })
        return False, errors


# Integration with existing axiom modules
def integrate_with_storage(storage_manager: StorageManager) -> DataQualityPipeline:
    """
    Create a quality pipeline integrated with axiom's storage.
    
    Args:
        storage_manager: Existing StorageManager instance
        
    Returns:
        Configured DataQualityPipeline
    """
    return DataQualityPipeline(storage_manager=storage_manager)


# Example usage
if __name__ == "__main__":
    # Example data
    sample_data = [
        {
            "url": "https://example.com/product/1",
            "name": "Wireless Headphones",
            "price": 99.99,
            "currency": "USD",
            "description": "High-quality wireless headphones with noise cancellation",
            "timestamp": datetime.now().isoformat(),
            "source": "example_store"
        },
        {
            "url": "https://example.com/product/2",
            "name": "Smart Watch",
            "price": 199.99,
            "currency": "USD",
            "description": "Feature-rich smartwatch with health monitoring",
            "timestamp": datetime.now().isoformat(),
            "source": "example_store"
        },
        {
            # Duplicate of first item with slight variation
            "url": "https://example.com/product/1-copy",
            "name": "Wireless Headphones",
            "price": 99.99,
            "currency": "USD",
            "description": "High quality wireless headphones with noise cancellation",  # Slight variation
            "timestamp": datetime.now().isoformat(),
            "source": "example_store"
        }
    ]
    
    # Create pipeline
    pipeline = create_quality_pipeline("ecommerce")
    
    # Process data
    print("Processing data through quality pipeline...")
    processed_data = pipeline.process_batch(sample_data, validate=True, check_duplicates=True, enrich=True)
    
    # Generate report
    report = pipeline.export_quality_report(processed_data)
    
    print(f"\nQuality Report Summary:")
    print(f"Total items: {report['summary']['total_items']}")
    print(f"Average quality score: {report['summary']['average_quality_score']}")
    print(f"Duplicate count: {report['summary']['duplicate_count']}")
    print(f"Duplicate rate: {report['summary']['duplicate_rate']:.2%}")
    
    print(f"\nQuality Distribution:")
    for level, stats in report['quality_distribution'].items():
        print(f"  {level}: {stats['count']} ({stats['percentage']}%)")
    
    print(f"\nPipeline Statistics:")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        if key != "quality_scores":
            print(f"  {key}: {value}")