"""Data Quality & Validation Pipeline for axiom.

This module provides comprehensive data validation, duplicate detection, enrichment,
and quality scoring capabilities for scraped data. Integrates with existing axiom
architecture while maintaining high performance and extensibility.

Key Features:
- JSON Schema validation via Pydantic models
- Near-duplicate detection using SimHash/MinHash algorithms
- Multi-source data enrichment with pluggable providers
- Automated quality scoring with configurable metrics
- Async-first design for high-throughput processing
"""

import asyncio
import hashlib
import json
import logging
import re
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    validator,
    root_validator,
)

from axiom.core.custom_types import (
    AdaptiveResponseType,
    SelectorGroupType,
    SpiderResponseType,
)
from axiom.core.utils._utils import (
    ensure_dict,
    ensure_list,
    safe_json_dumps,
)

# Optional dependencies with graceful fallback
try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    MinHash = None
    MinHashLSH = None

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"  # Data is unusable
    ERROR = "error"        # Major issues, data may be partially usable
    WARNING = "warning"    # Minor issues, data is usable with caution
    INFO = "info"          # Informational, data is good


class DuplicateMethod(str, Enum):
    """Methods for duplicate detection."""
    SIMHASH = "simhash"
    MINHASH = "minhash"
    FUZZY = "fuzzy"
    EXACT = "exact"


class QualityMetric(str, Enum):
    """Quality scoring metrics."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"


class ValidationIssue(BaseModel):
    """Individual validation issue with context."""
    field: Optional[str] = Field(None, description="Field path where issue occurred")
    message: str = Field(..., description="Human-readable issue description")
    severity: ValidationSeverity = Field(
        ValidationSeverity.ERROR,
        description="Issue severity level"
    )
    rule: Optional[str] = Field(None, description="Validation rule that failed")
    expected: Optional[Any] = Field(None, description="Expected value or type")
    actual: Optional[Any] = Field(None, description="Actual value found")
    code: Optional[str] = Field(None, description="Machine-readable error code")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for debugging"
    )

    class Config:
        use_enum_values = True


class ValidationResult(BaseModel):
    """Complete validation result with all issues and metadata."""
    is_valid: bool = Field(..., description="Overall validation status")
    issues: List[ValidationIssue] = Field(
        default_factory=list,
        description="List of validation issues found"
    )
    schema_name: Optional[str] = Field(
        None,
        description="Name of schema used for validation"
    )
    validation_time_ms: float = Field(
        ...,
        description="Time taken for validation in milliseconds"
    )
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of validation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional validation metadata"
    )

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return len([
            i for i in self.issues
            if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        ])

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return len([
            i for i in self.issues
            if i.severity == ValidationSeverity.WARNING
        ])

    def get_issues_by_severity(
        self,
        severity: ValidationSeverity
    ) -> List[ValidationIssue]:
        """Filter issues by severity level."""
        return [i for i in self.issues if i.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict()


class DataQualityScore(BaseModel):
    """Comprehensive quality score with breakdown."""
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall quality score (0-1)"
    )
    metric_scores: Dict[QualityMetric, float] = Field(
        default_factory=dict,
        description="Individual metric scores"
    )
    weighted_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted quality score"
    )
    grade: str = Field(..., description="Letter grade (A-F)")
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the score"
    )
    factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Factors contributing to the score"
    )
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of calculation"
    )

    @validator('grade', pre=True, always=True)
    def calculate_grade(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        """Calculate letter grade from overall score."""
        if v:
            return v

        score = values.get('overall_score', 0.0)
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

    def to_report(self) -> str:
        """Generate human-readable quality report."""
        lines = [
            "Data Quality Report",
            "=" * 50,
            f"Overall Score: {self.overall_score:.2%} ({self.grade})",
            f"Weighted Score: {self.weighted_score:.2%}",
            f"Confidence: {self.confidence:.2%}",
            "",
            "Metric Breakdown:",
        ]

        for metric, score in self.metric_scores.items():
            lines.append(f"  {metric.value}: {score:.2%}")

        if self.factors:
            lines.append("")
            lines.append("Contributing Factors:")
            for factor in self.factors:
                lines.append(f"  - {factor.get('name', 'Unknown')}: "
                           f"{factor.get('impact', 'N/A')}")

        return "\n".join(lines)


class DuplicateResult(BaseModel):
    """Result of duplicate detection analysis."""
    is_duplicate: bool = Field(
        ...,
        description="Whether item is considered a duplicate"
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (0-1)"
    )
    duplicate_of: Optional[str] = Field(
        None,
        description="ID of item this is duplicate of"
    )
    method_used: DuplicateMethod = Field(
        ...,
        description="Method used for detection"
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in duplicate detection"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific details"
    )


class EnrichmentResult(BaseModel):
    """Result of data enrichment operation."""
    success: bool = Field(..., description="Whether enrichment succeeded")
    source: str = Field(..., description="Name of enrichment source")
    enriched_fields: List[str] = Field(
        default_factory=list,
        description="Fields that were enriched"
    )
    new_fields: List[str] = Field(
        default_factory=list,
        description="New fields added by enrichment"
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in enriched data"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional enrichment metadata"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if enrichment failed"
    )


class BaseValidator(ABC):
    """Abstract base class for data validators."""

    @abstractmethod
    async def validate(
        self,
        data: Dict[str, Any],
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        **kwargs
    ) -> ValidationResult:
        """Validate data against schema or rules."""
        pass


class SchemaValidator(BaseValidator):
    """JSON Schema validator using Pydantic models."""

    def __init__(self):
        self._schemas: Dict[str, Type[BaseModel]] = {}
        self._compiled_validators: Dict[str, Any] = {}

    def register_schema(
        self,
        name: str,
        schema: Union[Type[BaseModel], Dict[str, Any]]
    ) -> None:
        """Register a validation schema."""
        if isinstance(schema, dict):
            # Convert JSON Schema dict to Pydantic model
            model = self._create_model_from_schema(name, schema)
            self._schemas[name] = model
        else:
            self._schemas[name] = schema

    def _create_model_from_schema(
        self,
        name: str,
        schema: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Create Pydantic model from JSON Schema dictionary."""
        fields = {}

        properties = schema.get('properties', {})
        required = set(schema.get('required', []))

        for field_name, field_schema in properties.items():
            field_type = self._get_python_type(field_schema)
            field_required = field_name in required

            field_kwargs = {
                'description': field_schema.get('description', ''),
            }

            if not field_required:
                field_kwargs['default'] = field_schema.get('default')

            if 'pattern' in field_schema:
                field_kwargs['regex'] = field_schema['pattern']

            if 'minimum' in field_schema:
                field_kwargs['ge'] = field_schema['minimum']
            if 'maximum' in field_schema:
                field_kwargs['le'] = field_schema['maximum']
            if 'minLength' in field_schema:
                field_kwargs['min_length'] = field_schema['minLength']
            if 'maxLength' in field_schema:
                field_kwargs['max_length'] = field_schema['maxLength']

            fields[field_name] = (field_type, Field(**field_kwargs))

        return type(name, (BaseModel,), {
            '__annotations__': {k: v[0] for k, v in fields.items()},
            **{k: v[1] for k, v in fields.items()}
        })

    def _get_python_type(self, field_schema: Dict[str, Any]) -> type:
        """Convert JSON Schema type to Python type."""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }

        json_type = field_schema.get('type', 'string')

        if json_type == 'array':
            items = field_schema.get('items', {})
            item_type = self._get_python_type(items)
            return List[item_type]
        elif json_type == 'object':
            return Dict[str, Any]

        return type_mapping.get(json_type, Any)

    async def validate(
        self,
        data: Dict[str, Any],
        schema: Optional[Union[Type[BaseModel], Dict[str, Any], str]] = None,
        **kwargs
    ) -> ValidationResult:
        """Validate data against schema."""
        start_time = datetime.now()
        issues = []
        schema_name = None

        try:
            if schema is None:
                raise ValueError("No schema provided for validation")

            if isinstance(schema, str):
                # Schema name reference
                if schema not in self._schemas:
                    raise ValueError(f"Schema '{schema}' not registered")
                model = self._schemas[schema]
                schema_name = schema
            elif isinstance(schema, dict):
                # Inline JSON Schema
                model = self._create_model_from_schema('InlineSchema', schema)
                schema_name = 'inline'
            elif isinstance(schema, type) and issubclass(schema, BaseModel):
                # Pydantic model class
                model = schema
                schema_name = schema.__name__
            else:
                raise ValueError(f"Invalid schema type: {type(schema)}")

            # Perform validation
            try:
                model(**data)
                is_valid = True
            except ValidationError as e:
                is_valid = False
                for error in e.errors():
                    field_path = ".".join(str(loc) for loc in error['loc'])
                    issues.append(ValidationIssue(
                        field=field_path,
                        message=error['msg'],
                        severity=ValidationSeverity.ERROR,
                        rule=error['type'],
                        actual=error.get('input'),
                        code=error['type'].upper(),
                        context={'ctx': error.get('ctx', {})}
                    ))

        except Exception as e:
            is_valid = False
            issues.append(ValidationIssue(
                message=f"Validation failed: {str(e)}",
                severity=ValidationSeverity.CRITICAL,
                code="VALIDATION_SYSTEM_ERROR"
            ))

        validation_time = (datetime.now() - start_time).total_seconds() * 1000

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            schema_name=schema_name,
            validation_time_ms=validation_time,
            metadata={'data_keys': list(data.keys()) if data else []}
        )


class DuplicateDetector:
    """Advanced duplicate detection using multiple algorithms."""

    def __init__(
        self,
        method: DuplicateMethod = DuplicateMethod.SIMHASH,
        threshold: float = 0.85,
        **kwargs
    ):
        self.method = method
        self.threshold = threshold
        self._fingerprint_index: Dict[str, Set[str]] = defaultdict(set)
        self._minhash_index: Optional[Any] = None

        if method == DuplicateMethod.MINHASH and not DATASKETCH_AVAILABLE:
            logger.warning(
                "datasketch not available, falling back to SIMHASH method"
            )
            self.method = DuplicateMethod.SIMHASH

        if method == DuplicateMethod.MINHASH:
            self._init_minhash_index(**kwargs)

    def _init_minhash_index(
        self,
        num_perm: int = 128,
        **kwargs
    ) -> None:
        """Initialize MinHash LSH index."""
        if DATASKETCH_AVAILABLE:
            self._minhash_index = MinHashLSH(
                threshold=self.threshold,
                num_perm=num_perm
            )

    def _compute_simhash(self, text: str) -> int:
        """Compute SimHash fingerprint for text."""
        if XXHASH_AVAILABLE:
            # Use xxhash for better performance
            return xxhash.xxh64(text).intdigest()
        else:
            # Fallback to standard hash
            return int(hashlib.sha256(text.encode()).hexdigest(), 16)

    def _compute_minhash(self, text: str) -> Any:
        """Compute MinHash signature for text."""
        if not DATASKETCH_AVAILABLE or MinHash is None:
            raise RuntimeError("datasketch not available for MinHash")

        # Tokenize text (simple whitespace + punctuation split)
        tokens = set(re.findall(r'\w+', text.lower()))
        m = MinHash(num_perm=128)
        for token in tokens:
            m.update(token.encode('utf-8'))
        return m

    def _compute_fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Compute fuzzy similarity between two texts."""
        return SequenceMatcher(None, text1, text2).ratio()

    def _extract_text(self, data: Dict[str, Any]) -> str:
        """Extract representative text from data dictionary."""
        parts = []
        for key, value in sorted(data.items()):
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}:{value}")
            elif isinstance(value, (list, dict)):
                parts.append(f"{key}:{safe_json_dumps(value)}")
        return " ".join(parts)

    async def detect_duplicates(
        self,
        items: List[Dict[str, Any]],
        id_field: Optional[str] = None
    ) -> List[DuplicateResult]:
        """Detect duplicates in a list of items."""
        results = []
        seen_hashes: Dict[str, str] = {}  # hash -> item_id

        for i, item in enumerate(items):
            item_id = item.get(id_field, str(i)) if id_field else str(i)
            text = self._extract_text(item)

            if self.method == DuplicateMethod.EXACT:
                # Exact match on full text
                item_hash = hashlib.sha256(text.encode()).hexdigest()
                is_duplicate = item_hash in seen_hashes
                duplicate_of = seen_hashes.get(item_hash)
                similarity = 1.0 if is_duplicate else 0.0

                if not is_duplicate:
                    seen_hashes[item_hash] = item_id

            elif self.method == DuplicateMethod.SIMHASH:
                # SimHash comparison
                item_hash = self._compute_simhash(text)
                is_duplicate = False
                duplicate_of = None
                max_similarity = 0.0

                for existing_hash, existing_id in seen_hashes.items():
                    # Hamming distance for SimHash
                    xor = item_hash ^ int(existing_hash)
                    distance = bin(xor).count('1')
                    similarity = 1 - (distance / 64)  # 64-bit hash

                    if similarity >= self.threshold:
                        is_duplicate = True
                        duplicate_of = existing_id
                        max_similarity = similarity
                        break

                if not is_duplicate:
                    seen_hashes[str(item_hash)] = item_id
                    similarity = 0.0
                else:
                    similarity = max_similarity

            elif self.method == DuplicateMethod.MINHASH:
                # MinHash LSH
                if self._minhash_index is None:
                    raise RuntimeError("MinHash index not initialized")

                minhash = self._compute_minhash(text)
                result = self._minhash_index.query(minhash)

                is_duplicate = len(result) > 0
                duplicate_of = result[0] if result else None
                similarity = 1.0 if is_duplicate else 0.0

                if not is_duplicate:
                    self._minhash_index.insert(item_id, minhash)

            elif self.method == DuplicateMethod.FUZZY:
                # Fuzzy string matching
                is_duplicate = False
                duplicate_of = None
                max_similarity = 0.0

                for existing_text, existing_id in seen_hashes.items():
                    similarity = self._compute_fuzzy_similarity(text, existing_text)
                    if similarity >= self.threshold:
                        is_duplicate = True
                        duplicate_of = existing_id
                        max_similarity = similarity
                        break

                if not is_duplicate:
                    seen_hashes[text] = item_id
                    similarity = 0.0
                else:
                    similarity = max_similarity

            else:
                raise ValueError(f"Unsupported duplicate detection method: {self.method}")

            results.append(DuplicateResult(
                is_duplicate=is_duplicate,
                similarity_score=similarity,
                duplicate_of=duplicate_of,
                method_used=self.method,
                confidence=0.95 if similarity > 0.9 else 0.8,
                details={
                    'item_index': i,
                    'text_length': len(text),
                    'method': self.method.value
                }
            ))

        return results

    async def is_duplicate_of(
        self,
        item: Dict[str, Any],
        reference: Dict[str, Any]
    ) -> DuplicateResult:
        """Check if item is duplicate of reference."""
        text1 = self._extract_text(item)
        text2 = self._extract_text(reference)

        if self.method == DuplicateMethod.EXACT:
            is_duplicate = text1 == text2
            similarity = 1.0 if is_duplicate else 0.0

        elif self.method == DuplicateMethod.SIMHASH:
            hash1 = self._compute_simhash(text1)
            hash2 = self._compute_simhash(text2)
            xor = hash1 ^ hash2
            distance = bin(xor).count('1')
            similarity = 1 - (distance / 64)
            is_duplicate = similarity >= self.threshold

        elif self.method == DuplicateMethod.MINHASH:
            if not DATASKETCH_AVAILABLE:
                raise RuntimeError("datasketch not available for MinHash")
            minhash1 = self._compute_minhash(text1)
            minhash2 = self._compute_minhash(text2)
            similarity = minhash1.jaccard(minhash2)
            is_duplicate = similarity >= self.threshold

        elif self.method == DuplicateMethod.FUZZY:
            similarity = self._compute_fuzzy_similarity(text1, text2)
            is_duplicate = similarity >= self.threshold

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        return DuplicateResult(
            is_duplicate=is_duplicate,
            similarity_score=similarity,
            duplicate_of="reference" if is_duplicate else None,
            method_used=self.method,
            confidence=min(1.0, similarity * 1.2) if is_duplicate else 1.0 - similarity
        )


class BaseEnrichmentSource(ABC):
    """Abstract base class for data enrichment sources."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

    @abstractmethod
    async def enrich(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], EnrichmentResult]:
        """Enrich data with additional information."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Clean up resources."""
        pass


class APIEnrichmentSource(BaseEnrichmentSource):
    """Enrichment source using external APIs."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers or {}
        self._session: Optional[aiohttp.ClientSession] = None

        if api_key:
            self.headers['Authorization'] = f"Bearer {api_key}"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            if not AIOHTTP_AVAILABLE:
                raise RuntimeError("aiohttp not installed for API enrichment")
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def enrich(
        self,
        data: Dict[str, Any],
        endpoint: str = "",
        method: str = "POST",
        **kwargs
    ) -> Tuple[Dict[str, Any], EnrichmentResult]:
        """Enrich data via API call."""
        try:
            session = await self._get_session()
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

            async with session.request(
                method,
                url,
                json=data,
                **kwargs
            ) as response:
                if response.status == 200:
                    enriched_data = await response.json()
                    merged_data = {**data, **enriched_data}

                    # Determine which fields were enriched
                    original_keys = set(data.keys())
                    enriched_keys = set(enriched_data.keys())
                    new_keys = enriched_keys - original_keys
                    updated_keys = enriched_keys & original_keys

                    result = EnrichmentResult(
                        success=True,
                        source=self.name,
                        enriched_fields=list(updated_keys),
                        new_fields=list(new_keys),
                        confidence=0.9,
                        metadata={
                            'status_code': response.status,
                            'endpoint': endpoint,
                            'response_time_ms': response.headers.get(
                                'X-Response-Time', 'N/A'
                            )
                        }
                    )

                    return merged_data, result
                else:
                    error_text = await response.text()
                    return data, EnrichmentResult(
                        success=False,
                        source=self.name,
                        error=f"API returned {response.status}: {error_text}",
                        metadata={'status_code': response.status}
                    )

        except Exception as e:
            logger.error(f"Enrichment failed for {self.name}: {str(e)}")
            return data, EnrichmentResult(
                success=False,
                source=self.name,
                error=str(e)
            )

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()


class QualityScorer:
    """Configurable quality scoring engine."""

    def __init__(
        self,
        weights: Optional[Dict[QualityMetric, float]] = None,
        custom_scorers: Optional[Dict[str, Callable]] = None
    ):
        self.weights = weights or {
            QualityMetric.COMPLETENESS: 0.25,
            QualityMetric.ACCURACY: 0.25,
            QualityMetric.CONSISTENCY: 0.20,
            QualityMetric.TIMELINESS: 0.15,
            QualityMetric.UNIQUENESS: 0.10,
            QualityMetric.VALIDITY: 0.05,
        }
        self.custom_scorers = custom_scorers or {}

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {
                k: v / total_weight
                for k, v in self.weights.items()
            }

    def _score_completeness(
        self,
        data: Dict[str, Any],
        required_fields: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Score data completeness."""
        if not data:
            return 0.0, {'empty_data': True}

        if required_fields:
            present = sum(1 for f in required_fields if f in data and data[f] is not None)
            score = present / len(required_fields) if required_fields else 1.0
            missing = [f for f in required_fields if f not in data or data[f] is None]
        else:
            # Score based on non-null values
            total_fields = len(data)
            non_null = sum(1 for v in data.values() if v is not None)
            score = non_null / total_fields if total_fields > 0 else 1.0
            missing = []

        return score, {
            'missing_fields': missing,
            'total_fields': len(data),
            'filled_fields': sum(1 for v in data.values() if v is not None)
        }

    def _score_accuracy(
        self,
        data: Dict[str, Any],
        validation_result: Optional[ValidationResult] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Score data accuracy based on validation."""
        if validation_result is None:
            return 1.0, {'no_validation': True}

        total_issues = len(validation_result.issues)
        if total_issues == 0:
            return 1.0, {'no_issues': True}

        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.ERROR: 0.8,
            ValidationSeverity.WARNING: 0.3,
            ValidationSeverity.INFO: 0.1
        }

        weighted_issues = sum(
            severity_weights.get(issue.severity, 0.5)
            for issue in validation_result.issues
        )

        # Score decreases with weighted issues
        max_possible = total_issues * 1.0  # All critical
        score = max(0.0, 1.0 - (weighted_issues / max_possible))

        return score, {
            'total_issues': total_issues,
            'weighted_issues': weighted_issues,
            'issues_by_severity': {
                sev.value: len(validation_result.get_issues_by_severity(sev))
                for sev in ValidationSeverity
            }
        }

    def _score_consistency(
        self,
        data: Dict[str, Any],
        reference_data: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Score data consistency."""
        if not reference_data:
            return 1.0, {'no_reference': True}

        # Check for consistent field types and formats
        consistency_issues = 0
        total_checks = 0

        for ref_item in reference_data[:10]:  # Sample first 10
            for key, value in data.items():
                if key in ref_item:
                    total_checks += 1
                    ref_value = ref_item[key]

                    # Type consistency
                    if type(value) != type(ref_value):
                        consistency_issues += 1
                        continue

                    # Format consistency for strings
                    if isinstance(value, str) and isinstance(ref_value, str):
                        # Check common patterns (emails, phones, dates)
                        if '@' in value and '@' not in ref_value:
                            consistency_issues += 1
                        elif re.match(r'^\d{3}-\d{3}-\d{4}$', value) and \
                             not re.match(r'^\d{3}-\d{3}-\d{4}$', ref_value):
                            consistency_issues += 1

        if total_checks == 0:
            return 1.0, {'no_comparable_fields': True}

        score = 1.0 - (consistency_issues / total_checks)
        return score, {
            'consistency_issues': consistency_issues,
            'total_checks': total_checks
        }

    def _score_timeliness(
        self,
        data: Dict[str, Any],
        timestamp_field: Optional[str] = None,
        max_age_days: Optional[int] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Score data timeliness."""
        if not timestamp_field or timestamp_field not in data:
            return 1.0, {'no_timestamp': True}

        try:
            timestamp = data[timestamp_field]
            if isinstance(timestamp, str):
                # Try common date formats
                for fmt in [
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d'
                ]:
                    try:
                        timestamp = datetime.strptime(timestamp, fmt)
                        break
                    except ValueError:
                        continue

            if isinstance(timestamp, datetime):
                now = datetime.now(timezone.utc)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)

                age_days = (now - timestamp).days

                if max_age_days is None:
                    # Default scoring: 1.0 for today, decreasing over time
                    score = max(0.0, 1.0 - (age_days / 365))
                else:
                    score = 1.0 if age_days <= max_age_days else 0.0

                return score, {
                    'age_days': age_days,
                    'timestamp': timestamp.isoformat(),
                    'max_age_days': max_age_days
                }

        except Exception as e:
            logger.debug(f"Timeliness scoring failed: {e}")

        return 0.5, {'timestamp_parse_error': True}

    def _score_uniqueness(
        self,
        data: Dict[str, Any],
        duplicate_result: Optional[DuplicateResult] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Score data uniqueness."""
        if duplicate_result is None:
            return 1.0, {'no_duplicate_check': True}

        if duplicate_result.is_duplicate:
            # Score decreases with similarity
            score = 1.0 - duplicate_result.similarity_score
        else:
            score = 1.0

        return score, {
            'is_duplicate': duplicate_result.is_duplicate,
            'similarity_score': duplicate_result.similarity_score
        }

    def _score_validity(
        self,
        data: Dict[str, Any],
        validation_result: Optional[ValidationResult] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Score data validity."""
        if validation_result is None:
            return 1.0, {'no_validation': True}

        if validation_result.is_valid:
            return 1.0, {'valid': True}
        else:
            # Partial score based on error count
            error_count = validation_result.error_count
            if error_count == 0:
                return 0.8, {'warnings_only': True}
            else:
                return max(0.0, 0.5 - (error_count * 0.1)), {
                    'error_count': error_count
                }

    async def calculate_score(
        self,
        data: Dict[str, Any],
        validation_result: Optional[ValidationResult] = None,
        duplicate_result: Optional[DuplicateResult] = None,
        enrichment_results: Optional[List[EnrichmentResult]] = None,
        **kwargs
    ) -> DataQualityScore:
        """Calculate comprehensive quality score."""
        metric_scores = {}
        factors = []

        # Calculate individual metric scores
        completeness_score, completeness_factors = self._score_completeness(
            data,
            kwargs.get('required_fields')
        )
        metric_scores[QualityMetric.COMPLETENESS] = completeness_score
        factors.append({
            'metric': QualityMetric.COMPLETENESS,
            'score': completeness_score,
            **completeness_factors
        })

        accuracy_score, accuracy_factors = self._score_accuracy(
            data,
            validation_result
        )
        metric_scores[QualityMetric.ACCURACY] = accuracy_score
        factors.append({
            'metric': QualityMetric.ACCURACY,
            'score': accuracy_score,
            **accuracy_factors
        })

        consistency_score, consistency_factors = self._score_consistency(
            data,
            kwargs.get('reference_data')
        )
        metric_scores[QualityMetric.CONSISTENCY] = consistency_score
        factors.append({
            'metric': QualityMetric.CONSISTENCY,
            'score': consistency_score,
            **consistency_factors
        })

        timeliness_score, timeliness_factors = self._score_timeliness(
            data,
            kwargs.get('timestamp_field'),
            kwargs.get('max_age_days')
        )
        metric_scores[QualityMetric.TIMELINESS] = timeliness_score
        factors.append({
            'metric': QualityMetric.TIMELINESS,
            'score': timeliness_score,
            **timeliness_factors
        })

        uniqueness_score, uniqueness_factors = self._score_uniqueness(
            data,
            duplicate_result
        )
        metric_scores[QualityMetric.UNIQUENESS] = uniqueness_score
        factors.append({
            'metric': QualityMetric.UNIQUENESS,
            'score': uniqueness_score,
            **uniqueness_factors
        })

        validity_score, validity_factors = self._score_validity(
            data,
            validation_result
        )
        metric_scores[QualityMetric.VALIDITY] = validity_score
        factors.append({
            'metric': QualityMetric.VALIDITY,
            'score': validity_score,
            **validity_factors
        })

        # Apply custom scorers
        for name, scorer_fn in self.custom_scorers.items():
            try:
                custom_score = await scorer_fn(data, **kwargs)
                metric_scores[QualityMetric(name)] = custom_score
                factors.append({
                    'metric': name,
                    'score': custom_score,
                    'custom': True
                })
            except Exception as e:
                logger.warning(f"Custom scorer {name} failed: {e}")

        # Calculate weighted score
        weighted_score = sum(
            metric_scores.get(metric, 0.0) * weight
            for metric, weight in self.weights.items()
        )

        # Calculate overall score (could be same as weighted or different)
        overall_score = weighted_score

        # Boost score for successful enrichments
        if enrichment_results:
            successful_enrichments = sum(
                1 for r in enrichment_results if r.success
            )
            if successful_enrichments > 0:
                enrichment_boost = min(0.1, successful_enrichments * 0.02)
                overall_score = min(1.0, overall_score + enrichment_boost)

        # Calculate confidence based on data availability
        confidence = 1.0
        if not validation_result:
            confidence *= 0.9
        if not duplicate_result:
            confidence *= 0.95

        return DataQualityScore(
            overall_score=overall_score,
            metric_scores=metric_scores,
            weighted_score=weighted_score,
            confidence=confidence,
            factors=factors
        )


class QualityPipeline:
    """Main pipeline orchestrating validation, deduplication, enrichment, and scoring."""

    def __init__(
        self,
        validator: Optional[BaseValidator] = None,
        duplicate_detector: Optional[DuplicateDetector] = None,
        enrichment_sources: Optional[List[BaseEnrichmentSource]] = None,
        quality_scorer: Optional[QualityScorer] = None,
        **kwargs
    ):
        self.validator = validator or SchemaValidator()
        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        self.enrichment_sources = enrichment_sources or []
        self.quality_scorer = quality_scorer or QualityScorer()
        self.config = kwargs

        # Pipeline state
        self._processed_count = 0
        self._error_count = 0
        self._start_time: Optional[datetime] = None

    async def process_item(
        self,
        item: Dict[str, Any],
        schema: Optional[Union[Type[BaseModel], Dict[str, Any], str]] = None,
        enrich: bool = True,
        check_duplicates: bool = True,
        calculate_score: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single item through the quality pipeline."""
        result = {
            'original': item.copy(),
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'pipeline_version': '1.0.0'
        }

        try:
            # Step 1: Validation
            validation_result = None
            if schema:
                validation_result = await self.validator.validate(
                    item,
                    schema=schema,
                    **kwargs
                )
                result['validation'] = validation_result.to_dict()

            # Step 2: Duplicate Detection
            duplicate_result = None
            if check_duplicates:
                # For single item, we need reference data
                reference_data = kwargs.get('reference_data', [])
                if reference_data:
                    duplicate_result = await self.duplicate_detector.is_duplicate_of(
                        item,
                        reference_data[0]  # Compare against first reference
                    )
                    result['duplicate_detection'] = duplicate_result.dict()

            # Step 3: Enrichment
            enrichment_results = []
            enriched_item = item.copy()

            if enrich and self.enrichment_sources:
                for source in self.enrichment_sources:
                    try:
                        enriched_item, enrichment_result = await source.enrich(
                            enriched_item,
                            **kwargs
                        )
                        enrichment_results.append(enrichment_result)
                    except Exception as e:
                        logger.error(
                            f"Enrichment source {source.name} failed: {e}"
                        )
                        enrichment_results.append(EnrichmentResult(
                            success=False,
                            source=source.name,
                            error=str(e)
                        ))

                result['enrichment'] = [r.dict() for r in enrichment_results]
                result['enriched_data'] = enriched_item

            # Step 4: Quality Scoring
            quality_score = None
            if calculate_score:
                quality_score = await self.quality_scorer.calculate_score(
                    enriched_item,
                    validation_result=validation_result,
                    duplicate_result=duplicate_result,
                    enrichment_results=enrichment_results,
                    **kwargs
                )
                result['quality_score'] = quality_score.dict()

            # Update pipeline metrics
            self._processed_count += 1

            # Add final data (enriched if available, otherwise original)
            result['final_data'] = enriched_item if enrich else item

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Pipeline processing failed: {e}")
            result['error'] = str(e)
            result['final_data'] = item
            return result

    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        schema: Optional[Union[Type[BaseModel], Dict[str, Any], str]] = None,
        enrich: bool = True,
        check_duplicates: bool = True,
        calculate_score: bool = True,
        batch_duplicate_check: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process a batch of items through the quality pipeline."""
        self._start_time = datetime.now()
        results = []

        # Batch duplicate detection if enabled
        batch_duplicate_results = None
        if check_duplicates and batch_duplicate_check:
            try:
                batch_duplicate_results = await self.duplicate_detector.detect_duplicates(
                    items,
                    id_field=kwargs.get('id_field')
                )
            except Exception as e:
                logger.error(f"Batch duplicate detection failed: {e}")

        # Process items concurrently
        tasks = []
        for i, item in enumerate(items):
            # Prepare item-specific kwargs
            item_kwargs = kwargs.copy()

            # Add batch duplicate result if available
            if batch_duplicate_results and i < len(batch_duplicate_results):
                item_kwargs['batch_duplicate_result'] = batch_duplicate_results[i]

            # Add reference data for single-item duplicate check
            if not batch_duplicate_check and i > 0:
                item_kwargs['reference_data'] = [items[0]]  # Compare against first

            task = self.process_item(
                item,
                schema=schema,
                enrich=enrich,
                check_duplicates=check_duplicates and not batch_duplicate_check,
                calculate_score=calculate_score,
                **item_kwargs
            )
            tasks.append(task)

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'original': items[i],
                    'error': str(result),
                    'final_data': items[i],
                    'processed_at': datetime.now(timezone.utc).isoformat()
                })
                self._error_count += 1
            else:
                # Add batch duplicate result if available
                if batch_duplicate_results and i < len(batch_duplicate_results):
                    result['batch_duplicate_detection'] = batch_duplicate_results[i].dict()
                processed_results.append(result)

        # Log batch statistics
        processing_time = (datetime.now() - self._start_time).total_seconds()
        logger.info(
            f"Batch processing complete: {len(items)} items, "
            f"{self._error_count} errors, "
            f"{processing_time:.2f}s"
        )

        return processed_results

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        return {
            'processed_count': self._processed_count,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(1, self._processed_count),
            'uptime_seconds': (
                datetime.now() - self._start_time
            ).total_seconds() if self._start_time else 0
        }

    async def close(self):
        """Clean up pipeline resources."""
        for source in self.enrichment_sources:
            await source.close()


# Factory functions for common configurations
def create_schema_validator() -> SchemaValidator:
    """Create a schema validator with common schemas."""
    validator = SchemaValidator()

    # Register common schemas
    product_schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string', 'minLength': 1, 'maxLength': 500},
            'price': {'type': 'number', 'minimum': 0},
            'currency': {'type': 'string', 'pattern': '^[A-Z]{3}$'},
            'url': {'type': 'string', 'pattern': '^https?://'},
            'description': {'type': 'string'},
            'sku': {'type': 'string'},
            'brand': {'type': 'string'},
            'category': {'type': 'string'},
            'in_stock': {'type': 'boolean'},
            'rating': {'type': 'number', 'minimum': 0, 'maximum': 5},
            'review_count': {'type': 'integer', 'minimum': 0},
            'images': {'type': 'array', 'items': {'type': 'string'}},
            'specifications': {'type': 'object'},
            'scraped_at': {'type': 'string', 'format': 'date-time'},
        },
        'required': ['name', 'price', 'url']
    }

    article_schema = {
        'type': 'object',
        'properties': {
            'title': {'type': 'string', 'minLength': 1},
            'content': {'type': 'string', 'minLength': 10},
            'author': {'type': 'string'},
            'published_at': {'type': 'string', 'format': 'date-time'},
            'url': {'type': 'string', 'pattern': '^https?://'},
            'source': {'type': 'string'},
            'category': {'type': 'string'},
            'tags': {'type': 'array', 'items': {'type': 'string'}},
            'word_count': {'type': 'integer', 'minimum': 0},
            'reading_time_minutes': {'type': 'number', 'minimum': 0},
        },
        'required': ['title', 'content', 'url']
    }

    person_schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string', 'minLength': 1},
            'email': {'type': 'string', 'pattern': '^[^@]+@[^@]+\\.[^@]+$'},
            'phone': {'type': 'string'},
            'company': {'type': 'string'},
            'job_title': {'type': 'string'},
            'location': {'type': 'string'},
            'linkedin_url': {'type': 'string'},
            'twitter_handle': {'type': 'string'},
            'bio': {'type': 'string'},
            'skills': {'type': 'array', 'items': {'type': 'string'}},
        },
        'required': ['name']
    }

    validator.register_schema('product', product_schema)
    validator.register_schema('article', article_schema)
    validator.register_schema('person', person_schema)

    return validator


def create_quality_pipeline(
    validation_schema: Optional[str] = None,
    duplicate_method: DuplicateMethod = DuplicateMethod.SIMHASH,
    duplicate_threshold: float = 0.85,
    enrichment_sources: Optional[List[BaseEnrichmentSource]] = None,
    **kwargs
) -> QualityPipeline:
    """Create a quality pipeline with common configuration."""
    validator = create_schema_validator()
    duplicate_detector = DuplicateDetector(
        method=duplicate_method,
        threshold=duplicate_threshold
    )
    scorer = QualityScorer()

    pipeline = QualityPipeline(
        validator=validator,
        duplicate_detector=duplicate_detector,
        enrichment_sources=enrichment_sources or [],
        quality_scorer=scorer,
        **kwargs
    )

    return pipeline


# Integration with axiom's existing components
class axiomQualityMixin:
    """Mixin to add quality pipeline capabilities to axiom components."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quality_pipeline: Optional[QualityPipeline] = None
        self._quality_config = kwargs.get('quality_config', {})

    def init_quality_pipeline(self, **kwargs) -> None:
        """Initialize the quality pipeline."""
        config = {**self._quality_config, **kwargs}
        self._quality_pipeline = create_quality_pipeline(**config)

    async def validate_and_score(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Validate and score data using the quality pipeline."""
        if self._quality_pipeline is None:
            self.init_quality_pipeline()

        if isinstance(data, list):
            return await self._quality_pipeline.process_batch(data, **kwargs)
        else:
            return await self._quality_pipeline.process_item(data, **kwargs)

    @property
    def quality_pipeline(self) -> Optional[QualityPipeline]:
        """Get the quality pipeline instance."""
        return self._quality_pipeline


# Example usage and testing
async def example_usage():
    """Example demonstrating the quality pipeline usage."""
    # Create sample data
    products = [
        {
            'name': 'Example Product',
            'price': 29.99,
            'currency': 'USD',
            'url': 'https://example.com/product/123',
            'description': 'A sample product description',
            'sku': 'PROD-123',
            'brand': 'Example Brand',
            'category': 'Electronics',
            'in_stock': True,
            'rating': 4.5,
            'review_count': 128,
            'scraped_at': datetime.now(timezone.utc).isoformat()
        },
        {
            'name': 'Another Product',
            'price': 49.99,
            'currency': 'USD',
            'url': 'https://example.com/product/456',
            'description': 'Another sample product',
            'sku': 'PROD-456',
            'brand': 'Another Brand',
            'category': 'Home & Garden',
            'in_stock': False,
            'rating': 3.8,
            'review_count': 42,
            'scraped_at': datetime.now(timezone.utc).isoformat()
        },
        {
            # Missing required fields
            'name': '',  # Empty name (will fail validation)
            'price': -10,  # Negative price (will fail validation)
            'url': 'invalid-url',  # Invalid URL format
            'currency': 'US',  # Invalid currency code
        }
    ]

    # Create quality pipeline
    pipeline = create_quality_pipeline(
        validation_schema='product',
        duplicate_method=DuplicateMethod.FUZZY,
        duplicate_threshold=0.9
    )

    try:
        # Process batch
        results = await pipeline.process_batch(
            products,
            schema='product',
            enrich=False,  # No enrichment sources configured
            check_duplicates=True,
            calculate_score=True
        )

        # Print results
        for i, result in enumerate(results):
            print(f"\n{'='*60}")
            print(f"Product {i+1}:")
            print(f"Valid: {result.get('validation', {}).get('is_valid', 'N/A')}")
            print(f"Quality Score: {result.get('quality_score', {}).get('overall_score', 'N/A'):.2%}")

            if 'validation' in result:
                issues = result['validation'].get('issues', [])
                if issues:
                    print(f"Validation Issues: {len(issues)}")
                    for issue in issues[:3]:  # Show first 3
                        print(f"  - {issue.get('field', 'N/A')}: {issue.get('message')}")

            if 'duplicate_detection' in result:
                dup = result['duplicate_detection']
                print(f"Duplicate: {dup.get('is_duplicate')} (score: {dup.get('similarity_score', 0):.2f})")

        # Get pipeline stats
        stats = pipeline.get_stats()
        print(f"\nPipeline Stats:")
        print(f"Processed: {stats['processed_count']}")
        print(f"Errors: {stats['error_count']}")
        print(f"Error Rate: {stats['error_rate']:.2%}")

    finally:
        await pipeline.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())