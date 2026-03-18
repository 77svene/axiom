"""Data Quality & Validation Pipeline for axiom.

This module provides automated data validation, duplicate detection, data enrichment,
and quality scoring for scraped data. It integrates with axiom's existing
architecture while adding robust data quality assurance capabilities.

Key Features:
- JSON Schema validation using Pydantic models
- Near-duplicate detection with SimHash/MinHash algorithms
- Data enrichment from configurable external sources
- Comprehensive quality scoring system
- Async-first design for high-performance processing

Example:
    >>> from axiom.quality.enrichment import DataQualityPipeline
    >>> pipeline = DataQualityPipeline()
    >>> validated_data = await pipeline.process(raw_data)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import aiohttp
from pydantic import BaseModel, Field, validator

from axiom.core.ai import AIEnrichmentProvider
from axiom.core.custom_types import JsonDict
from axiom.core.mixins import LoggerMixin
from axiom.core.storage import StorageManager
from axiom.core.utils._utils import generate_hash, retry_async

logger = logging.getLogger(__name__)


class QualityDimension(str, Enum):
    """Quality dimensions for scoring."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"


class DuplicateAlgorithm(str, Enum):
    """Algorithms for duplicate detection."""
    SIMHASH = "simhash"
    MINHASH = "minhash"
    COMBINED = "combined"


@dataclass
class QualityScore:
    """Comprehensive quality score for data records."""
    overall: float = 0.0
    dimensions: Dict[QualityDimension, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def calculate_overall(self, weights: Optional[Dict[QualityDimension, float]] = None) -> None:
        """Calculate weighted overall score from dimension scores."""
        if not self.dimensions:
            self.overall = 0.0
            return

        if weights is None:
            # Default weights emphasizing completeness and validity
            weights = {
                QualityDimension.COMPLETENESS: 0.25,
                QualityDimension.ACCURACY: 0.20,
                QualityDimension.CONSISTENCY: 0.15,
                QualityDimension.TIMELINESS: 0.10,
                QualityDimension.UNIQUENESS: 0.15,
                QualityDimension.VALIDITY: 0.15,
            }

        total_weight = sum(weights.get(dim, 0) for dim in self.dimensions.keys())
        if total_weight == 0:
            self.overall = 0.0
            return

        weighted_sum = sum(
            score * weights.get(dim, 0)
            for dim, score in self.dimensions.items()
        )
        self.overall = weighted_sum / total_weight


class DataSchema(BaseModel):
    """Base schema for data validation. Extend this for specific data types."""
    url: str = Field(..., description="Source URL of the data")
    timestamp: datetime = Field(default_factory=datetime.now)
    content_hash: Optional[str] = Field(None, description="Hash of the original content")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('url')
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class DuplicateDetector(LoggerMixin):
    """Detects near-duplicate content using various algorithms."""

    def __init__(
        self,
        algorithm: DuplicateAlgorithm = DuplicateAlgorithm.SIMHASH,
        threshold: float = 0.85,
        shingle_size: int = 3
    ):
        self.algorithm = algorithm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self._seen_hashes: Set[str] = set()
        self._minhash_permutations = 128

    def _text_shingles(self, text: str) -> Set[str]:
        """Create shingles from text for MinHash."""
        words = re.findall(r'\w+', text.lower())
        return {
            ' '.join(words[i:i + self.shingle_size])
            for i in range(len(words) - self.shingle_size + 1)
        }

    def _compute_simhash(self, text: str) -> int:
        """Compute SimHash for text content."""
        features = self._text_shingles(text)
        if not features:
            return 0

        # Initialize hash vector
        vector = [0] * 64

        for feature in features:
            # Hash the feature
            h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
            # Update vector
            for i in range(64):
                bit = (h >> i) & 1
                if bit:
                    vector[i] += 1
                else:
                    vector[i] -= 1

        # Compute final hash
        simhash = 0
        for i in range(64):
            if vector[i] > 0:
                simhash |= (1 << i)

        return simhash

    def _compute_minhash(self, text: str) -> List[int]:
        """Compute MinHash signature for text content."""
        shingles = self._text_shingles(text)
        if not shingles:
            return [0] * self._minhash_permutations

        signature = []
        for i in range(self._minhash_permutations):
            min_hash = float('inf')
            for shingle in shingles:
                # Create different hash functions by salting
                h = int(hashlib.md5(
                    f"{i}:{shingle}".encode()
                ).hexdigest(), 16)
                min_hash = min(min_hash, h)
            signature.append(min_hash)

        return signature

    def _simhash_similarity(self, hash1: int, hash2: int) -> float:
        """Calculate similarity between two SimHashes."""
        if hash1 == hash2:
            return 1.0

        # Hamming distance
        xor = hash1 ^ hash2
        distance = bin(xor).count('1')
        return 1.0 - (distance / 64.0)

    def _minhash_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Calculate Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            return 0.0

        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def is_duplicate(self, content: str, content_hash: Optional[str] = None) -> Tuple[bool, float]:
        """Check if content is a near-duplicate of previously seen content."""
        if content_hash is None:
            content_hash = generate_hash(content)

        # Check exact duplicates first
        if content_hash in self._seen_hashes:
            return True, 1.0

        # Compute hash based on algorithm
        if self.algorithm == DuplicateAlgorithm.SIMHASH:
            current_hash = self._compute_simhash(content)
            # Compare with all previous hashes (in production, use LSH for scalability)
            for seen_hash in self._seen_hashes:
                if len(seen_hash) == 64:  # SimHash stored as hex
                    seen_int = int(seen_hash, 16)
                    similarity = self._simhash_similarity(current_hash, seen_int)
                    if similarity >= self.threshold:
                        return True, similarity

        elif self.algorithm == DuplicateAlgorithm.MINHASH:
            current_sig = self._compute_minhash(content)
            # Compare signatures (in production, use MinHash LSH)
            for seen_hash in self._seen_hashes:
                if seen_hash.startswith('minhash:'):
                    stored_sig = json.loads(seen_hash[8:])
                    similarity = self._minhash_similarity(current_sig, stored_sig)
                    if similarity >= self.threshold:
                        return True, similarity

        # Store for future comparisons
        self._seen_hashes.add(content_hash)
        return False, 0.0

    def clear(self) -> None:
        """Clear the seen hashes cache."""
        self._seen_hashes.clear()


class DataEnricher(LoggerMixin):
    """Enriches data from multiple external sources."""

    def __init__(
        self,
        ai_provider: Optional[AIEnrichmentProvider] = None,
        api_keys: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.ai_provider = ai_provider or AIEnrichmentProvider()
        self.api_keys = api_keys or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        self._enrichment_cache: Dict[str, Any] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def enrich_from_google_knowledge_graph(
        self,
        query: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Enrich data using Google Knowledge Graph API."""
        if 'google_api_key' not in self.api_keys:
            self.logger.warning("Google API key not configured")
            return None

        cache_key = f"google_kg:{query}"
        if cache_key in self._enrichment_cache:
            return self._enrichment_cache[cache_key]

        session = await self._get_session()
        params = {
            'query': query,
            'key': self.api_keys['google_api_key'],
            'limit': 1,
            'indent': True,
            **kwargs
        }

        try:
            async with session.get(
                'https://kgsearch.googleapis.com/v1/entities:search',
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    result = self._parse_google_kg_response(data)
                    self._enrichment_cache[cache_key] = result
                    return result
        except Exception as e:
            self.logger.error(f"Google KG enrichment failed: {e}")

        return None

    def _parse_google_kg_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Google Knowledge Graph API response."""
        result = {}
        if 'itemListElement' in data and data['itemListElement']:
            element = data['itemListElement'][0]
            if 'result' in element:
                entity = element['result']
                result = {
                    'name': entity.get('name'),
                    'description': entity.get('description'),
                    'detailed_description': entity.get('detailedDescription', {}).get('articleBody'),
                    'url': entity.get('url'),
                    'image': entity.get('image', {}).get('contentUrl'),
                    'types': entity.get('@type', []),
                    'score': element.get('resultScore', 0)
                }
        return result

    async def enrich_from_wikipedia(self, title: str) -> Optional[Dict[str, Any]]:
        """Enrich data using Wikipedia API."""
        cache_key = f"wikipedia:{title}"
        if cache_key in self._enrichment_cache:
            return self._enrichment_cache[cache_key]

        session = await self._get_session()
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts|pageimages|info',
            'exintro': True,
            'explaintext': True,
            'pithumbsize': 300,
            'inprop': 'url',
            'format': 'json'
        }

        try:
            async with session.get(
                'https://en.wikipedia.org/w/api.php',
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    result = self._parse_wikipedia_response(data)
                    self._enrichment_cache[cache_key] = result
                    return result
        except Exception as e:
            self.logger.error(f"Wikipedia enrichment failed: {e}")

        return None

    def _parse_wikipedia_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Wikipedia API response."""
        result = {}
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            if page_id != '-1':  # Page exists
                result = {
                    'title': page_data.get('title'),
                    'extract': page_data.get('extract'),
                    'thumbnail': page_data.get('thumbnail', {}).get('source'),
                    'page_url': page_data.get('fullurl'),
                    'page_id': page_id
                }
                break
        return result

    async def enrich_with_ai(self, data: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Enrich data using AI capabilities."""
        try:
            return await self.ai_provider.enrich_data(data, context)
        except Exception as e:
            self.logger.error(f"AI enrichment failed: {e}")
            return data

    async def enrich(
        self,
        data: Dict[str, Any],
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Enrich data from multiple sources."""
        if sources is None:
            sources = ['ai']

        enriched = data.copy()

        for source in sources:
            try:
                if source == 'google_kg' and 'name' in data:
                    kg_data = await self.enrich_from_google_knowledge_graph(data['name'])
                    if kg_data:
                        enriched['knowledge_graph'] = kg_data

                elif source == 'wikipedia' and 'name' in data:
                    wiki_data = await self.enrich_from_wikipedia(data['name'])
                    if wiki_data:
                        enriched['wikipedia'] = wiki_data

                elif source == 'ai':
                    enriched = await self.enrich_with_ai(enriched)

            except Exception as e:
                self.logger.error(f"Enrichment from {source} failed: {e}")

        return enriched

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()


class DataQualityPipeline(LoggerMixin):
    """Main pipeline for data quality processing."""

    def __init__(
        self,
        schema: Optional[Type[BaseModel]] = None,
        duplicate_detector: Optional[DuplicateDetector] = None,
        enricher: Optional[DataEnricher] = None,
        storage_manager: Optional[StorageManager] = None,
        quality_weights: Optional[Dict[QualityDimension, float]] = None
    ):
        self.schema = schema or DataSchema
        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        self.enricher = enricher or DataEnricher()
        self.storage_manager = storage_manager
        self.quality_weights = quality_weights
        self._processing_stats: Dict[str, int] = defaultdict(int)

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate data against schema."""
        errors = []
        validated_data = {}

        try:
            # Create model instance for validation
            model_instance = self.schema(**data)
            validated_data = model_instance.dict()
            return True, validated_data, errors

        except Exception as e:
            errors.append(str(e))
            # Try to extract partial data
            for field_name in self.schema.__fields__.keys():
                if field_name in data:
                    validated_data[field_name] = data[field_name]
            return False, validated_data, errors

    def calculate_quality_score(
        self,
        data: Dict[str, Any],
        is_duplicate: bool = False,
        duplicate_similarity: float = 0.0,
        validation_errors: Optional[List[str]] = None
    ) -> QualityScore:
        """Calculate comprehensive quality score for data."""
        score = QualityScore()
        validation_errors = validation_errors or []

        # Completeness: percentage of non-empty fields
        if data:
            non_empty = sum(1 for v in data.values() if v is not None and v != "")
            score.dimensions[QualityDimension.COMPLETENESS] = non_empty / len(data)
        else:
            score.dimensions[QualityDimension.COMPLETENESS] = 0.0

        # Validity: based on validation errors
        if validation_errors:
            score.dimensions[QualityDimension.VALIDITY] = max(0, 1.0 - (len(validation_errors) * 0.2))
        else:
            score.dimensions[QualityDimension.VALIDITY] = 1.0

        # Uniqueness: penalize duplicates
        if is_duplicate:
            score.dimensions[QualityDimension.UNIQUENESS] = 1.0 - duplicate_similarity
        else:
            score.dimensions[QualityDimension.UNIQUENESS] = 1.0

        # Accuracy: heuristic based on data consistency
        accuracy_score = self._calculate_accuracy_heuristic(data)
        score.dimensions[QualityDimension.ACCURACY] = accuracy_score

        # Consistency: check for internal consistency
        consistency_score = self._calculate_consistency_heuristic(data)
        score.dimensions[QualityDimension.CONSISTENCY] = consistency_score

        # Timeliness: based on timestamp freshness
        timeliness_score = self._calculate_timeliness_score(data)
        score.dimensions[QualityDimension.TIMELINESS] = timeliness_score

        # Calculate overall score
        score.calculate_overall(self.quality_weights)

        # Add details
        score.details = {
            'validation_errors': validation_errors,
            'is_duplicate': is_duplicate,
            'duplicate_similarity': duplicate_similarity,
            'field_count': len(data),
            'non_empty_fields': sum(1 for v in data.values() if v is not None and v != "")
        }

        return score

    def _calculate_accuracy_heuristic(self, data: Dict[str, Any]) -> float:
        """Heuristic for data accuracy based on common patterns."""
        score = 1.0

        # Check for obviously incorrect patterns
        for key, value in data.items():
            if isinstance(value, str):
                # Check for placeholder text
                if any(phrase in value.lower() for phrase in ['lorem ipsum', 'test', 'example', 'placeholder']):
                    score -= 0.1

                # Check for HTML tags in non-HTML fields
                if '<' in value and '>' in value and 'html' not in key.lower():
                    score -= 0.05

            elif isinstance(value, (int, float)):
                # Check for unreasonable numbers
                if abs(value) > 1e10:  # Very large numbers
                    score -= 0.05

        return max(0.0, score)

    def _calculate_consistency_heuristic(self, data: Dict[str, Any]) -> float:
        """Heuristic for internal data consistency."""
        score = 1.0

        # Check for type consistency in lists
        for key, value in data.items():
            if isinstance(value, list) and value:
                # Check if all items are same type
                first_type = type(value[0])
                if not all(isinstance(item, first_type) for item in value):
                    score -= 0.1

        # Check for date consistency
        date_fields = [k for k in data.keys() if 'date' in k.lower() or 'time' in k.lower()]
        if len(date_fields) > 1:
            # Simple check: all date fields should be parseable
            for field in date_fields:
                if isinstance(data[field], str):
                    try:
                        datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        score -= 0.1

        return max(0.0, score)

    def _calculate_timeliness_score(self, data: Dict[str, Any]) -> float:
        """Calculate timeliness score based on data freshness."""
        # Default to 0.5 if no timestamp
        if 'timestamp' not in data:
            return 0.5

        try:
            if isinstance(data['timestamp'], str):
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            elif isinstance(data['timestamp'], datetime):
                timestamp = data['timestamp']
            else:
                return 0.5

            # Calculate age in days
            age_days = (datetime.now() - timestamp).days

            # Score decreases with age
            if age_days <= 1:
                return 1.0
            elif age_days <= 7:
                return 0.9
            elif age_days <= 30:
                return 0.7
            elif age_days <= 365:
                return 0.5
            else:
                return 0.3

        except (ValueError, AttributeError):
            return 0.5

    async def process(
        self,
        data: Dict[str, Any],
        enrich: bool = True,
        enrichment_sources: Optional[List[str]] = None,
        validate: bool = True,
        check_duplicates: bool = True
    ) -> Dict[str, Any]:
        """Process data through the quality pipeline."""
        self._processing_stats['total_processed'] += 1
        result = {
            'original': data,
            'processed_at': datetime.now().isoformat(),
            'pipeline_version': '1.0.0'
        }

        # Step 1: Validation
        validation_errors = []
        validated_data = data.copy()

        if validate:
            is_valid, validated_data, validation_errors = self.validate(data)
            result['validation'] = {
                'is_valid': is_valid,
                'errors': validation_errors
            }
            if not is_valid:
                self._processing_stats['validation_failures'] += 1

        # Step 2: Duplicate detection
        is_duplicate = False
        duplicate_similarity = 0.0

        if check_duplicates:
            content_hash = data.get('content_hash') or generate_hash(json.dumps(data, sort_keys=True))
            content_text = json.dumps(data, sort_keys=True)

            is_duplicate, duplicate_similarity = self.duplicate_detector.is_duplicate(
                content_text, content_hash
            )

            result['duplicate_detection'] = {
                'is_duplicate': is_duplicate,
                'similarity_score': duplicate_similarity,
                'content_hash': content_hash
            }

            if is_duplicate:
                self._processing_stats['duplicates_detected'] += 1

        # Step 3: Enrichment
        enriched_data = validated_data.copy()

        if enrich and not is_duplicate:  # Don't enrich duplicates
            try:
                enriched_data = await self.enricher.enrich(
                    validated_data,
                    sources=enrichment_sources
                )
                result['enrichment'] = {
                    'sources_used': enrichment_sources or ['ai'],
                    'success': True
                }
                self._processing_stats['enrichment_success'] += 1
            except Exception as e:
                self.logger.error(f"Enrichment failed: {e}")
                result['enrichment'] = {
                    'sources_used': [],
                    'success': False,
                    'error': str(e)
                }
                self._processing_stats['enrichment_failures'] += 1

        # Step 4: Quality scoring
        quality_score = self.calculate_quality_score(
            enriched_data,
            is_duplicate,
            duplicate_similarity,
            validation_errors
        )

        result['quality_score'] = {
            'overall': quality_score.overall,
            'dimensions': {dim.value: score for dim, score in quality_score.dimensions.items()},
            'details': quality_score.details,
            'timestamp': quality_score.timestamp.isoformat()
        }

        # Add processed data to result
        result['data'] = enriched_data

        # Store if storage manager is available
        if self.storage_manager and quality_score.overall >= 0.7:  # Only store good quality data
            try:
                await self.storage_manager.store(enriched_data)
                self._processing_stats['records_stored'] += 1
            except Exception as e:
                self.logger.error(f"Storage failed: {e}")

        return result

    async def process_batch(
        self,
        data_batch: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a batch of data records asynchronously."""
        tasks = []
        for data in data_batch:
            task = asyncio.create_task(self.process(data, **kwargs))
            tasks.append(task)

        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                yield result
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                yield {
                    'error': str(e),
                    'processed_at': datetime.now().isoformat()
                }

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return dict(self._processing_stats)

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._processing_stats.clear()

    async def close(self) -> None:
        """Clean up resources."""
        await self.enricher.close()
        self.duplicate_detector.clear()


# Factory function for easy pipeline creation
def create_quality_pipeline(
    schema: Optional[Type[BaseModel]] = None,
    duplicate_algorithm: DuplicateAlgorithm = DuplicateAlgorithm.SIMHASH,
    duplicate_threshold: float = 0.85,
    ai_provider: Optional[AIEnrichmentProvider] = None,
    api_keys: Optional[Dict[str, str]] = None,
    storage_manager: Optional[StorageManager] = None,
    **kwargs
) -> DataQualityPipeline:
    """Create a configured data quality pipeline.

    Args:
        schema: Pydantic model for validation
        duplicate_algorithm: Algorithm for duplicate detection
        duplicate_threshold: Similarity threshold for duplicates
        ai_provider: AI provider for enrichment
        api_keys: API keys for external services
        storage_manager: Storage manager for persisting data
        **kwargs: Additional arguments for pipeline configuration

    Returns:
        Configured DataQualityPipeline instance
    """
    duplicate_detector = DuplicateDetector(
        algorithm=duplicate_algorithm,
        threshold=duplicate_threshold
    )

    enricher = DataEnricher(
        ai_provider=ai_provider,
        api_keys=api_keys
    )

    return DataQualityPipeline(
        schema=schema,
        duplicate_detector=duplicate_detector,
        enricher=enricher,
        storage_manager=storage_manager,
        **kwargs
    )


# Integration with axiom's existing architecture
class axiomQualityIntegration:
    """Integration layer with axiom's existing components."""

    @staticmethod
    def create_for_axiom(
        storage_path: Optional[str] = None,
        ai_model: str = "gpt-4",
        **kwargs
    ) -> DataQualityPipeline:
        """Create a quality pipeline configured for axiom.

        Args:
            storage_path: Path for data storage
            ai_model: AI model to use for enrichment
            **kwargs: Additional configuration

        Returns:
            Configured DataQualityPipeline
        """
        from axiom.core.storage import StorageManager

        storage_manager = None
        if storage_path:
            storage_manager = StorageManager(storage_path)

        ai_provider = AIEnrichmentProvider(model=ai_model)

        return create_quality_pipeline(
            ai_provider=ai_provider,
            storage_manager=storage_manager,
            **kwargs
        )


# Export public API
__all__ = [
    'DataQualityPipeline',
    'DuplicateDetector',
    'DataEnricher',
    'QualityScore',
    'QualityDimension',
    'DuplicateAlgorithm',
    'DataSchema',
    'create_quality_pipeline',
    'axiomQualityIntegration',
]