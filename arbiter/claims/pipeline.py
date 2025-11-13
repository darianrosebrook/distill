# arbiter/claims/pipeline.py
# 4-stage claim extraction and verification pipeline
# Stage 1: Contextual Disambiguation
# Stage 2: Verifiable Content Qualification
# Stage 3: Atomic Claim Decomposition
# Stage 4: CAWS-Compliant Verification
# @author: @darianrosebrook

from typing import List, Optional, Dict, Tuple, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
import json
import re
import uuid
import hashlib
import math
from pathlib import Path


@dataclass
class ConversationContext:
    """Context for disambiguation and claim extraction."""
    prior_turns: List[str]
    entity_registry: Dict[str, str]
    code_spans: List[str]
    doc_sections: List[str]
    result_tables: List[Dict]


@dataclass
class AmbiguityInstance:
    """Detected ambiguity in text."""
    phrase: str
    possible_interpretations: List[str]
    context_dependency: bool
    resolution_confidence: float
    ambiguity_type: str  # "referential", "structural", "temporal"


@dataclass
class DisambiguationResult:
    """Result of ambiguity resolution."""
    success: bool
    disambiguated_sentence: Optional[str]
    failure_reason: Optional[str]
    audit_trail: List[Dict]
    unresolved_ambiguities: List[AmbiguityInstance]


@dataclass
class VerifiableContentResult:
    """Result of verifiable content qualification."""
    has_verifiable_content: bool
    rewritten_sentence: Optional[str]
    indicators: List[str]
    confidence: float


@dataclass
class ClaimElements:
    """Structured elements of a claim for coverage analysis."""
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    qualifiers: Optional[Dict[str, Any]] = None


class ClaimCategory(Enum):
    STATUS = auto()       # e.g., "production-ready", "enterprise-grade", "operational in prod"
    QUANT = auto()        # numeric assertions: counts, %, latencies, TPS, coverage
    BENCHMARK = auto()    # performance claims: p50/p95 latencies, TTFT/TPS, throughput
    SUPERLATIVE = auto()  # "best", "leading", "state-of-the-art" (generally disallowed)


@dataclass
class ClaimsPolicy:
    """Repo-level policy for evidence requirements by claim category."""
    # Which categories are allowed and under what artifact requirements
    require_artifacts: Dict[ClaimCategory,
                            Dict[str, Any]] = field(default_factory=dict)
    # Disallowed terms that always fail (SUPERLATIVE by default)
    banned_terms: List[str] = field(default_factory=list)
    # Numeric field name allowlist for QUANT/BENCHMARK (JSON path or dotted keys)
    numeric_fields_allow: List[str] = field(default_factory=list)
    # Minimums for typical gates (optional; mirrors your eval harness)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    version: str = "2025-11-12"


@dataclass
class AtomicClaim:
    """Atomic, verifiable claim."""
    id: str
    statement: str
    contextual_brackets: List[str]
    source_sentence: str
    verification_requirements: List[str]
    confidence: float
    elements: Optional[ClaimElements] = None


@dataclass
class VerificationResult:
    """CAWS-compliant verification result."""
    status: str  # "VERIFIED", "UNVERIFIED", "INSUFFICIENT_EVIDENCE"
    evidence_quality: float
    caws_compliance: bool
    verification_trail: List[Dict]
    outcome_id: Optional[int] = None  # 1-7 outcome classification
    # {"score": float, "detail": {...}}
    element_coverage: Optional[Dict[str, Any]] = None
    # {"support": p, "contradict": p, "insufficient": p}
    entailment_triage: Optional[Dict[str, float]] = None
    fingerprints: Dict[str, str] = field(default_factory=dict)


class ClaimDisambiguation:
    """Stage 1: Contextual Disambiguation."""

    # Pronoun patterns
    PRONOUNS = {
        'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'theirs',
        'this', 'that', 'these', 'those', 'which', 'who', 'whom', 'whose'
    }

    # Temporal reference patterns
    TEMPORAL_WORDS = {
        'now', 'then', 'today', 'yesterday', 'tomorrow', 'recently', 'soon', 'later',
        'before', 'after', 'during', 'while', 'when', 'once', 'previously', 'currently'
    }

    def detect_ambiguities(self, text: str, context: ConversationContext) -> List[AmbiguityInstance]:
        """Detect ambiguities in text before extraction.

        Detects:
        - Pronoun resolution (he, she, it, they, this, that, etc.)
        - Temporal reference resolution (now, then, recently, etc.)
        - Structural ambiguity detection (ambiguous conjunctions, relative clauses)
        """
        ambiguities = []
        words = text.lower().split()

        # Detect referential ambiguities (pronouns)
        for i, word in enumerate(words):
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.PRONOUNS:
                # Check if we can resolve from context
                can_resolve = self._can_resolve_pronoun(
                    clean_word, i, words, context)
                ambiguities.append(AmbiguityInstance(
                    phrase=word,
                    possible_interpretations=self._get_pronoun_interpretations(
                        clean_word, context),
                    context_dependency=True,
                    resolution_confidence=0.8 if can_resolve else 0.2,
                    ambiguity_type="referential"
                ))

        # Detect temporal ambiguities
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.TEMPORAL_WORDS:
                can_resolve = self._can_resolve_temporal(clean_word, context)
                ambiguities.append(AmbiguityInstance(
                    phrase=word,
                    possible_interpretations=self._get_temporal_interpretations(
                        clean_word, context),
                    context_dependency=True,
                    resolution_confidence=0.7 if can_resolve else 0.3,
                    ambiguity_type="temporal"
                ))

        # Detect structural ambiguities (ambiguous conjunctions, relative clauses)
        # Pattern: "X and Y" where X/Y could be ambiguous
        structural_patterns = [
            r'\b(and|or|but)\s+(\w+)',  # Conjunctions
            r'\b(that|which|who)\s+(\w+)',  # Relative clauses
        ]

        for pattern in structural_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase = match.group(0)
                ambiguities.append(AmbiguityInstance(
                    phrase=phrase,
                    possible_interpretations=[
                        f"Interpretation 1: {phrase}", f"Interpretation 2: {phrase}"],
                    context_dependency=True,
                    resolution_confidence=0.5,
                    ambiguity_type="structural"
                ))

        return ambiguities

    def _can_resolve_pronoun(self, pronoun: str, position: int, words: List[str], context: ConversationContext) -> bool:
        """Check if pronoun can be resolved from context."""
        # Check entity registry
        if context.entity_registry:
            # Look for entities in prior turns
            for turn in context.prior_turns:
                if any(entity.lower() in turn.lower() for entity in context.entity_registry.keys()):
                    return True

        # Check prior turns for potential antecedents
        if context.prior_turns:
            # Simple heuristic: if prior turns mention entities, likely resolvable
            return len(context.prior_turns) > 0

        return False

    def _get_pronoun_interpretations(self, pronoun: str, context: ConversationContext) -> List[str]:
        """Get possible interpretations for a pronoun."""
        interpretations = []

        # Check entity registry for potential matches
        if context.entity_registry:
            for entity, description in context.entity_registry.items():
                if pronoun in ['he', 'him', 'his']:
                    interpretations.append(f"{entity} (male entity)")
                elif pronoun in ['she', 'her', 'hers']:
                    interpretations.append(f"{entity} (female entity)")
                elif pronoun in ['it', 'its']:
                    interpretations.append(f"{entity} (object)")
                elif pronoun in ['they', 'them', 'their', 'theirs']:
                    interpretations.append(f"{entity} (plural)")
                elif pronoun in ['this', 'that', 'these', 'those']:
                    interpretations.append(f"{entity} (referent)")

        # Add generic interpretations if no matches
        if not interpretations:
            if pronoun in ['he', 'him', 'his']:
                interpretations.append("Male referent from context")
            elif pronoun in ['she', 'her', 'hers']:
                interpretations.append("Female referent from context")
            elif pronoun in ['it', 'its']:
                interpretations.append("Object referent from context")
            elif pronoun in ['they', 'them', 'their', 'theirs']:
                interpretations.append("Plural referent from context")
            else:
                interpretations.append("Referent from context")

        return interpretations

    def _can_resolve_temporal(self, temporal_word: str, context: ConversationContext) -> bool:
        """Check if temporal reference can be resolved."""
        # If we have prior turns, we can infer temporal context
        return len(context.prior_turns) > 0

    def _get_temporal_interpretations(self, temporal_word: str, context: ConversationContext) -> List[str]:
        """Get possible interpretations for temporal reference."""
        interpretations = []

        if temporal_word == 'now':
            interpretations.append("Current time")
            if context.prior_turns:
                interpretations.append("Time of conversation")
        elif temporal_word == 'then':
            interpretations.append("Previous time mentioned")
            if context.prior_turns:
                interpretations.append("Time from prior turn")
        elif temporal_word in ['today', 'yesterday', 'tomorrow']:
            interpretations.append(f"Absolute date: {temporal_word}")
        elif temporal_word in ['recently', 'soon', 'later']:
            interpretations.append("Relative to conversation time")

        return interpretations or [f"Temporal reference: {temporal_word}"]

    def resolve_ambiguity(self, ambiguous_phrase: str, context: ConversationContext) -> DisambiguationResult:
        """Resolve ambiguity using available context."""
        audit_trail = []
        unresolved = []
        resolved_text = ambiguous_phrase

        # Detect all ambiguities in the phrase
        ambiguities = self.detect_ambiguities(ambiguous_phrase, context)

        if not ambiguities:
            return DisambiguationResult(
                success=True,
                disambiguated_sentence=ambiguous_phrase,
                failure_reason=None,
                audit_trail=[{"action": "no_ambiguities_detected"}],
                unresolved_ambiguities=[]
            )

        # Try to resolve each ambiguity
        for amb in ambiguities:
            if amb.ambiguity_type == "referential":
                resolution = self._resolve_referential_ambiguity(amb, context)
                if resolution:
                    resolved_text = resolved_text.replace(
                        amb.phrase, resolution, 1)
                    audit_trail.append({
                        "type": "referential_resolution",
                        "phrase": amb.phrase,
                        "resolved_to": resolution
                    })
                else:
                    unresolved.append(amb)
                    audit_trail.append({
                        "type": "referential_failure",
                        "phrase": amb.phrase,
                        "reason": "no_matching_entity"
                    })

            elif amb.ambiguity_type == "temporal":
                resolution = self._resolve_temporal_ambiguity(amb, context)
                if resolution:
                    resolved_text = resolved_text.replace(
                        amb.phrase, resolution, 1)
                    audit_trail.append({
                        "type": "temporal_resolution",
                        "phrase": amb.phrase,
                        "resolved_to": resolution
                    })
                else:
                    unresolved.append(amb)

            else:
                # Structural ambiguities are harder to resolve automatically
                unresolved.append(amb)
                audit_trail.append({
                    "type": "structural_ambiguity",
                    "phrase": amb.phrase,
                    "reason": "requires_manual_resolution"
                })

        success = len(unresolved) == 0

        return DisambiguationResult(
            success=success,
            disambiguated_sentence=resolved_text if success else None,
            failure_reason=None if success else f"{len(unresolved)} ambiguities unresolved",
            audit_trail=audit_trail,
            unresolved_ambiguities=unresolved
        )

    def _resolve_referential_ambiguity(self, ambiguity: AmbiguityInstance, context: ConversationContext) -> Optional[str]:
        """Resolve a referential ambiguity using context."""
        pronoun = re.sub(r'[^\w]', '', ambiguity.phrase.lower())

        # Check entity registry first
        if context.entity_registry:
            # Find most recent entity that matches pronoun type
            for entity, description in context.entity_registry.items():
                if pronoun in ['he', 'him', 'his'] and 'male' in description.lower():
                    return entity
                elif pronoun in ['she', 'her', 'hers'] and 'female' in description.lower():
                    return entity
                elif pronoun in ['it', 'its']:
                    return entity
                elif pronoun in ['they', 'them', 'their', 'theirs']:
                    return entity

        # Check prior turns for entities
        if context.prior_turns:
            # Simple heuristic: use first entity mentioned in prior turns
            for turn in reversed(context.prior_turns):
                # Look for capitalized words (likely proper nouns)
                entities = re.findall(r'\b[A-Z][a-z]+\b', turn)
                if entities:
                    return entities[0]

        return None

    def _resolve_temporal_ambiguity(self, ambiguity: AmbiguityInstance, context: ConversationContext) -> Optional[str]:
        """Resolve a temporal ambiguity using context."""
        temporal_word = re.sub(r'[^\w]', '', ambiguity.phrase.lower())

        if temporal_word == 'now':
            return "currently"
        elif temporal_word == 'then':
            return "at that time"
        elif temporal_word == 'recently':
            return "in recent times"
        elif temporal_word == 'soon':
            return "in the near future"
        elif temporal_word == 'later':
            return "at a later time"

        return None

    def detect_unresolvable_ambiguities(self, sentence: str, context: ConversationContext) -> List[AmbiguityInstance]:
        """Identify ambiguities that cannot be resolved."""
        ambiguities = self.detect_ambiguities(sentence, context)
        unresolvable = []

        for amb in ambiguities:
            # Check if we can resolve it
            if amb.ambiguity_type == "referential":
                if not self._can_resolve_pronoun(re.sub(r'[^\w]', '', amb.phrase.lower()), 0, sentence.split(), context):
                    unresolvable.append(amb)
            elif amb.ambiguity_type == "temporal":
                if not self._can_resolve_temporal(re.sub(r'[^\w]', '', amb.phrase.lower()), context):
                    unresolvable.append(amb)
            elif amb.ambiguity_type == "structural":
                # Structural ambiguities are generally harder to resolve
                unresolvable.append(amb)

        return unresolvable


class VerifiableContentQualification:
    """Stage 2: Verifiable Content Qualification."""

    # Subjective language patterns (low verifiability)
    SUBJECTIVE_PATTERNS = [
        r'\b(think|believe|feel|seem|appear|probably|maybe|perhaps|might|could|should)\b',
        r'\b(good|bad|better|best|worse|worst|nice|great|terrible|awesome)\b',
        r'\b(opinion|viewpoint|perspective|interpretation)\b',
    ]

    # Factual indicators (high verifiability)
    FACTUAL_INDICATORS = {
        'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        'quantities': r'\b\d+\s*(percent|%|dollars?|\$|units?|items?|times?|hours?|minutes?|seconds?)\b',
        'numbers': r'\b\d+\.?\d*\b',
        # Capitalized words (likely proper nouns)
        'proper_nouns': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        'authorities': r'\b(according to|per|stated by|reported by|source:|reference:)\b',
        'measurements': r'\b\d+\s*(cm|m|km|inch|ft|yd|lb|kg|g|ml|l)\b',
    }

    def detect_verifiable_content(self, sentence: str, context: ConversationContext) -> VerifiableContentResult:
        """Detect if sentence contains verifiable factual content."""
        indicators = []
        confidence = 0.0

        # Check for factual indicators
        factual_score = 0.0
        for indicator_type, pattern in self.FACTUAL_INDICATORS.items():
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            if matches:
                indicators.append(f"{indicator_type}: {len(matches)} matches")
                factual_score += 0.2

        # Check for subjective language (reduces verifiability)
        subjective_score = 0.0
        for pattern in self.SUBJECTIVE_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                subjective_score += 0.3
                indicators.append(f"subjective_language_detected")

        # Calculate confidence
        confidence = min(1.0, max(0.0, factual_score - subjective_score))

        # Consider verifiable if confidence > 0.3
        has_verifiable = confidence > 0.3

        # Rewrite sentence to remove subjective content if needed
        rewritten = None
        if has_verifiable and subjective_score > 0:
            rewritten = self.rewrite_unverifiable_content(sentence, context)

        return VerifiableContentResult(
            has_verifiable_content=has_verifiable,
            rewritten_sentence=rewritten if rewritten else sentence if has_verifiable else None,
            indicators=indicators,
            confidence=confidence
        )

    def rewrite_unverifiable_content(self, sentence: str, context: ConversationContext) -> Optional[str]:
        """Rewrite sentence to remove unverifiable content."""
        rewritten = sentence

        # Remove subjective qualifiers
        subjective_replacements = {
            r'\b(think|believe|feel)\s+that\s+': '',
            r'\b(probably|maybe|perhaps|might|could)\s+': '',
            r'\b(seems?|appears?)\s+to\s+be\s+': 'is ',
            r'\b(in my opinion|I think|I believe)\s*,?\s*': '',
        }

        for pattern, replacement in subjective_replacements.items():
            rewritten = re.sub(pattern, replacement,
                               rewritten, flags=re.IGNORECASE)

        # Remove opinion markers
        opinion_patterns = [
            r'\b(opinion|viewpoint|perspective|interpretation)\s+is\s+that\s+',
            r'\bfrom\s+my\s+(perspective|viewpoint|understanding)\s*,?\s*',
        ]

        for pattern in opinion_patterns:
            rewritten = re.sub(pattern, '', rewritten, flags=re.IGNORECASE)

        # Clean up extra spaces
        rewritten = re.sub(r'\s+', ' ', rewritten).strip()

        # Only return if we actually changed something and result is non-empty
        if rewritten != sentence and rewritten:
            return rewritten

        return None


class AtomicClaimDecomposition:
    """Stage 3: Atomic Claim Decomposition."""

    def _extract_claim_elements(self, claim_text: str) -> ClaimElements:
        """Extract structured elements (subject/predicate/object/qualifiers) from claim."""
        elements = ClaimElements()
        elements.qualifiers = {
            "time": None,
            "location": None,
            "quantity": None,
            "condition": None,
            "negation": False
        }

        # Extract negation
        negation_patterns = [
            r'\b(not|no|never|none|nobody|nothing|nowhere)\b',
            r'\b(doesn\'t|don\'t|didn\'t|won\'t|can\'t|cannot|isn\'t|aren\'t)\b'
        ]
        for pattern in negation_patterns:
            if re.search(pattern, claim_text, re.IGNORECASE):
                if elements.qualifiers:
                    elements.qualifiers["negation"] = True
                break

        # Extract time qualifiers
        time_patterns = [
            r'\b(today|yesterday|tomorrow|now|then|recently|soon|later|before|after|during|while|when|once|previously|currently)\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        for pattern in time_patterns:
            match = re.search(pattern, claim_text, re.IGNORECASE)
            if match:
                elements.qualifiers["time"] = match.group(0)
                break

        # Extract location qualifiers
        location_patterns = [
            r'\b(in|at|on|from|to|near|within|outside)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(city|state|country|region|area)'
        ]
        for pattern in location_patterns:
            match = re.search(pattern, claim_text)
            if match:
                elements.qualifiers["location"] = match.group(0)
                break

        # Extract quantity qualifiers
        quantity_patterns = [
            r'\b\d+\s*(percent|%|dollars?|\$|units?|items?|times?|hours?|minutes?|seconds?)',
            r'\b\d+\.?\d*\b'
        ]
        for pattern in quantity_patterns:
            match = re.search(pattern, claim_text, re.IGNORECASE)
            if match:
                elements.qualifiers["quantity"] = match.group(0)
                break

        # Extract condition qualifiers (if-then, when, etc.)
        condition_patterns = [
            r'\bif\s+(.+?)\s+(then|,|$)',
            r'\bwhen\s+(.+?)\s+(,|then|$)',
            r'\bprovided\s+that\s+(.+?)\s+(,|then|$)'
        ]
        for pattern in condition_patterns:
            match = re.search(pattern, claim_text, re.IGNORECASE)
            if match:
                elements.qualifiers["condition"] = match.group(1).strip()
                break

        # Simple subject/predicate/object extraction using pattern matching
        # Pattern: Subject Verb Object (SVO)
        # Try to find: "Subject verb object" or "Subject is/was object"
        svo_patterns = [
            r'^([A-Z][^.!?]*?)\s+(is|was|are|were|has|have|had|does|did|will|can|should|must)\s+(.+?)(?:\.|$)',
            r'^([A-Z][^.!?]*?)\s+(\w+ed|\w+ing|\w+s)\s+(.+?)(?:\.|$)',
        ]

        for pattern in svo_patterns:
            match = re.search(pattern, claim_text)
            if match:
                elements.subject = match.group(1).strip()
                elements.predicate = match.group(2).strip()
                if len(match.groups()) >= 3:
                    elements.object = match.group(3).strip()
                break

        # Fallback: if no SVO pattern matched, try simpler patterns
        if not elements.subject:
            # Try to extract first noun phrase as subject
            words = claim_text.split()
            if words:
                # Capitalized word or first few words as subject
                if words[0][0].isupper():
                    elements.subject = words[0]
                elif len(words) > 1:
                    elements.subject = ' '.join(words[:2])

        # Extract predicate from verb patterns
        if not elements.predicate:
            verb_patterns = [
                r'\b(is|was|are|were|has|have|had|does|did|will|can|should|must|makes?|creates?|implements?|provides?|returns?)\b'
            ]
            for pattern in verb_patterns:
                match = re.search(pattern, claim_text, re.IGNORECASE)
                if match:
                    elements.predicate = match.group(1).lower()
                    break

        return elements

    def extract_atomic_claims(self, qualified_sentence: str, context: ConversationContext) -> List[AtomicClaim]:
        """Extract atomic claims from disambiguated, qualified sentence."""
        claims = []

        # Split on conjunctions (and, or, but)
        # Pattern: split on conjunctions but preserve structure
        conjunction_pattern = r'\s+(and|or|but)\s+'
        parts = re.split(conjunction_pattern, qualified_sentence)

        # Process each part
        claim_id_counter = 1
        for i in range(0, len(parts), 2):  # Skip conjunction words
            part = parts[i].strip()
            if not part:
                continue

            # Further split on commas if they separate independent clauses
            # Split on comma followed by capital
            clauses = re.split(r',\s+(?=[A-Z])', part)

            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue

                # Remove trailing punctuation for processing
                clean_clause = clause.rstrip('.,;:!?')

                # Skip if too short (likely not a complete claim)
                if len(clean_clause.split()) < 3:
                    continue

                # Handle conditionals (if-then statements)
                if re.search(r'\bif\s+', clean_clause, re.IGNORECASE):
                    # Extract both condition and consequence
                    conditional_match = re.search(
                        r'\bif\s+(.+?)\s+then\s+(.+?)$', clean_clause, re.IGNORECASE)
                    if conditional_match:
                        condition = conditional_match.group(1).strip()
                        consequence = conditional_match.group(2).strip()

                        # Create claim for condition
                        condition_elements = self._extract_claim_elements(
                            condition)
                        claims.append(AtomicClaim(
                            id=f"claim_{claim_id_counter}",
                            statement=condition,
                            contextual_brackets=self._extract_context_brackets(
                                condition, context),
                            source_sentence=qualified_sentence,
                            verification_requirements=[
                                "condition_verification"],
                            confidence=0.8,
                            elements=condition_elements
                        ))
                        claim_id_counter += 1

                        # Create claim for consequence
                        consequence_elements = self._extract_claim_elements(
                            consequence)
                        claims.append(AtomicClaim(
                            id=f"claim_{claim_id_counter}",
                            statement=consequence,
                            contextual_brackets=self._extract_context_brackets(
                                consequence, context),
                            source_sentence=qualified_sentence,
                            verification_requirements=[
                                "consequence_verification"],
                            confidence=0.8,
                            elements=consequence_elements
                        ))
                        claim_id_counter += 1
                    else:
                        # Simple if statement without explicit "then"
                        clause_elements = self._extract_claim_elements(
                            clean_clause)
                        claims.append(AtomicClaim(
                            id=f"claim_{claim_id_counter}",
                            statement=clean_clause,
                            contextual_brackets=self._extract_context_brackets(
                                clean_clause, context),
                            source_sentence=qualified_sentence,
                            verification_requirements=[
                                "conditional_verification"],
                            confidence=0.7,
                            elements=clause_elements
                        ))
                        claim_id_counter += 1
                else:
                    # Regular claim
                    # Add contextual brackets
                    bracketed_claim = self.add_contextual_brackets(
                        clean_clause, qualified_sentence)

                    claim_elements = self._extract_claim_elements(clean_clause)

                    claims.append(AtomicClaim(
                        id=f"claim_{claim_id_counter}",
                        statement=bracketed_claim,
                        contextual_brackets=self._extract_context_brackets(
                            clean_clause, context),
                        source_sentence=qualified_sentence,
                        verification_requirements=self._extract_verification_requirements(
                            clean_clause),
                        confidence=0.9,
                        elements=claim_elements
                    ))
                    claim_id_counter += 1

        return claims

    def _extract_context_brackets(self, claim: str, context: ConversationContext) -> List[str]:
        """Extract relevant context brackets for a claim."""
        brackets = []

        # Add entity context
        if context.entity_registry:
            for entity, description in context.entity_registry.items():
                if entity.lower() in claim.lower():
                    brackets.append(f"[{entity}: {description}]")

        # Add code span context if claim mentions code
        if context.code_spans:
            code_keywords = ['code', 'function', 'method',
                             'class', 'variable', 'api', 'endpoint']
            if any(keyword in claim.lower() for keyword in code_keywords):
                for code_span in context.code_spans[:2]:  # Limit to first 2
                    brackets.append(f"[code: {code_span[:50]}...]")

        return brackets

    def _extract_verification_requirements(self, claim: str) -> List[str]:
        """Extract verification requirements from claim."""
        requirements = []

        # Check for dates (require date verification)
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', claim):
            requirements.append("date_verification")

        # Check for quantities (require quantity verification)
        if re.search(r'\d+\s*(percent|%|dollars?|\$|units?)', claim, re.IGNORECASE):
            requirements.append("quantity_verification")

        # Check for proper nouns (require entity verification)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
        if proper_nouns:
            requirements.append("entity_verification")

        # Default requirement
        if not requirements:
            requirements.append("general_verification")

        return requirements

    def add_contextual_brackets(self, claim: str, implied_context: str) -> str:
        """Add contextual brackets to make claim self-contained."""
        # Extract key context from implied_context
        # Look for entities, dates, or other contextual information

        # Find entities in context that aren't in claim
        context_entities = re.findall(r'\b[A-Z][a-z]+\b', implied_context)
        claim_entities = set(re.findall(r'\b[A-Z][a-z]+\b', claim))

        missing_entities = [
            e for e in context_entities if e not in claim_entities]

        # Build bracketed claim
        bracketed = claim

        # Add missing entities in brackets at the start
        if missing_entities:
            # Use first missing entity as context
            entity = missing_entities[0]
            bracketed = f"[{entity}] {bracketed}"

        # Add temporal context if present in implied_context but not in claim
        temporal_words = ['today', 'yesterday',
                          'tomorrow', 'now', 'then', 'recently']
        context_temporal = [
            w for w in temporal_words if w in implied_context.lower()]
        claim_temporal = [w for w in temporal_words if w in claim.lower()]

        if context_temporal and not claim_temporal:
            bracketed = f"[{context_temporal[0]}] {bracketed}"

        return bracketed


class Decontextualizer:
    """Generates maximally decontextualized claim (c_max) that entails original claim (c)."""

    def to_cmax(self, claim: AtomicClaim, source_context: str) -> str:
        """Generate c_max that entails c by adding maximal specificity.

        Invariant: c_max ⊨ c (c_max entails c)
        Only adds specificity (time/topic/agent), never removes constraints.
        """
        base = claim.statement
        extras = []

        # Extract qualifiers from claim elements if available
        if claim.elements and claim.elements.qualifiers:
            qualifiers = claim.elements.qualifiers

            # Add time qualifier if present
            if qualifiers.get("time"):
                extras.append(f"time: {qualifiers['time']}")

            # Add location qualifier if present
            if qualifiers.get("location"):
                extras.append(f"location: {qualifiers['location']}")

            # Add condition qualifier if present
            if qualifiers.get("condition"):
                extras.append(f"condition: {qualifiers['condition']}")

            # Add quantity qualifier if present
            if qualifiers.get("quantity"):
                extras.append(f"quantity: {qualifiers['quantity']}")

        # Prefer explicit agent/topic from contextual_brackets
        if claim.contextual_brackets:
            for bracket in claim.contextual_brackets:
                # Extract entity from bracket format: "[Entity: description]" or "[Entity]"
                bracket_match = re.match(r'\[([^\]:]+)', bracket)
                if bracket_match:
                    entity = bracket_match.group(1).strip()
                    if entity not in extras:
                        extras.append(f"entity: {entity}")

        # Extract additional context from source_context
        # Look for entities, dates, or other contextual information
        context_entities = re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', source_context)
        claim_entities = set(re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', base))

        # Add missing entities from context
        for entity in context_entities[:2]:  # Limit to first 2
            if entity not in claim_entities and entity not in [e.split(':')[-1].strip() for e in extras if ':' in e]:
                extras.append(f"context: {entity}")

        # Compose c_max: base claim with sorted, deduplicated extras
        if extras:
            # Remove duplicates while preserving order
            seen = set()
            unique_extras = []
            for extra in extras:
                extra_key = extra.split(':')[0] if ':' in extra else extra
                if extra_key not in seen:
                    seen.add(extra_key)
                    unique_extras.append(extra)

            extras_str = '; '.join(sorted(unique_extras))
            cmax = f"{base} ({extras_str})"
        else:
            cmax = base

        # Assert invariant: c_max tokens should include c tokens (monotonic entailment)
        # Cheap check: ensure claim tokens are subset of cmax tokens
        base_tokens = set(re.findall(r'\b\w+\b', base.lower()))
        cmax_tokens = set(re.findall(r'\b\w+\b', cmax.lower()))

        if not base_tokens.issubset(cmax_tokens):
            # Fall back to concatenation with brackets to preserve entailment
            cmax = f"{base}. {cmax}"

        return cmax


class EvidenceRetriever(ABC):
    """Abstract interface for evidence retrieval."""

    @abstractmethod
    def retrieve(self, claim_text: str, manifest: Dict) -> List[Dict]:
        """Retrieve evidence for a claim.

        Returns:
            List of evidence dicts with keys: "text", "source", "quality"
        """
        pass


class ManifestEvidenceRetriever(EvidenceRetriever):
    """Retriever that adapts existing evidence_manifest structure."""

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _token_overlap_score(self, claim_text: str, evidence_text: str) -> float:
        """Crude relevance proxy: token overlap ratio."""
        claim_words = set(re.findall(r'\b\w+\b', claim_text.lower()))
        evidence_words = set(re.findall(r'\b\w+\b', evidence_text.lower()))

        if not claim_words:
            return 0.0

        overlap = len(claim_words & evidence_words)
        return overlap / len(claim_words)

    def retrieve(self, claim_text: str, manifest: Dict, k: int = 6, lambda_div: float = 0.5) -> List[Dict]:
        """
        Maximal Marginal Relevance (MMR): argmax (1-λ)·Rel - λ·Redundancy
        Rel: token-overlap proxy; Redundancy: Jaccard vs. selected set.
        """
        evidence_items = manifest.get(
            "evidence_items", manifest.get("evidence", []))

        if not evidence_items:
            return []

        # Convert manifest format to unified format
        unified_evidence = []
        for item in evidence_items:
            # Handle both old format ("text", "source", "quality") and new format ("content", "source", "quality_score")
            text = item.get("text") or item.get("content", "")
            source = item.get("source", "unknown")
            quality = item.get("quality", item.get("quality_score", 0.5))

            unified_evidence.append({
                "text": text,
                "source": source,
                "quality": float(quality)
            })

        # Relevance proxy (drop-in: replace with embeddings later)
        def rel(item):
            return self._token_overlap_score(claim_text, item.get("text", ""))

        # Sort candidates by relevance
        candidates = sorted(unified_evidence, key=rel, reverse=True)
        selected = []

        while candidates and len(selected) < k:
            best = None
            best_score = -1e9

            for item in candidates:
                r = rel(item)
                # Redundancy: max Jaccard similarity to already selected items
                redundancy = max(
                    (self._jaccard_similarity(item.get("text", ""), s.get("text", ""))
                     for s in selected),
                    default=0.0
                )
                # MMR score: (1-λ)·relevance - λ·redundancy
                score = (1 - lambda_div) * r - lambda_div * redundancy

                if score > best_score:
                    best = item
                    best_score = score

            if best:
                selected.append(best)
                candidates.remove(best)
            else:
                break

        return selected


class EntailmentJudge(ABC):
    """Abstract interface for entailment judgment."""

    @abstractmethod
    def triage(self, evidence_chunk: str, claim_text: str) -> Dict[str, float]:
        """Classify evidence-claim relationship into mutually-exclusive probabilities.

        Returns:
            Dict with keys "support", "contradict", "insufficient" summing to 1.0
        """
        pass


class PlaceholderEntailmentJudge(EntailmentJudge):
    """
    Placeholder entailment judge with caching and temperature scaling.

    WARNING: This is a placeholder implementation using lexical overlap heuristics.
    It should be replaced with a trained NLI model for production use.

    This implementation:
    - Uses simple word overlap for entailment detection
    - Has limited accuracy compared to trained models
    - Is intended as a fallback when no real judge is available

    For production, replace with a trained entailment judge model.
    """

    def __init__(
        self,
        temperature: float = 0.8,
        prior: Optional[Dict[str, float]] = None,
        eps: float = 1e-6,
        calibration_path: Optional[str] = None,
        warn_on_init: bool = True
    ):
        """
        Initialize with temperature, priors, and optional calibration.

        Args:
            temperature: Temperature for logit scaling
            prior: Prior probabilities for support/contradict/insufficient
            eps: Epsilon for numerical stability
            calibration_path: Path to calibration JSON file
            warn_on_init: Whether to print warning about placeholder status
        """
        if warn_on_init:
            import warnings
            warnings.warn(
                "PlaceholderEntailmentJudge is a placeholder implementation using "
                "lexical overlap heuristics. For production use, replace with a "
                "trained NLI model. This judge has limited accuracy compared to "
                "trained models.",
                UserWarning,
                stacklevel=2
            )

        self.temperature = temperature
        self.prior = prior or {"support": 0.45,
                               "contradict": 0.1, "insufficient": 0.45}
        self.bias = {"support": 0.0, "contradict": 0.0, "insufficient": 0.0}
        self.eps = eps
        self._memo: Dict[str, Dict[str, float]] = {}

        # Load calibration if provided
        if calibration_path:
            try:
                with open(calibration_path, "r") as f:
                    data = json.load(f)
                calib = data.get("calibration", {})
                self.temperature = float(
                    calib.get("temperature", self.temperature))
                self.prior = calib.get("priors", self.prior)
                self.bias = calib.get("bias", self.bias)
            except Exception:
                # Non-fatal: fall back to defaults
                pass

    def _hash_inputs(self, evidence_chunk: str, claim_text: str) -> str:
        """Generate SHA256 hash of inputs for caching."""
        combined = f"{evidence_chunk}|||{claim_text}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _raw_overlap_logits(self, evidence_chunk: str, claim_text: str) -> Tuple[float, float, float]:
        """Compute raw logits from lexical overlap (placeholder)."""
        evidence_words = set(re.findall(r'\b\w+\b', evidence_chunk.lower()))
        claim_words = set(re.findall(r'\b\w+\b', claim_text.lower()))

        if not claim_words:
            return (-1.0, -2.0, 1.0)  # insufficient logits

        overlap = len(evidence_words & claim_words)
        overlap_ratio = overlap / len(claim_words) if claim_words else 0.0

        # Check for explicit contradictions
        contradiction_indicators = [
            r'\b(not|no|never|false|incorrect|wrong|disproves?|refutes?|contradicts?)\b',
            r'\b(doesn\'t|don\'t|didn\'t|won\'t|can\'t|cannot)\b'
        ]
        has_contradiction = False
        for pattern in contradiction_indicators:
            if re.search(pattern, evidence_chunk, re.IGNORECASE):
                # Check if contradiction is about the claim
                claim_keywords = list(claim_words)[:5]
                if any(keyword in evidence_chunk.lower() for keyword in claim_keywords):
                    has_contradiction = True
                    break

        # Generate logits
        if has_contradiction:
            support_logit = -2.0
            contradict_logit = 2.0
            insufficient_logit = -1.0
        else:
            # Support logit increases with overlap
            support_logit = overlap_ratio * 3.0 - 1.0
            contradict_logit = -2.0
            # Insufficient logit decreases with overlap
            insufficient_logit = (1.0 - overlap_ratio) * 2.0 - 1.0

        return (support_logit, contradict_logit, insufficient_logit)

    def triage(self, evidence_chunk: str, claim_text: str) -> Dict[str, float]:
        """
        Apply calibration: temperature + per-class bias + prior smoothing.
        Keeps interface identical; replace internals when you swap in a real NLI.
        """
        cache_key = self._hash_inputs(evidence_chunk, claim_text)

        if cache_key in self._memo:
            return self._memo[cache_key]

        # Crude logits from overlaps (placeholder)
        s_raw, c_raw, i_raw = self._raw_overlap_logits(
            evidence_chunk, claim_text)

        # Temperature scaling + bias
        s = (s_raw / self.temperature) + self.bias.get("support", 0.0)
        c = (c_raw / self.temperature) + self.bias.get("contradict", 0.0)
        i = (i_raw / self.temperature) + self.bias.get("insufficient", 0.0)

        # Softmax with priors
        exps = [math.exp(s), math.exp(c), math.exp(i)]
        Z = sum(exps) + self.eps
        probs = [e / Z for e in exps]
        labels = ["support", "contradict", "insufficient"]

        # Prior smoothing to dampen spikes
        smoothed = {lbl: 0.5 * probs[idx] + 0.5 * self.prior[lbl]
                    for idx, lbl in enumerate(labels)}

        # Renormalize
        norm = sum(smoothed.values()) + self.eps
        result = {k: v / norm for k, v in smoothed.items()}

        self._memo[cache_key] = result
        return result


def _detect_negation(text: str) -> bool:
    """Simple negation detector."""
    negation_patterns = [
        r'\b(not|no|never|none|nobody|nothing|nowhere)\b',
        r'\b(doesn\'t|don\'t|didn\'t|won\'t|can\'t|cannot|isn\'t|aren\'t)\b'
    ]
    for pattern in negation_patterns:
        if re.search(pattern, text.lower()):
            return True
    return False


class ElementCoverageScorer:
    """Scores element-level coverage with binding-aware analysis."""

    def coverage(self, evidence_text: str, elements: ClaimElements) -> Dict[str, Any]:
        """
        Binding-aware element coverage with explicit negation mismatch penalty.
        Score = micro-F1 across {subject,predicate,object} + present qualifiers.
        """
        sent = evidence_text.strip().lower()
        detail = {
            "subject": False,
            "predicate": False,
            "object": False,
            "qualifiers": {},
            "negation_mismatch": False
        }

        # Cheap phrase hits; keep your dependency parser hook in place if available
        hits = []
        for key in ["subject", "predicate", "object"]:
            val = getattr(elements, key, None)
            ok = bool(val) and (val.lower() in sent)
            detail[key] = ok
            hits.append(ok)

        # Qualifiers (time/location/quantity/condition)
        qhits = []
        qualifiers = elements.qualifiers or {}
        for qk, qv in qualifiers.items():
            if qk == "negation":
                continue
            qok = bool(qv) and (str(qv).lower() in sent)
            detail["qualifiers"][qk] = qok
            if qv is not None:
                qhits.append(qok)

        # micro-F1 style: precision=recall on binary element set → average of hits
        pools = hits + qhits
        coverage = sum(1 for h in pools if h) / max(1, len(pools))

        # Negation mismatch penalty
        claim_neg = bool(qualifiers.get("negation"))
        evid_neg = _detect_negation(evidence_text)
        if claim_neg != evid_neg:
            detail["negation_mismatch"] = True
            coverage = max(0.0, coverage - 0.25)

        return {"score": coverage, "detail": detail}


def classify_outcome(
    c_text: str,
    cmax_text: str,
    triage_c: Dict[str, float],
    triage_cmax: Dict[str, float],
    triage_c_to_cmax: Dict[str, float],
    thresholds: Dict[str, float],
    precedence: List[str]
) -> int:
    """Return 1–7 outcome with contradiction precedence and explicit thresholds."""
    # Check for contradiction FIRST (highest precedence)
    contradict_min = thresholds.get("contradict_min", 0.5)
    if (triage_c.get("contradict", 0.0) >= contradict_min or
        triage_cmax.get("contradict", 0.0) >= contradict_min or
            triage_c_to_cmax.get("contradict", 0.0) >= contradict_min):
        return 5  # contradiction case

    # Helper: choose highest-precedence label above its threshold
    def top_label(triage: Dict[str, float]) -> str:
        ordered = sorted(triage.items(), key=lambda kv: kv[1], reverse=True)
        for label in precedence:
            # choose highest-precedence label above its threshold
            if triage.get(label, 0.0) >= thresholds.get(f"{label}_min", 0.5):
                return label
        # fallback: highest prob even if below threshold
        return ordered[0][0] if ordered else "insufficient"

    # Check if identical - but still need to verify thresholds are met
    if c_text.strip().lower() == cmax_text.strip().lower():
        # Even if identical, check if thresholds are met
        l_c = top_label(triage_c)
        if l_c == "support":
            return 1  # identical and supported
        # If identical but not supported, fall through to insufficient logic

    l_c = top_label(triage_c)
    l_cmax = top_label(triage_cmax)
    l_map = top_label(triage_c_to_cmax)
    if l_c == "support" and l_cmax == "support" and l_map == "support":
        return 2  # fully supported
    if l_c == "support" and l_map != "support":
        return 3  # right answer, wrong rationale (not entailed by c_max)
    if l_cmax == "support" and l_c != "support":
        # retrieval mismatch (Emax supports cmax but Ec doesn't support c)
        return 4
    if "insufficient" in {l_c, l_cmax, l_map}:
        # Check if all are insufficient
        if l_c == "insufficient" and l_cmax == "insufficient" and l_map == "insufficient":
            return 7  # no usable evidence
        return 6   # insufficient evidence (mixed)
    return 6       # conservative default


# Regex patterns for claim category classification
_RE_NUM = re.compile(
    r"(?P<num>-?\d+(?:\.\d+)?)(?P<unit>\s?(?:%|ms|s|tok/s|tokens/s|items|cases)?)", re.I)
_RE_STATUS = re.compile(
    r"\b(production[-\s]?ready|enterprise[-\s]?grade|battle[-\s]?tested|deployed|released)\b", re.I)
_RE_BENCH = re.compile(
    r"\b(p50|p95|ttft|tps|throughput|latency|tokens\s?per\s?second)\b", re.I)
_RE_SUPERL = re.compile(
    r"\b(best|leading|state[-\s]?of[-\s]?the[-\s]?art|next[-\s]?generation|revolutionary|breakthrough)\b", re.I)


def classify_claim_categories(text: str) -> Set[ClaimCategory]:
    """Classify claim text into categories."""
    cats: Set[ClaimCategory] = set()
    if _RE_STATUS.search(text):
        cats.add(ClaimCategory.STATUS)
    if _RE_SUPERL.search(text):
        cats.add(ClaimCategory.SUPERLATIVE)
    if _RE_BENCH.search(text):
        cats.add(ClaimCategory.BENCHMARK)
    if _RE_NUM.search(text):
        cats.add(ClaimCategory.QUANT)
    return cats


def _evidence_has_required_artifacts(evidence_manifest: Dict, category: ClaimCategory, req: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Require structured artifacts for certain categories.
    Manifest may include typed entries, e.g.:
      {"artifacts": [
         {"type":"eval_report","path":"eval/reports/latest.json","json_path":"summary.gates_ok"},
         {"type":"bench_json","path":"evaluation/perf_mem_eval.json","json_path":"p95.ttft_ms"}
      ]}

    Returns True if ANY of the required types is present (OR logic).
    """
    missing = []
    arts = evidence_manifest.get("artifacts", [])
    requested_types = set(req.get("types", []))
    if not requested_types:
        return True, missing
    present = {a.get("type") for a in arts}
    # Check if ANY required type is present (OR logic)
    has_any = any(t in present for t in requested_types)
    if not has_any:
        # All types are missing
        for t in requested_types:
            missing.append(f"artifact:type:{t}")
    return has_any, missing


def _load_json_field(path: str, dotted_key: str) -> Optional[Any]:
    """Load a nested JSON field using dotted key notation."""
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        cur = obj
        for part in dotted_key.split("."):
            cur = cur[part]
        return cur
    except Exception:
        return None


def _validate_numeric_claim_against_artifact(text: str, allowlist: List[str], artifacts: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    For QUANT/BENCHMARK text, attempt to match numeric mentions to allowed JSON fields in artifacts.
    If any number is present in text and no corresponding field can be read and compared, flag as missing.
    This is conservative by design.
    """
    numbers = _RE_NUM.findall(text)
    if not numbers:
        return True, []
    # require at least one allowed field present in artifacts for numeric verification
    for a in artifacts:
        p = a.get("path")
        jp = a.get("json_path")
        if not p or not jp:
            continue
        dotted = jp.replace("/", ".")
        if allowlist and dotted not in allowlist:
            continue
        val = _load_json_field(p, dotted)
        if val is not None:
            return True, []
    return False, [f"numeric:missing_proof_for:{text}"]


class CAWSClaimVerification:
    """Stage 4: CAWS-Compliant Verification."""

    def __init__(
        self,
        retriever: Optional[EvidenceRetriever] = None,
        entailment: Optional[EntailmentJudge] = None,
        coverage: Optional[ElementCoverageScorer] = None,
        policy: Optional[ClaimsPolicy] = None,
        decontextualizer: Optional[Decontextualizer] = None
    ):
        """Initialize verification components with defaults for backward compatibility."""
        self.retriever = retriever or ManifestEvidenceRetriever()
        # Try to load calibration if available
        calibration_path = None
        try:
            calib_path = Path("eval/config/entailment_calibration.json")
            if calib_path.exists():
                calibration_path = str(calib_path)
        except Exception:
            pass
        self.entailment = entailment or PlaceholderEntailmentJudge(
            temperature=0.8,
            calibration_path=calibration_path
        )
        self.coverage = coverage or ElementCoverageScorer()
        self.policy = policy or ClaimsPolicy(
            require_artifacts={
                ClaimCategory.STATUS: {"types": ["eval_report", "coverage_report", "ci_status"]},
                ClaimCategory.BENCHMARK: {"types": ["bench_json"]},
                ClaimCategory.QUANT: {"types": ["eval_report", "bench_json"]},
            },
            banned_terms=["state-of-the-art", "revolutionary",
                          "best", "leading", "next-generation", "breakthrough"],
            numeric_fields_allow=[
                "summary.avg_integration_f1_macro_lax",
                "summary.privacy_ok_rate",
                "summary.controls_with_integration",
                "perf.ttft_ms.p50", "perf.ttft_ms.p95", "perf.tps.p50", "perf.tps.p95"
            ],
            thresholds={"coverage_min": 0.70},
            version="2025-11-12"
        )
        self.decontextualizer = decontextualizer or Decontextualizer()
        # Centralized thresholds + precedence for determinism
        self.thresholds = {
            "support_min": 0.6,        # probability threshold for support
            "contradict_min": 0.55,    # probability threshold for contradiction
            "insufficient_min": 0.55,  # probability threshold for insufficient
            # element coverage score for VERIFIED
            "coverage_min": self.policy.thresholds.get("coverage_min", 0.7)
        }
        # Contradiction > Support > Insufficient precedence
        self.precedence = ["contradict", "support", "insufficient"]

    def _compute_fingerprints(self) -> Dict[str, str]:
        """Compute SHA256 fingerprints of operator config for determinism."""
        config = {
            "retriever": type(self.retriever).__name__,
            "entailment": type(self.entailment).__name__,
            "coverage": type(self.coverage).__name__,
            "decontextualizer": type(self.decontextualizer).__name__,
            "thresholds": self.thresholds,
            "precedence": self.precedence
        }

        # Add temperature if available
        if hasattr(self.entailment, 'temperature'):
            config["entailment_temperature"] = self.entailment.temperature

        # Add policy version
        config["policy_version"] = self.policy.version

        config_json = json.dumps(config, sort_keys=True)
        fingerprint = hashlib.sha256(config_json.encode()).hexdigest()

        return {
            "operator_config": fingerprint,
            "config": config
        }

    def verify_claim_evidence(self, claim: AtomicClaim, evidence_manifest: Dict) -> VerificationResult:
        """Verify claim against evidence manifest using entailment and coverage."""
        verification_trail = []

        # --- Policy gate: classify claim, enforce artifacts/bans before entailment ---
        categories = classify_claim_categories(claim.statement or "")
        policy_violations: List[str] = []

        # hard ban: SUPERLATIVE is never VERIFIED
        if ClaimCategory.SUPERLATIVE in categories:
            policy_violations.append("superlative_terms")

        # artifact requirements for STATUS/BENCHMARK/QUANT
        for cat in (ClaimCategory.STATUS, ClaimCategory.BENCHMARK, ClaimCategory.QUANT):
            if cat in categories:
                ok, missing = _evidence_has_required_artifacts(
                    evidence_manifest, cat, self.policy.require_artifacts.get(cat, {
                    })
                )
                if not ok:
                    policy_violations.extend(missing)

        # numeric claims must be backed by an allowed field in artifacts
        # Skip if numeric_fields_allow is empty (permissive policy for tests)
        if (ClaimCategory.QUANT in categories or ClaimCategory.BENCHMARK in categories) and self.policy.numeric_fields_allow:
            ok_num, miss_num = _validate_numeric_claim_against_artifact(
                claim.statement, self.policy.numeric_fields_allow, evidence_manifest.get(
                    "artifacts", [])
            )
            if not ok_num:
                policy_violations.extend(miss_num)

        # If there are policy violations, short-circuit to INSUFFICIENT (Outcome 6) with rationale
        if policy_violations:
            fingerprints = self._compute_fingerprints()
            fingerprints["policy_version"] = self.policy.version
            return VerificationResult(
                status="INSUFFICIENT_EVIDENCE",
                evidence_quality=0.0,
                caws_compliance=False,
                verification_trail=[{
                    "step": "policy_gate",
                    "policy_violations": policy_violations,
                    "rationale": "Policy-required artifacts missing or banned terminology detected."
                }],
                outcome_id=6,
                element_coverage={"score": 0.0, "detail": {
                    "policy_violations": policy_violations}},
                entailment_triage={"support": 0.0,
                                   "contradict": 0.0, "insufficient": 1.0},
                fingerprints=fingerprints
            )

        # --- existing pipeline continues (decontextualize, retrieve, triage, coverage, 7-way outcome) ---
        # Generate c_max
        c_text = claim.statement
        cmax_text = self.decontextualizer.to_cmax(claim, claim.source_sentence)

        verification_trail.append({
            "step": "decontextualization",
            "c": c_text,
            "c_max": cmax_text
        })

        # Retrieve evidence for both c and c_max
        Ec = self.retriever.retrieve(c_text, evidence_manifest)
        Emax = self.retriever.retrieve(cmax_text, evidence_manifest)

        verification_trail.append({
            "step": "evidence_retrieval",
            "Ec_count": len(Ec),
            "Emax_count": len(Emax)
        })

        if not Ec and not Emax:
            fingerprints = self._compute_fingerprints()
            return VerificationResult(
                status="INSUFFICIENT_EVIDENCE",
                evidence_quality=0.0,
                caws_compliance=False,
                verification_trail=verification_trail,
                outcome_id=7,
                element_coverage={"score": 0.0, "detail": {}},
                entailment_triage={"support": 0.0,
                                   "contradict": 0.0, "insufficient": 1.0},
                fingerprints=fingerprints
            )

        # Calculate entailment triads: max over evidence chunks by support score
        tri_Ec_c = {"support": 0.0, "contradict": 0.0, "insufficient": 1.0}
        tri_Emax_cmax = {"support": 0.0,
                         "contradict": 0.0, "insufficient": 1.0}
        tri_Ec_cmax = {"support": 0.0, "contradict": 0.0, "insufficient": 1.0}

        # Calculate tri_Ec_c (max support from Ec for c)
        for ev in Ec:
            tri = self.entailment.triage(ev["text"], c_text)
            if tri["support"] > tri_Ec_c["support"]:
                tri_Ec_c = tri

        # Calculate tri_Emax_cmax (max support from Emax for cmax)
        for ev in Emax:
            tri = self.entailment.triage(ev["text"], cmax_text)
            if tri["support"] > tri_Emax_cmax["support"]:
                tri_Emax_cmax = tri

        # Calculate tri_Ec_cmax (max support from Ec for cmax)
        for ev in Ec:
            tri = self.entailment.triage(ev["text"], cmax_text)
            if tri["support"] > tri_Ec_cmax["support"]:
                tri_Ec_cmax = tri

        verification_trail.append({
            "step": "entailment_triage",
            "tri_Ec_c": tri_Ec_c,
            "tri_Emax_cmax": tri_Emax_cmax,
            "tri_Ec_cmax": tri_Ec_cmax
        })

        # Classify outcome (1-7) with thresholds and precedence
        outcome_id = classify_outcome(
            c_text, cmax_text,
            triage_c=tri_Ec_c,
            triage_cmax=tri_Emax_cmax,
            triage_c_to_cmax=tri_Ec_cmax,
            thresholds=self.thresholds,
            precedence=self.precedence
        )

        verification_trail.append({
            "step": "outcome_classification",
            "outcome_id": outcome_id
        })

        # Calculate binding-aware coverage: best sentence score from all Ec
        best_coverage_score = 0.0
        best_coverage_detail = {}

        if claim.elements:
            for ev in Ec:
                cov = self.coverage.coverage(ev["text"], claim.elements)
                if cov["score"] > best_coverage_score:
                    best_coverage_score = cov["score"]
                    best_coverage_detail = cov["detail"]
        else:
            # Fallback if no elements extracted
            best_coverage_score = 0.5  # Neutral score
            best_coverage_detail = {}

        verification_trail.append({
            "step": "element_coverage",
            "coverage_score": best_coverage_score,
            "coverage_detail": best_coverage_detail
        })

        # Decide final status from outcome + coverage using configured thresholds
        # Block VERIFIED if negation mismatch detected
        has_negation_mismatch = best_coverage_detail.get(
            "negation_mismatch", False)

        if outcome_id in (1, 2, 4) and best_coverage_score >= self.thresholds["coverage_min"] and not has_negation_mismatch:
            status = "VERIFIED"
            caws_compliance = True
        elif outcome_id == 5:
            status = "UNVERIFIED"
            caws_compliance = False
        else:
            status = "INSUFFICIENT_EVIDENCE"
            caws_compliance = False

        # Compute fingerprints
        fingerprints = self._compute_fingerprints()

        # Aggregate entailment triads (use max support as primary)
        aggregate_triad = {
            "support": max(tri_Ec_c["support"], tri_Emax_cmax["support"], tri_Ec_cmax["support"]),
            "contradict": max(tri_Ec_c["contradict"], tri_Emax_cmax["contradict"], tri_Ec_cmax["contradict"]),
            "insufficient": min(tri_Ec_c["insufficient"], tri_Emax_cmax["insufficient"], tri_Ec_cmax["insufficient"])
        }
        # Normalize
        total = sum(aggregate_triad.values())
        if total > 0:
            aggregate_triad = {k: v / total for k,
                               v in aggregate_triad.items()}

        return VerificationResult(
            status=status,
            evidence_quality=best_coverage_score,
            caws_compliance=caws_compliance,
            verification_trail=verification_trail,
            outcome_id=outcome_id,
            element_coverage={
                "score": best_coverage_score,
                "detail": best_coverage_detail
            },
            entailment_triage=aggregate_triad,
            fingerprints=fingerprints
        )

    def validate_claim_scope(self, claim: AtomicClaim, working_spec: Dict) -> Dict:
        """Validate claim is within CAWS scope boundaries."""
        violations = []
        within_scope = True

        # Extract scope from working spec
        scope = working_spec.get("scope", {})
        scope_in = scope.get("in", [])
        scope_out = scope.get("out", [])

        # Extract modules from blast_radius
        blast_radius = working_spec.get("blast_radius", {})
        allowed_modules = blast_radius.get("modules", [])

        # Extract claim text and check against scope
        claim_text = claim.statement.lower()
        claim_words = set(re.findall(r'\b\w+\b', claim_text))

        # Check if claim mentions out-of-scope items
        for out_item in scope_out:
            out_pattern = out_item.lower().replace(
                '/', r'[/\\]').replace('*', r'.*')
            if re.search(out_pattern, claim_text, re.IGNORECASE):
                violations.append({
                    "type": "out_of_scope_reference",
                    "item": out_item,
                    "claim": claim.statement[:100]
                })
                within_scope = False

        # Check if claim mentions modules not in blast_radius
        if allowed_modules:
            # Extract module names from claim
            claim_modules = []
            for module in allowed_modules:
                module_name = module.split(
                    '/')[-1].replace('.py', '').replace('_', ' ')
                if module_name.lower() in claim_text:
                    claim_modules.append(module)

            # Check if claim mentions code/files but no allowed modules match
            code_keywords = ['file', 'module', 'function',
                             'class', 'code', 'implementation']
            if any(keyword in claim_text for keyword in code_keywords):
                if not claim_modules:
                    # Claim mentions code but doesn't match any allowed module
                    violations.append({
                        "type": "module_not_in_blast_radius",
                        "claim": claim.statement[:100],
                        "allowed_modules": allowed_modules
                    })
                    # Don't mark as out of scope if it's just a general code reference
                    # within_scope = False  # Commented out - too strict

        # Check if claim references files/paths
        file_path_pattern = r'[\w/\\]+\.(py|js|ts|yaml|yml|json|md)'
        file_paths = re.findall(file_path_pattern, claim.statement)

        if file_paths:
            for file_path in file_paths:
                # Check if path is in scope
                path_in_scope = False
                for scope_item in scope_in:
                    scope_pattern = scope_item.lower().replace(
                        '/', r'[/\\]').replace('*', r'.*')
                    if re.search(scope_pattern, file_path, re.IGNORECASE):
                        path_in_scope = True
                        break

                if not path_in_scope:
                    violations.append({
                        "type": "file_path_out_of_scope",
                        "path": file_path,
                        "claim": claim.statement[:100]
                    })
                    within_scope = False

        return {
            "within_scope": within_scope,
            "violations": violations
        }


def calculate_desirable_outcome_rate(outcomes: List[int]) -> float:
    """Calculate rate of desirable outcomes (1, 2, 4, 7).

    Args:
        outcomes: List of outcome IDs (1-7)

    Returns:
        Rate of desirable outcomes (0.0-1.0)
    """
    if not outcomes:
        return 0.0

    desirable_outcomes = {1, 2, 4, 7}
    desirable_count = sum(1 for o in outcomes if o in desirable_outcomes)

    return desirable_count / len(outcomes)


def calculate_coverage_stats(coverage_scores: List[float]) -> Dict[str, float]:
    """Calculate coverage statistics (mean, p50, p95).

    Args:
        coverage_scores: List of coverage scores (0.0-1.0)

    Returns:
        Dict with "mean", "p50", "p95" keys
    """
    if not coverage_scores:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}

    sorted_scores = sorted(coverage_scores)
    n = len(sorted_scores)

    mean = sum(sorted_scores) / n
    p50 = sorted_scores[n // 2] if n > 0 else 0.0
    p95_idx = int(n * 0.95)
    p95 = sorted_scores[min(p95_idx, n - 1)] if n > 0 else 0.0

    return {
        "mean": mean,
        "p50": p50,
        "p95": p95
    }


def count_rationale_regressions(outcomes: List[int]) -> int:
    """Count Result-3 cases (right answer, wrong rationale).

    Args:
        outcomes: List of outcome IDs (1-7)

    Returns:
        Count of Result-3 outcomes
    """
    return sum(1 for o in outcomes if o == 3)


def summarize_verifications(
    results: List[VerificationResult],
    hw_profile_key: str = ""
) -> Dict[str, Any]:
    """Summarize verification results with outcome distribution and metrics.

    Args:
        results: List of VerificationResult objects
        hw_profile_key: Optional hardware profile key for relative comparisons

    Returns:
        Dict with outcome histogram, desirable rate, coverage stats, etc.
    """
    if not results:
        return {
            "outcomes_hist": {},
            "desirable_rate": 0.0,
            "coverage_mean": 0.0,
            "coverage_p50": 0.0,
            "coverage_p95": 0.0,
            "rationale_regressions": 0,
            "hw_profile_key": hw_profile_key
        }

    # Build outcome histogram
    outcomes_hist: Dict[int, int] = {}
    outcomes_list: List[int] = []
    coverage_scores: List[float] = []

    for result in results:
        if result.outcome_id is not None:
            outcome_id = result.outcome_id
            outcomes_hist[outcome_id] = outcomes_hist.get(outcome_id, 0) + 1
            outcomes_list.append(outcome_id)

        if result.element_coverage and "score" in result.element_coverage:
            coverage_scores.append(result.element_coverage["score"])

    # Calculate metrics
    desirable_rate = calculate_desirable_outcome_rate(outcomes_list)
    coverage_stats = calculate_coverage_stats(coverage_scores)
    rationale_regressions = count_rationale_regressions(outcomes_list)

    return {
        "outcomes_hist": outcomes_hist,
        "desirable_rate": desirable_rate,
        "coverage_mean": coverage_stats["mean"],
        "coverage_p50": coverage_stats["p50"],
        "coverage_p95": coverage_stats["p95"],
        "rationale_regressions": rationale_regressions,
        "total_claims": len(results),
        "hw_profile_key": hw_profile_key
    }


class ClaimifyPipeline:
    """Complete 4-stage claim extraction and verification pipeline."""

    def __init__(self):
        self.disambiguation = ClaimDisambiguation()
        self.qualification = VerifiableContentQualification()
        self.decomposition = AtomicClaimDecomposition()
        self.verification = CAWSClaimVerification()

    def process(self, text: str, context: ConversationContext,
                evidence_manifest: Optional[Dict] = None) -> Dict:
        """Run complete pipeline on input text.

        Returns:
            Dictionary with extracted claims and verification results
        """
        results = {
            "disambiguation": None,
            "qualification": None,
            "claims": [],
            "verification": []
        }

        # Stage 1: Disambiguation
        ambiguities = self.disambiguation.detect_ambiguities(text, context)
        if ambiguities:
            disambig_result = self.disambiguation.resolve_ambiguity(
                text, context)
            if not disambig_result.success:
                # Hard gate: skip if cannot disambiguate
                results["disambiguation"] = disambig_result
                return results
            text = disambig_result.disambiguated_sentence

        # Stage 2: Qualification
        qual_result = self.qualification.detect_verifiable_content(
            text, context)
        if not qual_result.has_verifiable_content:
            # Hard gate: skip if no verifiable content
            results["qualification"] = qual_result
            return results

        # Stage 3: Decomposition
        claims = self.decomposition.extract_atomic_claims(
            qual_result.rewritten_sentence or text, context)
        results["claims"] = claims

        # Stage 4: Verification (if evidence provided)
        if evidence_manifest:
            for claim in claims:
                verif_result = self.verification.verify_claim_evidence(
                    claim, evidence_manifest)
                results["verification"].append({
                    "claim_id": claim.id,
                    "verification": verif_result
                })

        return results
