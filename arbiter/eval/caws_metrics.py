# arbiter/eval/caws_metrics.py
# CAWS evaluation metrics: pairwise accuracy, clause F1, claim P/R/F1
# @author: @darianrosebrook

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PairwiseAccuracyMetrics:
    """Metrics for pairwise ranking evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    tie_handling: str  # "exclude", "count_as_correct", "penalize"


@dataclass
class ClauseMappingMetrics:
    """Metrics for CAWS clause mapping."""

    f1_score: float
    precision: float
    recall: float
    per_clause_metrics: Dict[str, Dict[str, float]]


@dataclass
class ClaimVerificationMetrics:
    """Metrics for claim extraction and verification."""

    extraction_precision: float
    extraction_recall: float
    extraction_f1: float
    verification_precision: float
    verification_recall: float
    verification_f1: float
    coverage: float  # Factual coverage metric


def compute_pairwise_accuracy(
    predictions: List[int], labels: List[int], tie_strategy: str = "exclude"
) -> PairwiseAccuracyMetrics:
    """Compute pairwise ranking accuracy.

    Args:
        predictions: Predicted preferences (0=A, 1=B, -1=tie)
        labels: Gold preferences
        tie_strategy: How to handle ties in evaluation

    Returns:
        PairwiseAccuracyMetrics
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Predictions and labels must have same length: {len(predictions)} != {len(labels)}"
        )

    correct = 0
    total = 0

    # Count true positives, false positives, false negatives for precision/recall
    # Treat preference for A (0) as positive class
    tp_prefer_a = 0  # Predicted A, labeled A
    fp_prefer_a = 0  # Predicted A, labeled B
    fn_prefer_a = 0  # Predicted B, labeled A

    # Treat preference for B (1) as positive class
    tp_prefer_b = 0  # Predicted B, labeled B
    fp_prefer_b = 0  # Predicted B, labeled A
    fn_prefer_b = 0  # Predicted A, labeled B

    for pred, label in zip(predictions, labels):
        # Handle ties based on strategy
        if tie_strategy == "exclude":
            if pred == -1 or label == -1:
                continue
        elif tie_strategy == "count_as_correct":
            if pred == -1 and label == -1:
                correct += 1
                total += 1
                continue
            elif pred == -1 or label == -1:
                total += 1
                continue
        elif tie_strategy == "penalize":
            if pred == -1 or label == -1:
                total += 1
                continue

        # Count accuracy
        if pred == label:
            correct += 1
        total += 1

        # Count for precision/recall (treating as binary classification)
        # For preference A (0)
        if pred == 0 and label == 0:
            tp_prefer_a += 1
        elif pred == 0 and label == 1:
            fp_prefer_a += 1
        elif pred == 1 and label == 0:
            fn_prefer_a += 1

        # For preference B (1)
        if pred == 1 and label == 1:
            tp_prefer_b += 1
        elif pred == 1 and label == 0:
            fp_prefer_b += 1
        elif pred == 0 and label == 1:
            fn_prefer_b += 1

    accuracy = correct / total if total > 0 else 0.0

    # Compute macro-averaged precision/recall/F1 across both classes
    # Precision for A
    prec_a = tp_prefer_a / (tp_prefer_a + fp_prefer_a) if (tp_prefer_a + fp_prefer_a) > 0 else 0.0
    # Recall for A
    rec_a = tp_prefer_a / (tp_prefer_a + fn_prefer_a) if (tp_prefer_a + fn_prefer_a) > 0 else 0.0

    # Precision for B
    prec_b = tp_prefer_b / (tp_prefer_b + fp_prefer_b) if (tp_prefer_b + fp_prefer_b) > 0 else 0.0
    # Recall for B
    rec_b = tp_prefer_b / (tp_prefer_b + fn_prefer_b) if (tp_prefer_b + fn_prefer_b) > 0 else 0.0

    # Macro-averaged metrics
    precision = (prec_a + prec_b) / 2.0
    recall = (rec_a + rec_b) / 2.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return PairwiseAccuracyMetrics(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1, tie_handling=tie_strategy
    )


def compute_clause_f1(
    predicted_clauses: List[List[str]], gold_clauses: List[List[str]]
) -> ClauseMappingMetrics:
    """Compute F1 score for clause mapping.

    Args:
        predicted_clauses: List of predicted clause lists per example
        gold_clauses: List of gold clause lists per example

    Returns:
        ClauseMappingMetrics
    """
    if len(predicted_clauses) != len(gold_clauses):
        raise ValueError(
            f"Predicted and gold clauses must have same length: {len(predicted_clauses)} != {len(gold_clauses)}"
        )

    # Use set-based precision/recall/F1 for multi-label classification
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Track per-clause metrics
    clause_tp = {}  # clause -> true positive count
    clause_fp = {}  # clause -> false positive count
    clause_fn = {}  # clause -> false negative count

    for pred_list, gold_list in zip(predicted_clauses, gold_clauses):
        pred_set = set(pred_list)
        gold_set = set(gold_list)

        # Overall metrics
        tp = pred_set & gold_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set

        true_positives += len(tp)
        false_positives += len(fp)
        false_negatives += len(fn)

        # Per-clause metrics
        for clause in tp:
            clause_tp[clause] = clause_tp.get(clause, 0) + 1
        for clause in fp:
            clause_fp[clause] = clause_fp.get(clause, 0) + 1
        for clause in fn:
            clause_fn[clause] = clause_fn.get(clause, 0) + 1

    # Overall precision/recall/F1
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute per-clause metrics
    all_clauses = set(clause_tp.keys()) | set(clause_fp.keys()) | set(clause_fn.keys())
    per_clause_metrics = {}

    for clause in all_clauses:
        tp_count = clause_tp.get(clause, 0)
        fp_count = clause_fp.get(clause, 0)
        fn_count = clause_fn.get(clause, 0)

        clause_precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        clause_recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        clause_f1 = (
            2 * clause_precision * clause_recall / (clause_precision + clause_recall)
            if (clause_precision + clause_recall) > 0
            else 0.0
        )

        per_clause_metrics[clause] = {
            "precision": clause_precision,
            "recall": clause_recall,
            "f1": clause_f1,
            "support": tp_count + fn_count,  # Number of gold instances
        }

    return ClauseMappingMetrics(
        f1_score=f1, precision=precision, recall=recall, per_clause_metrics=per_clause_metrics
    )


def compute_claim_metrics(
    extracted_claims: List[Dict], gold_claims: List[Dict], verification_results: List[Dict]
) -> ClaimVerificationMetrics:
    """Compute precision/recall/F1 for claim extraction and verification.

    Args:
        extracted_claims: List of extracted claim dictionaries (must have 'id' and 'statement' fields)
        gold_claims: List of gold claim dictionaries (must have 'id' and 'statement' fields)
        verification_results: List of verification result dictionaries (must have 'claim_id' and 'status' fields)

    Returns:
        ClaimVerificationMetrics
    """
    # Match extracted to gold claims using ID and statement similarity
    # For extraction metrics: treat as set matching problem
    # For verification metrics: use verification_results status

    gold_statements = {claim.get("statement", "") for claim in gold_claims}
    extracted_statements = {claim.get("statement", "") for claim in extracted_claims}

    # Extraction metrics: match by statement (exact match)
    # True positives: statements that appear in both extracted and gold
    tp_extraction = len(extracted_statements & gold_statements)
    # False positives: statements extracted but not in gold
    fp_extraction = len(extracted_statements - gold_statements)
    # False negatives: statements in gold but not extracted
    fn_extraction = len(gold_statements - extracted_statements)

    extraction_precision = (
        tp_extraction / (tp_extraction + fp_extraction)
        if (tp_extraction + fp_extraction) > 0
        else 0.0
    )
    extraction_recall = (
        tp_extraction / (tp_extraction + fn_extraction)
        if (tp_extraction + fn_extraction) > 0
        else 0.0
    )
    extraction_f1 = (
        2 * extraction_precision * extraction_recall / (extraction_precision + extraction_recall)
        if (extraction_precision + extraction_recall) > 0
        else 0.0
    )

    # Verification metrics: based on verification_results
    # Build mapping from claim_id to verification status
    verification_by_claim_id = {}
    for result in verification_results:
        claim_id = result.get("claim_id", "")
        status = result.get("status", "UNVERIFIED")
        verification_by_claim_id[claim_id] = status

    # Count verification metrics
    # True positives: claims verified (status == "VERIFIED")
    # False positives: claims marked verified but shouldn't be (need gold verification status)
    # False negatives: claims that should be verified but aren't

    # If gold claims have verification status, use that
    # Otherwise, assume all gold claims should be verified
    verified_count = 0
    unverified_count = 0
    insufficient_evidence_count = 0

    for claim_id, status in verification_by_claim_id.items():
        if status == "VERIFIED":
            verified_count += 1
        elif status == "UNVERIFIED":
            unverified_count += 1
        elif status == "INSUFFICIENT_EVIDENCE":
            insufficient_evidence_count += 1

    # For verification metrics, we need gold verification status
    # If not available, we can only compute based on extracted claims
    # Assume all extracted claims that match gold should be verified
    # True positives: verified claims that match gold
    tp_verification = 0
    fp_verification = 0
    fn_verification = 0

    matched_claim_ids = set()
    for claim in extracted_claims:
        claim_id = claim.get("id", "")
        statement = claim.get("statement", "")

        # Check if this claim matches a gold claim
        matches_gold = statement in gold_statements

        # Get verification status
        status = verification_by_claim_id.get(claim_id, "UNVERIFIED")

        if matches_gold:
            matched_claim_ids.add(claim_id)
            if status == "VERIFIED":
                tp_verification += 1
            else:
                fn_verification += 1
        else:
            if status == "VERIFIED":
                fp_verification += 1

    # Count false negatives: gold claims that weren't extracted or weren't verified
    for claim in gold_claims:
        claim_id = claim.get("id", "")
        statement = claim.get("statement", "")

        if statement not in extracted_statements:
            # Not extracted at all - this is an extraction false negative, not verification
            continue

        if claim_id not in matched_claim_ids:
            # Extracted but doesn't match - already counted
            continue

        # Check if it was verified
        status = verification_by_claim_id.get(claim_id, "UNVERIFIED")
        if status != "VERIFIED":
            fn_verification += 1

    verification_precision = (
        tp_verification / (tp_verification + fp_verification)
        if (tp_verification + fp_verification) > 0
        else 0.0
    )
    verification_recall = (
        tp_verification / (tp_verification + fn_verification)
        if (tp_verification + fn_verification) > 0
        else 0.0
    )
    verification_f1 = (
        2
        * verification_precision
        * verification_recall
        / (verification_precision + verification_recall)
        if (verification_precision + verification_recall) > 0
        else 0.0
    )

    # Coverage: fraction of gold claims that were extracted and verified
    total_gold = len(gold_claims)
    extracted_and_verified = sum(
        1
        for claim in gold_claims
        if claim.get("statement", "") in extracted_statements
        and verification_by_claim_id.get(claim.get("id", ""), "") == "VERIFIED"
    )
    coverage = extracted_and_verified / total_gold if total_gold > 0 else 0.0

    return ClaimVerificationMetrics(
        extraction_precision=extraction_precision,
        extraction_recall=extraction_recall,
        extraction_f1=extraction_f1,
        verification_precision=verification_precision,
        verification_recall=verification_recall,
        verification_f1=verification_f1,
        coverage=coverage,
    )
