"""
PII redaction and URL allowlist utilities for dataset generation.

Author: @darianrosebrook
"""
import re

URL_ALLOW = {"example.com", "example.org", "example.net"}


def normalize_hostname(hostname: str) -> str:
    """Normalize hostname: lowercase and strip www."""
    hostname = hostname.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def redact_pii(text: str) -> str:
    """Redact PII (emails, UUIDs, phones, credit cards) from text."""
    # Emails
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[REDACTED_EMAIL]", text)

    # UUIDs
    text = re.sub(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
        "[REDACTED_UUID]",
        text,
    )

    # Phone numbers (US format: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX.XXX.XXXX)
    phone_patterns = [
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US phone
        # International
        r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
    ]
    for pattern in phone_patterns:
        text = re.sub(pattern, "[REDACTED_PHONE]", text)

    # Credit card numbers (13-19 digits, may have spaces/dashes)
    # Match sequences that look like credit cards but exclude obvious non-card numbers
    cc_pattern = r"\b(?:\d[ -]?){13,19}\b"
    # More specific: 4 groups of 4 digits
    cc_pattern_specific = r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"
    text = re.sub(cc_pattern_specific, "[REDACTED_CC]", text)
    # Also catch longer sequences
    text = re.sub(cc_pattern, lambda m: "[REDACTED_CC]" if len(
        re.sub(r'[ -]', '', m.group())) >= 13 else m.group(), text)

    return text


def allowlist_urls(text: str) -> bool:
    """Check if all URLs in text are in allowlist (with hostname normalization)."""
    urls = re.findall(r"https?://([^/\s]+)", text)
    normalized_allow = {normalize_hostname(h) for h in URL_ALLOW}
    for host in urls:
        normalized_host = normalize_hostname(host)
        if normalized_host not in normalized_allow:
            return False
    return True


def scan_safety(text: str) -> dict:
    """Scan text for safety issues and return scan results."""
    # Check URLs
    urls = re.findall(r"https?://([^/\s]+)", text)
    normalized_allow = {normalize_hostname(h) for h in URL_ALLOW}
    urls_ok = all(normalize_hostname(host)
                  in normalized_allow for host in urls)

    # Check for PII patterns (count hits before redaction)
    email_hits = len(re.findall(r"[\w\.-]+@[\w\.-]+", text))
    uuid_hits = len(re.findall(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
        text,
    ))
    phone_hits = len(re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text))
    cc_hits = len(re.findall(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", text))
    pii_hits = email_hits + uuid_hits + phone_hits + cc_hits

    return {
        "urls_ok": urls_ok,
        "pii_hits": pii_hits,
    }
