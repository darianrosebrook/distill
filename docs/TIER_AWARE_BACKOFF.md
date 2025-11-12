# Tier-Aware Exponential Backoff

## Overview

The `TeacherClient` now implements tier-aware exponential backoff for rate limit handling. Instead of using a fixed exponential backoff (1s, 2s, 4s, 8s...), the system:

1. **Detects API tier** from response headers (or defaults to FREE tier)
2. **Uses tier-specific base delay** as the starting point for exponential backoff
3. **Logs tier information** when detected and when rate limited

## Tier-Specific Delays

| Tier | Base Delay | Backoff Sequence (6 retries) |
|------|------------|-------------------------------|
| FREE | 20.0s | 20s, 40s, 80s, 160s, 320s, 640s |
| TIER1 | 0.3s | 0.3s, 0.6s, 1.2s, 2.4s, 4.8s, 9.6s |
| TIER2 | 0.12s | 0.12s, 0.24s, 0.48s, 0.96s, 1.92s, 3.84s |
| TIER3 | 0.012s | 0.012s, 0.024s, 0.048s, 0.096s, 0.192s, 0.384s |
| TIER4 | 0.012s | 0.012s, 0.024s, 0.048s, 0.096s, 0.192s, 0.384s |
| TIER5 | 0.006s | 0.006s, 0.012s, 0.024s, 0.048s, 0.096s, 0.192s |

## Implementation Details

### Default Tier on Initialization

The client now defaults to **FREE tier** on initialization, ensuring tier-aware backoff works immediately:

```python
# Default to FREE tier so tier-aware backoff works from the start
self._tier: Optional[APITier] = APITier.FREE
self._tier_limits: Optional[TierLimits] = TIER_LIMITS[APITier.FREE]
```

### Tier Detection

Tier is detected from API response headers (including 429 responses):

```python
def _update_tier_from_response(self, response: requests.Response):
    """Update tier information from API response headers."""
    # Extracts RPM, TPM, TPD from headers
    # Infers tier based on RPM thresholds
    # Updates self._tier and self._tier_limits
```

### Tier-Aware Backoff Logic

When a 429 (rate limit) is encountered:

1. **Try to extract tier from response headers** (may contain rate limit info)
2. **Check for Retry-After header** (if present, use that)
3. **Otherwise, use tier-aware exponential backoff**:
   ```python
   base_delay = self._tier_limits.delay  # e.g., 20s for FREE tier
   wait_time = base_delay * (self._retry_backoff_factor ** retry_count)
   ```
4. **Log tier info** in the wait message

### Logging

Tier information is logged:

- **When tier is detected**: `[TeacherClient] Detected tier: free (RPM: 3, delay: 20.0s)`
- **When rate limited**: `[TeacherClient] Rate limited (429), waiting 20.0s before retry 1/6 (tier: free, base: 20.0s)`

## Example Output

### Successful API Call (Tier Detected)

```
[test_api_once] ✅ API Test Results:
  ✅ API connectivity: OK
  ✅ Authentication: OK

  API Tier Information:
    Tier: free
    RPM: 3
    TPM: 500,000
    TPD: 1,500,000
    Concurrency: 1
    Recommended delay: 20.0s
    Backoff strategy: Tier-aware (starts at 20.0s, then exponential)
```

### Rate Limited (Tier-Aware Backoff)

```
[TeacherClient] Rate limited (429), waiting 20.0s before retry 1/6 (tier: free, base: 20.0s)
[TeacherClient] Rate limited (429), waiting 40.0s before retry 2/6 (tier: free, base: 20.0s)
[TeacherClient] Rate limited (429), waiting 80.0s before retry 3/6 (tier: free, base: 20.0s)
```

## Benefits

1. **Appropriate delays**: FREE tier waits 20s (not 1s) before first retry
2. **Faster recovery**: Higher tiers use shorter delays appropriate to their limits
3. **Automatic adaptation**: Tier detection happens automatically from API responses
4. **Clear visibility**: Logging shows which tier and backoff strategy is being used

## Testing

Run the API test to verify tier detection and backoff:

```bash
python -m scripts.test_api_once
```

The test will:
- Make a single API call
- Detect tier from response headers
- Display tier information
- Show tier-aware backoff strategy

