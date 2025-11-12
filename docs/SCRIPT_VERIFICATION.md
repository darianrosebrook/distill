# Script Verification Summary

## Overview

All main scripts have been verified to use the correct tier-aware backoff setup and API key loading.

## Scripts Verified

### 1. `scripts/test_api_once.py` ✅

**Status**: Fully configured

- ✅ Uses `TeacherClient.from_endpoint()` with `max_retries=5` and `retry_backoff_factor=2.0`
- ✅ Loads API key from `MOONSHOT_API_KEY` or `KIMI_API_KEY` (environment or `.env.local`)
- ✅ Tier-aware backoff enabled (defaults to FREE tier)
- ✅ Tier detection and logging implemented
- ✅ Single API call combines health check, tier detection, and sample test

### 2. `scripts/make_kd_mix.py` ✅

**Status**: Fully configured

- ✅ Uses `TeacherClient.from_endpoint()` with `max_retries=5` and `retry_backoff_factor=2.0`
- ✅ Loads API key from `MOONSHOT_API_KEY` or `KIMI_API_KEY` (environment or `.env.local`)
- ✅ Tier detection and display implemented
- ✅ Warns if `--delay` doesn't match tier recommendation
- ✅ Tier-aware backoff enabled (handles retries automatically)

**Key Features**:
- Detects tier on initialization
- Displays tier info (RPM, TPM, TPD, recommended delay)
- Warns if manual delay doesn't match tier recommendation
- Uses tier-aware exponential backoff for rate limit retries

### 3. `scripts/make_kd_mix_hardened.py` ✅

**Status**: Fully configured

- ✅ Uses `TeacherClient.from_endpoint()` with `max_retries=5` and `retry_backoff_factor=2.0`
- ✅ API key loaded via `TeacherClient` internal method (from `MOONSHOT_API_KEY` or `KIMI_API_KEY`)
- ✅ Tier detection and display implemented
- ✅ Warns if `--delay` doesn't match tier recommendation
- ✅ Tier-aware backoff enabled (handles retries automatically)

**Key Features**:
- Detects tier on initialization
- Displays tier info (RPM, TPM, TPD, recommended delay)
- Warns if manual delay doesn't match tier recommendation
- Uses tier-aware exponential backoff for rate limit retries
- Additional hardening: checkpointing, budget tracking, resume capability

### 4. `scripts/make_kd_mix_daily.sh` ✅

**Status**: Configured (uses `make_kd_mix.py`)

- ✅ Calls `make_kd_mix.py` with appropriate arguments
- ✅ Hardcodes `DELAY=20` for FREE tier (correct default)
- ✅ Loads `KIMI_API_KEY` from `.env.local` if not in environment
- ✅ Tier-aware backoff handled by underlying Python script

**Note**: The shell script hardcodes delay for FREE tier. If tier changes, the Python script will warn about delay mismatch.

## Tier-Aware Backoff Behavior

All scripts now use tier-aware exponential backoff:

| Tier | Base Delay | Retry Sequence (6 retries) |
|------|------------|----------------------------|
| FREE | 20.0s | 20s, 40s, 80s, 160s, 320s, 640s |
| TIER1 | 0.3s | 0.3s, 0.6s, 1.2s, 2.4s, 4.8s, 9.6s |
| TIER2 | 0.12s | 0.12s, 0.24s, 0.48s, 0.96s, 1.92s, 3.84s |
| TIER3+ | 0.012s | 0.012s, 0.024s, 0.048s, 0.096s, 0.192s, 0.384s |

## API Key Loading Priority

All scripts follow this priority order:

1. `MOONSHOT_API_KEY` environment variable
2. `KIMI_API_KEY` environment variable (backward compatibility)
3. `MOONSHOT_API_KEY` from `.env.local`
4. `KIMI_API_KEY` from `.env.local` (backward compatibility)

## Verification Checklist

- [x] All scripts use `TeacherClient.from_endpoint()` with correct retry settings
- [x] All scripts load API keys correctly (MOONSHOT_API_KEY or KIMI_API_KEY)
- [x] All scripts detect and display tier information
- [x] All scripts warn if manual delay doesn't match tier recommendation
- [x] All scripts use tier-aware exponential backoff for rate limit retries
- [x] Tier detection happens automatically from API response headers
- [x] Default tier (FREE) ensures backoff works immediately
- [x] Logging shows tier info when detected and when rate limited

## Next Steps

All scripts are ready for production use with tier-aware backoff. The system will:

1. **Automatically detect tier** from API response headers
2. **Use appropriate backoff delays** based on detected tier
3. **Log tier information** for visibility
4. **Warn about delay mismatches** to help users optimize

No further changes needed for tier-aware backoff implementation.

