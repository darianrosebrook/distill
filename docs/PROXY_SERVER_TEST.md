# Proxy Server Test Guide

## Overview

The proxy server (`capture/proxy_server.py`) acts as a reverse proxy between the `TeacherClient` and the Kimi API, capturing all API interactions for tool-use trace analysis.

## Test Setup

### 1. Start Proxy Server

In one terminal, start the proxy server:

```bash
cd /Users/darianrosebrook/Desktop/Projects/distill
source venv/bin/activate

python -m capture.proxy_server \
    --upstream https://api.moonshot.ai/v1 \
    --bind 127.0.0.1 \
    --port 8081 \
    --out capture/raw
```

The proxy will:
- Listen on `http://127.0.0.1:8081`
- Forward requests to `https://api.moonshot.ai/v1`
- Capture all requests/responses to `capture/raw/`
- Create trace files: `{trace_id}.jsonl` and `{trace_id}.meta.json`

### 2. Run Test Script

In another terminal, run the test:

```bash
cd /Users/darianrosebrook/Desktop/Projects/distill
source venv/bin/activate

python -m scripts.test_proxy_server
```

Or set custom proxy URL:

```bash
PROXY_URL=http://127.0.0.1:8081 python -m scripts.test_proxy_server
```

## Expected Behavior

### ✅ Success Case

1. **Proxy Server Running**
   ```
   [test_proxy_server] ✅ Proxy server is running
   ```

2. **API Call Through Proxy**
   ```
   [test_proxy_server] Making API call through proxy...
     Prompt: What is 2+2? Answer briefly.
     ✅ API call successful
     Response: 4...
   ```

3. **Capture Files Created**
   ```
   [test_proxy_server] ✅ Capture files created:
     Trace file: capture/raw/{trace_id}.jsonl
     Meta file: capture/raw/{trace_id}.meta.json
     Trace lines: {number}
     First line keys: ['raw']
   ```

4. **Meta File Contents**
   ```
   [test_proxy_server] Meta file contents:
     Trace ID: {uuid}
     Method: POST
     Body length: {bytes}
     Timestamps: {'start': ..., 'end': ...}
   ```

### ❌ Failure Cases

1. **Proxy Not Running**
   ```
   [test_proxy_server] ❌ Proxy server not running at http://127.0.0.1:8081
   ```
   **Solution**: Start the proxy server first (see step 1)

2. **API Key Missing**
   ```
   [test_proxy_server] ERROR: MOONSHOT_API_KEY or KIMI_API_KEY not found
   ```
   **Solution**: Set `MOONSHOT_API_KEY` in `.env.local` or environment

3. **API Call Failed**
   ```
   [test_proxy_server] ❌ API call failed: {error}
   ```
   **Possible causes**:
   - Rate limit (429) - wait 20s and retry
   - Network error - check connectivity
   - Invalid API key - verify key is correct

## Verification Steps

### 1. Check Capture Files

```bash
ls -lh capture/raw/
```

Should see:
- `*.jsonl` files (trace data)
- `*.meta.json` files (metadata)

### 2. Inspect Trace File

```bash
head -5 capture/raw/{trace_id}.jsonl
```

Should see JSON lines with `{"raw": "..."}` format.

### 3. Inspect Meta File

```bash
cat capture/raw/{trace_id}.meta.json
```

Should see:
- `trace_id`: UUID
- `timestamps`: Start/end times
- `request`: Method, headers, body length

### 4. Test Normalization

After capturing traces, test normalization:

```bash
python -m capture.normalize_trace \
    --in capture/raw \
    --out traces/normalized.jsonl \
    --schema schemas/trace.schema.json
```

## Proxy Server Fixes

### Fixed Issues

1. **URL Path Construction**: Now correctly appends request path to upstream URL
   - Before: `url = CONFIG["upstream"]` (missing path)
   - After: `url = f"{upstream_base}{upstream_path}?{query_string}"`

2. **Host Header**: Removed host header to avoid conflicts
   - Before: Forwarded all headers including `host`
   - After: Removes `host` header before forwarding

## Integration with TeacherClient

The `TeacherClient` can be configured to use the proxy:

```python
from models.teacher.teacher_client import TeacherClient

# Point to proxy instead of direct API
client = TeacherClient.from_endpoint(
    endpoint="http://127.0.0.1:8081",  # Proxy URL
    api_key=api_key,
    max_retries=5,
    retry_backoff_factor=2.0
)

# All API calls go through proxy and are captured
results = client.sample(["What is 2+2?"])
```

## Next Steps

After successful proxy test:

1. ✅ **Proxy Server**: Working and capturing traces
2. ⏭️ **Normalize Traces**: Convert raw traces to normalized format
3. ⏭️ **Pack Datasets**: Split into tool_select, post_tool, final stages
4. ⏭️ **Train Models**: Use stage-specific datasets for training

## Troubleshooting

### Proxy Server Won't Start

**Error**: `Address already in use`

**Solution**: 
```bash
# Find process using port 8081
lsof -i :8081

# Kill process
kill -9 {PID}

# Or use different port
python -m capture.proxy_server --port 8082 ...
```

### No Capture Files Created

**Possible causes**:
- Proxy server not receiving requests (check logs)
- Write permissions on `capture/raw/` directory
- API calls failing before reaching proxy

**Solution**:
```bash
# Check proxy server logs
# Verify directory exists and is writable
mkdir -p capture/raw
chmod 755 capture/raw
```

### API Calls Timing Out

**Possible causes**:
- Proxy server not forwarding correctly
- Network issues
- API rate limiting

**Solution**:
- Check proxy server logs
- Test direct API call (bypass proxy)
- Add delays between requests

