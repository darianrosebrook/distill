
from __future__ import annotations
import asyncio, json, uuid, time, argparse, os
from typing import AsyncIterator, Dict, Any, List
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()
CONFIG = {"upstream": "", "out_dir": "capture/raw"}

def now() -> float:
    return time.time()

async def stream_upstream(req: Request, path: str) -> StreamingResponse:
    # Construct upstream URL: base + path + query string
    upstream_base = CONFIG["upstream"].rstrip("/")
    upstream_path = f"/{path}" if path else ""
    query_string = str(req.url.query)
    url = f"{upstream_base}{upstream_path}" + (f"?{query_string}" if query_string else "")
    
    method = req.method
    headers = dict(req.headers)
    # Remove host header to avoid conflicts
    headers.pop("host", None)
    body = await req.body()

    trace_id = str(uuid.uuid4())
    start = now()
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    raw_path = os.path.join(CONFIG["out_dir"], f"{trace_id}.jsonl")

    async with httpx.AsyncClient(timeout=None) as client:
        upstream = await client.stream(method, url, headers=headers, content=body)
        async def agen() -> AsyncIterator[bytes]:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    # echo downstream
                    yield chunk
                    # also write to disk line-by-line for later normalization
                    with open(raw_path, "ab") as f:
                        f.write(json.dumps({"raw": chunk.decode("utf-8", errors="ignore")}).encode("utf-8"))
                        f.write(b"\n")

        resp = StreamingResponse(agen(), status_code=upstream.status_code, headers=dict(upstream.headers))

    meta = {
        "trace_id": trace_id,
        "timestamps": {"start": start, "end": now()},
        "request": {"method": method, "headers": headers, "body_len": len(body)}
    }
    with open(raw_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return resp

@app.api_route("/{path:path}", methods=["GET","POST","PUT","PATCH","DELETE"])
async def handle(req: Request, path: str):
    return await stream_upstream(req, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--out", dest="out_dir", default="capture/raw")
    args = ap.parse_args()
    CONFIG.update({"upstream": args.upstream, "out_dir": args.out_dir})
    uvicorn.run(app, host=args.bind, port=args.port)

if __name__ == "__main__":
    main()
