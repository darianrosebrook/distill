
from __future__ import annotations
import json
import time
import os
import functools
from typing import Callable, Any

def log_tool(out_dir: str = "capture/tools"):
    os.makedirs(out_dir, exist_ok=True)
    def decorator(fn: Callable[..., Any]):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            status = "ok"
            err = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as ex:
                status = "error"
                err = repr(ex)
                raise
            finally:
                rec = {
                    "tool": fn.__name__,
                    "args": kwargs,
                    "start": start,
                    "end": time.time(),
                    "latency_ms": int((time.time()-start)*1000),
                    "status": status,
                    "error": err
                }
                fname = f"{int(start*1000)}_{fn.__name__}.json"
                with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
        return wrapper
    return decorator
