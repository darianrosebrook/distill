
from __future__ import annotations
import os, json, argparse, glob, time
from typing import Dict, Any, List
from capture.validators import validate_trace, redact

def parse_raw_lines(lines: List[str]) -> List[dict]:
    events = []
    for ln in lines:
        s = ln.strip()
        if not s: 
            continue
        # Try JSON payload first
        try:
            obj = json.loads(s)
            t = obj.get("type")
            if t in ("assistant.delta","assistant.final","assistant.tool_call","tool.result"):
                events.append(obj); continue
        except Exception:
            pass
        # Try SSE: "data: {json}"
        if s.startswith("data:"):
            payload = s[5:].strip()
            try:
                obj = json.loads(payload)
                t = obj.get("type")
                if t in ("assistant.delta","assistant.final","assistant.tool_call","tool.result"):
                    events.append(obj); continue
            except Exception:
                events.append({"type":"assistant.delta","text":payload}); continue
        # Fallback: treat line as free-text delta
        events.append({"type":"assistant.delta","text": s})
    return events

def build_trace(raw_file: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    meta_file = raw_file + ".meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    trace_id = meta["trace_id"]
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_lines = [json.loads(x)["raw"] for x in f if x.strip()]
    events = parse_raw_lines(raw_lines)

    trace = {
        "trace_id": trace_id,
        "consent": True,
        "timestamps": meta.get("timestamps", {"start": time.time(), "end": time.time()}),
        "context": {"system": "", "tools": []},
        "history": [],
        "events": [],
        "telemetry": {"prompt_tokens": 0, "completion_tokens": 0, "tool_calls": 0, "json_validity_errors": 0}
    }

    tool_calls = 0
    for e in events:
        et = e.get("type")
        if et == "assistant.tool_call": tool_calls += 1
        trace["events"].append(e)

    trace["telemetry"]["tool_calls"] = tool_calls
    trace = redact(trace)
    try:
        validate_trace(trace, schema)
    except Exception as ex:
        trace["_validation_error"] = str(ex)
    return trace

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--schema", required=True)
    args = ap.parse_args()

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    files = sorted(glob.glob(os.path.join(args.indir, "*.jsonl")))
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as out:
        for fp in files:
            trace = build_trace(fp, schema)
            out.write(json.dumps(trace, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
