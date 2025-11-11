import json, typer
def main(out: str = typer.Option("data/kd_mix.jsonl"), teacher: str = typer.Option(...)):
    print(f"Sampling teacher at {teacher}; writing {out} (stub).")
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt":"hello","teacher":"stub"})+"\n")
if __name__ == "__main__":
    typer.run(main)
