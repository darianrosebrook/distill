import typer, os
def main(model: str = typer.Option(...), out: str = typer.Option("dist/")):
    os.makedirs(out, exist_ok=True)
    print(f"Packaging {model} into {out} (stub).")
if __name__ == "__main__":
    typer.run(main)
