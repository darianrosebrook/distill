# training/distill_kd.py
# @author: @darianrosebrook

import json
import typer
from typing import List

app = typer.Typer()


@app.command()
def main(config: List[str]):
    print("KD configs:", config)
    # Load configs here (YAML/JSON), set up dataset/teacher, run KD loop.


if __name__ == "__main__":
    app()
