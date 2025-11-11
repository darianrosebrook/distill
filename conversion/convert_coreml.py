import typer
def main(config: str = typer.Argument(...)):
    print("CoreML conversion stub:", config)
if __name__ == "__main__":
    typer.run(main)
