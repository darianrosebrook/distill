import typer
def main(config: str = typer.Argument(...)):
    print("Intermediate distillation stub:", config)
if __name__ == "__main__":
    typer.run(main)
