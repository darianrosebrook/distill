import typer
def main(config: str = typer.Argument(...)):
    print("Process supervision stub:", config)
if __name__ == "__main__":
    typer.run(main)
