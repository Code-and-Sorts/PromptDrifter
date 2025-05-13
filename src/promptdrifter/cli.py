import asyncio
import importlib.resources
import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.text import Text

from promptdrifter.runner import Runner

app = typer.Typer(no_args_is_help=True)
console = Console()


def get_version():
    try:
        with open("pyproject.toml", "r") as f:
            for line in f:
                if line.startswith("version = "):
                    return line.split("=")[1].strip().strip('"')
    except Exception:
        return "unknown"


@app.command()
def version():
    """Display the current version of PromptDrifter."""
    console.print(f"PromptDrifter version: [bold blue]{get_version()}[/bold blue]")


@app.command()
def init(
    ctx: typer.Context,
    target_path_str: str = typer.Argument(
        ".", help="The directory to initialize the project in. Defaults to current directory."
    ),
):
    """Initialize a new promptdrifter project with a sample config."""
    target_path = Path(target_path_str).resolve()
    config_file_path = target_path / "promptdrifter.yaml"

    # Sanitize path strings before use with Rich
    sane_target_path_str = str(target_path).replace("\u00A0", " ")
    sane_config_file_path_str = str(config_file_path).replace("\u00A0", " ")

    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
        message = Text("Created directory: ", style="green")
        path_text = Text(sane_target_path_str, style="green", no_wrap=True, overflow="ignore")
        message.append(path_text)
        console.print(message)
    elif not target_path.is_dir():
        console.print(
            f"[bold red]Error: Target path '{sane_target_path_str}' exists but is not a directory.[/bold red]"
        )
        raise typer.Exit(code=1)

    if config_file_path.exists():
        console.print(
            f"[yellow]Warning: Configuration file '{sane_config_file_path_str}' already exists. Skipping.[/yellow]"
        )
        return

    try:
        sample_config_content = (
            importlib.resources.files("promptdrifter")
            .joinpath("schema", "sample", "v0.1.yaml")
            .read_text()
        )
    except FileNotFoundError:
        console.print(
            "[bold red]Error: Sample configuration file not found in the package.[/bold red]"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error reading sample configuration: {e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        with open(config_file_path, "w") as f:
            f.write(sample_config_content)
        message = Text("Successfully created sample configuration: ", style="green")
        path_text = Text(sane_config_file_path_str, style="green", no_wrap=True, overflow="ignore")
        message.append(path_text)
        console.print(message)
        console.print("You can now edit this file and run 'promptdrifter run'.")
    except IOError as e:
        message = Text("Error writing configuration file to '", style="bold red")
        path_text = Text(sane_config_file_path_str, style="bold red", no_wrap=True, overflow="ignore")
        message.append(path_text)
        message.append(f"': {e}", style="bold red")
        console.print(message)
        raise typer.Exit(code=1)


async def _run_async(
    files: List[Path],
    no_cache: bool,
    cache_db: Optional[Path],
    config_dir: Path,
):
    """Async implementation of the run command."""
    if not files:
        console.print("[bold red]Error: No YAML files provided.[/bold red]")
        console.print("\n[bold]Usage:[/bold]")
        console.print("  promptdrifter run [OPTIONS] <file1.yaml> [file2.yaml ...]")
        console.print("\n[bold]Example:[/bold]")
        console.print("  promptdrifter run ./tests/promptdrifter.yaml")
        console.print("  promptdrifter run -c ./config ./tests/*.yaml")
        console.print("\n[bold]Options:[/bold]")
        console.print("  -c, --config-dir PATH    Directory containing config files")
        console.print("  --no-cache              Disable response caching")
        console.print("  --cache-db PATH         Path to cache database file")
        raise typer.Exit(code=1)

    yaml_files_str = []
    invalid_files = []
    for f_path in files:
        if not f_path.exists():
            invalid_files.append((f_path, "File not found"))
            continue
        if not f_path.is_file():
            invalid_files.append((f_path, "Path is not a file"))
            continue
        if not (f_path.name.endswith(".yaml") or f_path.name.endswith(".yml")):
            invalid_files.append((f_path, "Not a YAML file"))
            continue
        yaml_files_str.append(str(f_path))

    if invalid_files:
        console.print("[bold red]Error: Invalid file(s) provided:[/bold red]")
        for file_path_obj, reason in invalid_files:
            sane_file_path_str = str(file_path_obj).replace("\u00A0", " ")
            console.print(f"  • {sane_file_path_str}: {reason}")
        console.print("\n[bold]Please provide valid YAML test files.[/bold]")
        raise typer.Exit(code=1)

    runner_instance: Optional[Runner] = None
    try:
        runner_instance = Runner(
            config_dir=config_dir, cache_db_path=cache_db, use_cache=not no_cache
        )
        overall_success = await runner_instance.run_suite(yaml_files_str)
        if not overall_success:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(
            f"[bold red]An unexpected error occurred during CLI run: {e}[/bold red]"
        )
        raise typer.Exit(code=1)
    finally:
        if runner_instance:
            try:
                await runner_instance.close_cache_connection()
            except Exception as close_e:
                console.print(
                    f"[bold yellow]Warning: Failed to close cache connection: {close_e}[/bold yellow]"
                )


@app.command()
def run(
    files: List[Path] = typer.Argument(..., help="Paths to YAML test suite files."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable response caching"),
    cache_db: Optional[Path] = typer.Option(
        None, "--cache-db", help="Path to cache database file"
    ),
    config_dir: Path = typer.Option(
        Path("."), "--config-dir", "-c", help="Directory containing config files"
    ),
):
    """Run a suite of prompt tests from one or more YAML files."""
    asyncio.run(_run_async(files, no_cache, cache_db, config_dir))


@app.command()
def record(
    output_file: Path = typer.Option(
        Path("recorded_interactions.json"),
        "--output",
        "-o",
        help="Path to save recorded interactions",
    ),
    adapter: str = typer.Option(
        ..., "--adapter", "-a", help="LLM adapter to use (e.g., openai, ollama)"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Model to use with the adapter"
    ),
):
    """Record interactions with an LLM for later use in test cases."""
    if not adapter or not model:
        console.print(
            "[bold red]Error: Both adapter and model are required.[/bold red]"
        )
        raise typer.Exit(code=1)

    console.print("[bold]Recording mode started. Type 'exit' to finish.[/bold]")
    interactions = []

    while True:
        try:
            prompt = typer.prompt("\nEnter your prompt")
            if prompt.lower() == "exit":
                break

            response = typer.prompt("Enter the expected response")

            interaction = {
                "prompt": prompt,
                "expected_response": response,
                "adapter": adapter,
                "model": model,
            }
            interactions.append(interaction)
            console.print("[green]Interaction recorded![/green]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Recording interrupted.[/yellow]")
            break

    if interactions:
        try:
            with open(output_file, "w") as f:
                json.dump(interactions, f, indent=2)
            # Sanitize output_file path string before use with Rich
            sane_output_file_str = str(output_file).replace("\u00A0", " ")
            console.print(
                f"[green]Recorded {len(interactions)} interactions to {sane_output_file_str}[/green]"
            )
            console.print("\n[bold]Next steps:[/bold]")
            console.print("1. Review the recorded interactions")
            console.print("2. Convert them to YAML test cases")
            console.print(
                "3. Run your tests with: promptdrifter run <your-test-file.yaml>"
            )
        except Exception as e:
            console.print(f"[bold red]Error saving interactions: {e}[/bold red]")
            raise typer.Exit(code=1)
    else:
        console.print("[yellow]No interactions were recorded.[/yellow]")


if __name__ == "__main__":
    app()
