import asyncio
import importlib.metadata
import importlib.resources
import json
import os
import tomllib
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from promptdrifter.runner import Runner

app = typer.Typer(no_args_is_help=True)
console = Console()


def get_version():
    """Get the package version from pyproject.toml or package metadata."""
    try:
        package_root = Path(__file__).parent.parent.parent
        pyproject_path = package_root / "pyproject.toml"

        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                toml_data = tomllib.load(f)
                if "project" in toml_data and "version" in toml_data["project"]:
                    version_raw = toml_data["project"]["version"]
                    try:
                        from packaging.version import Version
                        return Version(version_raw).public
                    except ImportError:
                        return version_raw
        return importlib.metadata.version("promptdrifter")
    except Exception:
        return "0.0.1"


@app.command()
def version():
    """Display the current version of PromptDrifter."""
    ascii_logo = """
                                             [bold #4bcbf1]▒▒▒▒▒▒▒▒▒▒▒▒▒
                                          ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                                        ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                                       ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                                      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒
                                      ▒▒▒▒▒▒▒▒▒▒▒        ▒▒▒▒▒▒▒▒
                                     ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒   ▒▒▒▒▒▒▒▒▒▒
                                     ▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒▒
                                      ▒▒▒▒▒▒▒▒        ▒▒▒▒▒▒▒▒▒▒▒
                                      ▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                                       ▒▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                                        ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                                         ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                                            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                                                ▒▒▒▒▒▒▒[/bold #4bcbf1]

 [bold #9f4cf2]▒▒▒▒▒▒▒                                          ▒▒▒  ▒▒▒▒▒▒▒▒         ▒▒▒▒  ▒▒▒▒▒  ▒▒▒
 ▒▒  ▒▒▒ ▒▒▒▒▒▒▒  ▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒  ▒▒▒ ▒▒▒▒▒▒      ▒▒▒▒▒▒ ▒▒▒▒▒▒   ▒▒▒▒▒▒  ▒▒▒▒▒▒
 ▒▒▒▒▒▒▒ ▒▒▒▒▒▒▒ ▒▒▒  ▒▒▒ ▒▒▒▒ ▒▒▒▒ ▒▒▒ ▒▒▒ ▒▒▒▒  ▒▒▒  ▒▒▒▒  ▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒  ▒▒▒▒   ▒▒▒  ▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒▒▒
 ▒▒▒▒▒   ▒▒▒▒    ▒▒▒  ▒▒▒ ▒▒▒▒ ▒▒▒▒ ▒▒▒ ▒▒▒  ▒▒▒  ▒▒▒  ▒▒▒▒  ▒▒▒ ▒▒▒    ▒▒▒▒  ▒▒▒▒   ▒▒▒  ▒▒▒▒▒▒▒▒▒  ▒▒▒
 ▒▒      ▒▒▒▒    ▒▒▒  ▒▒▒ ▒▒▒▒ ▒▒▒▒ ▒▒▒ ▒▒▒  ▒▒▒  ▒▒▒  ▒▒▒▒ ▒▒▒▒ ▒▒▒    ▒▒▒▒  ▒▒▒▒   ▒▒▒  ▒▒▒▒       ▒▒▒
 ▒▒      ▒▒▒▒     ▒▒▒▒▒▒  ▒▒▒▒ ▒▒▒▒ ▒▒▒ ▒▒▒▒▒▒▒   ▒▒▒▒ ▒▒▒▒▒▒▒▒  ▒▒▒    ▒▒▒▒  ▒▒▒▒   ▒▒▒▒  ▒▒▒▒▒▒▒   ▒▒▒
                                        ▒▒▒
                                        ▒▒▒[/bold #9f4cf2]
    """
    console.print(ascii_logo)
    console.print(f"Version: [bold #4bcbf1]{get_version()}[/bold #4bcbf1]")


def _print_api_key_security_warning():
    """Prints a security warning if API keys are passed via CLI."""
    warning_message = (
        "[bold yellow]SECURITY WARNING:[/bold yellow]\n"
        "Passing API keys directly via command-line arguments can expose them in your shell history "
        "or process list. It is recommended to use environment variables for API keys where possible."
    )
    console.print(
        Panel(warning_message, title="[bold red]Warning[/bold red]", border_style="red")
    )


@app.command()
def init(
    ctx: typer.Context,
    target_path_str: str = typer.Argument(
        ".",
        help="The directory to initialize the project in. Defaults to current directory.",
    ),
):
    """Initialize a new promptdrifter project with a sample config."""
    target_path = Path(target_path_str).resolve()
    # TODO: Check all yaml files in the target path and dynamically determine if they are valid promptdrifter files
    config_file_path = target_path / "promptdrifter.yaml"

    sane_target_path_str = str(target_path).replace("\u00a0", " ")
    sane_config_file_path_str = str(config_file_path).replace("\u00a0", " ")

    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
        message = Text("Created directory: ", style="green")
        path_text = Text(
            sane_target_path_str, style="green", no_wrap=True, overflow="ignore"
        )
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
        path_text = Text(
            sane_config_file_path_str, style="green", no_wrap=True, overflow="ignore"
        )
        message.append(path_text)
        console.print(message)
        console.print("You can now edit this file and run 'promptdrifter run'.")
    except IOError as e:
        message = Text("Error writing configuration file to '", style="bold red")
        path_text = Text(
            sane_config_file_path_str, style="bold red", no_wrap=True, overflow="ignore"
        )
        message.append(path_text)
        message.append(f"': {e}", style="bold red")
        console.print(message)
        raise typer.Exit(code=1)


def is_from_env(param_name: str, env_var: str) -> bool:
    """Check if a parameter was provided via environment variable rather than CLI."""
    return env_var in os.environ


async def _run_async(
    files: List[Path],
    no_cache: bool,
    cache_db: Optional[Path],
    config_dir: Path,
    openai_api_key: Optional[str],
    gemini_api_key: Optional[str],
    qwen_api_key: Optional[str],
    claude_api_key: Optional[str],
    grok_api_key: Optional[str],
    deepseek_api_key: Optional[str],
    mistral_api_key: Optional[str],
    # llama_api_key: Optional[str],
):
    """Async implementation of the run command."""
    if (
        (openai_api_key and not is_from_env("openai_api_key", "OPENAI_API_KEY"))
        or (gemini_api_key and not is_from_env("gemini_api_key", "GEMINI_API_KEY"))
        or (qwen_api_key and not is_from_env("qwen_api_key", "QWEN_API_KEY"))
        or (claude_api_key and not is_from_env("claude_api_key", "CLAUDE_API_KEY"))
        or (grok_api_key and not is_from_env("grok_api_key", "GROK_API_KEY"))
        or (deepseek_api_key and not is_from_env("deepseek_api_key", "DEEPSEEK_API_KEY"))
        or (mistral_api_key and not is_from_env("mistral_api_key", "MISTRAL_API_KEY"))
        # or (llama_api_key and not is_from_env("llama_api_key", "LLAMA_API_KEY"))
    ):
        _print_api_key_security_warning()

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
            sane_file_path_str = str(file_path_obj).replace("\u00a0", " ")
            console.print(f"  • {sane_file_path_str}: {reason}")
        console.print("\n[bold]Please provide valid YAML test files.[/bold]")
        raise typer.Exit(code=1)

    runner_instance: Optional[Runner] = None
    try:
        runner_instance = Runner(
            config_dir=config_dir,
            cache_db_path=cache_db,
            use_cache=not no_cache,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            qwen_api_key=qwen_api_key,
            claude_api_key=claude_api_key,
            grok_api_key=grok_api_key,
            deepseek_api_key=deepseek_api_key,
            mistral_api_key=mistral_api_key,
            # llama_api_key=llama_api_key,
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
    openai_api_key: Optional[str] = typer.Option(
        None,
        "--openai-api-key",
        help="OpenAI API key. Overrides OPENAI_API_KEY env var. Warning: Exposes key in shell history.",
        envvar="OPENAI_API_KEY",
        rich_help_panel="API Keys",
    ),
    gemini_api_key: Optional[str] = typer.Option(
        None,
        "--gemini-api-key",
        help="Google Gemini API key. Overrides GEMINI_API_KEY env var. Warning: Exposes key in shell history.",
        envvar="GEMINI_API_KEY",
        rich_help_panel="API Keys",
    ),
    qwen_api_key: Optional[str] = typer.Option(
        None,
        "--qwen-api-key",
        help="Qwen API key (DashScope). Overrides QWEN_API_KEY or DASHSCOPE_API_KEY env var. Warning: Exposes key in shell history.",
        envvar="QWEN_API_KEY",
        rich_help_panel="API Keys",
    ),
    claude_api_key: Optional[str] = typer.Option(
        None,
        "--claude-api-key",
        help="Anthropic Claude API key. Overrides CLAUDE_API_KEY env var. Warning: Exposes key in shell history.",
        envvar="CLAUDE_API_KEY",
        rich_help_panel="API Keys",
    ),
    grok_api_key: Optional[str] = typer.Option(
        None,
        "--grok-api-key",
        help="Grok API key (can also be set via GROK_API_KEY env var. Warning: Exposes key in shell history.)",
        envvar="GROK_API_KEY",
        rich_help_panel="API Keys",
    ),
    deepseek_api_key: Optional[str] = typer.Option(
        None,
        "--deepseek-api-key",
        help="DeepSeek API key (can also be set via DEEPSEEK_API_KEY env var. Warning: Exposes key in shell history.)",
        envvar="DEEPSEEK_API_KEY",
        rich_help_panel="API Keys",
    ),
    mistral_api_key: Optional[str] = typer.Option(
        None,
        "--mistral-api-key",
        help="Mistral API key (can also be set via MISTRAL_API_KEY env var. Warning: Exposes key in shell history.)",
        envvar="MISTRAL_API_KEY",
        rich_help_panel="API Keys",
    ),
    # llama_api_key: Optional[str] = typer.Option(
    #     None,
    #     "--llama-api-key",
    #     help="Meta Llama API key (can also be set via LLAMA_API_KEY env var. Warning: Exposes key in shell history.)",
    #     envvar="LLAMA_API_KEY",
    #     rich_help_panel="API Keys",
    # ),
):
    """Run a suite of prompt tests from one or more YAML files."""
    asyncio.run(
        _run_async(
            files,
            no_cache,
            cache_db,
            config_dir,
            openai_api_key,
            gemini_api_key,
            qwen_api_key,
            claude_api_key,
            grok_api_key,
            deepseek_api_key,
            mistral_api_key,
            # llama_api_key,
        )
    )


@app.command()
def record(
    output_file: Path = typer.Option(
        Path("recorded_interactions.json"),
        "--output",
        "-o",
        help="Path to save recorded interactions",
    ),
    adapter: str = typer.Option(
        ..., "--adapter", "-a", help="LLM adapter to use (e.g., openai, ollama, claude, grok)"
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
            sane_output_file_str = str(output_file).replace("\u00a0", " ")
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
