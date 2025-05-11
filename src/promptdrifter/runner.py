from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from rich.console import Console
from rich.table import Table

from .adapters.base import Adapter
from .adapters.ollama import OllamaAdapter  # Example
from .adapters.openai import OpenAIAdapter  # Example, dynamically load based on config
from .assertions import exact_match, regex_match
from .cache import PromptCache
from .yaml_loader import YamlFileLoader

# A mapping from adapter names (as used in YAML) to their classes
ADAPTER_REGISTRY: Dict[str, Type[Adapter]] = {
    "openai": OpenAIAdapter,
    "ollama": OllamaAdapter,
    # Add other adapters here as they are created
}


class Runner:
    """Orchestrates loading test suites, running them, and reporting results."""

    def __init__(
        self,
        config_dir: Path,
        cache_db_path: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self.config_dir = config_dir  # Directory containing YAML test files
        self.yaml_loader = (
            YamlFileLoader()
        )  # Schema path is defaulted in YamlFileLoader
        if use_cache:
            if cache_db_path is not None:
                self.cache = PromptCache(db_path=cache_db_path)
            else:
                self.cache = PromptCache()
        else:
            self.cache = None
        self.console = Console()
        self.results: List[Dict[str, Any]] = []  # To store results of each test case
        self.overall_success = True  # Track if all tests passed

    async def close_cache_connection(self):
        """Closes the database connection if cache is enabled and connection exists."""
        if self.cache and hasattr(self.cache, "close"):
            self.cache.close()

    def _get_adapter_instance(self, adapter_name: str) -> Optional[Adapter]:
        """Retrieves an initialized adapter instance from the registry."""
        adapter_class = ADAPTER_REGISTRY.get(adapter_name.lower())
        if adapter_class:
            try:
                # Adapters might require specific API keys or configs during __init__
                # This part may need to be enhanced to pass config to adapters if needed
                # For now, assuming adapters can be initialized or get config from env vars
                return adapter_class()
            except Exception as e:
                self.console.print(
                    f"[bold red]Error initializing adapter '{adapter_name}': {e}[/bold red]"
                )
                return None
        self.console.print(f"[bold red]Unknown adapter: {adapter_name}[/bold red]")
        return None

    async def _run_single_test_case(
        self, test_case_path: Path, test_case_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs a single test case definition."""
        case_id = test_case_data.get("id", "unknown_id")
        prompt_template = test_case_data.get("prompt", "")
        inputs = test_case_data.get("inputs", {})
        adapter_name = test_case_data.get("adapter")
        model = test_case_data.get("model")
        temperature = test_case_data.get("temperature")
        max_tokens = test_case_data.get("max_tokens")
        expect_exact = test_case_data.get("expect_exact")
        expect_regex = test_case_data.get("expect_regex")

        result_details = {
            "file": str(test_case_path.name),
            "id": case_id,
            "prompt_template": prompt_template,
            "inputs": inputs,
            "adapter": adapter_name,
            "model": model,
            "status": "SKIPPED",  # Default status
            "reason": "",
            "actual_response": None,
        }

        if not prompt_template:
            result_details["reason"] = "No prompt defined."
            return result_details
        if not adapter_name:
            result_details["reason"] = "No adapter specified."
            return result_details
        if not (expect_exact or expect_regex):
            result_details["reason"] = (
                "No assertion (expect_exact or expect_regex) defined."
            )
            return result_details

        # Substitute inputs into prompt
        prompt_text = prompt_template
        for var, value in inputs.items():
            prompt_text = prompt_text.replace(f"{{{{{var}}}}}", str(value))
        result_details["rendered_prompt"] = prompt_text

        adapter_instance = self._get_adapter_instance(adapter_name)
        if not adapter_instance:
            result_details["reason"] = (
                f"Adapter '{adapter_name}' could not be initialized or found."
            )
            return result_details

        adapter_options = {
            # Only include if not None, to let adapter defaults take precedence
            k: v
            for k, v in {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }.items()
            if v is not None
        }
        # Pass through any other top-level fields from YAML that aren't core runner fields as adapter options
        for key, value in test_case_data.items():
            if key not in [
                "version",
                "id",
                "prompt",
                "inputs",
                "expect_exact",
                "expect_regex",
                "adapter",
                "model",
                "temperature",
                "max_tokens",
                "tags",
            ]:
                adapter_options[key] = value

        llm_response_data: Optional[Dict[str, Any]] = None

        # Cache lookup
        if self.cache:
            cached_response = self.cache.get(
                prompt_text, adapter_name, model, adapter_options
            )
            if cached_response:
                llm_response_data = cached_response
                result_details["cache_status"] = "HIT"

        if not llm_response_data:
            if self.cache:
                result_details["cache_status"] = "MISS"
            try:
                llm_response_data = await adapter_instance.execute(
                    prompt_text,
                    model=model,
                    **adapter_options,  # Pass constructed options
                )
                if (
                    self.cache
                    and llm_response_data
                    and not llm_response_data.get("error")
                ):
                    self.cache.put(
                        prompt_text,
                        adapter_name,
                        model,
                        adapter_options,
                        llm_response_data,
                    )
            except Exception as e:
                result_details["status"] = "ERROR"
                result_details["reason"] = f"Adapter execution error: {e}"
                if hasattr(adapter_instance, "close"):
                    await adapter_instance.close()
                return result_details

        if hasattr(adapter_instance, "close"):  # Ensure adapter client is closed
            await adapter_instance.close()

        if not llm_response_data:
            result_details["status"] = "ERROR"
            result_details["reason"] = "No response from adapter."
            return result_details

        if llm_response_data.get("error"):
            result_details["status"] = "ERROR"
            result_details["reason"] = f"Adapter error: {llm_response_data['error']}"
            result_details["actual_response"] = llm_response_data.get("raw_response")
            return result_details

        actual_text_response = llm_response_data.get("text_response")
        result_details["actual_response"] = (
            actual_text_response  # Could be full JSON or just text
        )
        result_details["raw_adapter_response"] = llm_response_data.get("raw_response")

        if actual_text_response is None:
            result_details["status"] = "FAIL"
            result_details["reason"] = "Adapter returned no text_response."
            return result_details

        # Perform assertions
        passed = False
        if expect_exact:
            passed = exact_match(expect_exact, actual_text_response)
            if not passed:
                result_details["reason"] = (
                    f"Exact match failed. Expected: '{expect_exact}'"
                )
        elif expect_regex:
            passed = regex_match(expect_regex, actual_text_response)
            if not passed:
                result_details["reason"] = (
                    f"Regex match failed. Pattern: '{expect_regex}'"
                )

        result_details["status"] = "PASS" if passed else "FAIL"
        if not passed:
            self.overall_success = False

        return result_details

    async def run_suite(self, test_file_paths: List[Path]):
        """Loads YAML files from a directory and runs all test cases defined within them."""
        self.results = []
        self.overall_success = True
        ran_any_test = False

        for test_file_path in test_file_paths:
            if isinstance(test_file_path, str):
                test_file_path = Path(test_file_path)
            if not test_file_path.is_file() or not (
                test_file_path.name.endswith(".yaml")
                or test_file_path.name.endswith(".yml")
            ):
                self.console.print(
                    f"[yellow]Skipping non-YAML file: {test_file_path}[/yellow]"
                )
                continue

            self.console.print(f"[cyan]Processing test file: {test_file_path}[/cyan]")
            try:
                test_suite_data = self.yaml_loader.load_and_validate_yaml(
                    test_file_path
                )
                result = await self._run_single_test_case(
                    test_file_path, test_suite_data
                )
                self.results.append(result)
                ran_any_test = True
            except ValueError as e:
                # Check if this is our custom schema validation error
                if str(e).startswith("[bold red]Schema Validation Error"):
                    self.console.print(
                        str(e)
                    )  # Print the pre-formatted message directly
                else:
                    # It's some other ValueError, use the standard formatting
                    self.console.print(
                        f"[bold red]Error processing {test_file_path}: {e}[/bold red]"
                    )
                self.overall_success = False
                self.results.append(
                    {
                        "file": str(test_file_path.name),
                        "status": "ERROR",
                        "reason": str(e),
                    }
                )
            except Exception as e:  # Catch other non-ValueError exceptions
                self.console.print(
                    f"[bold red]Error processing {test_file_path}: {e}[/bold red]"
                )
                self.overall_success = False
                self.results.append(
                    {
                        "file": str(test_file_path.name),
                        "status": "ERROR",
                        "reason": str(e),
                    }
                )

        if ran_any_test:
            self._report_results()
        return self.overall_success

    def _report_results(self):
        """Prints a summary of all test results using Rich table."""
        if not self.results:
            self.console.print("[yellow]No test results to report.[/yellow]")
            return

        table = Table(title="PromptDrifter Test Results")
        table.add_column("File", style="dim", width=20)
        table.add_column("ID", style="cyan", width=20)
        table.add_column("Adapter", style="magenta", width=10)
        table.add_column("Model", style="blue", width=15)
        table.add_column("Status", justify="center")
        table.add_column("Reason/Details", width=50)
        # table.add_column("Cached", justify="center")

        summary = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIPPED": 0, "TOTAL": 0}

        for result in self.results:
            status = result.get("status", "SKIPPED")
            summary[status] = summary.get(status, 0) + 1
            summary["TOTAL"] += 1

            status_color = (
                "green" if status == "PASS" else "red" if status == "FAIL" else "yellow"
            )

            reason = str(result.get("reason", ""))
            if status == "FAIL" and result.get("actual_response") is not None:
                reason += f"\nActual: '{str(result.get('actual_response'))[:100]}...'"
            elif status == "ERROR" and result.get("actual_response"):
                # For ERROR status, result.get("actual_response") contains the raw adapter response if available
                reason += f"\nAdapter Raw Response: '{str(result.get('actual_response'))[:100]}...'"

            table.add_row(
                result.get("file", "N/A"),
                result.get("id", "N/A"),
                result.get("adapter", "N/A"),
                result.get("model", "N/A"),
                f"[{status_color}]{status}[/{status_color}]",
                reason,
                # result.get("cache_status", "N/A")
            )

        self.console.print(table)
        self.console.print("\n[bold]Summary:[/bold]")
        for status, count in summary.items():
            if status == "TOTAL":
                continue
            color = (
                "green" if status == "PASS" else "red" if status == "FAIL" else "yellow"
            )
            self.console.print(f"  {status}: [{color}]{count}[/{color}]")
        self.console.print(f"  TOTAL: {summary['TOTAL']}")

        if not self.overall_success:
            self.console.print(
                "\n[bold red]Some tests failed or encountered errors.[/bold red]"
            )
        else:
            self.console.print(
                "\n[bold green]All tests passed successfully![/bold green]"
            )


# Example of how this might be invoked from cli.py (run command)
# async def main_run(test_files: List[Path]):
#     runner = Runner(config_dir=Path(".")) # Assuming test files are relative to cwd or specific paths given
#     success = await runner.run_suite(test_files)
#     if not success:
#         raise typer.Exit(code=1)

# if __name__ == '__main__':
#     # This is a placeholder for actual invocation
#     # You would typically get test file paths from CLI arguments
#     # Create dummy YAML for testing runner directly (if needed)
#     # Ensure schema/v0.1.json exists or YamlFileLoader will fail

#     async def run_example():
#         example_test_dir = Path("example_tests")
#         example_test_dir.mkdir(exist_ok=True)
#         schema_dir = Path("schema")
#         schema_dir.mkdir(exist_ok=True)

#         if not (schema_dir / "v0.1.json").exists():
#             print("Creating dummy schema/v0.1.json for runner example")
#             with open(schema_dir / "v0.1.json", "w") as f:
#                 json.dump({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
#                            "properties": {"version": {"type": "string"}, "id": {"type": "string"}, "prompt": {"type": "string"}},
#                            "required": ["version", "id", "prompt"]
#                           }, f)

#         test_yaml_content = """
# version: '0.1'
# id: 'sample-test-001'
# prompt: 'What is 2+2?'
# adapter: 'ollama' # Make sure ollama is running and has a default model like llama2
# model: 'llama2' # Or your default ollama model
# expect_exact: '4'
#         """
#         with open(example_test_dir / "test1.yaml", "w") as f:
#             f.write(test_yaml_content)

#         test_yaml_error_content = """
# version: '0.1'
# id: 'error-test-openai'
# prompt: 'Translate to French: Hello'
# adapter: 'openai' # Requires OPENAI_API_KEY
# model: 'gpt-3.5-turbo'
# expect_exact: 'Bonjour'
#         """
#         with open(example_test_dir / "test_openai_error.yaml", "w") as f:
#             f.write(test_yaml_error_content)

#         runner = Runner(config_dir=example_test_dir, use_cache=False)
#         await runner.run_suite([example_test_dir / "test1.yaml", example_test_dir / "test_openai_error.yaml"])

#         # Clean up
#         # (example_test_dir / "test1.yaml").unlink()
#         # (example_test_dir / "test_openai_error.yaml").unlink()
#         # example_test_dir.rmdir()
#         # (schema_dir / "v0.1.json").unlink(missing_ok=True) # if dummy created
#         # schema_dir.rmdir(missing_ok=True)

#     asyncio.run(run_example())
