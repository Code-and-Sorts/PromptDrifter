from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from rich.console import Console
from rich.table import Table

from .adapters.base import Adapter
from .adapters.gemini import GeminiAdapter
from .adapters.ollama import OllamaAdapter  # Example
from .adapters.openai import OpenAIAdapter  # Example, dynamically load based on config
from .assertions import exact_match, regex_match
from .cache import PromptCache
from .yaml_loader import YamlFileLoader

# A mapping from adapter names (as used in YAML) to their classes
ADAPTER_REGISTRY: Dict[str, Type[Adapter]] = {
    "openai": OpenAIAdapter,
    "ollama": OllamaAdapter,
    "gemini": GeminiAdapter,
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
    ) -> List[Dict[str, Any]]:
        """Runs a single test case definition for all its adapters."""
        case_id = test_case_data.get("id", "unknown_id")
        prompt_template = test_case_data.get("prompt", "")
        inputs = test_case_data.get("inputs", {})
        adapter_configs = test_case_data.get("adapter", [])
        expect_exact = test_case_data.get("expect_exact")
        expect_regex = test_case_data.get("expect_regex")
        expect_substring = test_case_data.get("expect_substring")
        expect_substring_case_insensitive = test_case_data.get("expect_substring_case_insensitive")

        base_result_details = {
            "file": str(test_case_path.name),
            "id": case_id,
            "prompt_template": prompt_template,
            "inputs": inputs,
        }

        all_adapter_results: List[Dict[str, Any]] = []

        if not prompt_template:
            all_adapter_results.append({
                **base_result_details,
                "status": "SKIPPED",
                "reason": "No prompt defined.",
                "adapter": "N/A",
                "model": "N/A",
            })
            return all_adapter_results
        if not adapter_configs:
            all_adapter_results.append({
                **base_result_details,
                "status": "SKIPPED",
                "reason": "No adapter specified.",
                "adapter": "N/A",
                "model": "N/A",
            })
            return all_adapter_results
        if not (expect_exact or expect_regex or expect_substring or expect_substring_case_insensitive):
            all_adapter_results.append({
                **base_result_details,
                "status": "SKIPPED",
                "reason": "No assertion defined.",
                "adapter": "N/A",
                "model": "N/A",
            })
            return all_adapter_results

        prompt_text = prompt_template
        for var, value in inputs.items():
            prompt_text = prompt_text.replace(f"{{{{{var}}}}}", str(value))

        for adapter_config in adapter_configs:
            current_run_details = {
                **base_result_details,
                "rendered_prompt": prompt_text,
                "status": "SKIPPED",
                "reason": "",
                "actual_response": None,
                "cache_status": "N/A",
            }

            adapter_name = adapter_config.get("type")
            model = adapter_config.get("model")
            temperature = adapter_config.get("temperature")
            max_tokens = adapter_config.get("max_tokens")
            additional_adapter_kwargs = {
                k: v for k, v in adapter_config.items() if k not in ["type", "model", "temperature", "max_tokens"]
            }

            current_run_details["adapter"] = adapter_name if adapter_name else "N/A"
            current_run_details["model"] = model if model else "N/A"

            adapter_instance = self._get_adapter_instance(adapter_name)
            if not adapter_instance:
                current_run_details["status"] = "ERROR"
                current_run_details["reason"] = (
                    f"Adapter '{adapter_name}' could not be initialized or found."
                )
                all_adapter_results.append(current_run_details)
                self.overall_success = False
                continue

            adapter_options = {
                k: v
                for k, v in {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }.items()
                if v is not None
            }
            adapter_options.update(additional_adapter_kwargs)

            llm_response_data: Optional[Dict[str, Any]] = None

            if self.cache:
                cache_options_key = frozenset(adapter_options.items())
                cached_response = self.cache.get(
                    prompt_text, adapter_name, model, cache_options_key
                )
                if cached_response:
                    llm_response_data = cached_response
                    current_run_details["cache_status"] = "HIT"

            if not llm_response_data:
                if self.cache:
                    current_run_details["cache_status"] = "MISS"
                try:
                    llm_response_data = await adapter_instance.execute(
                        prompt_text,
                        model=model,
                        **adapter_options,
                    )
                    if (
                        self.cache
                        and llm_response_data
                        and not llm_response_data.get("error")
                    ):
                        cache_options_key_put = frozenset(adapter_options.items())
                        self.cache.put(
                            prompt_text,
                            adapter_name,
                            model,
                            cache_options_key_put,
                            llm_response_data,
                        )
                except Exception as e:
                    current_run_details["status"] = "ERROR"
                    current_run_details["reason"] = f"Adapter execution error: {e}"
                    all_adapter_results.append(current_run_details)
                    self.overall_success = False
                    if hasattr(adapter_instance, "close"):
                        await adapter_instance.close()
                    continue

            if hasattr(adapter_instance, "close"):
                await adapter_instance.close()

            if not llm_response_data:
                current_run_details["status"] = "ERROR"
                current_run_details["reason"] = "No response from adapter."
                all_adapter_results.append(current_run_details)
                self.overall_success = False
                continue

            if llm_response_data.get("error"):
                current_run_details["status"] = "ERROR"
                current_run_details["reason"] = f"Adapter error: {llm_response_data['error']}"
                current_run_details["actual_response"] = llm_response_data.get("raw_response")
                all_adapter_results.append(current_run_details)
                self.overall_success = False
                continue

            actual_text_response = llm_response_data.get("text_response")
            current_run_details["actual_response"] = actual_text_response
            current_run_details["raw_adapter_response"] = llm_response_data.get("raw_response")

            if actual_text_response is None:
                current_run_details["status"] = "FAIL"
                current_run_details["reason"] = "Adapter returned no text_response."
                all_adapter_results.append(current_run_details)
                self.overall_success = False
                continue

            passed = False
            assertion_reason = ""
            if expect_exact:
                passed = exact_match(expect_exact, actual_text_response)
                if not passed:
                    assertion_reason = f"Exact match failed. Expected: '{expect_exact}'"
            elif expect_regex:
                passed = regex_match(expect_regex, actual_text_response)
                if not passed:
                    assertion_reason = f"Regex match failed. Pattern: '{expect_regex}'"
            elif expect_substring:
                passed = expect_substring in actual_text_response
                if not passed:
                    assertion_reason = f"Substring match failed. Expected to find: '{expect_substring}'"
            elif expect_substring_case_insensitive:
                passed = expect_substring_case_insensitive.lower() in actual_text_response.lower()
                if not passed:
                    assertion_reason = f"Case-insensitive substring match failed. Expected to find: '{expect_substring_case_insensitive}'"

            current_run_details["status"] = "PASS" if passed else "FAIL"
            if not passed:
                current_run_details["reason"] = assertion_reason
                self.overall_success = False
            
            all_adapter_results.append(current_run_details)

        return all_adapter_results

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
                # If adapters key exists and is a list, treat each as a test case
                if isinstance(test_suite_data, dict) and isinstance(test_suite_data.get("adapters"), list):
                    for test_case in test_suite_data["adapters"]:
                        results = await self._run_single_test_case(
                            test_file_path, test_case
                        )
                        self.results.extend(results)
                        ran_any_test = True
                else:
                    results = await self._run_single_test_case(
                        test_file_path, test_suite_data
                    )
                    self.results.extend(results)
                    ran_any_test = True
            except ValueError as e:
                if str(e).startswith("[bold red]Schema Validation Error"):
                    self.console.print(str(e))
                else:
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
            except Exception as e:
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
