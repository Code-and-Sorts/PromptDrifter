from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from jinja2 import Template
from rich.console import Console
from rich.table import Table

from promptdrifter.models.config import TestCase

from .adapters.base import Adapter
from .adapters.claude import ClaudeAdapter
from .adapters.deepseek import DeepSeekAdapter
from .adapters.gemini import GeminiAdapter
from .adapters.grok import GrokAdapter
from .adapters.llama import LlamaAdapter
from .adapters.mistral import MistralAdapter
from .adapters.ollama import OllamaAdapter
from .adapters.openai import OpenAIAdapter
from .adapters.qwen import QwenAdapter
from .assertions import exact_match, regex_match
from .cache import PromptCache
from .yaml_loader import YamlFileLoader

ADAPTER_REGISTRY: Dict[str, Type[Adapter]] = {
    "openai": OpenAIAdapter,
    "ollama": OllamaAdapter,
    "gemini": GeminiAdapter,
    "qwen": QwenAdapter,
    "claude": ClaudeAdapter,
    "grok": GrokAdapter,
    "deepseek": DeepSeekAdapter,
    "llama": LlamaAdapter,
    "mistral": MistralAdapter,
}


class Runner:
    """Orchestrates loading test suites, running them, and reporting results."""

    def __init__(
        self,
        config_dir: Path,
        cache_db_path: Optional[Path] = None,
        use_cache: bool = True,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        qwen_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        grok_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        meta_llama_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
    ):
        self.config_dir = config_dir
        self.yaml_loader = YamlFileLoader()
        if use_cache:
            if cache_db_path is not None:
                self.cache = PromptCache(db_path=cache_db_path)
            else:
                self.cache = PromptCache()
        else:
            self.cache = None
        self.console = Console()
        self.results: List[Dict[str, Any]] = []
        self.overall_success = True
        self.cli_openai_key = openai_api_key
        self.cli_gemini_key = gemini_api_key
        self.cli_qwen_key = qwen_api_key
        self.cli_anthropic_key = anthropic_api_key
        self.cli_grok_key = grok_api_key
        self.cli_deepseek_key = deepseek_api_key
        self.cli_meta_llama_key = meta_llama_api_key
        self.cli_mistral_key = mistral_api_key

    async def close_cache_connection(self):
        """Closes the database connection if cache is enabled and connection exists."""
        if self.cache and hasattr(self.cache, "close"):
            self.cache.close()

    def _get_adapter_instance(self, adapter_name: str) -> Optional[Adapter]:
        """Retrieves an initialized adapter instance from the registry."""
        adapter_class = ADAPTER_REGISTRY.get(adapter_name.lower())
        if adapter_class:
            try:
                api_key_to_pass = None
                if adapter_name.lower() == "openai" and self.cli_openai_key:
                    api_key_to_pass = self.cli_openai_key
                elif adapter_name.lower() == "gemini" and self.cli_gemini_key:
                    api_key_to_pass = self.cli_gemini_key
                elif adapter_name.lower() == "qwen" and self.cli_qwen_key:
                    api_key_to_pass = self.cli_qwen_key
                elif adapter_name.lower() == "claude" and self.cli_anthropic_key:
                    api_key_to_pass = self.cli_anthropic_key
                elif adapter_name.lower() == "grok" and self.cli_grok_key:
                    api_key_to_pass = self.cli_grok_key
                elif adapter_name.lower() == "deepseek" and self.cli_deepseek_key:
                    api_key_to_pass = self.cli_deepseek_key
                elif adapter_name.lower() == "llama" and self.cli_meta_llama_key:
                    api_key_to_pass = self.cli_meta_llama_key
                elif adapter_name.lower() == "mistral" and self.cli_mistral_key:
                    api_key_to_pass = self.cli_mistral_key

                if api_key_to_pass:
                    return adapter_class(api_key=api_key_to_pass)
                else:
                    return adapter_class()
            except Exception as e:
                self.console.print(
                    f"[bold red]Error initializing adapter '{adapter_name}': {e}[/bold red]"
                )
                return None
        self.console.print(f"[bold red]Unknown adapter: {adapter_name}[/bold red]")
        return None

    async def _run_single_test_case(
        self,
        test_case_path: Path,
        test_case_model: TestCase,
    ) -> List[Dict[str, Any]]:
        all_adapter_results = []

        test_id = test_case_model.id
        base_prompt = test_case_model.prompt
        inputs = test_case_model.inputs

        prompt_text = base_prompt
        if inputs:
            try:
                template = Template(base_prompt)
                prompt_text = template.render(**inputs)
            except Exception as e:
                self.console.print(
                    f"[bold red]Error rendering prompt for test ID '{test_id}' in file '{test_case_path.name}': {e}[/bold red]"
                )
                return [
                    {
                        "file": str(test_case_path.name),
                        "id": test_id,
                        "adapter": "N/A",
                        "model": "N/A",
                        "status": "ERROR",
                        "reason": f"Prompt templating error: {e}",
                        "prompt": base_prompt,
                        "inputs": inputs,
                    }
                ]

        expect_exact = test_case_model.expect_exact
        expect_regex = test_case_model.expect_regex
        expect_substring = test_case_model.expect_substring
        expect_substring_case_insensitive = (
            test_case_model.expect_substring_case_insensitive
        )

        for adapter_config_model in test_case_model.adapter_configurations:
            adapter_name = adapter_config_model.adapter_type
            model_name = adapter_config_model.model

            current_run_details = {
                "file": str(test_case_path.name),
                "id": test_id,
                "adapter": adapter_name,
                "model": model_name,
                "status": "SKIPPED",
                "reason": "",
                "prompt": prompt_text,
                "expected": expect_exact
                or expect_regex
                or expect_substring
                or expect_substring_case_insensitive,
                "inputs": inputs,
                "cache_status": "N/A",
                "actual_response": None,
                "raw_adapter_response": None,
            }

            adapter_instance = self._get_adapter_instance(adapter_name)
            if adapter_instance is None:
                current_run_details["reason"] = (
                    f"Adapter '{adapter_name}' not found or failed to initialize."
                )
                current_run_details["status"] = "ERROR"
                all_adapter_results.append(current_run_details)
                self.overall_success = False
                continue

            all_adapter_params = adapter_config_model.model_dump(
                by_alias=True, exclude_none=True
            )

            known_options_to_pass = {}
            if "temperature" in all_adapter_params:
                known_options_to_pass["temperature"] = all_adapter_params.pop(
                    "temperature"
                )
            if "max_tokens" in all_adapter_params:
                known_options_to_pass["max_tokens"] = all_adapter_params.pop(
                    "max_tokens"
                )

            all_adapter_params.pop("type", None)
            all_adapter_params.pop("model", None)

            additional_adapter_kwargs = all_adapter_params

            adapter_options = {**known_options_to_pass, **additional_adapter_kwargs}

            llm_response_data: Optional[Dict[str, Any]] = None
            cache_key_options_component = None

            if self.cache:
                assertion_details_for_cache = []
                if test_case_model.expect_exact is not None:
                    assertion_details_for_cache.append(("_assertion_type", "exact"))
                    assertion_details_for_cache.append(
                        ("_assertion_value", test_case_model.expect_exact)
                    )
                elif test_case_model.expect_regex is not None:
                    assertion_details_for_cache.append(("_assertion_type", "regex"))
                    assertion_details_for_cache.append(
                        ("_assertion_value", test_case_model.expect_regex)
                    )
                elif test_case_model.expect_substring is not None:
                    assertion_details_for_cache.append(("_assertion_type", "substring"))
                    assertion_details_for_cache.append(
                        ("_assertion_value", test_case_model.expect_substring)
                    )
                elif test_case_model.expect_substring_case_insensitive is not None:
                    assertion_details_for_cache.append(
                        ("_assertion_type", "substring_case_insensitive")
                    )
                    assertion_details_for_cache.append(
                        (
                            "_assertion_value",
                            test_case_model.expect_substring_case_insensitive,
                        )
                    )

                sorted_adapter_options_items = sorted(
                    list(adapter_options.items()), key=lambda item: item[0]
                )
                combined_options_for_cache_key = (
                    sorted_adapter_options_items + assertion_details_for_cache
                )
                cache_key_options_component = frozenset(combined_options_for_cache_key)

                cached_response = self.cache.get(
                    prompt_text, adapter_name, model_name, cache_key_options_component
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
                        model=model_name,
                        **adapter_options,
                    )
                    if (
                        self.cache
                        and cache_key_options_component is not None
                        and llm_response_data
                        and not llm_response_data.get("error")
                    ):
                        self.cache.put(
                            prompt_text,
                            adapter_name,
                            model_name,
                            cache_key_options_component,
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
                current_run_details["reason"] = (
                    f"Adapter error: {llm_response_data['error']}"
                )
                current_run_details["actual_response"] = llm_response_data.get(
                    "raw_response"
                )
                all_adapter_results.append(current_run_details)
                self.overall_success = False
                continue

            actual_text_response = llm_response_data.get("text_response")
            current_run_details["actual_response"] = actual_text_response
            current_run_details["raw_adapter_response"] = llm_response_data.get(
                "raw_response"
            )

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
                passed = (
                    expect_substring_case_insensitive.lower()
                    in actual_text_response.lower()
                )
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
                config_model = self.yaml_loader.load_and_validate_yaml(test_file_path)

                if config_model and config_model.tests:
                    for test_case_model in config_model.tests:
                        results = await self._run_single_test_case(
                            test_file_path, test_case_model
                        )
                        self.results.extend(results)
                        ran_any_test = True
                elif not config_model.tests:
                    self.console.print(
                        f"[yellow]Warning: No tests found in {test_file_path} (version: {config_model.version if config_model else 'N/A'}).[/yellow]"
                    )
            except ValueError as e:
                self.console.print(str(e))
                self.overall_success = False
                self.results.append(
                    {
                        "file": str(test_file_path.name),
                        "id": "YAML_LOAD_ERROR",
                        "adapter": "N/A",
                        "model": "N/A",
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
        table.add_column("Reason/Details", width=50, overflow="fold")
        table.add_column("Cached", justify="center")

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
                reason += f"\n----\nActual: '{str(result.get('actual_response'))}'"
            elif status == "ERROR" and result.get("raw_adapter_response"):
                reason += f"\nAdapter Raw Response: '{str(result.get('raw_adapter_response'))}'"

            table.add_row(
                result.get("file", "N/A"),
                result.get("id", "N/A"),
                result.get("adapter", "N/A"),
                result.get("model", "N/A"),
                f"[{status_color}]{status}[/{status_color}]",
                reason,
                result.get("cache_status", "N/A"),
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
