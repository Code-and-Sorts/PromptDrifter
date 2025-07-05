import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template
from rich.console import Console
from rich.table import Table

from promptdrifter.models.config import TestCase

from .adapter_manager import get_adapter_manager
from .adapters.base import Adapter
from .cache import PromptCache
from .yaml_loader import YamlFileLoader


class Runner:
    """Orchestrates loading test suites, running them, and reporting results."""

    def __init__(
        self,
        config_dir: Path,
        cache_db_path: Optional[Path] = None,
        use_cache: bool = True,
        max_concurrent_prompt_tests: int = 10,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        qwen_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        grok_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        # llama_api_key: Optional[str] = None,
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

        # Store API keys for adapter manager
        self.api_keys = {
            "openai": openai_api_key,
            "gemini": gemini_api_key,
            "qwen": qwen_api_key,
            "claude": claude_api_key,
            "grok": grok_api_key,
            "deepseek": deepseek_api_key,
            "mistral": mistral_api_key,
        }

        # Initialize high-performance adapter manager
        self.adapter_manager = get_adapter_manager()

        # Configure concurrency limits for prompt test execution
        self.max_concurrent_prompt_tests = max_concurrent_prompt_tests
        self._semaphore = asyncio.Semaphore(max_concurrent_prompt_tests)

    async def close_cache_connection(self):
        """Closes the database connection if cache is enabled and connection exists."""
        if self.cache and hasattr(self.cache, "close"):
            self.cache.close()

        # Close all adapter connections
        await self.adapter_manager.close_all_adapters()

    async def _get_adapter_instance(self, adapter_name: str, base_url: Optional[str] = None) -> Optional[Adapter]:
        """Retrieves an initialized adapter instance using the optimized adapter manager."""
        adapter_name = adapter_name.lower()
        api_key = self.api_keys.get(adapter_name)

        try:
            adapter = await self.adapter_manager.get_adapter(
                adapter_type=adapter_name,
                api_key=api_key,
                base_url=base_url
            )

            if adapter is None:
                self.console.print(f"[bold red]Unknown adapter: {adapter_name}[/bold red]")

            return adapter

        except Exception as e:
            self.console.print(
                f"[bold red]Error initializing adapter '{adapter_name}': {e}[/bold red]"
            )
            return None

    async def _run_single_test_case_with_semaphore(
        self,
        test_case_path: Path,
        test_case_model: TestCase,
    ) -> List[Dict[str, Any]]:
        """Wrapper for _run_single_test_case with concurrency control."""
        async with self._semaphore:
            return await self._run_single_test_case(test_case_path, test_case_model)

    async def _run_single_test_case(
        self,
        test_case_path: Path,
        test_case_model: TestCase,
    ) -> List[Dict[str, Any]]:
        all_adapter_results = []

        test_id = test_case_model.id
        base_prompt = test_case_model.prompt
        inputs = test_case_model.inputs
        tags = test_case_model.tags or []

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
                        "tags": tags,
                    }
                ]

        expect_exact = test_case_model.expect_exact
        expect_regex = test_case_model.expect_regex
        expect_substring = test_case_model.expect_substring
        expect_substring_case_insensitive = (
            test_case_model.expect_substring_case_insensitive
        )

        adapter_instances = []
        adapter_details = []

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
                "tags": tags,
            }

            all_adapter_params = adapter_config_model.model_dump(
                by_alias=True, exclude_none=True
            )

            base_url = all_adapter_params.get("base_url")

            adapter_instance = await self._get_adapter_instance(adapter_name, base_url)
            if adapter_instance is None:
                current_run_details["reason"] = (
                    f"Adapter '{adapter_name}' not found or failed to initialize."
                )
                current_run_details["status"] = "ERROR"
                all_adapter_results.append(current_run_details)
                self.overall_success = False
                continue

            known_options_to_pass = {}
            if "temperature" in all_adapter_params:
                known_options_to_pass["temperature"] = all_adapter_params.pop(
                    "temperature"
                )
            if "max_tokens" in all_adapter_params:
                known_options_to_pass["max_tokens"] = all_adapter_params.pop(
                    "max_tokens"
                )

            all_adapter_params.pop("base_url", None)
            all_adapter_params.pop("type", None)
            all_adapter_params.pop("model", None)

            additional_adapter_kwargs = all_adapter_params

            adapter_options = {**known_options_to_pass, **additional_adapter_kwargs}

            if base_url:
                adapter_options["base_url"] = base_url

            if adapter_options.pop("skip", False):
                current_run_details["status"] = "SKIPPED"
                current_run_details["reason"] = "Adapter explicitly skipped via configuration."
                all_adapter_results.append(current_run_details)
                continue

            cache_key_options_component = None
            cached_response = None

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
                    current_run_details["cache_status"] = "HIT"

            if cached_response:
                current_run_details["cache_status"] = "HIT"
                all_adapter_results.append(self._process_adapter_response(
                    current_run_details,
                    cached_response,
                    expect_exact,
                    expect_regex,
                    expect_substring,
                    expect_substring_case_insensitive
                ))
            else:
                if self.cache:
                    current_run_details["cache_status"] = "MISS"

                try:
                    config_override = type(adapter_instance.config)(
                        default_model=model_name,
                        **adapter_options
                    )

                    adapter_details.append({
                        "run_details": current_run_details,
                        "adapter_instance": adapter_instance,
                        "prompt": prompt_text,
                        "config_override": config_override,
                        "cache_key": cache_key_options_component
                    })
                    adapter_instances.append(adapter_instance)
                except Exception as e:
                    current_run_details["status"] = "ERROR"
                    current_run_details["reason"] = f"Adapter configuration error: {e}"
                    all_adapter_results.append(current_run_details)
                    self.overall_success = False

        if adapter_details:
            tasks = [
                self._execute_adapter_task(
                    details["adapter_instance"],
                    details["prompt"],
                    details["config_override"],
                    details["run_details"],
                    details["cache_key"],
                    expect_exact,
                    expect_regex,
                    expect_substring,
                    expect_substring_case_insensitive
                )
                for details in adapter_details
            ]

            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in task_results:
                if isinstance(result, Exception):
                    error_details = {
                        "file": str(test_case_path.name),
                        "id": test_id,
                        "adapter": "N/A",
                        "model": "N/A",
                        "status": "ERROR",
                        "reason": f"Concurrent execution error: {str(result)}",
                    }
                    all_adapter_results.append(error_details)
                    self.overall_success = False
                else:
                    all_adapter_results.append(result)


        return all_adapter_results

    async def _execute_adapter_task(
        self,
        adapter_instance,
        prompt_text,
        config_override,
        run_details,
        cache_key_options_component,
        expect_exact,
        expect_regex,
        expect_substring,
        expect_substring_case_insensitive
    ):
        """Execute a single adapter task and process its result."""
        try:
            response = await adapter_instance.execute(
                prompt_text,
                config_override=config_override
            )
            llm_response_data = {
                "text_response": response.text_response,
                "raw_response": response.raw_response,
                "error": response.error
            }

            result = self._process_adapter_response(
                run_details.copy(),
                llm_response_data,
                expect_exact,
                expect_regex,
                expect_substring,
                expect_substring_case_insensitive
            )

            if (
                self.cache
                and cache_key_options_component is not None
                and llm_response_data
                and not llm_response_data.get("error")
                and result["status"] == "PASS"
            ):
                self.cache.put(
                    prompt_text,
                    run_details["adapter"],
                    run_details["model"],
                    cache_key_options_component,
                    llm_response_data,
                )

            return result
        except Exception as e:
            run_details["status"] = "ERROR"
            run_details["reason"] = f"Adapter execution error: {e}"
            self.overall_success = False
            return run_details

    def _process_adapter_response(
        self,
        run_details,
        llm_response_data,
        expect_exact,
        expect_regex,
        expect_substring,
        expect_substring_case_insensitive
    ):
        """Process an adapter response and determine test status."""
        # lazy import
        from .drift_types import exact_match, regex_match

        if llm_response_data.get("error"):
            run_details["status"] = "ERROR"
            run_details["reason"] = f"Adapter error: {llm_response_data['error']}"
            run_details["actual_response"] = llm_response_data.get("raw_response")
            self.overall_success = False
            return run_details

        actual_text_response = llm_response_data.get("text_response")
        run_details["actual_response"] = actual_text_response
        run_details["raw_adapter_response"] = llm_response_data.get("raw_response")

        if actual_text_response is None:
            run_details["status"] = "FAIL"
            run_details["reason"] = "Adapter returned no text_response."
            self.overall_success = False
            return run_details

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

        run_details["status"] = "PASS" if passed else "FAIL"
        if not passed:
            run_details["reason"] = assertion_reason
            self.overall_success = False

        return run_details

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
                    tasks = [
                        self._run_single_test_case_with_semaphore(test_file_path, test_case_model)
                        for test_case_model in config_model.tests
                    ]

                    test_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in test_results:
                        if isinstance(result, Exception):
                            error_result = {
                                "file": str(test_file_path.name),
                                "id": "PARALLEL_EXECUTION_ERROR",
                                "adapter": "N/A",
                                "model": "N/A",
                                "status": "ERROR",
                                "reason": f"Parallel execution error: {str(result)}",
                            }
                            self.results.append(error_result)
                            self.overall_success = False
                        else:
                            self.results.extend(result)

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

        table = Table(title="PromptDrifter Test Results", padding=(1, 1, 1, 1))
        table.add_column("File", style="dim", width=20, no_wrap=False, overflow="fold")
        table.add_column("ID", style="cyan", width=20, no_wrap=False, overflow="fold")
        table.add_column("Adapter", style="magenta", width=10, no_wrap=True)
        table.add_column("Model", style="blue", width=20, no_wrap=False, overflow="fold")
        table.add_column("Status", justify="center", no_wrap=True)
        table.add_column("Failure Details", width=50, overflow="fold", no_wrap=False)
        table.add_column("Cache", justify="center", no_wrap=True, width=10)
        table.add_column("Tags", width=15, overflow="fold", no_wrap=False)

        summary = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIPPED": 0, "TOTAL": 0}

        results_by_id = {}
        for result in self.results:
            file_name = result.get("file", "N/A")
            test_id = result.get("id", "N/A")
            key = f"{file_name}:{test_id}"

            if key not in results_by_id:
                results_by_id[key] = []
            results_by_id[key].append(result)

        for idx, (key, group_results) in enumerate(results_by_id.items()):
            file_name = group_results[0].get("file", "N/A")
            test_id = group_results[0].get("id", "N/A")

            group_results.sort(key=lambda x: (x.get("adapter", ""), x.get("model", "")))
            middle_row = len(group_results) // 2

            for i, result in enumerate(group_results):
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

                display_file = file_name if i == middle_row else ""
                display_id = test_id if i == middle_row else ""

                cache_status = result.get("cache_status", "N/A")
                if cache_status == "HIT":
                    cache_display = f"[green]{cache_status}[/green]"
                elif cache_status == "MISS":
                    cache_display = f"[yellow]{cache_status}[/yellow]"
                else:
                    cache_display = f"[dim]{cache_status}[/dim]"

                tags = result.get("tags", [])
                if tags:
                    formatted_tags = "\n".join([f"[bold #d655fe]#{tag}[/bold #d655fe]" for tag in tags])
                else:
                    formatted_tags = ""

                table.add_row(
                    display_file,
                    display_id,
                    result.get("adapter", "N/A"),
                    result.get("model", "N/A"),
                    f"[{status_color}]{status}[/{status_color}]",
                    reason,
                    cache_display,
                    formatted_tags,
                )

            # Add a separator between different test IDs (except after the last group)
            if idx < len(results_by_id) - 1:
                table.add_row(
                    "─" * 20,  # Separator for File column
                    "─" * 20,  # Separator for ID column
                    "─" * 10,  # Separator for Adapter column
                    "─" * 20,  # Separator for Model column
                    "─" * 10,  # Separator for Status column
                    "─" * 50,  # Separator for Reason column
                    "─" * 10,  # Separator for Cache column
                    "─" * 15,  # Separator for Tags column
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
