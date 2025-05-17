# What is PromptDrifter?

[PromptDrifter](https://github.com/Code-and-Sorts/PromptDrifter) is a command-line tool to test and evaluate Large Language Model (LLM) prompts across different models and providers. It helps catch "prompt drift" when LLM responses change unexpectedly.

By comparing LLM outputs against version-controlled expected responses or predefined patterns, PromptDrifter helps ensure your LLM-powered features behave predictably.

## Features

- **Automated Guardrails**: Integrates LLM response validation directly into your development and CI/CD workflows.
- **Early Drift Detection**: Catches deviations by comparing LLM outputs against version-controlled expected responses or predefined patterns (like regex).
- **Model Agnostic Design**: Through a flexible adapter system, PromptDrifter can interact with various LLM providers and models (e.g., OpenAI, Ollama, Gemini).
- **Declarative Test Suites**: Define your prompt tests in easy-to-understand YAML files, making them simple to create, manage, and version.
- **CLI Interface**: Simple command-line tool for easy integration and use.
- **Response Caching**: Speeds up test runs by caching responses from LLMs.

## Getting Started

To start using PromptDrifter:

1. **Install the package:**
    ```bash
    pip install promptdrifter
    ```

2. **Initialize a sample configuration:**
    ```bash
    promptdrifter init
    ```

This will create a `promptdrifter.yaml` file in your current directory.

3. **Edit `promptdrifter.yaml`** to define your prompts and expected outcomes.

4. **Run your tests:**
    ```bash
    promptdrifter run promptdrifter.yaml
    ```

## License

PromptDrifter is licensed under the MIT License.
