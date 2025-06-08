# Introduction to PromptDrifter

PromptDrifter is a one-command CI guardrail that catches prompt drift and fails the build when your LLM answers change unexpectedly. It helps you maintain consistent AI responses across your application, while providing powerful tools to compare models and test/validate prompts.

With PromptDrifter, you can easily compare responses across different LLM providers, temperature settings, and prompt variations side-by-side. This makes it simple to identify which model performs best for your specific use case, measure the impact of subtle prompt changes, and validate that your prompts produce reliable, high-quality outputs before deploying to production.

## The Challenge: LLM Instability

Large Language Models (LLMs) like GPT-4, Claude, and others are constantly evolving. This evolution can lead to unexpected changes in their responses, even when using identical prompts.

### What is Prompt Drift?

Prompt drift occurs when the responses from a Language Model change unexpectedly, often due to:

- **Model Updates**: New versions with different weights or architectures.
- **Training Changes**: Updates to the model's underlying training data.
- **Parameter Shifts**: Different default settings for temperature, top-p, etc.
- **Context Handling**: Changes in how context or system prompts are processed.

These inconsistencies can lead to:
- Broken workflows
- Inconsistent user experiences
- Unexpected application behavior
- Failed tests and deployments

## How PromptDrifter Helps

PromptDrifter provides a simple, reliable way to detect when your LLM outputs have drifted from expected responses. It:

1. **Establishes baselines** for expected LLM responses.
2. **Monitors changes** in responses over time.
3. **Alerts you** when responses drift beyond acceptable thresholds.
4. **Integrates with CI/CD** to prevent problematic deployments and can be enabled for scheduled monitoring.

## Getting Started

### Installation

Install PromptDrifter using pip:

```bash
pip install promptdrifter
```

### Basic Usage

Create a configuration file:

```bash
promptdrifter init
```

This generates a `promptdrifter.yaml` file that you can customize with your prompts and expectations.

Run your tests:

```bash
promptdrifter test
```

## Key Features

- **Multiple Test Types**: Exact match, regex, substring, and semantic similarity tests.
- **CI/CD Integration**: Plug and play GitHub Action and can be integrated with GitLab CI, Jenkins, and more.
- **Various LLM Providers**: Support for OpenAI, Anthropic, Google, and others.
- **Configurable Thresholds**: Set your own tolerance for acceptable drift.
- **Detailed Reporting**: Get clear information about what changed and why.
