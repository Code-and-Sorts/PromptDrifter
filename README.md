<p align="center">
  <img src=".docs/promptdrifer-logo.svg" alt="PromptDrifter Logo" width="500"/>
</p>

<br />

<p align="center">
  <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields" />
  <img alt="Python" src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" />
</p>

<p align="center">
  <a href="#why-promptdrifter">Why PromptDrifter?</a> - <a href="#quickstart">Quickstart</a> - <a href="#docs">Docs</a> - <a href="https://github.com/Code-and-Sorts/PromptDrifter/issues/new?assignees=&labels=bug&template=bug_report.md">Bug Reports</a>
</p>

## 🧭 PromptDrifter is a one-command CI guardrail, open source platform for catching prompt drift and fails if your LLM answers change.

## ❓ Why PromptDrifter?

The landscape of Large Language Models (LLMs) is one of rapid evolution. While exciting, this constant change introduces a critical challenge for applications relying on them: **prompt drift**.

Over time, updates to LLM versions, or even subtle shifts in their training data or internal architecture, can cause their responses to identical prompts to change. These changes can range from minor formatting differences to significant alterations in content or structure, potentially breaking downstream processes causing issues with the integrity of your application.

**Undetected prompt drift can lead to**

- 🚨 **Unexpected Failures:** Applications or CI/CD pipelines may break silently or with cryptic errors when LLM outputs deviate from expected formats or content.
- 📉 **Degraded User Experience:** Features relying on consistent LLM responses can malfunction, leading to user frustration.
- ⏱️ **Increased Maintenance:** Engineers spend valuable time diagnosing issues, tracing them back to changed LLM behavior rather addressing features and bugs in code.
- 🚧 **Blocked Deployments:** Uncertainty about LLM stability can slow down development cycles and deployment frequency.

**PromptDrifter tackles these challenges head-on by providing**

- 🛡️ **Automated Guardrails:** A simple, command-line driven tool to integrate LLM response validation directly into your development and CI/CD workflows.
- 🔍 **Early Drift Detection:** By comparing LLM outputs against version-controlled expected responses or predefined patterns (like regex), **PromptDrifter** catches deviations as soon as they occur.
- ⚙️ **Consistent and Reliable Applications:** Ensures that your LLM-powered features behave predictably by failing builds when significant response changes are detected, *before* they impact users or production systems.
- 🔌 **Model Agnostic Design:** Through a flexible adapter system, PromptDrifter can interact with various LLM providers and models (e.g., OpenAI, Ollama, and more to come).
- 📝 **Declarative Test Suites:** Define your prompt tests in easy-to-understand YAML files, making them simple to create, manage, and version alongside your codebase.
- 😌 **Developer Peace of Mind:** Build with greater confidence, knowing you have a safety net that monitors the stability of your critical prompt interactions.

By making prompt-response testing a straightforward and automated part of your workflow, **PromptDrifter** helps you harness the power of LLMs while mitigating the risks associated with their dynamic nature.

## 🏃 Quickstart


## 🤖 Supported LLM Adapters

PromptDrifter is designed to be extensible to various Large Language Models through its adapter system. Here's a current list of supported and planned adapters:

| Provider / Model Family | Adapter Status | Details / Model Examples                                  | Linked Issue |
| :---------------------- | :------------- | :-------------------------------------------------------- | :------------------------------------- |
| **OpenAI**              | ✅ Available   | `gpt-3.5-turbo`, `gpt-4`, `gpt-4o`, etc.                  | N/A                                    |
| **Ollama**              | ✅ Available   | Supports self-hosted models like `llama3`, `mistral`, `gemma` | N/A                                |
| **Anthropic (Claude)**  | ⏳ Coming Soon | `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`      | `[Track Issue](#)` Placeholder         |
| **Google (Gemini)**     | ⏳ Coming Soon | `gemini-1.5-pro`, `gemini-1.0-pro`                        | `[Track Issue](#)` Placeholder         |

If there's a model or provider you'd like to see supported, please [open a feature request](https://github.com/Code-and-Sorts/promptdrifter/issues/new/choose) or consider contributing an adapter!

## 📚 Docs

// TODO: Link to docs

## 🧑‍💻 Contributing

Follow the [contributing guide](CONTRIBUTING.md).

## 🔖 Code of Conduct

Please make sure you read the [Code of Conduct guide](CODE-OF-CONDUCT.md).
