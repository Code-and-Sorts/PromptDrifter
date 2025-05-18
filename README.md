<p align="center">
  <img src=".docs/promptdrifer-logo.svg" alt="PromptDrifter Logo" width="500"/>
</p>

<br />

<p align="center">
  <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" />
  <img alt="PRs welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" />
  <img alt="Python" src="https://img.shields.io/badge/Made with-Python-3776AB?style=flat-square&logo=python&logoColor=white" />
</p>

<p align="center">
  <img alt="Build and Test CI" src="https://img.shields.io/github/actions/workflow/status/Code-and-Sorts/PromptDrifter/main-ci.yaml?branch=main&label=Build%20and%20Test&style=flat-square" />
</p>

<p align="center">
  <a href="#-quick-start">🏃 Quick-Start</a> - <a href="#-why-promptdrifter">❓ Why PromptDrifter?</a> - <a href="#-demo">🎬 Demo</a> - <a href="#-docs">📚 Docs</a>
  <br />
  <a href="https://github.com/Code-and-Sorts/PromptDrifter/issues/new?assignees=&template=bug_report.md">🐛 Bug Report</a> - <a href="https://github.com/Code-and-Sorts/PromptDrifter/issues/new?assignees=&template=feature_request.md">✨ Feature Request</a>
</p>

### PromptDrifter is a one-command CI guardrail, open source platform for catching prompt drift and fails if your LLM answers change.

> [!IMPORTANT]
> **Development Notice**: This project is under active development. Breaking changes may occur between versions. Please check the changelog and release notes before updating.

## 🏃 Quick-Start

TODO: add quick-start

## ❓ Why PromptDrifter?

The landscape of Large Language Models (LLMs) is one of rapid evolution. While exciting, this constant change introduces a critical challenge for applications relying on them: **prompt drift**.

Over time, updates to LLM versions, or even subtle shifts in their training data or internal architecture, can cause their responses to identical prompts to change. These changes can range from minor formatting differences to significant alterations in content or structure, potentially breaking downstream processes causing issues with the integrity of your application.

---

### Undetected prompt drift can lead to:

#### 🚨 Unexpected Failures
Applications or CI/CD pipelines may break silently or with cryptic errors when LLM outputs deviate from expected formats or content.

#### 📉 Degraded User Experience
Features relying on consistent LLM responses can malfunction, leading to user frustration.

#### ⏱️ Increased Maintenance
Engineers spend valuable time diagnosing issues, tracing them back to changed LLM behavior rather addressing features and bugs in code.

#### 🚧 Blocked Deployments
Uncertainty about LLM stability can slow down development cycles and deployment frequency.

---

### PromptDrifter tackles these challenges head-on by providing:

#### 🛡️ Automated Guardrails
A simple, command-line driven tool to integrate LLM response validation directly into your development and CI/CD workflows.

#### 🔍 Early Drift Detection
By comparing LLM outputs against version-controlled expected responses or predefined patterns (like regex), **PromptDrifter** catches deviations as soon as they occur.

#### ⚙️ Consistent and Reliable Applications
Ensures that your LLM-powered features behave predictably by failing builds when significant response changes are detected, *before* they impact users or production systems.

#### 🔌 Model Agnostic Design
Through a flexible adapter system, PromptDrifter can interact with various LLM providers and models (e.g., OpenAI, Ollama, and more to come).

#### 📝 Declarative Test Suites
Define your prompt tests in easy-to-understand YAML files, making them simple to create, manage, and version alongside your codebase.

#### 😌 Developer Peace of Mind
Build with greater confidence, knowing you have a safety net that monitors the stability of your critical prompt interactions.

> [!NOTE]
> By making prompt-response testing a straightforward and automated part of your workflow, **PromptDrifter** helps you harness the power of LLMs while mitigating the risks associated with their dynamic nature.

## 🎬 Demo

TODO: add demo with screenshots and GIFs

## 🤖 Supported LLM Adapters

PromptDrifter is designed to be extensible to various Large Language Models through its adapter system. Here's a current list of supported and planned adapters:

| Provider / Model Family | Adapter Status | Details / Model Examples | Linked Issue |
| :---------------------- | :------------- | :----------------------- | :----------- |
| **GPT (OpenAI)**        | ✅ Available   | `gpt-3.5-turbo`, `gpt-4`, `gpt-4o`, etc.                         | N/A                            |
| **Ollama**              | ✅ Available   | `llama3`, `mistral`, `gemma`, etc.                               | N/A                            |
| **Claude (Anthropic)**  | ✅ Available  | `claude-3-7-sonnet`, `claude-3-5-sonnet`, `claude-3-opus`        | N/A                            |
| **Gemini (Google)**     | ✅ Available   | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash-thinking`| N/A                            |
| **Qwen**                | ✅ Available | `qwen3-30b-a3b`, `qwq-32b`                                        | N/A  |
| **Grok (xAI)**          | ✅ Available | `grok-3`, `grok-2`, etc.                                         | N/A |
| **DeepSeek**            | ✅ Available | `deepseek-r1`, `deepseek-v3-0324`, etc.                          | N/A |
| **Llama (Meta)**        | ✅ Available | `llama-4-maverick`, `llama-4-scout`, etc.                        | N/A |
| **Mistral**             | ⏳ Coming Soon | `mistral-small-24b-instruct-2501`, `mistral-small-3.1-24b-instruct-2503` | N/a |


If there's a model or provider you'd like to see supported, please [open a feature request](https://github.com/Code-and-Sorts/PromptDrifter/issues/new?assignees=&template=feature_request.md) or consider contributing an adapter!

## 🧪 Supported Drift Tests

| Name | Config key | Description | Implementation Status|
| :---------------------- | :------------- | :-------------------------------------------------------- | :------------------------------------- |
| **Exact Match** | `expect_exact` | Output should match be an exact match | ✅ Available |
| **Regex** | `expect_regex` | Output should match regex pattern  | ✅ Available |
| **Substring** | `expect_substring` | Output should contain the substring  | ✅ Available |
| **Substring Case Insensitive** | `expect_substring_case_insensitive` | Output should contain the substring that is case insensitive (e.g. "hElLO" === "hello")  | ✅ Available |

## ⚙️ GitHub Action

Automate your prompt drift detection by integrating PromptDrifter directly into your GitHub workflows!

We provide a reusable GitHub Action that makes it easy to run your PromptDrifter tests on every push or pull request.

➡️ **Find the PromptDrifter GitHub Action and usage instructions here: [CodeAndSorts/promptdrifter-action](https://github.com/CodeAndSorts/promptdrifter-action)** (Replace this URL with the actual one once the action is published in its own repository or on the GitHub Marketplace).

This action allows you to:
*   Install a specific version of PromptDrifter or use the latest.
*   Specify your test files and configurations.
*   Control caching behavior.

By using the action, you can ensure that any changes to your LLM's responses that violate your defined tests will automatically flag your CI builds, preventing unexpected issues from reaching production.

## 📚 Docs

// TODO: Link to docs. In docs, add references.

## 🧑‍💻 Contributing

Follow the [contributing guide](./.github/CONTRIBUTING.md).

## 🔖 Code of Conduct

Please make sure you read the [Code of Conduct guide](./.github/CODE-OF-CONDUCT.md).
