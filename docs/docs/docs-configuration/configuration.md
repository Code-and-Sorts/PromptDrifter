# Configuration File

PromptDrifter is configured using a YAML file. By default, it looks for a `promptdrifter.yaml` file in the current directory.

## Overview

Configuration files in PromptDrifter define:

- **Test cases**: Individual prompts and expected responses
- **Adapter configurations**: Settings for different LLM providers
- **Input variables**: Dynamic values that can be injected into prompts
- **Expected outputs**: Validation criteria for responses
- **Test environment settings**: Global settings for test execution

Configuration files can be split across multiple files and directories to organize complex test suites. They support environment variable injection for secure credential management.

## YAML Configuration Examples

### Basic Configuration

A basic test configuration with a single adapter:

```yaml
version: "0.1"
adapters:
  - id: "basic-test"
    prompt: "What is the capital of France?"
    expect_exact: "Paris"
    adapter:
      - type: "openai"
        model: "gpt-3.5-turbo"
        temperature: 0.3
        max_tokens: 100
```

### Multiple Adapters

Testing the same prompt across multiple adapters:

```yaml
version: "0.1"
adapters:
  - id: "multi-adapter-test"
    prompt: "Write a haiku about programming."
    expect_substring: "code"
    adapter:
      - type: "openai"
        model: "gpt-4"
        temperature: 0.7
      - type: "claude"
        model: "claude-3-sonnet"
        temperature: 0.7
      - type: "gemini"
        model: "gemini-2.5-pro"
        temperature: 0.7
```

### Template Variables

Using template variables in prompts:

```yaml
version: "0.1"
adapters:
  - id: "template-test"
    prompt: "Write a {{length}} poem about {{topic}}."
    inputs:
      length: "short"
      topic: "artificial intelligence"
    expect_substring: "intelligence"
    adapter:
      - type: "openai"
        model: "gpt-4"
```

### Using Regex for Validation

Testing responses using regular expressions:

```yaml
version: "0.1"
adapters:
  - id: "regex-test"
    prompt: "List three prime numbers."
    expect_regex: "\\b(2|3|5|7|11|13|17|19)\\b.*\\b(2|3|5|7|11|13|17|19)\\b.*\\b(2|3|5|7|11|13|17|19)\\b"
    adapter:
      - type: "openai"
        model: "gpt-3.5-turbo"
```

This regex checks that the response contains three prime numbers from the list.

## Environment Variables

You can reference environment variables in your configuration using the `${VAR_NAME}` syntax. This is useful for storing sensitive information like API keys.

## Creating a New Configuration File

### Using the CLI

PromptDrifter provides a CLI command to initialize a new configuration file with a basic template:

```bash
promptdrifter init
```

This will create a `promptdrifter.yaml` file in the current directory with a basic structure.

You can specify a different filename or path:

```bash
promptdrifter init --output custom-config.yaml
```

To initialize with a specific adapter type:

```bash
promptdrifter init --adapter openai
```

For a more comprehensive starter template:

```bash
promptdrifter init --template full
```

### Configuration File Structure

The generated configuration file includes commented sections explaining each component and providing examples for common use cases.

## Multiple Test Files

You can split your tests across multiple files and run them all:

```bash
promptdrifter run tests/*.yaml
```
