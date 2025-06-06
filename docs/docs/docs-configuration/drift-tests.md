# Drift Tests

## Overview

Drift tests in PromptDrifter help you detect when your LLM responses have changed unexpectedly. These tests provide a systematic way to catch "prompt drift" - situations where models begin generating different outputs for the same inputs over time, which can break your application's functionality.

## Why Drift Testing Matters

LLM outputs can change for many reasons:
- Model updates by providers (e.g., GPT-3.5 to GPT-4, or silent updates)
- Changes in model weights or training data
- Modifications to system prompts
- Changes in model parameters (temperature, top_p, etc.)
- Knowledge cutoff date changes for newer models

Drift testing helps ensure your application remains stable despite these changes.

## How Drift Tests Work

PromptDrifter compares actual LLM responses against expected outputs using various validation methods. The testing process follows these steps:

1. **Define Tests**: Create YAML configurations with prompts and expected outputs
2. **Run Tests**: Execute tests against one or more LLM providers
3. **Detect Drift**: Identify when responses no longer match expectations
4. **Alert**: Report changes that indicate problematic drift

## Setting Up Drift Tests

Drift tests use the same YAML configuration format as other PromptDrifter tests. Each test specifies a prompt, the expected response pattern, and one or more adapter configurations.

### Basic Configuration

```yaml
version: "0.1"
adapters:
  - id: "basic-drift-test"
    prompt: "What is the capital of France?"
    expect_exact: "Paris"
    adapter:
      - type: "openai"
        model: "gpt-3.5-turbo"
        temperature: 0.3
```

### Validation Methods

PromptDrifter supports five validation methods for detecting drift. You must choose exactly one for each test:

#### 1. Exact Match

Tests if the response matches the expected output exactly:

```yaml
version: "0.1"
adapters:
  - id: "exact-match-test"
    prompt: "What is 2+2?"
    expect_exact: "4"
    adapter:
      - type: "openai"
        model: "gpt-4"
```

#### 2. Regex Match

Tests if the response matches a regular expression:

```yaml
version: "0.1"
adapters:
  - id: "regex-match-test"
    prompt: "List three prime numbers"
    expect_regex: "\\b(2|3|5|7|11|13|17|19|23|29|31|37|41|43|47|53|59|61|67|71|73|79|83|89|97)\\b.*\\b(2|3|5|7|11|13|17|19|23|29|31|37|41|43|47|53|59|61|67|71|73|79|83|89|97)\\b.*\\b(2|3|5|7|11|13|17|19|23|29|31|37|41|43|47|53|59|61|67|71|73|79|83|89|97)\\b"
    adapter:
      - type: "gemini"
        model: "gemini-2.5-pro"
```

#### 3. Substring Match

Tests if the response contains a specific substring:

```yaml
version: "0.1"
adapters:
  - id: "substring-test"
    prompt: "Write a haiku about programming"
    expect_substring: "code"
    adapter:
      - type: "claude"
        model: "claude-3-sonnet"
```

#### 4. Case-Insensitive Substring Match

Tests if the response contains a specific substring, ignoring case:

```yaml
version: "0.1"
adapters:
  - id: "case-insensitive-test"
    prompt: "Explain what HTTP stands for"
    expect_substring_case_insensitive: "hypertext transfer protocol"
    adapter:
      - type: "ollama"
        model: "llama3"
```

#### 5. Text Similarity

Tests if the response is semantically similar to the expected text, using a similarity threshold:

```yaml
version: "0.1"
adapters:
  - id: "similarity-test"
    prompt: "Explain the concept of machine learning"
    text_similarity:
      text: "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
      threshold: 0.8  # Similarity threshold (0.0-1.0)
    adapter:
      - type: "openai"
        model: "gpt-4"
```

### Using Template Variables

You can use template variables in your prompts for more dynamic testing:

```yaml
version: "0.1"
adapters:
  - id: "template-variable-test"
    prompt: "Summarize the following text: {{content}}"
    inputs:
      content: "Transformers are neural network architectures that revolutionized natural language processing through their self-attention mechanisms. They process entire sequences simultaneously rather than sequentially, enabling better capture of long-range dependencies in text. Models like BERT, GPT, and T5 are all based on the transformer architecture and have achieved state-of-the-art results across numerous language tasks. Transformers have expanded beyond NLP into computer vision, audio processing, and multimodal applications, becoming one of the most influential architectural innovations in modern machine learning."
    expect_substring: "summary"
    adapter:
      - type: "openai"
        model: "gpt-4o"
```

### Testing Multiple Adapters

Test the same prompt across multiple LLM providers to compare responses:

```yaml
version: "0.1"
adapters:
  - id: "multi-adapter-test"
    prompt: "Explain quantum computing briefly"
    expect_substring: "superposition"
    adapter:
      - type: "openai"
        model: "gpt-4"
      - type: "claude"
        model: "claude-3-opus"
      - type: "gemini"
        model: "gemini-2.5-pro"
        skip: true  # This adapter will be skipped during test execution
```

## Best Practices

1. **Choose the Right Validation Method**:
   - Use `expect_exact` for precise, short responses.
   - Use `expect_regex` for responses with variable parts but known patterns.
   - Use `expect_substring` when you need to verify specific content appears.
   - Use `expect_substring_case_insensitive` when case doesn't matter.
   - Use `text_similarity` when you need to check semantic meaning rather than exact wording. This method takes longer to run because it needs to load and use a neural network model to compute semantic embeddings.

2. **Start Simple**: Begin with critical prompts your application relies on.

3. **Version Control**: Store drift test configurations in your repository.

4. **Regular Testing**: Schedule frequent tests for early detection of issues. You can implement regular testing in several ways:
   - **CI/CD Integration**: Run drift tests as part of your deployment pipeline.
   - **Scheduled Cron Jobs**: Set up scheduled workflows in your CI/CD pipeline.
   - **Model Deployment Hooks**: Trigger tests whenever new model versions are deployed.

5. **Adapt Expectations**: Update your expected outputs when model changes are intentional.

## Troubleshooting

### False Positives

If you're getting too many failures for acceptable changes:

- Switch from `expect_exact` to `expect_substring` for more flexibility.
- Use `expect_regex` with carefully crafted patterns.
- Use `text_similarity` with an appropriate threshold for meaning-based comparison.
- Update your expected outputs to match new but acceptable responses.

### Handling Model Updates

When a model provider releases a significant update:

1. Run your tests to identify changes.
2. Review the changes to determine if they're problematic.
3. Update your expected outputs for non-problematic changes.
4. Address any problematic changes in your application logic.

## Conclusion

Drift testing is essential for maintaining stable LLM-powered applications. By implementing regular drift tests with PromptDrifter, you can:
- Quickly identify unexpected changes in model behavior.
- Ensure consistent application performance.
- Build confidence in your AI-powered features.
- Document model behavior over time.
