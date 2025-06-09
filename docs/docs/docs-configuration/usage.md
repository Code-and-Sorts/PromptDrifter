# CLI Usage

After understanding drift tests, you'll need to effectively use PromptDrifter's command-line interface (CLI) to run these tests, manage results, and monitor for drift. This section covers the CLI commands and options in detail.

## Installation

First, install PromptDrifter using pip:

```bash
pip install promptdrifter
```

## Basic Commands

### Running Tests

The most basic command to run tests:

```bash
promptdrifter run path/to/tests.yaml
```

Run all tests in a directory:

```bash
promptdrifter run tests/*.yaml
```

<!-- TODO: Add screenshot of running tests with output -->

### Test Statuses

Tests can have the following statuses:

```
✅ PASS  - The test meets the expected output criteria
❌ FAIL  - The test does not meet the expected output criteria
⚠️ ERROR - An error occurred during test execution (API errors, etc.)
⏩ SKIP  - The test was skipped (missing API keys, etc.)
```

<!-- TODO: Add screenshot showing different test statuses -->

### Viewing Failures

When tests fail, PromptDrifter shows detailed information about what went wrong:

```bash
promptdrifter run path/to/tests.yaml
```

The output includes:
- Expected output
- Actual output
- Specific reason for failure
- Detailed diff for better comparison

<!-- TODO: Add screenshot of failure output -->

## Command Options

PromptDrifter CLI supports various options:

### Global Options

```bash
# Display help
promptdrifter --help

# Show version
promptdrifter version
```

### Run Command Options

```bash
# Set configuration directory
promptdrifter run --config-dir ./config path/to/tests.yaml

# Disable caching
promptdrifter run --no-cache path/to/tests.yaml

# Set custom cache file location
promptdrifter run --cache-db ./custom-cache.json path/to/tests.yaml

# Set API keys for specific providers
promptdrifter run --openai-api-key YOUR_KEY --claude-api-key YOUR_KEY path/to/tests.yaml
```

## Response Caching

PromptDrifter can cache LLM responses to save time and API costs during repeated test runs.

### How Caching Works

By default, PromptDrifter caches responses based on:
- The prompt text (after template substitution)
- The adapter type (e.g., OpenAI, Claude)
- The model name
- Model parameters (temperature, max_tokens, etc.)

Responses are stored in a local cache file, and subsequent identical requests will use the cached responses instead of making new API calls.

### Configuring Cache

```bash
# Enable caching (on by default)
promptdrifter run path/to/tests.yaml

# Disable caching
promptdrifter run --no-cache path/to/tests.yaml

# Set custom cache file location
promptdrifter run --cache-db .custom-cache.json path/to/tests.yaml
```

### Cache Time-to-Live (TTL)

PromptDrifter uses a default cache TTL (Time-to-Live) of 24 hours for cached responses. Currently, the TTL is not configurable via the CLI, but the default implementation ensures that:

- Cached responses expire after 24 hours.
- Expired entries are automatically removed when accessing the cache.
- The system will fetch fresh responses once the cache entries expire.

### Cache Behavior with Failures

**Important**: All tests that previously failed will always be re-run, regardless of cache settings. This ensures you always get the latest result for failing tests, which is crucial for:

- Verifying if issues have been resolved.
- Detecting intermittent failures.
- Properly tracking drift over time.

## Environment Variables

You can set environment variables instead of using command-line flags:

```bash
# Set API keys
export OPENAI_API_KEY=your-key
export CLAUDE_API_KEY=your-key
export GEMINI_API_KEY=your-key
export QWEN_API_KEY=your-key
export GROK_API_KEY=your-key
export DEEPSEEK_API_KEY=your-key
export MISTRAL_API_KEY=your-key

# Run tests
promptdrifter run path/to/tests.yaml
```

## Advanced Usage

### Drift Testing

Test specific drift types:

```bash
promptdrifter drift-type text_similarity "Expected text" "Actual response"
```

Supported drift types:
- exact_match
- regex_match
- expect_substring
- expect_substring_case_insensitive
- text_similarity

### Initializing a Project

Create a new PromptDrifter project with a sample configuration:

```bash
promptdrifter init ./my-project
```

## Troubleshooting

### Common Issues

- **API Key Errors**: Ensure API keys are set correctly
- **Rate Limiting**: Consider adding delays between tests
- **Timeouts**: Retry tests after a delay
- **Cache Issues**: Try clearing the cache with `--no-cache`
