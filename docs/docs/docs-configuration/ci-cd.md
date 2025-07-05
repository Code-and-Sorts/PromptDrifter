# CI/CD Integration

Integrating PromptDrifter into your CI/CD pipeline helps you catch prompt drift early and ensures LLM outputs remain consistent as your application evolves.

## GitHub Actions Integration

PromptDrifter provides an official GitHub Action that makes it easy to integrate prompt drift detection into your CI/CD workflow.

### Quick Setup

1. **Add the Action to your workflow** (e.g., `.github/workflows/prompt-drift.yml`):

```yaml
name: Prompt Drift Detection

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  prompt-drift-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Run PromptDrifter
      uses: Code-and-Sorts/PromptDrifter-action@v0.0.2
      with:
        files: 'tests/promptdrifter.yaml'
        openai-api-key: ${{ secrets.OPENAI_API_KEY }}
```

2. **Set up your API keys** in GitHub repository secrets:
   - Go to your repository Settings > Secrets and variables > Actions
   - Add your LLM provider API keys (e.g., `OPENAI_API_KEY`, `CLAUDE_API_KEY`)

3. **Create your test configuration** file (`tests/promptdrifter.yaml`) with your prompts and expected outputs.

### Action Features

- **Multi-provider Support**: OpenAI, Claude, Gemini, and other supported LLM providers
- **Secure API Key Handling**: Uses GitHub secrets for secure API key management
- **Flexible Configuration**: Specify test files, caching behavior, and more
- **Version Control**: Pin to specific versions or use latest

### Find the Action

The official PromptDrifter GitHub Action is available on the [GitHub Marketplace](https://github.com/marketplace/actions/promptdrifter).

## Best Practices

1. **Secure API Keys**: Never commit API keys to your repository. Always use environment variables or secrets.

2. **Periodic Testing**: Schedule regular drift tests (daily/weekly) to catch slow drift over time.

3. **Selective Testing**: In pre-merge workflows, consider running only a subset of critical tests to save time and API costs.

4. **Notifications**: Configure alerts when drift is detected so your team can address issues quickly.

5. **Keep Cache Small**: If using caching, periodically clear old cached responses to prevent excessive cache size.

## Troubleshooting

- **API Rate Limits**: If you hit rate limits, consider spacing out your tests or using caching more aggressively.

- **Test Failures**: If tests fail in CI but pass locally, check for environment differences or API key issues.
