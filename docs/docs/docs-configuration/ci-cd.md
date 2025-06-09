# CI/CD Integration

Integrating PromptDrifter into your CI/CD pipeline helps you catch prompt drift early and ensures LLM outputs remain consistent as your application evolves.

## GitHub Actions Integration

<!-- TODO: Add GHA docs and links once implemented. -->

## Best Practices

1. **Secure API Keys**: Never commit API keys to your repository. Always use environment variables or secrets.

2. **Periodic Testing**: Schedule regular drift tests (daily/weekly) to catch slow drift over time.

3. **Selective Testing**: In pre-merge workflows, consider running only a subset of critical tests to save time and API costs.

4. **Notifications**: Configure alerts when drift is detected so your team can address issues quickly.

5. **Keep Cache Small**: If using caching, periodically clear old cached responses to prevent excessive cache size.

## Troubleshooting

- **API Rate Limits**: If you hit rate limits, consider spacing out your tests or using caching more aggressively.

- **Test Failures**: If tests fail in CI but pass locally, check for environment differences or API key issues.
