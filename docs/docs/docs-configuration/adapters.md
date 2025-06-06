# Adapters

## Overview

Adapters in PromptDrifter provide a consistent interface for interacting with different Large Language Model (LLM) providers. They abstract away the provider-specific implementation details, allowing you to seamlessly switch between different LLM providers while maintaining the same code structure.

## Supported Adapters

## Adapter Configuration

Each adapter requires specific configuration parameters. Below are the configuration options for each supported adapter. Note that PromptDrifter supports all models available through each provider's API - the models listed are just examples of what's available.

### Common Configuration Options

These options can be used with any adapter type:

```yaml
adapter:
  - type: "any-adapter-type"
    model: "model-name"
    skip: true                        # Optional: When true, this adapter will be skipped (default: false)
    temperature: 0.7                  # Optional: Controls randomness (0.0-1.0)
    max_tokens: 1024                  # Optional: Maximum tokens in the response
```

### OpenAI (GPT)

[OpenAI](https://openai.com/) provides state-of-the-art language models like GPT-3.5 and GPT-4. Their models excel at a wide range of tasks including content generation, summarization, and conversational AI.

#### Example Models

The following are examples of available models. PromptDrifter supports any model offered by OpenAI's API:

- `gpt-3.5-turbo` - Fast and cost-effective model for most tasks
- `gpt-4` - More capable model with improved reasoning
- `gpt-4o` - Latest model with enhanced capabilities across text, vision, and audio
- `gpt-4-turbo` - Cost-effective alternative to GPT-4 with similar capabilities

#### Configuration Options

```yaml
adapter:
  - type: "openai"
    model: "gpt-4o"                    # The model to use (default: gpt-3.5-turbo)
    temperature: 0.7                   # Optional: Controls randomness (0.0-1.0)
    max_tokens: 2048                   # Optional: Maximum tokens in the response
    system_prompt: "Your instruction"  # Optional: System prompt for the conversation
```

The API key is typically set via the `OPENAI_API_KEY` environment variable.

### Claude (Anthropic)

[Anthropic Claude](https://docs.anthropic.com/en/api/overview/) offers a family of AI assistants known for their conversational abilities, reasoning, and instruction-following capabilities.

#### Example Models

The following are examples of available models. PromptDrifter supports any model offered by Anthropic's Claude API:

- `claude-3-opus` - High-capability model for complex tasks
- `claude-3-sonnet` - Balanced model for most tasks
- `claude-3-haiku` - Fast, cost-effective model
- `claude-3-5-sonnet` - Enhanced model with improved reasoning
- `claude-3-7-sonnet` - Latest model with state-of-the-art capabilities

#### Configuration Options

```yaml
adapter:
  - type: "claude"
    model: "claude-3-sonnet"           # The model to use (default: claude-3-opus-20240229)
    temperature: 0.7                   # Optional: Controls randomness (0.0-1.0)
    max_tokens: 1024                   # Optional: Maximum tokens in the response
    system_prompt: "Your instruction"  # Optional: System prompt for the conversation
```

The API key is typically set via the `CLAUDE_API_KEY` environment variable.

### Gemini (Google)

[Google's Gemini](https://ai.google.dev/gemini-api) is a family of multimodal AI models capable of reasoning across text, images, audio, video, and code.

#### Example Models

The following are examples of available models. PromptDrifter supports any model offered by Google AI's API:

- `gemini-2.5-pro` - Advanced reasoning and multimodal capabilities
- `gemini-2.5-flash` - Fast, economical alternative to Pro
- `gemini-2.0-flash-thinking` - Specialized for step-by-step reasoning

#### Configuration Options

```yaml
adapter:
  - type: "gemini"
    model: "gemini-2.5-pro"            # The model to use (default: gemini-1.5-flash-latest)
    temperature: 0.7                   # Optional: Controls randomness (0.0-1.0)
    max_tokens: 1024                   # Optional: Maximum tokens in the response
    system_prompt: "Your instruction"  # Optional: System prompt for the conversation
```

The API key is typically set via the `GEMINI_API_KEY` environment variable.

### Ollama

[Ollama](https://ollama.com/) allows you to run open-source large language models locally. It supports a variety of models including Llama, Mistral, and Gemma.

#### Example Models

The following are examples of available models. PromptDrifter supports any model you have installed in your Ollama instance:

- `llama3` - Meta's Llama 3 model
- `mistral` - Mistral AI's models
- `gemma` - Google's Gemma models
- Any other models installed in your Ollama instance

#### Configuration Options

```yaml
adapter:
  - type: "ollama"
    model: "llama3"                    # The model to use (default: llama3)
    temperature: 0.7                   # Optional: Controls randomness (0.0-1.0)
    host: "http://localhost:11434"     # Optional: Ollama host URL
    system_prompt: "Your instruction"  # Optional: System prompt for the conversation
```

Due to Ollama running locally, there is not need to pass in a API key.

### Qwen (Alibaba)

[Alibaba Qwen](https://www.alibabacloud.com/help/en/model-studio/use-qwen-by-calling-api/) is a series of large language models developed by Alibaba Cloud, known for strong multilingual capabilities and coding performance.

#### Example Models

The following are examples of available models. PromptDrifter supports any model offered by Qwen's API:

- `qwen3-30b-a3b` - High-capacity model with advanced reasoning
- `qwq-32b` - Specialized variant with enhanced instruction following

#### Configuration Options

```yaml
adapter:
  - type: "qwen"
    model: "qwen-plus"                 # The model to use (default: qwen-plus)
    temperature: 0.7                   # Optional: Controls randomness (0.0-1.0)
    max_tokens: 1024                   # Optional: Maximum tokens in the response
    system_prompt: "Your instruction"  # Optional: System prompt for the conversation
```

The API key is typically set via the `QWEN_API_KEY` environment variable.

### Grok (xAI)

[xAI Grok](https://grok.x.ai/) is developed and designed to answer questions with humor and personality while having up-to-date knowledge.

#### Example Models

The following are examples of available models. PromptDrifter supports any model offered by Grok's API:

- `grok-3` - Latest model with enhanced capabilities
- `grok-2` - Previous generation model

#### Configuration Options

```yaml
adapter:
  - type: "grok"
    model: "grok-1"                    # The model to use (default: grok-1)
    temperature: 0.7                   # Optional: Controls randomness (0.0-1.0)
    max_tokens: 1024                   # Optional: Maximum tokens in the response
    system_prompt: "Your instruction"  # Optional: System prompt for the conversation
```

The API key is typically set via the `GROK_API_KEY` environment variable.

### DeepSeek

[DeepSeek](https://www.deepseek.com/) offers state-of-the-art language models with strong reasoning and problem-solving capabilities.

#### Example Models

The following are examples of available models. PromptDrifter supports any model offered by DeepSeek's API:

- `deepseek-r1` - Research model with advanced reasoning
- `deepseek-v3-0324` - Enterprise-grade model for various applications

#### Configuration Options

```yaml
adapter:
  - type: "deepseek"
    model: "deepseek-chat"             # The model to use (default: deepseek-chat)
    temperature: 0.7                   # Optional: Controls randomness (0.0-1.0)
    max_tokens: 1024                   # Optional: Maximum tokens in the response
    system_prompt: "Your instruction"  # Optional: System prompt for the conversation
```

The API key is typically set via the `DEEPSEEK_API_KEY` environment variable.

### Mistral

[Mistral AI](https://mistral.ai/) provides efficient, high-performance language models with strong reasoning capabilities.

#### Example Models

The following are examples of available models. PromptDrifter supports any model offered by Mistral's API:

- `mistral-small-24b-instruct-2501` - 24B parameter model optimized for instruction following
- `mistral-small-3.1-24b-instruct-2503` - Updated 24B parameter model with enhanced capabilities

#### Configuration Options

```yaml
adapter:
  - type: "mistral"
    model: "mistral-large-latest"      # The model to use (default: mistral-large-latest)
    temperature: 0.7                   # Optional: Controls randomness (0.0-1.0)
    max_tokens: 1024                   # Optional: Maximum tokens in the response
    system_prompt: "Your instruction"  # Optional: System prompt for the conversation
```

The API key is typically set via the `MISTRAL_API_KEY` environment variable.

### Azure OpenAI

[Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service/) provides enterprise-grade OpenAI models through Microsoft's Azure cloud platform with added security and compliance features.

#### Example Models

PromptDrifter supports any model deployed to your Azure OpenAI service:

- Same models as OpenAI (GPT-3.5, GPT-4, etc.)
- Custom deployments configured in your Azure account

#### Configuration Options

```yaml
adapter:
  - type: "azure"
    endpoint: "https://your-resource.openai.azure.com/" # Your Azure endpoint
    api_version: "2023-05-15"          # Azure OpenAI API version
    deployment: "your-deployment-name" # Your Azure deployment name
    temperature: 0.7                   # Optional: Controls randomness (0.0-1.0)
    max_tokens: 1024                   # Optional: Maximum tokens in the response
    system_prompt: "Your instruction"  # Optional: System prompt for the conversation
```

The API key is typically set via the `AZURE_OPENAI_API_KEY` environment variable.

## Adapter Best Practices

- **API Keys**: Store API keys securely using environment variables
- **Error Handling**: Implement proper error handling for API rate limits and service outages
- **Caching**: Use response caching for frequent identical requests
- **Monitoring**: Monitor usage to manage costs and performance
