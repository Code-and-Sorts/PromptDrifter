# Configuration for Adapters

# OpenAI
OPENAI_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
API_KEY_ENV_VAR_OPENAI = "OPENAI_API_KEY"

# Google Gemini
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest"
API_KEY_ENV_VAR_GEMINI = "GEMINI_API_KEY"

# Qwen (Tongyi Qianwen via DashScope)
API_KEY_ENV_VAR_QWEN = "QWEN_API_KEY"
QWEN_API_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"

# Ollama
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"

# Mistral
MISTRAL_API_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MISTRAL_MODEL = "mistral-large-latest"
API_KEY_ENV_VAR_MISTRAL = "MISTRAL_API_KEY"

# Claude (Anthropic)
CLAUDE_API_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_CLAUDE_MODEL = "claude-3-opus-20240229"
API_KEY_ENV_VAR_CLAUDE = "CLAUDE_API_KEY"
CLAUDE_API_VERSION = "2023-06-01"

# Grok (xAI)
GROK_API_BASE_URL = "https://api.x.ai"
DEFAULT_GROK_MODEL = "grok-1"
API_KEY_ENV_VAR_GROK = "GROK_API_KEY"

# DeepSeek
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
API_KEY_ENV_VAR_DEEPSEEK = "DEEPSEEK_API_KEY"

# Azure OpenAI
AZURE_OPENAI_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_AZURE_OPENAI_MODEL = "gpt-3.5-turbo"
API_KEY_ENV_VAR_AZURE_OPENAI = "AZURE_OPENAI_API_KEY"

# Llama (Meta)
LLAMA_API_BASE_URL = "https://llama-api.meta.ai/v1"
DEFAULT_LLAMA_MODEL = "llama-3-70b-instruct"
API_KEY_ENV_VAR_LLAMA = "META_LLAMA_API_KEY"
