# LangChain Text Generation Project

A comprehensive LangChain setup for text generation with multiple LLM providers (OpenAI, Anthropic, and local models).

## Setup

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Files

- **text_generator.py** - Basic text generation with multiple providers
- **advanced_chain.py** - Advanced patterns (chains, structured output, variations)
- **requirements.txt** - Project dependencies
- **.env.example** - Environment variable template

## Quick Start

### Basic Text Generation
```python
from text_generator import TextGenerator

# OpenAI
generator = TextGenerator(provider="openai")
result = generator.generate("Write a short story about AI")
print(result)

# Anthropic
generator = TextGenerator(provider="anthropic")
result = generator.generate("Explain quantum computing")
print(result)
```

### With System Message
```python
result = generator.generate(
    prompt="What is machine learning?",
    system_message="You are a helpful educator."
)
```

### Advanced Chains
```python
from advanced_chain import AdvancedTextGenerator

generator = AdvancedTextGenerator()
result = generator.sequential_generation("Climate Change")
print(result['outline'])
print(result['article'])
```

## Providers

- **OpenAI**: gpt-4, gpt-3.5-turbo
- **Anthropic**: claude-3-opus, claude-3-sonnet
- **Local**: Requires Ollama setup

## API Keys

Get your API keys from:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

## Running Examples

```bash
python text_generator.py
python advanced_chain.py
```
