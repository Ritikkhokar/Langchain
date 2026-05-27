"""
LangChain Text Generation with Multiple LLM Providers
Supports: OpenAI, Anthropic, and local models
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

load_dotenv()


class TextGenerator:
    def __init__(self, provider: str = "openai", model: str = None, streaming: bool = False):
        """
        Initialize text generator with specified provider.
        
        Args:
            provider: 'openai' or 'anthropic'
            model: specific model name (e.g., 'gpt-4', 'claude-3-opus')
            streaming: whether to stream output
        """
        self.provider = provider
        self.streaming = streaming
        self.llm = self._init_provider(model)
    
    def _init_provider(self, model: str):
        """Initialize the LLM provider."""
        callbacks = [StreamingStdOutCallbackHandler()] if self.streaming else []
        
        if self.provider == "openai":
            return ChatOpenAI(
                model=model or "gpt-3.5-turbo",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY"),
                callbacks=callbacks
            )
        elif self.provider == "anthropic":
            return ChatAnthropic(
                model=model or "claude-3-sonnet-20240229",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                callbacks=callbacks
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(self, prompt: str, system_message: str = None) -> str:
        """
        Generate text based on prompt.
        
        Args:
            prompt: user prompt
            system_message: optional system message
            
        Returns:
            Generated text
        """
        if system_message:
            template = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "{input}")
            ])
            chain = template | self.llm
            response = chain.invoke({"input": prompt})
        else:
            response = self.llm.invoke(prompt)
        
        return response.content
    
    def generate_with_context(self, prompt: str, context: str) -> str:
        """Generate text with additional context."""
        system_msg = f"You have the following context:\n\n{context}\n\nUse this to inform your response."
        return self.generate(prompt, system_msg)


def main():
    """Example usage."""
    # Example 1: Simple text generation with OpenAI
    print("=" * 60)
    print("Example 1: OpenAI Text Generation")
    print("=" * 60)
    
    try:
        generator = TextGenerator(provider="openai", model="gpt-3.5-turbo")
        result = generator.generate("Write a short poem about artificial intelligence")
        print(result)
    except Exception as e:
        print(f"OpenAI example failed: {e}")
    
    # Example 2: With system message
    print("\n" + "=" * 60)
    print("Example 2: With System Message")
    print("=" * 60)
    
    try:
        generator = TextGenerator(provider="openai")
        result = generator.generate(
            prompt="What is machine learning?",
            system_message="You are a helpful AI educator. Explain concepts clearly and concisely."
        )
        print(result)
    except Exception as e:
        print(f"System message example failed: {e}")
    
    # Example 3: With context
    print("\n" + "=" * 60)
    print("Example 3: Generation with Context")
    print("=" * 60)
    
    try:
        generator = TextGenerator(provider="openai")
        context = "Python is a high-level programming language known for its simplicity and readability."
        result = generator.generate_with_context(
            prompt="Why should beginners learn Python?",
            context=context
        )
        print(result)
    except Exception as e:
        print(f"Context example failed: {e}")


if __name__ == "__main__":
    main()
