"""
Advanced LangChain patterns for text generation.
Includes chains, output parsing, and memory management.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from typing import List

load_dotenv()


class AdvancedTextGenerator:
    """Advanced text generation with chains and structured output."""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm = self._init_llm(provider)
    
    def _init_llm(self, provider: str):
        """Initialize LLM."""
        if provider == "openai":
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        elif provider == "anthropic":
            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.7)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def sequential_generation(self, topic: str) -> dict:
        """
        Generate content in sequence:
        1. Create outline
        2. Generate full text from outline
        """
        # Step 1: Create outline
        outline_prompt = PromptTemplate(
            input_variables=["topic"],
            template="Create a brief outline for an article about {topic}. List 3-4 main points."
        )
        
        # Step 2: Expand outline
        expansion_prompt = PromptTemplate(
            input_variables=["outline", "topic"],
            template="Based on this outline:\n{outline}\n\nWrite a detailed article about {topic}."
        )
        
        chain1 = LLMChain(llm=self.llm, prompt=outline_prompt, output_key="outline")
        chain2 = LLMChain(llm=self.llm, prompt=expansion_prompt, output_key="article")
        
        sequential_chain = SequentialChain(
            chains=[chain1, chain2],
            input_variables=["topic"],
            output_variables=["outline", "article"]
        )
        
        return sequential_chain.invoke({"topic": topic})
    
    def generate_variations(self, text: str, num_variations: int = 3) -> List[str]:
        """Generate multiple variations of text."""
        prompt = ChatPromptTemplate.from_template(
            "Generate a {num} word alternative version of this text that maintains the meaning:\n{text}"
        )
        
        variations = []
        for i in range(num_variations):
            chain = LLMChain(llm=self.llm, prompt=prompt)
            variation = chain.run(text=text, num="100-150")
            variations.append(variation.strip())
        
        return variations
    
    def structured_extraction(self, text: str) -> List[str]:
        """Extract structured information as a list."""
        output_parser = CommaSeparatedListOutputParser()
        
        prompt = ChatPromptTemplate.from_template(
            "Extract the key topics from this text as a comma-separated list:\n{text}\n"
            "Format: topic1, topic2, topic3"
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=output_parser
        )
        
        return chain.run(text=text)


def main():
    print("=" * 60)
    print("Advanced Text Generation Examples")
    print("=" * 60)
    
    generator = AdvancedTextGenerator(provider="openai")
    
    # Example 1: Sequential generation
    print("\n1. Sequential Generation (Outline → Full Article)")
    print("-" * 60)
    try:
        result = generator.sequential_generation("Quantum Computing")
        print(f"Outline:\n{result['outline']}\n")
        print(f"Full Article:\n{result['article'][:500]}...")
    except Exception as e:
        print(f"Sequential generation failed: {e}")
    
    # Example 2: Generate variations
    print("\n2. Generate Text Variations")
    print("-" * 60)
    try:
        original = "Artificial intelligence is transforming how we work and live."
        variations = generator.generate_variations(original, num_variations=2)
        print(f"Original: {original}\n")
        for i, var in enumerate(variations, 1):
            print(f"Variation {i}: {var}\n")
    except Exception as e:
        print(f"Text variations failed: {e}")
    
    # Example 3: Structured extraction
    print("\n3. Structured Information Extraction")
    print("-" * 60)
    try:
        sample_text = """
        Machine learning enables computers to learn from data without being explicitly programmed.
        Deep learning uses neural networks with multiple layers. Natural language processing helps 
        computers understand and generate human language. Computer vision enables machines to interpret images.
        """
        topics = generator.structured_extraction(sample_text)
        print(f"Extracted topics: {topics}")
    except Exception as e:
        print(f"Structured extraction failed: {e}")


if __name__ == "__main__":
    main()
