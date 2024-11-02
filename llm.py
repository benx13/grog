from abc import ABC, abstractmethod
from ollama_llm import OllamaLLM


class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, context_graph, context_vector, question):
        pass


class LLMFactory:
    @staticmethod
    def create_llm(llm_type, **kwargs):
        if llm_type.lower() == 'ollama':
            return OllamaLLM(**kwargs)
        # Add more LLM types here as needed
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

class LLM:
    def __init__(self, llm_type, **kwargs):
        self.llm = LLMFactory.create_llm(llm_type, **kwargs)

    def invoke(self, context_graph, context_vector, question):
        return self.llm.invoke(context_graph, context_vector, question)

if __name__ == "__main__":
    # Example usage
    llm = LLM('ollama', model='qwen2.5:0.5b-instruct', temperature=0)
    
    # Sample contexts and question
    context_graph = "Pete Warden is a software engineer and entrepreneur known for his work in machine learning and mobile technology."
    context_vector = "Pete Warden co-founded Jetpac, a startup that used AI to analyze Instagram photos for travel recommendations."
    question = "Who is Pete Warden and what is he known for?"

    result = llm.invoke(context_graph, context_vector, question)
    print("LLM Response:")
    print(result)
