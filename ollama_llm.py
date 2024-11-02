from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class OllamaLLM:
    def __init__(self, model='llama3.1:8b-instruct-q8_0', temperature=0):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )
        self.prompt = PromptTemplate(
            template="""Using the provided contexts, respond concisely and exactly to the question. Answer only from the two contexts given below.

            context1: {context_graph}

            context2: {context_vector}

            question: {question}
            """,
            input_variables=["context_graph", "context_vector", "question"],
        )
        self.rag_chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, context_graph, context_vector, question):
        try:
            result = self.rag_chain.invoke({
                "context_graph": context_graph,
                "context_vector": context_vector,
                "question": question
            })
            return result
        except Exception as e:
            print(f"An error occurred while invoking the LLM: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def switch_model(self, new_model: str):
        """
        Switch the current model to a new one.
        
        Args:
            new_model (str): The name of the new model to use.
        """
        self.llm = ChatOllama(
            model=new_model,
            temperature=self.llm.temperature,
        )
        print(f"Model switched to: {new_model}")

if __name__ == "__main__":
    # Example usage
    ollama_llm = OllamaLLM(model='qwen2.5:0.5b-instruct')
    
    # Sample contexts and question
    context_graph = "Pete Warden is a software engineer and entrepreneur known for his work in machine learning and mobile technology."
    context_vector = "Pete Warden co-founded Jetpac, a startup that used AI to analyze Instagram photos for travel recommendations."
    question = "Who is Pete Warden and what is he known for?"

    result = ollama_llm.invoke(context_graph, context_vector, question)
    print("LLM Response:")
    print(result)





