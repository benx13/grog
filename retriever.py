from abc import ABC, abstractmethod
from graph_retriever import GraphRetriever
from vector_retriever import VectorRetriever

class BaseRetriever(ABC):
    @abstractmethod
    def invoke(self, input_prompt):
        pass

    @abstractmethod
    def close(self):
        pass

class RetrieverFactory:
    @staticmethod
    def create_retriever(retriever_type, **kwargs):
        if retriever_type.lower() == 'graph':
            return GraphRetriever(**kwargs)
        elif retriever_type.lower() == 'vector':
            return VectorRetriever(**kwargs)
        # Add more retriever types here as needed
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")

class Retriever:
    def __init__(self, retriever_type, **kwargs):
        self.retriever = RetrieverFactory.create_retriever(retriever_type, **kwargs)

    def invoke(self, input_prompt):
        return self.retriever.invoke(input_prompt)

    def close(self):
        self.retriever.close()

if __name__ == "__main__":
    # Example usage for Graph Retriever
    graph_retriever = Retriever('graph', uri="bolt://localhost:7687", user="neo4j", password="strongpassword")
    
    # Example usage for Vector Retriever
    vector_retriever = Retriever('vector', uri="tcp://192.168.1.96:19530", collection_name="context_path_chunks_noblogs____valid")

    # Use graph retriever
    query = "pete warden"
    graph_results = graph_retriever.invoke(query)
    print("Results from Graph Retriever:")
    print(graph_results)

    # Use vector retriever
    vector_results = vector_retriever.invoke(query)
    print("\nResults from Vector Retriever:")
    print(vector_results)

    # Close the connections
    graph_retriever.close()
    vector_retriever.close()
