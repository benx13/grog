from retriever import Retriever
from llm import LLM
from sentence_splitter import SentenceSplitter
from sentence_classifier import SentenceClassifier

class RAGPipeline:
    def __init__(self, config):
        self.components = {}
        self.pipeline = config['pipeline']
        self.build_pipeline(config['components'])

    def build_pipeline(self, config):
        # Initialize Graph Retriever
        if 'graph_retriever' in config:
            self.components['graph_retriever'] = Retriever(
                'graph',
                **config['graph_retriever']
            )

        # Initialize Vector Retriever
        if 'vector_retriever' in config:
            self.components['vector_retriever'] = Retriever(
                'vector',
                **config['vector_retriever']
            )

        # Initialize Sentence Splitter
        if 'sentence_splitter' in config:
            self.components['sentence_splitter'] = SentenceSplitter(
                **config['sentence_splitter']
            )

        # Initialize Sentence Classifier
        if 'sentence_classifier' in config:
            self.components['sentence_classifier'] = SentenceClassifier(
                **config['sentence_classifier']
            )

        # Initialize LLM
        if 'llm' in config:
            self.components['llm'] = LLM(
                config['llm']['type'],
                **config['llm']['params']
            )

        # Initialize other components (examples)
        if 'router' in config:
            self.components['router'] = self.create_router(config['router'])

        if 'prompt_splitter' in config:
            self.components['prompt_splitter'] = self.create_prompt_splitter(config['prompt_splitter'])

        # Add more components as needed

    def create_router(self, config):
        # Placeholder for router creation
        # Implement the actual router logic based on your needs
        class Router:
            def route(self, query):
                # Add routing logic here
                pass
        return Router()

    def create_prompt_splitter(self, config):
        # Placeholder for prompt splitter creation
        # Implement the actual prompt splitter logic based on your needs
        class PromptSplitter:
            def split(self, prompt):
                # Add splitting logic here
                pass
        return PromptSplitter()

    def process(self, query):
        results = {}
        queries = {}

        # Route the query if a router is present
        if self.pipeline['router']:
            query = self.components['router'].route(query)

        # Split the prompt if a prompt splitter is present
        if self.pipeline['prompt_splitter']:
            split_queries = self.components['prompt_splitter'].invoke(query)
            

        if self.pipeline['sentence_classifier']:

        # Retrieve from graph
        if self.pipeline['graph_retriever']:
            results['graph_context'] = self.components['graph_retriever'].invoke(query)

        # Retrieve from vector store
        if self.pipeline['vector_retriever']:
            results['vector_context'] = self.components['vector_retriever'].invoke(query)

        # Process with LLM
        if self.pipeline['llm']:
            results['llm_response'] = self.components['llm'].invoke(
                results.get('graph_context', ''),
                results.get('vector_context', ''),
                query
            )
        return results

    def close(self):
        # Close all retriever connections
        for component in self.components.values():
            if isinstance(component, Retriever):
                component.close()

if __name__ == "__main__":
    # Example configuration
    config = {
        'pipeline': {
            'prompt_splitter': False,
            'router': False,
            'graph_retriever': True,
            'vector_retriever': True,
            'sentence_classifier': True,
            'llm': True
        },
        'components': {
            'graph_retriever': {
                'uri': "bolt://localhost:7687",
            'user': "neo4j",
            'password': "strongpassword"
            },
            'vector_retriever': {
                'uri': "tcp://192.168.1.96:19530",
                'collection_name': "context_path_chunks_noblogs____valid"
            },
            'sentence_splitter': {
                'split_char': '.'  # Optional configuration
            },
            'sentence_classifier': {
                'model_dir': '/path/to/your/model',
                'device': 'cuda',  # or 'cpu'
                'max_length': 512,
                'batch_size': 32
            },
            'llm': {
                'type': 'ollama',
                'params': {
                    'model': 'qwen2.5:0.5b-instruct',
                    'temperature': 0
                }
            },
            'router': {},  # Add router configuration if needed
            'prompt_splitter': {}  # Add prompt splitter configuration if needed
        }
    }

    # Create and use the pipeline
    pipeline = RAGPipeline(config)
    query = "hello assitant, Who is Wassim Kezai?"
    results = pipeline.process(query)

    print("Graph Context:", results.get('graph_context', ''))
    print("Vector Context:", results.get('vector_context', ''))
    print("LLM Response:", results.get('llm_response', ''))

    # Close the pipeline connections
    pipeline.close()
