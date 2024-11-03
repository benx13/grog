import networkx as nx
import json
import yaml
from typing import Dict, Any

# Import base components
from components import QueryInput, ResponseOutput
from retriever import Retriever
from llm import LLM

class RAGPipeline:
    def __init__(self, barfi_path="schema.barfi", components_path="components.yaml"):
        print("Initializing RAG Pipeline...")
        
        # Load configurations
        with open(components_path, 'r') as f:
            self.components_config = yaml.safe_load(f)['components']
        
        # Component class mapping
        self.component_classes = {
            'query_input': QueryInput,
            'response_output': ResponseOutput,
            'vector_retriever': lambda params: Retriever('vector', **params),
            'graph_retriever': lambda params: Retriever('graph', **params),
            'llm': lambda params: LLM('ollama', **params)
        }
        
        self.pipeline_graph = self._load_barfi_schema(barfi_path)
        self.components = {}
        self._initialize_components()
        
        print("Pipeline initialized successfully")

    def _load_barfi_schema(self, barfi_path: str) -> nx.DiGraph:
        """Load the pipeline graph from a barfi schema file"""
        print(f"Loading pipeline schema from {barfi_path}")
        
        with open(barfi_path, 'r', encoding='utf-8') as f:
            barfi_data = json.load(f)
            
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node_data in barfi_data['nodes'].items():
            component_type = node_data['type'].lower().replace(' ', '_')
            print(f"Adding node: {node_id} of type {component_type}")
            G.add_node(
                node_id, 
                type=component_type,
                parameters=node_data.get('parameters', {}),
                interfaces=node_data.get('interfaces', {})
            )
        
        # Add edges
        for edge in barfi_data['edges']:
            print(f"Adding edge: {edge['source']} -> {edge['target']}")
            G.add_edge(
                edge['source'], 
                edge['target'],
                source_port=edge['source_port'],
                target_port=edge['target_port']
            )
            
        return G

    def _initialize_components(self):
        """Initialize component instances"""
        print("\nInitializing components...")
        
        for node_id, node_data in self.pipeline_graph.nodes(data=True):
            component_type = node_data['type'].lower().replace(' ', '_')
            parameters = node_data.get('parameters', {})
            
            try:
                if 'retriever' in component_type:
                    # Extract retriever type from component type (e.g., 'vector' from 'vector_retriever')
                    retriever_type = component_type.split('_')[0]
                    print(f"Initializing {retriever_type} retriever for node {node_id}")
                    self.components[node_id] = Retriever(retriever_type, **parameters)
                
                elif component_type == 'llm':
                    print(f"Initializing LLM for node {node_id}")
                    self.components[node_id] = LLM('ollama', **parameters)
                
                elif component_type in ['query_input', 'response_output']:
                    print(f"Initializing {component_type} for node {node_id}")
                    component_class = self.component_classes[component_type]
                    self.components[node_id] = component_class()
                
                else:
                    print(f"Warning: Unknown component type {component_type}")
                    
            except Exception as e:
                print(f"Error initializing component {node_id}: {str(e)}")
                raise

    def process(self, query: str) -> Dict[str, Any]:
        print("\nStarting pipeline execution...")
        print(f"Initial query: {query}")
        
        results = {}
        
        # Get processing order
        try:
            processing_order = list(nx.topological_sort(self.pipeline_graph))
            print(f"Processing order: {processing_order}")
        except nx.NetworkXUnfeasible:
            raise ValueError("Pipeline graph must be acyclic")

        # Process each node in order
        for node_id in processing_order:
            print(f"\nProcessing node: {node_id}")
            
            node_data = self.pipeline_graph.nodes[node_id]
            component_type = node_data['type'].lower().replace(' ', '_')
            component = self.components.get(node_id)
            
            if component is None:
                print(f"Warning: No component found for node {node_id}")
                continue
                
            # Collect inputs for the component
            inputs = {'query': query}  # Pass original query to all components
            for pred in self.pipeline_graph.predecessors(node_id):
                edge_data = self.pipeline_graph[pred][node_id]
                source_port = edge_data['source_port']
                target_port = edge_data['target_port']
                
                # Get the output from the predecessor's results
                pred_output = results.get(f"{pred}_{source_port}")
                if pred_output is not None:
                    print(f"  Input {target_port}: {pred_output}")
                    inputs[target_port] = pred_output
            
            # Invoke the component
            try:
                print(f"  Invoking {component_type} component")
                component_results = component.invoke(**inputs)
                
                # Store results
                for output_name, output_value in component_results.items():
                    result_key = f"{node_id}_{output_name}"
                    results[result_key] = output_value
                    print(f"  Output {output_name}: {output_value}")
                    
            except Exception as e:
                print(f"Error processing node {node_id}: {str(e)}")
                raise
        
        print("\nPipeline execution completed")
        return results

    def close(self):
        """Close all components that have a close method"""
        for component in self.components.values():
            if hasattr(component, 'close'):
                component.close()

if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline()
    results = pipeline.process("What is the capital of France?")
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))

