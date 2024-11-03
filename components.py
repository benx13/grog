from typing import Dict, Any
from abc import ABC, abstractmethod

class BaseComponent(ABC):
    """Base class for all pipeline components"""
    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}

    @abstractmethod
    def invoke(self, **kwargs) -> Dict[str, Any]:
        """Execute the component's main functionality"""
        pass

class QueryInput(BaseComponent):
    """Component that handles the initial query input"""
    def invoke(self, query: str = None, **kwargs) -> Dict[str, Any]:
        """
        Process the initial query.
        
        Args:
            query (str): The input query
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Dictionary containing the query
        """
        print(f"[QueryInput] Processing query: {query}")
        if not query:
            raise ValueError("Query cannot be empty")
        return {"query": query}

class ResponseOutput(BaseComponent):
    """Component that handles the final response output"""
    def invoke(self, response: str = None, **kwargs) -> Dict[str, Any]:
        """
        Process the final response.
        
        Args:
            response (str): The response to output
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Dictionary containing the response
        """
        print(f"[ResponseOutput] Final response: {response}")
        if not response:
            raise ValueError("Response cannot be empty")
        return {"response": response}

# Example usage
if __name__ == "__main__":
    # Test QueryInput
    query_input = QueryInput()
    result = query_input.invoke(query="What is the capital of France?")
    print(result)
    
    # Test ResponseOutput
    response_output = ResponseOutput()
    result = response_output.invoke(response="The capital of France is Paris.")
    print(result)
