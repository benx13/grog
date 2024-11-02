import os
from wtpsplit import SaT
import torch


os.environ["TOKENIZERS_PARALLELISM"] = "true"

class SentenceSplitter:
    def __init__(self):

        
        # Use CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.splitter = SaT("sat-3l-sm").half().to(self.device)

    def invoke(self, text: str) -> list:
        """
        Splits text into sentences using the SaT model.
        Optimized for CUDA and parallel processing.
        
        Args:
            text (str): The text to split into sentences
            
        Returns:
            list: List of sentences
        """
        try:
            if not text:
                return []
            
            # Split text using SaT model
            sentences = [s.strip() for s in self.splitter.split(text)]
            sentences = self.splitter.split(text)
            return sentences
            
        except Exception as e:
            print(f"An error occurred in sentence splitting: {e}")
            import traceback
            print(traceback.format_exc())
            return [text]  # Return original text as single sentence if splitting fails

    def invoke_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Splits multiple texts into sentences using the SaT model.
        More efficient than processing one text at a time.
        
        Args:
            texts (list[str]): List of texts to split
            
        Returns:
            list[list[str]]: List of sentence lists
        """
        try:
            if not texts:
                return []
            
            # Process batch of texts
            return [[s.strip() for s in sentences] 
                   for sentences in self.splitter.split(texts)]
            
        except Exception as e:
            print(f"An error occurred in sentence splitting: {e}")
            import traceback
            print(traceback.format_exc())
            return [texts]  # Return original texts if splitting fails

if __name__ == "__main__":
    # Example usage
    splitter = SentenceSplitter()
    
    test_text = """This is a very important sentence about the topic. 
                   This is another sentence. 
                   Here's a third piece of text."""
    
    sentences = splitter.invoke(test_text)
    print("Split sentences:")
    for sentence in sentences:
        print(f"- {sentence}") 