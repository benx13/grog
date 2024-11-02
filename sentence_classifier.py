import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple

class SentenceClassifier:
    def __init__(
        self,
        model_dir: str,
        device: str = None,
        max_length: int = 512,
        batch_size: int = 32
    ):
        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.tokenizer = None
        self.inverse_label_mapping = {0: 'unnecessary', 1: 'necessary'}
        self.load_trained_model()

    def load_trained_model(self):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(self.model_dir, 'model')
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.model_dir, 'tokenizer')
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"An error occurred during model loading: {e}")
            raise

    def _predict_batch(self, prompt: str, sentences: List[str]) -> List[Tuple[str, str]]:
        try:
            input_texts = [f"{prompt} [SEP] {sentence}" for sentence in sentences]
            inputs = self.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            return [(sentence, self.inverse_label_mapping.get(pred, "unknown"))
                    for sentence, pred in zip(sentences, predictions)]
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            raise

    def invoke(self, sentences: List[str], prompt: str = None) -> List[str]:
        """
        Classifies pre-split sentences as necessary or unnecessary.
        
        Args:
            sentences (List[str]): List of sentences to classify
            prompt (str, optional): The prompt to use for classification
            
        Returns:
            List[str]: List of necessary sentences
        """
        try:
            if not sentences:
                return []
            
            # Use a default prompt if none provided
            prompt = prompt or "Is this sentence necessary for understanding the context?"
            
            necessary_sentences = []
            
            # Process sentences in batches
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i:i + self.batch_size]
                predictions = self._predict_batch(prompt, batch)
                necessary_sentences.extend(sent for sent, label in predictions if label == 'necessary')
            
            return necessary_sentences
            
        except Exception as e:
            print(f"An error occurred in sentence classification: {e}")
            import traceback
            print(traceback.format_exc())
            return sentences  # Return original sentences if classification fails
            
    def close(self):
        """
        Cleanup method (if needed)
        """
        pass

if __name__ == "__main__":
    # Example usage
    model_directory = '/path/to/your/model'
    classifier = SentenceClassifier(model_dir=model_directory)
    
    test_sentences = [
        "This is a very important sentence about the topic.",
        "This is irrelevant information.",
        "Here's another crucial piece of information."
    ]
    
    filtered_sentences = classifier.invoke(test_sentences)
    print("Filtered sentences:")
    for sentence in filtered_sentences:
        print(f"- {sentence}") 