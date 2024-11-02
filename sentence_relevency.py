import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple

class DistilBERTClassifier:
    def __init__(
        self,
        model_dir: str,
        device: str = None,
        max_length: int = 512,
        batch_size: int = 32
    ):
        """
        Initializes the DistilBERTClassifier.

        :param model_dir: Directory where the trained model and tokenizer are saved.
                          It should contain 'model' and 'tokenizer' subdirectories.
        :param device: Device to run the model on ('cuda' or 'cpu'). If None, automatically detects.
        :param max_length: Maximum token length for the tokenizer.
        :param batch_size: Number of sentences to process in a batch during prediction.
        """
        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size

        # Determine device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.tokenizer = None
        self.inverse_label_mapping = {0: 'unnecessary', 1: 'necessary'}  # Adjust based on your label encoding

    def load_trained_model(self):
        """
        Loads the trained DistilBERT model and tokenizer from the specified directory.
        """
        try:
            print("Loading the trained model and tokenizer for prediction...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(self.model_dir, 'model')
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.model_dir, 'tokenizer')
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"An error occurred during model loading: {e}")
            raise

    def predict_sentences(
        self,
        prompts: List[str],
        nested_sentences: List[List[str]]
    ) -> List[List[Tuple[str, str]]]:
        """
        Predicts labels for a nested list of sentences.

        :param prompts: A list of prompt strings.
        :param nested_sentences: A list of lists, where each inner list contains sentences corresponding to each prompt.
        :return: A list of lists containing tuples of (sentence, label).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before making predictions.")

        if len(prompts) != len(nested_sentences):
            raise ValueError("The number of prompts must match the number of sentence lists.")

        results = []

        for idx, (prompt, sentence_list) in enumerate(zip(prompts, nested_sentences)):
            print(f"Processing prompt {idx + 1}/{len(prompts)}...")
            batch_results = []
            for i in range(0, len(sentence_list), self.batch_size):
                batch_sentences = sentence_list[i:i + self.batch_size]
                predictions = self._predict_batch(prompt, batch_sentences)
                batch_results.extend(predictions)
            results.append(batch_results)

        return results

    def _predict_batch(
        self,
        prompt: str,
        sentences: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Predicts labels for a batch of sentences.

        :param prompt: The prompt string to concatenate with each sentence.
        :param sentences: A list of sentences.
        :return: A list of tuples containing (sentence, label).
        """
        try:
            # Construct input_text as prompt + " [SEP] " + sentence
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
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()

            labeled_sentences = [
                (sentence, self.inverse_label_mapping.get(pred, "unknown"))
                for sentence, pred in zip(sentences, predictions)
            ]

            return labeled_sentences

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            raise


# Initialize the classifier
model_directory = '/content/drive/MyDrive/models/model_necessary'  # Replace with your model directory
classifier = DistilBERTClassifier(
    model_dir=model_directory,
    max_length=512,
    batch_size=32  # Adjust based on your memory constraints
)

# Load the trained model and tokenizer
classifier.load_trained_model()

# Example list of prompts


# Ensure the lengths match
if len(questions) != len(split_output):
    raise ValueError("The number of prompts must match the number of split sentence lists.")

# Perform predictions
predictions = classifier.predict_sentences(questions, split_output)

# Display results
for i, batch in enumerate(predictions):
    print(f"\nPrompt {i + 1}: {questions[i]}")
    for sentence, label in batch:
        print(f"Sentence: {sentence}\nLabel: {label}\n")


