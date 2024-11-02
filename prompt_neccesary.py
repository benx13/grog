import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset as HFDataset
import warnings
import numpy as np
warnings.filterwarnings("ignore")
# multi_model_pipeline.py
from transformers import pipeline
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import datetime

import logging
from tqdm import tqdm





# Configure the logging
logging.basicConfig(
    filename='app.log',              # Log file name
    filemode='a',                     # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO                # Log level
)

# Create a logger
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    task: str
    path: str
    label_mapping: Optional[Dict[int, str]] = field(default_factory=dict)
    additional_config: Optional[Dict[str, Any]] = field(default_factory=dict)




class SequenceClassifier:
    """
    A sequence classifier for categorizing text data into predefined classes using Hugging Face Transformers.
    This class handles data loading, preprocessing, tokenization, model training, evaluation, and inference.
    """

    def __init__(
        self,
        csv_file_path,
        output_model_dir,
        model_name='bert-base-uncased',
        max_length=128,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3,
        weight_decay=0.01,
        seed=42,
        device=None,
        log_file='sequence_classifier.log'
    ):
        """
        Initializes the SequenceClassifier with the given parameters.

        Parameters:
        - csv_file_path (str): Path to the input CSV file containing prompts and sentences.
        - output_model_dir (str): Directory where the trained model and tokenizer will be saved.
        - model_name (str): Pre-trained model name from Hugging Face (e.g., 'bert-base-uncased').
        - max_length (int): Maximum token length for inputs.
        - batch_size (int): Batch size for training and evaluation.
        - learning_rate (float): Learning rate for the optimizer.
        - num_epochs (int): Number of training epochs.
        - weight_decay (float): Weight decay for the optimizer.
        - seed (int): Random seed for reproducibility.
        - device (str): Device to use ('cuda' or 'cpu'). If None, automatically detected.
        - log_file (str): File to save training logs.
        """
        # Configure logging
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

        # Set seed for reproducibility
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Determine device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.logger.info(f"Using device: {self.device}")

        # Initialize parameters
        self.csv_file_path = csv_file_path
        self.output_model_dir = output_model_dir
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay

        # Initialize label mapping
        self.label_mapping = {'unnecessary': 0, 'necessary': 1}
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

        # Initialize tokenizer and model (to be loaded later)
        self.tokenizer = None
        self.model = None

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None

        # Initialize data collator
        self.data_collator = None

        # Initialize Trainer
        self.trainer = None

    def load_and_preprocess_data(self):
        """
        Loads the dataset from the CSV file, preprocesses it by combining prompts and sentences,
        cleans labels, maps labels to integers, and splits the data into training and validation sets.
        """
        try:
            self.logger.info("Loading dataset...")
            df = pd.read_csv(self.csv_file_path)

            # Combine the prompt and sentence for input
            self.logger.info("Combining prompts and sentences...")
            df['input_text'] = df['prompt'].astype(str) + " [SEP] " + df['sentence'].astype(str)

            # Clean the labels
            self.logger.info("Cleaning labels...")
            df['label'] = df['label'].astype(str).str.strip().str.lower()

            # Map labels to integers
            self.logger.info("Mapping labels to integers...")
            df['label'] = df['label'].map(self.label_mapping)

            # Drop rows with NaN labels after mapping
            initial_count = len(df)
            df.dropna(subset=['label'], inplace=True)
            final_count = len(df)
            self.logger.info(f"Dropped {initial_count - final_count} rows due to invalid labels.")
            print(f"Dropped {initial_count - final_count} rows due to invalid labels.")

            # Convert labels to integers
            df['label'] = df['label'].astype(int)

            # Split the data into train and validation sets
            self.logger.info("Splitting data into training and validation sets...")
            train_df, val_df = train_test_split(
                df,
                test_size=0.2,
                random_state=self.seed,
                stratify=df['label']
            )

            # Convert to Hugging Face Dataset format
            self.logger.info("Converting data to Hugging Face Dataset format...")
            train_dataset = HFDataset.from_pandas(train_df.reset_index(drop=True))
            val_dataset = HFDataset.from_pandas(val_df.reset_index(drop=True))

            self.train_df = train_df
            self.val_df = val_df
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

            self.logger.info("Data loading and preprocessing completed successfully.")

        except Exception as e:
            self.logger.error(f"Error during data loading and preprocessing: {e}")
            print(f"An error occurred during data loading and preprocessing: {e}")
            raise

    def tokenize_datasets(self):
        """
        Initializes the tokenizer and tokenizes the training and validation datasets.
        """
        try:
            self.logger.info("Initializing tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Tokenization function
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['input_text'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                )

            # Tokenize the datasets with progress bar
            self.logger.info("Tokenizing training dataset...")
            self.train_dataset = self.train_dataset.map(
                tokenize_function,
                batched=True,
                desc="Tokenizing training data"
            )
            self.logger.info("Tokenizing validation dataset...")
            self.val_dataset = self.val_dataset.map(
                tokenize_function,
                batched=True,
                desc="Tokenizing validation data"
            )

            # Remove columns other than the ones required
            columns_to_remove = list(set(self.train_dataset.column_names) - set(['input_ids', 'attention_mask', 'label']))
            self.train_dataset = self.train_dataset.remove_columns(columns_to_remove)
            self.val_dataset = self.val_dataset.remove_columns(columns_to_remove)

            # Set the format for PyTorch tensors
            self.logger.info("Setting dataset format for PyTorch tensors...")
            self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            self.val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            self.logger.info("Tokenization completed successfully.")

        except Exception as e:
            self.logger.error(f"Error during tokenization: {e}")
            print(f"An error occurred during tokenization: {e}")
            raise

    def compute_class_weights(self):
        """
        Computes class weights to handle class imbalance.
        """
        try:
            self.logger.info("Computing class weights for handling class imbalance...")
            label_counts = self.train_df['label'].value_counts().sort_index()
            class_weights = 1.0 / label_counts
            class_weights = class_weights.to_numpy()
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            self.class_weights = class_weights
            self.logger.info(f"Class weights: {self.class_weights}")
            print(f"Class weights: {self.class_weights}")

        except Exception as e:
            self.logger.error(f"Error during class weight computation: {e}")
            print(f"An error occurred during class weight computation: {e}")
            raise

    def initialize_model_and_trainer(self):
        """
        Initializes the model, defines metrics, sets up training arguments, and initializes the Trainer.
        """
        try:
            self.logger.info("Initializing model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2
            )
            self.model.to(self.device)

            # Define metrics using the evaluate library
            self.logger.info("Loading evaluation metrics...")
            self.metric_accuracy = evaluate.load("accuracy")
            self.metric_precision = evaluate.load("precision")
            self.metric_recall = evaluate.load("recall")
            self.metric_f1 = evaluate.load("f1")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                accuracy = self.metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
                precision = self.metric_precision.compute(predictions=predictions, references=labels, average='binary')['precision']
                recall = self.metric_recall.compute(predictions=predictions, references=labels, average='binary')['recall']
                f1 = self.metric_f1.compute(predictions=predictions, references=labels, average='binary')['f1']
                return {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            # Define training arguments
            self.logger.info("Setting training arguments...")
            training_args = TrainingArguments(
                output_dir=os.path.join(self.output_model_dir, 'results'),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.num_epochs,
                weight_decay=self.weight_decay,
                logging_dir=os.path.join(self.output_model_dir, 'logs'),
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                greater_is_better=True,
                logging_steps=50,
                save_total_limit=2,
                report_to="none",  # To disable logging to WandB or other services
            )

            # Define data collator
            self.logger.info("Initializing data collator...")
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            # Define custom Trainer to handle class weights
            class CustomTrainer(Trainer):
                def __init__(self, class_weights, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.class_weights = class_weights

                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.get("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                    return (loss, outputs) if return_outputs else loss

            # Initialize the Trainer
            self.logger.info("Initializing Trainer...")
            self.trainer = CustomTrainer(
                class_weights=self.class_weights,
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )

            self.logger.info("Model and Trainer initialized successfully.")

        except Exception as e:
            self.logger.error(f"Error during model and Trainer initialization: {e}")
            print(f"An error occurred during model and Trainer initialization: {e}")
            raise

    def train(self):
        """
        Trains the model using the Trainer.
        """
        try:
            self.logger.info("Starting training...")
            print("Starting training...")
            self.trainer.train()
            self.logger.info("Training completed successfully.")
            print("Training completed successfully.")

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            print(f"An error occurred during training: {e}")
            raise

    def evaluate(self):
        """
        Evaluates the model on the validation set and prints metrics.
        """
        try:
            self.logger.info("Evaluating the model...")
            print("Evaluating the model...")
            metrics = self.trainer.evaluate()

            # Print evaluation metrics
            self.logger.info("Evaluation Metrics:")
            for key, value in metrics.items():
                if key.startswith("eval_"):
                    metric_name = key.replace("eval_", "").capitalize()
                    self.logger.info(f"{metric_name}: {value:.4f}")
                    print(f"{metric_name}: {value:.4f}")

            self.logger.info("Evaluation completed successfully.")
            print("Evaluation completed successfully.")

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            print(f"An error occurred during evaluation: {e}")
            raise

    def generate_classification_report_and_confusion_matrix(self):
        """
        Generates and prints the classification report and plots the confusion matrix.
        """
        try:
            self.logger.info("Generating predictions on validation set...")
            print("Generating predictions on validation set...")
            predictions_output = self.trainer.predict(self.val_dataset)
            preds = np.argmax(predictions_output.predictions, axis=-1)
            true_labels = predictions_output.label_ids

            # Classification report
            self.logger.info("Generating classification report...")
            report = classification_report(true_labels, preds, target_names=self.label_mapping.keys())
            self.logger.info(f"Classification Report:\n{report}")
            print("\nClassification Report:")
            print(report)

            # Confusion matrix
            self.logger.info("Plotting confusion matrix...")
            cm = confusion_matrix(true_labels, preds)
            cm_labels = list(self.label_mapping.keys())

            # Plot confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.show()
            self.logger.info("Confusion matrix plotted successfully.")

        except Exception as e:
            self.logger.error(f"Error during classification report and confusion matrix generation: {e}")
            print(f"An error occurred during classification report and confusion matrix generation: {e}")
            raise

    def save_model(self):
        """
        Saves the trained model and tokenizer to the specified directory.
        """
        try:
            self.logger.info("Saving the model and tokenizer...")
            print("Saving the model and tokenizer...")
            os.makedirs(self.output_model_dir, exist_ok=True)
            self.model.save_pretrained(os.path.join(self.output_model_dir, 'model'))
            self.tokenizer.save_pretrained(os.path.join(self.output_model_dir, 'tokenizer'))
            self.logger.info("Model and tokenizer saved successfully.")
            print("Model and tokenizer saved successfully.")

        except Exception as e:
            self.logger.error(f"Error during model saving: {e}")
            print(f"An error occurred during model saving: {e}")
            raise

    def load_trained_model(self):
        """
        Loads the trained model and tokenizer from the specified directory.
        """
        try:
            self.logger.info("Loading the trained model and tokenizer for prediction...")
            print("Loading the trained model and tokenizer for prediction...")
            self.model = AutoModelForSequenceClassification.from_pretrained(os.path.join(self.output_model_dir, 'model'))
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.output_model_dir, 'tokenizer'))
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("Model and tokenizer loaded successfully.")
            print("Model and tokenizer loaded successfully.")

        except Exception as e:
            self.logger.error(f"Error during model loading: {e}")
            print(f"An error occurred during model loading: {e}")
            raise

    def predict_prompt_sentence(self, prompt, sentence):
        """
        Predicts the label for a given prompt and sentence.

        Parameters:
        - prompt (str): The user prompt.
        - sentence (str): The sentence to classify.

        Returns:
        - str: The predicted label ('necessary' or 'unnecessary').
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer must be loaded before making predictions.")

            # Prepare the input
            input_text = prompt + " [SEP] " + sentence
            inputs = self.tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                predicted_label = self.inverse_label_mapping.get(prediction, "unknown")

            return predicted_label

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            print(f"An error occurred during prediction: {e}")
            raise

    def interactive_prediction(self):
        """
        Provides an interactive interface for users to input prompts and sentences for prediction.
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer must be loaded before making predictions.")

            self.logger.info("Starting interactive prediction interface.")
            print("Enter your prompts and sentences for classification. Type 'exit' to quit.")
            while True:
                user_prompt = input("Enter the prompt (or 'exit' to quit): ")
                if user_prompt.lower() == 'exit':
                    self.logger.info("Exiting interactive prediction interface.")
                    break
                user_sentence = input("Enter the sentence: ")
                if user_sentence.lower() == 'exit':
                    self.logger.info("Exiting interactive prediction interface.")
                    break
                try:
                    result = self.predict_prompt_sentence(user_prompt, user_sentence)
                    print(f"Predicted Label: {result}\n")
                except Exception as e:
                    print(f"An error occurred during prediction: {e}\n")
                    self.logger.error(f"Error during interactive prediction: {e}")

        except Exception as e:
            self.logger.error(f"Error in interactive_prediction method: {e}")
            print(f"An error occurred in interactive prediction: {e}")
            raise



class Classifier:
    """Binary Sequence Classifier using Hugging Face Transformers."""

    def __init__(
        self,
        data_path,
        output_model_dir,
        model_choice="1",
        max_length=128,
        batch_size=32,
        learning_rate=2e-5,
        num_epochs=3,
        weight_decay=0.01,
        seed=42,
    ):
        """
        Initializes the Classifier with the given parameters.

        Parameters:
        - data_path (str): Path to the input CSV file containing 'prompt', 'sentence', and 'label'.
        - output_model_dir (str): Directory where the trained model and tokenizer will be saved.
        - model_choice (str): Choice of pre-trained model. Options: "1", "2", "3".
        - max_length (int): Maximum token length for inputs.
        - batch_size (int): Batch size for training and evaluation.
        - learning_rate (float): Learning rate for the optimizer.
        - num_epochs (int): Number of training epochs.
        - weight_decay (float): Weight decay for the optimizer.
        - seed (int): Random seed for reproducibility.
        """
        self.data_path = data_path
        self.output_model_dir = output_model_dir
        self.model_choice = model_choice
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.seed = seed

        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize label encoder
        self.label_encoder = LabelEncoder()

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_args = None

        # Placeholders for datasets
        self.train_df = None
        self.eval_df = None
        self.train_dataset = None
        self.eval_dataset = None

    def load_data(self):
        """Loads the dataset from a CSV file."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded with {len(self.df)} records.")

    def preprocess_data(self):
        """Preprocesses the data by combining 'prompt' and 'sentence', dropping missing values, and encoding labels."""
        # Combine the prompt and sentence for input
        print("Combining 'prompt' and 'sentence' into 'input_text'...")
        self.df['input_text'] = self.df['prompt'] + " [SEP] " + self.df['sentence']

        # Drop missing values
        print("Dropping rows with missing 'input_text' or 'label'...")
        initial_length = len(self.df)
        self.df.dropna(subset=["input_text", "label"], inplace=True)
        print(
            f"Dropped {initial_length - len(self.df)} rows. Remaining records: {len(self.df)}."
        )

        # Encode labels
        print("Encoding labels...")
        self.df["label"] = self.label_encoder.fit_transform(self.df["label"])
        print(
            "Label mapping:",
            dict(
                zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_),
                )
            ),
        )

    def balance_data(self):
        """Balances the dataset by oversampling the minority class."""
        print("Balancing the dataset...")
        from sklearn.utils import resample

        # Separate majority and minority classes
        majority_class = self.df[
            self.df["label"] == self.df["label"].value_counts().idxmax()
        ]
        minority_class = self.df[
            self.df["label"] != self.df["label"].value_counts().idxmax()
        ]

        # Oversample minority class
        minority_oversampled = resample(
            minority_class,
            replace=True,  # Sample with replacement
            n_samples=len(majority_class),  # To match majority class
            random_state=42,  # For reproducibility
        )

        # Combine majority and oversampled minority class
        self.df = pd.concat([majority_class, minority_oversampled])
        print("Data balanced. Class distribution:")
        print(self.df["label"].value_counts())

    def split_data(self, test_size=0.2, random_state=42):
        """Splits the data into training and evaluation sets."""
        print("Splitting data into training and evaluation sets...")
        train_df, eval_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df["label"],
        )
        print(f"Training set: {len(train_df)} records")
        print(f"Evaluation set: {len(eval_df)} records")
        self.train_df = train_df
        self.eval_df = eval_df
        return train_df, eval_df

    def choose_model(self):
        """Chooses a pre-trained model based on user choice."""
        print("Please choose a model:")
        print("1: distilbert-base-uncased")
        print("2: huawei-noah/TinyBERT_General_4L_312D")
        print("3: Qwen/Qwen2.5-0.5B")  # Ensure you have access to this model
        # model_choice is passed in __init__
        if self.model_choice == "1":
            model_name = "distilbert-base-uncased"
        elif self.model_choice == "2":
            model_name = "huawei-noah/TinyBERT_General_4L_312D"
        elif self.model_choice == "3":
            model_name = "Qwen/Qwen2.5-0.5B"  # Ensure you have access
        else:
            print("Invalid choice. Using default model: distilbert-base-uncased")
            model_name = "distilbert-base-uncased"
        return model_name

    def load_model_and_tokenizer(self):
        """Loads the pre-trained tokenizer and model based on user choice."""
        model_name = self.choose_model()
        self.model_name = model_name
        print(f"Loading tokenizer and model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,  # To handle possible size mismatches
        )
        print("Tokenizer and model loaded.")

        # Set pad_token_id if not already set
        if self.tokenizer.pad_token_id is None:
            print("Setting pad_token_id to eos_token_id...")
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                # Set pad_token_id to a default value (e.g., 0)
                self.tokenizer.pad_token_id = 0
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        print(f"pad_token_id set to {self.tokenizer.pad_token_id}")

    def tokenize_dataset(self, dataset, max_length=128):
        """Tokenizes the dataset."""
        print("Tokenizing dataset...")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["input_text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing data",
        )
        print("Dataset tokenized.")
        return dataset

    def prepare_dataset(self, train_dataset, eval_dataset):
        """Prepares the dataset for training."""
        # Rename 'label' to 'labels' as expected by the Trainer
        print("Renaming 'label' column to 'labels'...")
        train_dataset = train_dataset.rename_column("label", "labels")
        eval_dataset = eval_dataset.rename_column("label", "labels")
        print("Columns renamed.")

        # Set the format of the datasets to torch tensors
        print("Setting dataset format to PyTorch tensors...")
        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        eval_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        print("Dataset format set.")
        return train_dataset, eval_dataset

    def compute_metrics(self, eval_pred):
        """Computes evaluation metrics."""
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def set_training_args(
        self, output_dir="./results", num_train_epochs=3, per_device_train_batch_size=32
    ):
        """Sets the training arguments."""
        print("Setting up training arguments...")
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=self.weight_decay,
            logging_dir="./logs",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU is available
            report_to="none",  # Disable logging to external services
        )
        print("Training arguments set.")

    def initialize_trainer(self, train_dataset, eval_dataset):
        """Initializes the Trainer."""
        print("Initializing Trainer...")

        class CustomTrainer(Trainer):
            def __init__(self, *args, class_weights=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        # Handle class imbalance by computing class weights
        print("Computing class weights for handling class imbalance...")
        label_counts = self.train_df['label'].value_counts().sort_index()
        class_weights = 1.0 / label_counts
        class_weights = torch.tensor(class_weights.values, dtype=torch.float).to(self.device)
        print(f"Class weights: {class_weights}")

        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights=class_weights,
        )
        print("Trainer initialized.")

    def train_model(self):
        """Trains the model."""
        print("Starting training...")
        self.trainer.train()
        print("Training completed.")

    def evaluate_model(self):
        """Evaluates the model."""
        print("Evaluating the model...")
        evaluation_results = self.trainer.evaluate()
        print("Evaluation results:", evaluation_results)

    def generate_classification_report_and_confusion_matrix(self, eval_dataset):
        """Generates classification report and plots confusion matrix."""
        print("Generating classification report and confusion matrix...")
        predictions, labels, _ = self.trainer.predict(eval_dataset)
        preds = torch.argmax(torch.tensor(predictions), dim=-1)

        # Classification report
        report = classification_report(
            labels, preds, target_names=self.get_label_names()
        )
        print("Classification Report:\n", report)

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.get_label_names(),
            yticklabels=self.get_label_names(),
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    def get_label_names(self):
        """Returns the list of label names."""
        return list(self.label_encoder.classes_)

    def save_model_and_tokenizer(self, save_directory=None):
        """Saves the trained model and tokenizer."""
        if save_directory is None:
            save_directory = self.output_model_dir
        print("Saving the trained model and tokenizer...")
        self.trainer.save_model(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print("Model and tokenizer saved.")

    def load_model_and_tokenizer_for_prediction(self, save_directory):
        """Loads the trained model and tokenizer from the specified directory."""
        print("Loading the trained model and tokenizer for prediction...")
        self.tokenizer = AutoTokenizer.from_pretrained(save_directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(save_directory)
        self.model.to(self.device)
        self.model.eval()
        print("Model and tokenizer loaded.")

    def predict_prompt_sentence(self, prompt, sentence):
        """
        Predicts the label for a given prompt and sentence.

        Parameters:
        - prompt (str): The user prompt.
        - sentence (str): The sentence to classify.

        Returns:
        - str: The predicted label ('necessary' or 'unnecessary').
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before making predictions.")

        # Prepare the input
        input_text = prompt + " [SEP] " + sentence
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        return predicted_label

    def interactive_prediction(self):
        """
        Provides an interactive interface for users to input prompts and sentences for prediction.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before making predictions.")

        print("Enter your prompts and sentences for classification. Type 'exit' to quit.")
        while True:
            user_prompt = input("Enter the prompt (or 'exit' to quit): ")
            if user_prompt.lower() == 'exit':
                break
            user_sentence = input("Enter the sentence: ")
            if user_sentence.lower() == 'exit':
                break
            try:
                result = self.predict_prompt_sentence(user_prompt, user_sentence)
                print(f"Predicted Label: {result}\n")
            except Exception as e:
                print(f"An error occurred during prediction: {e}\n")

    def run_pipeline(self):
        """
        Runs the entire pipeline: load data, preprocess, balance, split, tokenize, prepare, train, evaluate, report, save, and start prediction.
        """
        self.load_data()
        self.preprocess_data()
        self.balance_data()
        train_df, eval_df = self.split_data()

        # Convert to Hugging Face dataset format
        print("Converting to Hugging Face datasets...")
        train_dataset = HFDataset.from_pandas(train_df)
        eval_dataset = HFDataset.from_pandas(eval_df)
        print("Datasets converted.")

        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Tokenize datasets
        train_dataset = self.tokenize_dataset(train_dataset, max_length=self.max_length)
        eval_dataset = self.tokenize_dataset(eval_dataset, max_length=self.max_length)

        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_dataset(train_dataset, eval_dataset)

        # Set training arguments
        self.set_training_args(per_device_train_batch_size=self.batch_size)

        # Initialize Trainer
        self.initialize_trainer(train_dataset, eval_dataset)

        # Train the model
        self.train_model()

        # Evaluate the model
        self.evaluate_model()

        # Generate classification report and confusion matrix
        self.generate_classification_report_and_confusion_matrix(eval_dataset)

        # Save the model and tokenizer
        self.save_model_and_tokenizer()

        # Load the trained model and tokenizer for prediction
        self.load_model_and_tokenizer_for_prediction(self.output_model_dir)

        # Start interactive prediction
        self.interactive_prediction()



class MultiModelPipeline:
    def __init__(self, model_configs: List[ModelConfig]):
        """
        Initialize the MultiModelPipeline with a list of ModelConfig instances.

        Args:
            model_configs (List[ModelConfig]): List of model configurations to load.
        """
        self.model_configs = model_configs
        self.pipelines: Dict[str, Dict[str, Any]] = {}  # {model_name: {'pipeline': pipeline_obj, 'label_mapping': dict}}
    
    def display_available_models(self):
        """
        Display the list of available models.
        """
        print("\nAvailable Models:")
        for idx, config in enumerate(self.model_configs, start=1):
            print(f"  {idx}. {config.name} (Task: {config.task})")
        print()
    
    def select_models(self) -> List[ModelConfig]:
        """
        Allow the user to select which models to load.

        Returns:
            List[ModelConfig]: List of selected model configurations.
        """
        while True:
            self.display_available_models()
            user_input = input("Enter the numbers of the models you want to load (e.g., 1,3) or 'all' to load all models: ").strip()

            if user_input.lower() == 'exit':
                print("Exiting the script.")
                sys.exit()

            if user_input.lower() == 'all':
                return self.model_configs.copy()

            if not user_input:
                print("No input detected. Please enter valid numbers corresponding to the models.\n")
                continue

            try:
                selections = [int(num.strip()) for num in user_input.split(',')]
                if not selections:
                    raise ValueError

                selected_models = []
                for num in selections:
                    if num < 1 or num > len(self.model_configs):
                        raise ValueError(f"Number {num} is out of range.")
                    selected_models.append(self.model_configs[num - 1])
                
                return selected_models

            except ValueError as ve:
                print(f"Invalid input: {ve}\nPlease enter valid numbers separated by commas (e.g., 1,2) or 'all'.\n")
    
    def load_pipelines(self, selected_models: List[ModelConfig]):
        """
        Load Hugging Face pipelines for each selected model.

        Args:
            selected_models (List[ModelConfig]): List of selected model configurations.
        """
        for config in selected_models:
            try:
                print(f"Loading {config.name} for task '{config.task}' from '{config.path}'...")
                device = 0 if torch.cuda.is_available() else -1

                # Initialize the pipeline with additional configurations if any
                clf = pipeline(
                    task=config.task,
                    model=config.path,
                    tokenizer=config.path,
                    device=device,
                    **config.additional_config  # Unpack additional configurations
                )

                # Store the pipeline and its label mapping
                self.pipelines[config.name] = {
                    'pipeline': clf,
                    'label_mapping': config.label_mapping
                }

                print(f"{config.name} loaded successfully.\n")

            except Exception as e:
                print(f"Error loading {config.name} from '{config.path}': {e}\n")
    
    def get_user_input(self) -> Optional[str]:
        """
        Prompt the user to input data for prediction.

        Returns:
            Optional[str]: The user's input or None if exiting.
        """
        try:
            return input("Enter input data (or type 'exit' to quit): ")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit()
    
    def predict(self, input_data: str) -> Dict[str, Any]:
        """
        Perform predictions using all loaded pipelines.

        Args:
            input_data (str): The input data for prediction.

        Returns:
            Dict[str, Any]: Predictions from each model.
        """
        results = {}
        for model_name, components in self.pipelines.items():
            clf = components['pipeline']
            label_mapping = components.get('label_mapping', {})
            try:
                output = clf(input_data)[0]
                task = clf.task

                if task == 'text-classification':
                    label = output['label']
                    # Handle label mapping if provided
                    if label.isdigit():
                        label = label_mapping.get(int(label), label)
                    elif '_' in label:
                        # Example: 'LABEL_0', 'LABEL_1', etc.
                        label_num = int(label.split('_')[-1])
                        label = label_mapping.get(label_num, label)
                    score = output['score']
                    results[model_name] = {'label': label, 'score': score}
                
                elif task == 'text-generation':
                    generated_text = output['generated_text']
                    score = output.get('score', None)  # Not all generation tasks provide a score
                    results[model_name] = {'generated_text': generated_text, 'score': score}
                
                else:
                    # Handle other tasks as needed
                    results[model_name] = {'output': output}
            
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def run(self):
        """
        Run the multi-model pipeline.
        """
        print("Welcome to the Multi-Model Pipeline!\n")
        print("You can load and interact with multiple models for various NLP tasks.\n")
        print("To exit at any prompt, type 'exit'.\n")

        # Step 1: Model Selection
        selected_models = self.select_models()

        if not selected_models:
            print("No models selected. Exiting.")
            return

        # Step 2: Load Pipelines
        self.load_pipelines(selected_models)

        if not self.pipelines:
            print("No models loaded successfully. Exiting.")
            return

        print("All selected models loaded successfully. You can start entering data for processing.\n")

        # Step 3: Processing Loop
        while True:
            user_input = self.get_user_input()
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            if not user_input.strip():
                print("Empty input. Please enter valid data.\n")
                continue

            predictions = self.predict(user_input)

            print("\nResults:")
            for model_name, result in predictions.items():
                if 'error' in result:
                    print(f"  {model_name}: Error - {result['error']}")
                else:
                    if 'label' in result:
                        label = result['label']
                        score = result['score']
                        print(f"  {model_name}: {label} (Confidence: {score:.4f})")
                    elif 'generated_text' in result:
                        generated_text = result['generated_text']
                        score = result.get('score', 'N/A')
                        print(f"  {model_name}: Generated Text - {generated_text} (Score: {score})")
                    else:
                        print(f"  {model_name}: {result}")
            print("\n" + "-"*50 + "\n")


# multiclass_classifier.py



class MulticlassClassifier:
    """
    A comprehensive multiclass text classifier using DistilBERT.
    This class handles data preprocessing, model training, evaluation, prediction,
    and saving the trained model.
    """
    
    class TextDataset(Dataset):
        """
        Custom Dataset class for handling text data and labels.
        """
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts.reset_index(drop=True)
            self.labels = labels.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    def __init__(self, 
                 model_name='distilbert-base-uncased', 
                 max_len=128, 
                 batch_size=16, 
                 epochs=3, 
                 lr=2e-5, 
                 weight_decay=0.01, 
                 save_dir='saved_models', 
                 log_file='training.log'):
        """
        Initializes the MulticlassClassifier with specified parameters.
        
        Args:
            model_name (str): Pretrained model name from Hugging Face.
            max_len (int): Maximum token length for inputs.
            batch_size (int): Batch size for training and evaluation.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            save_dir (str): Directory to save the trained model.
            log_file (str): File to save training logs.
        """
        # Configure logging
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

        # Check for GPU availability
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using device: {self.device}")
        self.logger.info(f"Using device: {self.device}")

        # Initialize parameters
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_dir = save_dir

        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)

    def train_and_evaluate(self, df):
        """
        Trains the model and evaluates its performance on validation data.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'Text' and 'Label' columns.
        """
        # Data Preprocessing
        self.logger.info("Starting data preprocessing")
        label_encoder = LabelEncoder()
        df['Label_encoded'] = label_encoder.fit_transform(df['Label'])
        self.class_names = label_encoder.classes_
        X_train, X_val, y_train, y_val = train_test_split(
            df['Text'], df['Label_encoded'],
            test_size=0.2,
            random_state=42,
            stratify=df['Label_encoded']
        )

        # Create datasets and dataloaders
        train_dataset = self.TextDataset(X_train, y_train, self.tokenizer, self.max_len)
        val_dataset = self.TextDataset(X_val, y_val, self.tokenizer, self.max_len)

        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Model Initialization
        self.logger.info("Initializing the model")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.class_names)
        )
        self.model.to(self.device)

        # Optimizer and Scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = len(train_data_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        # Training Loop
        self.logger.info("Starting training")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            train_loss, train_acc = self._train_epoch(train_data_loader, optimizer, scheduler, loss_fn)
            val_loss, val_acc = self._eval_epoch(val_data_loader, loss_fn)

            self.logger.info(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        # Predictions
        self.logger.info("Generating predictions on validation set")
        predictions, real_labels = self.predict(val_data_loader)

        # Evaluation Metrics
        report = classification_report(real_labels, predictions, target_names=self.class_names)
        self.logger.info(f"Classification Report:\n{report}")
        print(f"Classification Report:\n{report}")

        # Plot Confusion Matrix
        self.plot_confusion_matrix(real_labels, predictions)

        # Save the trained model
        self.save_model()

    def _train_epoch(self, data_loader, optimizer, scheduler, loss_fn):
        """
        Trains the model for one epoch.
        
        Args:
            data_loader (DataLoader): DataLoader for training data.
            optimizer (Optimizer): Optimizer for updating model weights.
            scheduler (Scheduler): Learning rate scheduler.
            loss_fn (Loss): Loss function.
        
        Returns:
            Tuple[float, float]: Average loss and accuracy for the epoch.
        """
        self.model.train()
        losses, correct_predictions = 0, 0
        for batch in tqdm(data_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            _, preds = torch.max(outputs.logits, dim=1)

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses += loss.item()
            correct_predictions += torch.sum(preds == labels)

        avg_loss = losses / len(data_loader)
        avg_acc = correct_predictions.double() / len(data_loader.dataset)
        return avg_loss, avg_acc.item()

    def _eval_epoch(self, data_loader, loss_fn):
        """
        Evaluates the model on validation data.
        
        Args:
            data_loader (DataLoader): DataLoader for validation data.
            loss_fn (Loss): Loss function.
        
        Returns:
            Tuple[float, float]: Average loss and accuracy for the validation set.
        """
        self.model.eval()
        losses, correct_predictions = 0, 0
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                _, preds = torch.max(outputs.logits, dim=1)

                losses += loss.item()
                correct_predictions += torch.sum(preds == labels)

        avg_loss = losses / len(data_loader)
        avg_acc = correct_predictions.double() / len(data_loader.dataset)
        return avg_loss, avg_acc.item()

    def predict(self, data_loader):
        """
        Generates predictions for the given data.
        
        Args:
            data_loader (DataLoader): DataLoader for the data to predict.
        
        Returns:
            Tuple[List[int], List[int]]: Predicted labels and true labels.
        """
        self.model.eval()
        predictions, real_labels = [], []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                real_labels.extend(labels.cpu().numpy())
        return predictions, real_labels

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plots a confusion matrix using seaborn.
        
        Args:
            y_true (List[int]): True labels.
            y_pred (List[int]): Predicted labels.
        """
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

    def save_model(self):
        """
        Saves the trained model and tokenizer to the specified directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.save_dir, f"DistilBERT_{timestamp}")
        os.makedirs(model_path, exist_ok=True)
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        self.logger.info(f"Model saved to {model_path}")
        print(f"Model saved to {model_path}")

