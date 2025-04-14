import logging
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
        
    
def prepare_dataset(data_config: dict, tokenizer: AutoTokenizer) -> Dataset:
    """Prepare and tokenize dataset for training.

    Args:
        data_config (dict): Configuration with dataset path and max_length.
        tokenizer (AutoTokenizer): Tokenizer for encoding data.

    Returns:
        Dataset: Tokenized dataset split into train and test.

    Raises:
        ValueError: If dataset lacks required columns or has missing values.
    """
    
    
    json_file = data_config["dataset_path"]
    max_length = data_config["max_length"]
    
    logger.info(f"Loading dataset from {json_file}...")
    
    df = pd.read_json(json_file)
    
    if "cpp_code" not in df.columns or "rust_code" not in df.columns:
        raise ValueError("Dataset must have 'cpp_code' and 'rust_code' columns.")
    if df["cpp_code"].isnull().any() or df["rust_code"].isnull().any():
        raise ValueError("Missing values detected in dataset.")
    
    df = df.sample(frac=1, random_state=28).reset_index(drop=True)
    
    def tokenize_function(examples):
        prompts = [str(x) for x in examples["cpp_code"]]
        targets = [str(x) for x in examples["rust_code"]]

        # Tokenize both the input (prompts) and targets (rust_code) with truncation and padding
        input_encodings = tokenizer(prompts, truncation=True, padding="max_length", max_length=data_config["max_length"])
        target_encodings = tokenizer(targets, truncation=True, padding="max_length", max_length=data_config["max_length"])

        # Ensure that padding is correctly handled by masking labels
        labels = target_encodings["input_ids"]
        labels = [
            [(token if token != tokenizer.pad_token_id else -100) for token in seq]
            for seq in labels
        ]

        input_encodings["labels"] = labels
        return input_encodings


    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=df.columns.tolist(), desc="Tokenizing dataset")
    
    
    logger.info("Splitting dataset into training and validation...")
    dataset = dataset.train_test_split(test_size=0.2, shuffle=False)
    logger.info(f"Training samples: {len(dataset['train'])}, Validation samples: {len(dataset['test'])}")
    
    return dataset