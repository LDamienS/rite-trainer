import logging
import time
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import Dataset

logger = logging.getLogger(__name__)

def fine_tune_model(model, tokenizer: AutoTokenizer, dataset: Dataset, training_config: dict):
    """Train the model with the provided dataset and configuration.

    Args:
        model: The model to train.
        tokenizer (AutoTokenizer): Tokenizer for decoding/debugging.
        dataset (Dataset): Training and validation dataset.
        training_config (dict): Training arguments configuration.
    """
    logger.info("Starting fine-tuning with Trainer...")
    
    training_args = TrainingArguments(**training_config)
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        args=training_args
    )
    
    # Debug batch
    batch = trainer.get_train_dataloader().__iter__().__next__()
    logger.info(f"input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"labels shape: {batch['labels'].shape}")
    logger.debug(f"Raw batch input_ids (first 40): {batch['input_ids'][0][:40]}")
    logger.debug(f"Raw batch labels (first 40): {batch['labels'][0][:40]}")
    
    decoded_input = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
    decoded_label = tokenizer.decode([t for t in batch["labels"][0] if t != -100], skip_special_tokens=True)
    logger.debug(f"Decoded Input (C++): {decoded_input}")
    logger.debug(f"Decoded Label (Rust): {decoded_label}")
   
    start_time = time.time()
    trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Total training time: {time.time() - start_time:.2f} seconds.")
    
    output_dir = training_config["output_dir"]
    
    logger.info("Saving final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Final model and tokenizer saved to {output_dir}.")
    