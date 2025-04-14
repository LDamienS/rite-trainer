import logging
import argparse
import shutil
import wandb
from datetime import datetime
from pathlib import Path
from cpp_to_rust.config_utils import load_config
from cpp_to_rust.model_utils import load_model_and_tokenizer, apply_lora
from cpp_to_rust.data_utils import prepare_dataset
from cpp_to_rust.train import fine_tune_model

def setup_logging(log_file: str, debug: bool = False) -> logging.Logger:
    """Configure logging to file and console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main(config_path: str, dataset_path: str = None, use_wandb: bool = False, debug: bool = False ) -> None:
    """Main function to execute the training pipeline.

    Args:
        config_path (str): Path to the configuration YAML file.
    """
    
    config = load_config(config_path)
    
    config_name = Path(config_path).stem
    
    
    
    # Override config file if dataset argument is passed
    if dataset_path:
        config["data"]["dataset_path"] = dataset_path
    
    
    
    
    output_dir = Path(f"training_results/{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    # Create directory for final results/logs for each training session
    config["training"]["output_dir"] = f"{output_dir}/model"
    config["training"]["logging_dir"] = f"{output_dir}/training.log"
    shutil.copy(config_path, f"{output_dir}/config.yaml")
    
    logger = setup_logging(config["training"]["logging_dir"], debug)
    logger.info(f"Model, config file, and logs will be saved to {output_dir}")

    
    # Initial W&B if --local flag isnt used
    if not use_wandb:
        config["training"]["report_to"] = "wandb"
        config["training"]["run_name"] = output_dir.name
        
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            config=config,
            name=output_dir.name,
            dir=output_dir
        )
            
        logger.info("Tracking metrics with W&B!!!")
    else:
        logger.info("Locally fine-tuning, not reporting to W&B!!!")
    
    
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config["model"])
    # model = apply_lora(model, config["lora"])
    
    # Prepare dataset
    dataset = prepare_dataset(config["data"], tokenizer)
    
    fine_tune_model(model, tokenizer, dataset, config["training"])
    
    if not use_wandb:
        wandb.finish()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for C++ to Rust translation.")
    
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to the configuration YAML file (defaults to configs/default_config.yaml).")
    
    parser.add_argument("--dataset", type=str,
                        help="Path to dataset JSON file to be used for training (defaults to training.dataset_path in selected config.")
    
    parser.add_argument("--local", action="store_true",
                        help="Disables Weights & Biases logging for local/test runs.") 
    
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enables debugging messages.")
    
    args = parser.parse_args()
    
    main(args.config, args.dataset, args.local, args.verbose)