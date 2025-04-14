import logging
import torch
from transformers import AutoTokenizer, Mamba2ForCausalLM
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_config: dict) -> tuple:
    """Load pre-trained model and tokenizer with given config.

    Args:
        model_config (dict): Configuration for model and tokenizer.
    """
    logger.info("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        from_slow=True,
        legacy=False,
        device_map=model_config["device_map"],
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = Mamba2ForCausalLM.from_pretrained(
        model_config["model_id"],
        device_map=model_config["device_map"],
        torch_dtype=getattr(torch, model_config["torch_dtype"])
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    # prints out the models layers
    # for name, param in model.named_parameters():
    #     logger.info(f"Layer: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    
    return model, tokenizer

def apply_lora(model, lora_config: dict):
    lora = LoraConfig(**lora_config)
    return get_peft_model(model, lora)