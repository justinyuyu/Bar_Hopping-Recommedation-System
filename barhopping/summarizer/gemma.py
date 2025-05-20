import torch
import re
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from barhopping.config import GEMMA_MODEL, HF_TOKEN
from barhopping.logger import logger

# Global instances
_tokenizer = None
_model = None

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        logger.info("Using MPS device")
        return torch.device("mps")
    logger.info("Using CPU device")
    return torch.device("cpu")

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoProcessor.from_pretrained(
            GEMMA_MODEL,
            use_auth_token=HF_TOKEN
        )
    return _tokenizer

def get_model(device: torch.device):
    global _model
    if _model is None:
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            GEMMA_MODEL,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float32
        ).eval().to(device)
    return _model

def build_prompt(reviews: list[str], photos: list[str]) -> str:
    reviews_text = "\n".join(reviews)
    photos_text = "\n".join(photos) if photos else "No photos available"
    return (
        "Based on the following reviews and photos, provide a concise summary of this bar:\n\n"
        f"Reviews:\n{reviews_text}\n\n"
        f"Photos:\n{photos_text}\n\nSummary:"
    )

def summarize_bar(reviews: list[str], photos: list[str]) -> str:
    """Generate a bar summary based on reviews and photos."""
    try:
        device = get_device()
        tokenizer = get_tokenizer()
        model = get_model(device)

        prompt = build_prompt(reviews, photos)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Summary:")[-1]
        return re.sub(r"\s+", " ", summary).strip()
        
    except Exception as e:
        logger.error(f"Error in summarize_bar: {e}")
        if "out of memory" in str(e).lower():
            logger.warning("Switching to CPU due to memory issues")
            torch.cuda.empty_cache()
            return summarize_bar(reviews, photos)
        return "Unable to generate summary at this time."