"""
Load the fine-tuned model and classify CT slices.

Used by both the evaluation script and the Gradio demo, so that they run
the model in exactly the same way.

Research use only. Not for clinical use.
"""

from dataclasses import dataclass
from typing import List

# Unsloth must be imported before transformers and peft.
from unsloth import FastVisionModel

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image

from training.prompt import build_messages, parse_answer

BASE_MODEL = "unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit"
ADAPTER = "mku1988/medgemma-ct-fibrosis"

# The fine-tuned model answers in a few tokens. The base model tends to
# explain its reasoning, so it needs a longer limit.
MAX_NEW_TOKENS_FINE_TUNED = 16
MAX_NEW_TOKENS_BASE = 256


@dataclass
class SlicePrediction:
    answer: str        # FIBROSIS DETECTED, NO FIBROSIS DETECTED or UNPARSEABLE
    confidence: float  # mean top-token probability; NOT a calibrated probability
    raw_response: str


class FibrosisPredictor:
    """
    Args:
        adapter: Hugging Face repository or local folder with the LoRA
            adapter. None runs the base model without fine-tuning.
        base_model: The 4-bit base model the adapter was trained on.
    """

    def __init__(self, adapter: str = ADAPTER, base_model: str = BASE_MODEL):
        model, self.processor = FastVisionModel.from_pretrained(base_model, load_in_4bit=True)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        FastVisionModel.for_inference(model)
        self.model = model
        self.max_new_tokens = MAX_NEW_TOKENS_FINE_TUNED if adapter else MAX_NEW_TOKENS_BASE

    def predict(self, image: Image.Image) -> SlicePrediction:
        """Classify one lung-windowed CT slice."""
        image = image.convert("RGB")
        inputs = self._prepare_inputs(image)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        new_tokens = output.sequences[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(new_tokens, skip_special_tokens=True)
        return SlicePrediction(parse_answer(response), _mean_top_token_probability(output.scores), response)

    def _prepare_inputs(self, image: Image.Image) -> dict:
        """Apply the chat template and move tensors to the GPU (floats in bfloat16)."""
        prompt = self.processor.apply_chat_template(
            build_messages(image), tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return {
            key: value.to("cuda", dtype=torch.bfloat16) if value.is_floating_point() else value.to("cuda")
            for key, value in inputs.items()
        }


def _mean_top_token_probability(scores: List[torch.Tensor]) -> float:
    """
    Average, over generated tokens, of the probability of the chosen token.

    This shows how decisive the model was. It is not a calibrated probability
    of disease: the model has been wrong with values above 0.97.
    """
    if not scores:
        return 0.0
    top_probabilities = [F.softmax(step[0].float(), dim=-1).max().item() for step in scores]
    return sum(top_probabilities) / len(top_probabilities)
