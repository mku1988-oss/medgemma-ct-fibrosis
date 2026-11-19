"""
LoRA fine-tuning of MedGemma 1.5 4B on an 8 GB GPU.

Reference training script. It reproduces the setup of the released model
(4-bit base model, frozen vision encoder, LoRA on the language model, one
slice per example, cosine schedule) with a neutral prompt throughout.

Hardware used for the released model: NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM.
Peak use was about 7.4 GB and the run took about 34 minutes.

Usage:
    python -m training.train --manifest data/processed/manifest.json \
        --splits-dir data/splits --output-dir models/medgemma-ct-fibrosis

    python -m training.train ... --quick-test     # 20-step smoke test

Research use only. Not for clinical use.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

# Unsloth must be imported before transformers and trl.
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import pandas as pd
import torch
from transformers import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer

from .dataset import SliceDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingSettings:
    """Hyperparameters of the released model."""

    base_model: str = "unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit"

    # LoRA on all linear layers of the language model; vision encoder frozen.
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # One image per example and batch size 1 keep memory under 8 GB.
    # Gradient accumulation gives an effective batch size of 16.
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    fibrosis_oversampling: int = 2
    early_stopping_patience: int = 3
    seed: int = 42


class PeakMemoryCallback(TrainerCallback):
    """Log reserved GPU memory so the 8 GB budget can be checked during training."""

    def __init__(self):
        self.peak_gb = 0.0

    def on_log(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            self.peak_gb = max(self.peak_gb, reserved)
            logger.info("step %d: GPU reserved %.2f GB (peak %.2f GB)",
                        state.global_step, reserved, self.peak_gb)


def train(manifest: Path, splits_dir: Path, output_dir: Path,
          settings: TrainingSettings, quick_test: bool = False) -> None:
    """Fine-tune the model and save the LoRA adapter to output_dir."""
    train_ids = pd.read_csv(splits_dir / "train_split.csv")["patient_id"].astype(str)
    val_ids = pd.read_csv(splits_dir / "val_split.csv")["patient_id"].astype(str)

    train_data = SliceDataset(manifest, train_ids, settings.fibrosis_oversampling)
    val_data = SliceDataset(manifest, val_ids)

    model, processor = load_model(settings)

    callbacks = [PeakMemoryCallback()]
    if not quick_test:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=settings.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=trainer_config(settings, output_dir, quick_test),
        callbacks=callbacks,
    )

    trainer.train()

    if not quick_test:
        model.save_pretrained(str(output_dir))
        processor.save_pretrained(str(output_dir))
        logger.info("Saved LoRA adapter to %s", output_dir)


def load_model(settings: TrainingSettings):
    """Load the 4-bit base model and attach LoRA adapters to the language layers."""
    model, processor = FastVisionModel.from_pretrained(
        settings.base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,   # keep MedGemma's medical image encoder unchanged
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        random_state=settings.seed,
    )
    return model, processor


def trainer_config(settings: TrainingSettings, output_dir: Path, quick_test: bool) -> SFTConfig:
    """Training arguments. bfloat16 is required: float16 is unstable with Gemma 3 models."""
    return SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        num_train_epochs=settings.epochs,
        max_steps=20 if quick_test else -1,
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        logging_steps=1 if quick_test else 5,
        eval_strategy="no" if quick_test else "epoch",
        save_strategy="no" if quick_test else "epoch",
        save_total_limit=settings.epochs,
        load_best_model_at_end=not quick_test,
        metric_for_best_model="eval_loss",
        max_seq_length=settings.max_seq_length,
        gradient_checkpointing=True,
        seed=settings.seed,
        report_to="none",
        # The dataset already returns chat messages; the Unsloth collator handles them.
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma 1.5 4B for CT fibrosis detection.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/medgemma-ct-fibrosis"))
    parser.add_argument("--quick-test", action="store_true", help="Run 20 steps without saving")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(args.manifest, args.splits_dir, args.output_dir, TrainingSettings(), args.quick_test)


if __name__ == "__main__":
    main()
