"""
Knowledge Distillation: Train Student Model from Teacher

This script demonstrates a complete distillation workflow:
1. Load a pre-trained teacher model (BERT-base)
2. Create a smaller student model (DistilBERT or custom)
3. Distill knowledge using combined hard+soft target loss
4. Evaluate and compare student vs teacher performance

Usage:
    python train_student.py \
        --teacher bert-base-uncased \
        --student distilbert-base-uncased \
        --dataset sst2 \
        --temperature 4.0 \
        --alpha 0.3 \
        --epochs 3 \
        --batch_size 32
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


@dataclass
class DistillationTrainingConfig:
    """Configuration for distillation training."""

    teacher_model: str = "bert-base-uncased"
    student_model: str = "distilbert-base-uncased"
    dataset: str = "sst2"
    temperature: float = 4.0
    alpha: float = 0.3
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 32
    max_seq_length: int = 128
    warmup_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./distillation_output"
    seed: int = 42


class DistillationTrainer:
    """
    Knowledge Distillation Trainer for sequence classification.
    """

    def __init__(
        self,
        config: DistillationTrainingConfig,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Set random seed
        self._set_seed(config.seed)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)

        # Initialize models
        self.teacher = self._load_teacher()
        self.student = self._load_student()

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.student.parameters(), lr=config.learning_rate, weight_decay=0.01
        )

        total_steps = len(train_loader) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.best_score = 0.0
        self.history = []

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_teacher(self) -> nn.Module:
        """Load pre-trained teacher model."""
        print(f"Loading teacher model: {self.config.teacher_model}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.teacher_model, num_labels=2
        )
        return model.to(self.config.device)

    def _load_student(self) -> nn.Module:
        """Load student model (can be smaller architecture)."""
        print(f"Loading student model: {self.config.student_model}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.student_model, num_labels=2
        )
        return model.to(self.config.device)

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        temperature: float,
        alpha: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined distillation loss.

        Total Loss = α * Hard_Loss + (1-α) * Soft_Loss

        Args:
            student_logits: Student model output logits
            teacher_logits: Teacher model output logits
            hard_labels: Ground truth labels
            temperature: Temperature for soft target distillation
            alpha: Weight for hard target loss

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Hard loss: standard cross entropy
        hard_loss = F.cross_entropy(student_logits, hard_labels)

        # Soft loss: KL divergence with temperature scaling
        # P_teacher = softmax(logits_T / T)
        # P_student = softmax(logits_S / T)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

        soft_loss = F.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean"
        ) * (temperature**2)

        # Combined loss
        total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

        loss_components = {
            "total_loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
            "temperature": temperature,
        }

        return total_loss, loss_components

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        self.teacher.eval()

        total_loss = 0.0
        total_hard = 0.0
        total_soft = 0.0
        num_batches = 0

        progress_bar = None
        try:
            from tqdm import tqdm

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        except ImportError:
            progress_bar = self.train_loader

        for batch in progress_bar:
            # Move data to device
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            # Teacher forward (no gradients)
            with torch.no_grad():
                teacher_output = self.teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                teacher_logits = teacher_output.logits

            # Student forward
            student_output = self.student(
                input_ids=input_ids, attention_mask=attention_mask
            )
            student_logits = student_output.logits

            # Compute distillation loss
            loss, loss_components = self.compute_distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=self.config.temperature,
                alpha=self.config.alpha,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Accumulate
            total_loss += loss.item()
            total_hard += loss_components["hard_loss"].item()
            total_soft += loss_components["soft_loss"].item()
            num_batches += 1

            # Update progress bar
            if progress_bar is not None and hasattr(progress_bar, "set_postfix"):
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "hard": f"{loss_components['hard_loss'].item():.4f}",
                        "soft": f"{loss_components['soft_loss'].item():.4f}",
                    }
                )

        avg_metrics = {
            "train_loss": total_loss / num_batches,
            "train_hard": total_hard / num_batches,
            "train_soft": total_soft / num_batches,
        }

        return avg_metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate student model."""
        self.student.eval()
        self.teacher.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                # Student prediction
                student_output = self.student(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                loss = F.cross_entropy(student_output.logits, labels)
                total_loss += loss.item()

                predictions = student_output.logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return {
            "eval_loss": total_loss / len(self.eval_loader),
            "accuracy": correct / total,
        }

    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        print("\n" + "=" * 60)
        print("Starting Knowledge Distillation Training")
        print("=" * 60)
        print(f"Teacher: {self.config.teacher_model}")
        print(f"Student: {self.config.student_model}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Alpha: {self.config.alpha}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch Size: {self.config.batch_size}")
        print("=" * 60 + "\n")

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Evaluate
            if self.eval_loader is not None:
                eval_metrics = self.evaluate()
            else:
                eval_metrics = {
                    "accuracy": 0.0,
                    "eval_loss": train_metrics["train_loss"],
                }

            epoch_time = time.time() - start_time

            # Log
            print(f"\nEpoch {epoch + 1}/{self.config.epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"    - Hard Loss: {train_metrics['train_hard']:.4f}")
            print(f"    - Soft Loss: {train_metrics['train_soft']:.4f}")
            print(f"  Eval Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}")

            # Save best
            if eval_metrics["accuracy"] > self.best_score:
                self.best_score = eval_metrics["accuracy"]
                self.save_checkpoint("best_model")
                print(f"  -> New best model saved!")

            self.history.append(
                {"epoch": epoch + 1, "train": train_metrics, "eval": eval_metrics}
            )

        return self.history

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, f"{name}.pt")
        torch.save(
            {
                "student_state_dict": self.student.state_dict(),
                "config": self.config,
                "best_accuracy": self.best_score,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")


class DummyTextDataset(Dataset):
    """
    Dummy dataset for demonstration purposes.
    In production, replace with actual dataset loading.
    """

    def __init__(self, tokenizer, num_samples=1000, max_length=128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

        # Generate dummy text data
        self.texts = [
            "This is a sample sentence for sentiment classification . " * 3
            for _ in range(num_samples)
        ]
        self.labels = [i % 2 for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_data_loaders(config: DistillationTrainingConfig):
    """
    Create data loaders for training and evaluation.

    In production, replace with actual dataset loading:
        from datasets import load_dataset
        dataset = load_dataset(config.dataset)
    """
    print(f"\nPreparing {config.dataset} dataset...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)

    # For demonstration, we use a dummy dataset
    # In production, replace with:
    # dataset = load_dataset(config.dataset)
    # train_dataset = dataset['train'].map(preprocess_function, ...)

    train_size = 800
    eval_size = 200

    train_dataset = DummyTextDataset(
        tokenizer, num_samples=train_size, max_length=config.max_seq_length
    )
    eval_dataset = DummyTextDataset(
        tokenizer, num_samples=eval_size, max_length=config.max_seq_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )

    return train_loader, eval_loader


def compare_models(teacher, student, eval_loader, device):
    """
    Compare teacher and student model performance.
    """
    print("\n" + "=" * 60)
    print("Model Comparison: Teacher vs Student")
    print("=" * 60)

    teacher.eval()
    student.eval()

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    teacher_params = count_params(teacher)
    student_params = count_params(student)

    print(f"\nModel Sizes:")
    print(f"  Teacher: {teacher_params:,} parameters")
    print(f"  Student: {student_params:,} parameters")
    print(f"  Compression: {teacher_params / student_params:.2f}x")

    # Evaluate both
    def evaluate(model):
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = output.logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total

    teacher_acc = evaluate(teacher)
    student_acc = evaluate(student)

    print(f"\nAccuracy:")
    print(f"  Teacher: {teacher_acc:.4f}")
    print(f"  Student: {student_acc:.4f}")
    print(f"  Retention: {student_acc / teacher_acc * 100:.1f}%")

    return {
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": teacher_params / student_params,
        "teacher_accuracy": teacher_acc,
        "student_accuracy": student_acc,
        "knowledge_retention_pct": student_acc / teacher_acc * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument(
        "--teacher",
        type=str,
        default="bert-base-uncased",
        help="Teacher model name or path",
    )
    parser.add_argument(
        "--student",
        type=str,
        default="distilbert-base-uncased",
        help="Student model name or path",
    )
    parser.add_argument("--dataset", type=str, default="sst2", help="Dataset name")
    parser.add_argument(
        "--temperature", type=float, default=4.0, help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3, help="Weight for hard target loss"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distillation_output",
        help="Output directory",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=128, help="Maximum sequence length"
    )

    args = parser.parse_args()

    # Create config
    config = DistillationTrainingConfig(
        teacher_model=args.teacher,
        student_model=args.student,
        dataset=args.dataset,
        temperature=args.temperature,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
    )

    # Get data loaders
    train_loader, eval_loader = get_data_loaders(config)

    # Create trainer
    trainer = DistillationTrainer(config, train_loader, eval_loader)

    # Train
    history = trainer.train()

    # Compare models
    results = compare_models(
        trainer.teacher, trainer.student, eval_loader, config.device
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Best Student Accuracy: {trainer.best_score:.4f}")
    print(f"  Teacher Accuracy: {results['teacher_accuracy']:.4f}")
    print(f"  Compression Ratio: {results['compression_ratio']:.2f}x")
    print(f"  Knowledge Retention: {results['knowledge_retention_pct']:.1f}%")
    print(f"\nOutput saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
