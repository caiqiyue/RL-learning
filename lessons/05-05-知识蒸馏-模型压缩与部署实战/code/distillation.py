"""
Knowledge Distillation Training Module

This module implements a complete knowledge distillation pipeline with:
- Temperature scaling for soft target generation
- KL divergence loss for distillation
- Combined hard-target + soft-target loss
- Response-based and feature-based distillation support

Usage:
    python distillation.py --teacher bert-base --student distilbert --temperature 4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, Dict, List
import math


class DistillationConfig:
    """Configuration for knowledge distillation training."""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        beta: float = 0.0,
        distill_layers: Optional[List[int]] = None,
        hard_loss_weight: float = 0.3,
        soft_loss_weight: float = 0.7,
        use_adaptive_temperature: bool = False,
    ):
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.distill_layers = distill_layers
        self.hard_loss_weight = hard_loss_weight
        self.soft_loss_weight = soft_loss_weight
        self.use_adaptive_temperature = use_adaptive_temperature


class TemperatureScaledSoftmax(nn.Module):
    """Temperature-scaled softmax for soft target generation."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=-1)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    Compute KL divergence distillation loss between teacher and student.

    The loss is computed as:
        KL(P_teacher || P_student) = Σ P_teacher * log(P_teacher / P_student)

    Args:
        student_logits: Student model output logits [batch, num_classes]
        teacher_logits: Teacher model output logits [batch, num_classes]
        temperature: Temperature scaling factor T
        reduction: Loss reduction method ('batchmean', 'sum', 'mean', 'none')

    Returns:
        KL divergence loss (scaled by T²)
    """
    # Get probability distributions with temperature scaling
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute KL divergence: KL(P || Q) = Σ P * log(P/Q)
    # Note: F.kl_div expects log-probabilities as input
    kl_div = F.kl_div(
        student_probs.log(),  # log(Q) - student distribution in log space
        teacher_probs,  # P - teacher distribution
        reduction=reduction,
    )

    # Scale by T² to compensate for the gradient scaling effect
    # During backprop, gradients are scaled by 1/T², so we multiply by T²
    scaled_loss = kl_div * (temperature**2)

    return scaled_loss


def combined_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined distillation loss with hard target + soft target.

    Total Loss = α * Hard_Loss + (1-α) * Soft_Loss

    Args:
        student_logits: Student model output logits [batch, num_classes]
        teacher_logits: Teacher model output logits [batch, num_classes]
        hard_labels: Ground truth labels [batch]
        temperature: Temperature for soft target distillation
        alpha: Weight for hard target loss (0.3 means 30% hard, 70% soft)

    Returns:
        Tuple of (total_loss, loss_dict with individual components)
    """
    # Hard loss: standard cross entropy with true labels
    hard_loss = F.cross_entropy(student_logits, hard_labels, reduction="mean")

    # Soft loss: KL divergence between teacher and student soft outputs
    soft_loss = distillation_loss(
        student_logits, teacher_logits, temperature, reduction="batchmean"
    )

    # Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

    loss_dict = {
        "total_loss": total_loss,
        "hard_loss": hard_loss,
        "soft_loss": soft_loss,
        "temperature": torch.tensor(temperature),
    }

    return total_loss, loss_dict


def feature_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projector: Optional[nn.Module] = None,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute feature-based distillation loss for intermediate layers.

    Args:
        student_hidden: Student model hidden states [batch, seq_len, hidden_dim_S]
        teacher_hidden: Teacher model hidden states [batch, seq_len, hidden_dim_T]
        projector: Optional module to project student dimensions to teacher dimensions
        loss_type: Type of loss ('mse', 'cosine', 'kl')

    Returns:
        Feature distillation loss
    """
    if projector is not None:
        student_hidden = projector(student_hidden)

    if loss_type == "mse":
        loss = F.mse_loss(student_hidden, teacher_hidden)
    elif loss_type == "cosine":
        loss = (
            1
            - F.cosine_similarity(
                student_hidden.flatten(1), teacher_hidden.flatten(1)
            ).mean()
        )
    elif loss_type == "kl":
        # Treat as probability distributions
        student_dist = F.softmax(student_hidden, dim=-1)
        teacher_dist = F.softmax(teacher_hidden, dim=-1)
        loss = F.kl_div(student_dist.log(), teacher_dist, reduction="batchmean")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss


def attention_distillation_loss(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    temperature: float = 2.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute attention-based distillation loss.

    Args:
        student_attention: Student attention scores [batch, heads, seq_len, seq_len]
        teacher_attention: Teacher attention scores [batch, heads, seq_len, seq_len]
        temperature: Temperature for softening attention distributions
        attention_mask: Optional mask for padding positions [batch, seq_len]

    Returns:
        Attention distillation loss
    """
    # Normalize attention scores to probability distributions
    student_attn = F.softmax(student_attention / temperature, dim=-1)
    teacher_attn = F.softmax(teacher_attention / temperature, dim=-1)

    # Apply mask if provided (set masked positions to 0 in loss)
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        mask = mask.expand_as(student_attn)
        student_attn = student_attn * mask
        teacher_attn = teacher_attn * mask
        # Normalize after masking
        student_attn = student_attn / (student_attn.sum(dim=-1, keepdim=True) + 1e-8)
        teacher_attn = teacher_attn / (teacher_attn.sum(dim=-1, keepdim=True) + 1e-8)

    # Compute KL divergence for each head and average
    kl_loss = F.kl_div(student_attn.log(), teacher_attn, reduction="batchmean") * (
        temperature**2
    )

    return kl_loss


class MultiLayerDistillationLoss(nn.Module):
    """
    Multi-layer distillation loss that combines:
    - Response-based distillation (final logits)
    - Feature-based distillation (hidden states)
    - Attention-based distillation (attention weights)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        distill_layers: List[int] = None,
        layer_loss_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.distill_layers = distill_layers or []
        self.layer_loss_weights = layer_loss_weights or []

        # Projectors for dimension mismatch
        self.projectors = nn.ModuleDict()

    def add_projection(self, layer_idx: int, student_dim: int, teacher_dim: int):
        """Add a projection layer for dimension matching."""
        if student_dim != teacher_dim:
            self.projectors[str(layer_idx)] = nn.Linear(student_dim, teacher_dim)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        student_hidden_states: Optional[List[torch.Tensor]] = None,
        teacher_hidden_states: Optional[List[torch.Tensor]] = None,
        student_attentions: Optional[List[torch.Tensor]] = None,
        teacher_attentions: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined multi-layer distillation loss.
        """
        total_loss = 0.0
        loss_components = {}

        # 1. Response-based distillation (final layer)
        soft_loss = distillation_loss(
            student_logits, teacher_logits, self.temperature, reduction="batchmean"
        )
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        response_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        total_loss += response_loss
        loss_components["response_loss"] = response_loss

        # 2. Feature-based distillation (intermediate layers)
        if student_hidden_states and teacher_hidden_states:
            feature_loss = 0.0
            for layer_idx in self.distill_layers:
                s_hidden = student_hidden_states[layer_idx]
                t_hidden = teacher_hidden_states[layer_idx]

                # Project if dimensions don't match
                if str(layer_idx) in self.projectors:
                    s_hidden = self.projectors[str(layer_idx)](s_hidden)

                # Compute MSE loss
                layer_loss = F.mse_loss(s_hidden, t_hidden)

                # Apply layer weight if provided
                if layer_idx < len(self.layer_loss_weights):
                    layer_loss = layer_loss * self.layer_loss_weights[layer_idx]

                feature_loss += layer_loss

            total_loss += feature_loss
            loss_components["feature_loss"] = feature_loss

        # 3. Attention-based distillation
        if student_attentions and teacher_attentions:
            attention_loss = 0.0
            for layer_idx in self.distill_layers:
                s_attn = student_attentions[layer_idx]
                t_attn = teacher_attentions[layer_idx]
                layer_loss = attention_distillation_loss(
                    s_attn, t_attn, self.temperature
                )

                if layer_idx < len(self.layer_loss_weights):
                    layer_loss = layer_loss * self.layer_loss_weights[layer_idx]

                attention_loss += layer_loss

            total_loss += attention_loss
            loss_components["attention_loss"] = attention_loss

        loss_components["total_loss"] = total_loss
        return total_loss, loss_components


def adaptive_temperature_schedule(
    epoch: int, total_epochs: int, temp_start: float = 8.0, temp_end: float = 2.0
) -> float:
    """
    Compute adaptive temperature for curriculum distillation.

    Temperature starts high (more exploration, smoother distributions)
    and decreases over time (more exploitation, sharper distributions).

    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        temp_start: Starting temperature
        temp_end: Ending temperature

    Returns:
        Current temperature value
    """
    progress = epoch / max(total_epochs - 1, 1)
    temperature = temp_start - (temp_start - temp_end) * progress
    return temperature


class DistillationTrainer:
    """
    Complete distillation training loop.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.config = config
        self.optimizer = optimizer
        self.device = device

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Multi-layer distillation loss
        self.distillation_criterion = MultiLayerDistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha,
            distill_layers=config.distill_layers,
        )

        self.history = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        use_adaptive_temp: bool = False,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        epoch_losses = {
            "total": 0.0,
            "hard": 0.0,
            "soft": 0.0,
            "feature": 0.0,
            "attention": 0.0,
        }
        num_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Adaptive temperature
            temperature = self.config.temperature
            if use_adaptive_temp:
                temperature = adaptive_temperature_schedule(
                    epoch,
                    len(train_loader) * epoch,  # placeholder
                )

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_output = self.teacher(inputs)
                teacher_logits = (
                    teacher_output.logits
                    if hasattr(teacher_output, "logits")
                    else teacher_output
                )

            # Student forward
            student_output = self.student(inputs)
            student_logits = (
                student_output.logits
                if hasattr(student_output, "logits")
                else student_output
            )

            # Compute losses
            total_loss, loss_dict = combined_distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=temperature,
                alpha=self.config.alpha,
            )

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            epoch_losses["total"] += loss_dict["total_loss"].item()
            epoch_losses["hard"] += loss_dict["hard_loss"].item()
            epoch_losses["soft"] += loss_dict["soft_loss"].item()
            num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate student model."""
        self.student.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                student_output = self.student(inputs)
                student_logits = (
                    student_output.logits
                    if hasattr(student_output, "logits")
                    else student_output
                )

                loss = F.cross_entropy(student_logits, labels)
                total_loss += loss.item()

                predictions = student_logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return {"eval_loss": total_loss / len(eval_loader), "accuracy": correct / total}

    def save_student(self, path: str):
        """Save student model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.student.state_dict(),
                "config": self.config,
                "history": self.history,
            },
            path,
        )


def demo_distillation():
    """
    Demo: Simple distillation example with dummy data.
    """
    print("=" * 60)
    print("Knowledge Distillation Demo")
    print("=" * 60)

    # Create dummy teacher and student models
    class DummyTeacher(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=3072, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    class DummyStudent(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=1536, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    # Initialize models
    teacher = DummyTeacher(input_dim=512, hidden_dim=2048, num_classes=10)
    student = DummyStudent(input_dim=512, hidden_dim=1024, num_classes=10)

    print(f"\nTeacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(
        f"Compression ratio: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.2f}x"
    )

    # Create dummy data
    batch_size = 32
    seq_len = 128
    input_dim = 512

    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    dummy_labels = torch.randint(0, 10, (batch_size,))

    # Simulate teacher logits (with "dark knowledge")
    with torch.no_grad():
        teacher_logits = teacher(dummy_input.mean(dim=1))  # [batch, 10]
        # Add structure to teacher output - some classes are more similar
        teacher_logits[:, 5:] += torch.randn(batch_size, 5) * 0.5  # Add noise

    student_logits = student(dummy_input.mean(dim=1))

    print("\n" + "-" * 40)
    print("Testing distillation loss...")
    print("-" * 40)

    # Test different temperatures
    temperatures = [1.0, 2.0, 4.0, 8.0]
    for T in temperatures:
        loss = distillation_loss(student_logits.clone(), teacher_logits.clone(), T)
        print(f"T={T}: KL divergence loss = {loss.item():.4f}")

    # Test combined loss
    print("\n" + "-" * 40)
    print("Testing combined loss (hard + soft)...")
    print("-" * 40)

    for alpha in [0.1, 0.3, 0.5]:
        total_loss, loss_dict = combined_distillation_loss(
            student_logits.clone(),
            teacher_logits.clone(),
            dummy_labels,
            temperature=4.0,
            alpha=alpha,
        )
        print(
            f"alpha={alpha}: Total={total_loss.item():.4f}, Hard={loss_dict['hard_loss'].item():.4f}, Soft={loss_dict['soft_loss'].item():.4f}"
        )

    # Test adaptive temperature
    print("\n" + "-" * 40)
    print("Adaptive temperature schedule...")
    print("-" * 40)

    for epoch in range(5):
        T = adaptive_temperature_schedule(
            epoch, total_epochs=5, temp_start=8.0, temp_end=2.0
        )
        print(f"Epoch {epoch}: T = {T:.2f}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo_distillation()
