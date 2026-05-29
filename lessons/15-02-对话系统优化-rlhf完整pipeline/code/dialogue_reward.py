"""
Dialogue Reward Model: Training a model to evaluate dialogue quality

This module handles:
1. Reward model architecture
2. Human preference data processing
3. Reward model training
4. Preference pair generation
"""

import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import torch.nn.functional as F


@dataclass
class PreferencePair:
    """A pair of responses with human preference annotation"""

    prompt: str
    chosen_response: str
    rejected_response: str
    preference: str  # "chosen" or "rejected"
    annotator_id: Optional[str] = None
    confidence: Optional[float] = None


class DialogueRewardModel(nn.Module):
    """
    Reward Model for Dialogue Quality

    Architecture: Base LM + Reward Head
    Takes a dialogue and outputs a scalar reward representing quality
    """

    def __init__(
        self, base_model_path: str, reward_scale: float = 2.5, freeze_base: bool = True
    ):
        super().__init__()
        self.reward_scale = reward_scale

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Freeze base model if specified (common for reward models)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Get hidden size
        hidden_size = self.base_model.config.hidden_size

        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Identity(),  # Output raw score, scale later
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward for dialogue

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            rewards: [batch_size] - reward value per sample
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use the last hidden state
        last_hidden = outputs.hidden_states[-1]

        # Take the last token's hidden state for reward prediction
        # This captures the "final" representation of the sequence
        last_token_hidden = last_hidden[:, -1, :]

        # Compute reward
        reward = self.reward_head(last_token_hidden).squeeze(-1)

        return reward * self.reward_scale

    def forward_with_response(
        self,
        prompt_tokens: torch.Tensor,
        response_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards for prompt and response separately
        Useful for understanding contribution of each part
        """
        # Concatenate prompt and response
        full_tokens = torch.cat([prompt_tokens, response_tokens], dim=-1)
        full_mask = torch.cat([prompt_mask, response_mask], dim=-1)

        outputs = self.base_model(
            input_ids=full_tokens,
            attention_mask=full_mask,
        )

        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden).squeeze(-1)

        return reward * self.reward_scale, last_hidden


class RewardLossFn:
    """Reward model loss function"""

    @staticmethod
    def compute_pair_loss(
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute Bradley-Terry style preference loss

        Loss = -log(sigmoid(reward_chosen - reward_rejected - margin))

        Encourages chosen response to have higher reward than rejected
        """
        diff = chosen_rewards - rejected_rewards - margin

        # Using sigmoid with log to create proper ranking loss
        loss = -F.logsigmoid(diff)

        return loss.mean()

    @staticmethod
    def compute_margin_loss(
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        margin: float = 0.0,
    ) -> torch.Tensor:
        """
        Simple margin-based ranking loss
        """
        return torch.clamp(margin - (chosen_rewards - rejected_rewards), min=0).mean()


class PreferenceDataset(Dataset):
    """Dataset for preference-based training"""

    def __init__(
        self,
        preference_data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
    ):
        self.data = preference_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        prompt = sample["prompt"]
        chosen = sample["chosen_response"]
        rejected = sample["rejected_response"]

        # Format: prompt + chosen (for chosen reward)
        # Format: prompt + rejected (for rejected reward)

        # Tokenize chosen
        chosen_text = f"{prompt}\n{chosen}"
        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize rejected
        rejected_text = f"{prompt}\n{rejected}"
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_ids": chosen_enc["input_ids"].squeeze(),
            "chosen_mask": chosen_enc["attention_mask"].squeeze(),
            "rejected_ids": rejected_enc["input_ids"].squeeze(),
            "rejected_mask": rejected_enc["attention_mask"].squeeze(),
        }


class RewardModelTrainer:
    """Trainer for Reward Model"""

    def __init__(
        self,
        base_model_path: str,
        output_dir: str = "./checkpoints/reward_model",
        learning_rate: float = 1e-5,
        batch_size: int = 16,
        epochs: int = 1,
        warmup_steps: int = 100,
        margin: float = 0.5,
    ):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.margin = margin

        self.tokenizer = None
        self.model = None
        self.trainer = None

    def setup(self):
        """Initialize tokenizer and model"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = DialogueRewardModel(
            base_model_path=self.base_model_path,
            reward_scale=2.5,
            freeze_base=False,  # Unfreeze for reward training
        )

    def compute_metrics(self, eval_pred):
        """Compute training metrics"""
        chosen_rewards = eval_pred.predictions[0]
        rejected_rewards = eval_pred.predictions[1]

        # Accuracy: how often chosen > rejected
        accuracy = (chosen_rewards > rejected_rewards).mean()

        # Average reward margin
        margin = (chosen_rewards - rejected_rewards).mean()

        return {"accuracy": accuracy, "reward_margin": margin}

    def train(
        self,
        train_dataset: PreferenceDataset,
        eval_dataset: Optional[PreferenceDataset] = None,
    ):
        """Train the reward model"""

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            optimizer="adamw",
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            logging_steps=10,
            eval_steps=100,
            save_steps=500,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            bf16=True,
            report_to=["tensorboard"],
        )

        # Custom trainer to handle pair-wise loss
        class RewardTrainer(Trainer):
            def compute_loss(self, model, inputs, **kwargs):
                chosen_ids = inputs["chosen_ids"]
                chosen_mask = inputs["chosen_mask"]
                rejected_ids = inputs["rejected_ids"]
                rejected_mask = inputs["rejected_mask"]

                # Compute rewards for both responses
                chosen_rewards = model(chosen_ids, chosen_mask)
                rejected_rewards = model(rejected_ids, rejected_mask)

                # Compute ranking loss
                loss = RewardLossFn.compute_pair_loss(
                    chosen_rewards, rejected_rewards, margin=0.5
                )

                return loss

        self.trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

    def save_model(self, path: Optional[str] = None):
        """Save the trained reward model"""
        save_path = path or self.output_dir
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)


class PreferenceAnnotator:
    """Generate synthetic preference data for training reward model"""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_preference_pair(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        quality_a: float,
        quality_b: float,
    ) -> Dict[str, Any]:
        """
        Generate a preference pair based on quality scores

        Args:
            prompt: The user prompt
            response_a: First response
            response_b: Second response
            quality_a: Quality score for response_a (0-1)
            quality_b: Quality score for response_b (0-1)
        """
        # Determine preference based on quality scores
        if quality_a > quality_b:
            preference = "a"
            chosen = response_a
            rejected = response_b
        elif quality_b > quality_a:
            preference = "b"
            chosen = response_b
            rejected = response_a
        else:
            preference = "equal"
            chosen = response_a
            rejected = response_b

        return {
            "prompt": prompt,
            "chosen_response": chosen,
            "rejected_response": rejected,
            "preference": preference,
            "quality_a": quality_a,
            "quality_b": quality_b,
            "margin": abs(quality_a - quality_b),
        }

    def batch_generate(
        self, dialogues: List[Dict], num_pairs_per_dialogue: int = 2
    ) -> List[Dict]:
        """
        Generate preference pairs from dialogue dataset

        Args:
            dialogues: List of dialogue samples
            num_pairs_per_dialogue: Number of pairs to generate per dialogue
        """
        preference_data = []

        for dialogue in dialogues:
            prompt = self._extract_prompt(dialogue)
            responses = self._extract_responses(dialogue)

            for _ in range(num_pairs_per_dialogue):
                # Randomly select two responses
                import random

                selected = random.sample(responses, min(2, len(responses)))

                if len(selected) < 2:
                    continue

                response_a, response_b = selected

                # Simulate quality scores (in real scenario, from human annotators)
                quality_a = random.uniform(0.3, 1.0)
                quality_b = random.uniform(0.3, 1.0)

                pair = self.generate_preference_pair(
                    prompt, response_a, response_b, quality_a, quality_b
                )
                pair["dialogue_id"] = dialogue.get("id", "unknown")

                preference_data.append(pair)

        return preference_data

    def _extract_prompt(self, dialogue: Dict) -> str:
        """Extract the user prompt from dialogue"""
        turns = dialogue.get("turns", [])
        for turn in turns:
            if turn.get("role") == "user":
                return turn["content"]
        return dialogue.get("prompt", "")

    def _extract_responses(self, dialogue: Dict) -> List[str]:
        """Extract assistant responses from dialogue"""
        responses = []
        turns = dialogue.get("turns", [])
        for turn in turns:
            if turn.get("role") == "assistant":
                responses.append(turn["content"])
        return responses


def load_preference_data(path: str) -> List[Dict]:
    """Load preference data from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_reward_training_dataset(
    dialogues: List[Dict], annotator: Optional[PreferenceAnnotator] = None
) -> List[Dict]:
    """
    Create preference dataset from dialogue samples

    Args:
        dialogues: List of dialogue samples with turns
        annotator: Optional annotator to generate synthetic preferences

    Returns:
        List of preference pairs ready for training
    """
    if annotator is None:
        annotator = PreferenceAnnotator()

    return annotator.batch_generate(dialogues)


def main():
    """Example usage"""
    config = {
        "base_model_path": "meta-llama/Llama-2-7b",
        "output_dir": "./checkpoints/reward_model",
        "learning_rate": 1e-5,
        "batch_size": 16,
        "epochs": 1,
    }

    print("Dialogue Reward Model Module loaded")
    print("Use RewardModelTrainer to train your reward model")


if __name__ == "__main__":
    main()
