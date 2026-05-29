"""
Dialogue RLHF: Complete RLHF Pipeline for Dialogue Systems

This module implements:
1. PPO (Proximal Policy Optimization) for dialogue
2. GRPO (Group Relative Policy Optimization) alternative
3. Multi-objective reward computation
4. Complete training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOTrainingArguments
from tqdm import tqdm
import numpy as np


@dataclass
class DialogueRewardSignal:
    """Multi-objective reward signal for dialogue"""

    rm_reward_weight: float = 0.5
    safety_weight: float = 0.3
    coherence_weight: float = 0.2
    safety_threshold: float = 0.0

    # Penalty weights
    excessive_length_penalty: float = 0.01
    repetition_penalty: float = 0.1

    # Reward model (set externally)
    reward_model: Optional[Any] = None

    # Safety classifier (set externally)
    safety_classifier: Optional[Any] = None

    def compute_total_reward(
        self, prompt: str, response: str, rm_reward: float
    ) -> Tuple[float, Dict]:
        """
        Compute combined reward from multiple signals

        Args:
            prompt: User prompt
            response: Assistant response
            rm_reward: Reward model score

        Returns:
            total_reward: Combined reward
            breakdown: Dict with individual components
        """
        breakdown = {}

        # 1. Reward Model component
        breakdown["rm_reward"] = rm_reward * self.rm_reward_weight

        # 2. Safety component
        safety_reward = self._compute_safety_reward(response)
        breakdown["safety_reward"] = safety_reward * self.safety_weight

        # 3. Coherence component
        coherence_reward = self._compute_coherence_reward(response)
        breakdown["coherence_reward"] = coherence_reward * self.coherence_weight

        # 4. Length penalty
        length_penalty = self._compute_length_penalty(response)
        breakdown["length_penalty"] = length_penalty

        # 5. Repetition penalty
        repetition_penalty = self._compute_repetition_penalty(response)
        breakdown["repetition_penalty"] = repetition_penalty

        # Total reward
        total_reward = (
            breakdown["rm_reward"]
            + breakdown["safety_reward"]
            + breakdown["coherence_reward"]
            + breakdown["length_penalty"]
            + breakdown["repetition_penalty"]
        )

        breakdown["total"] = total_reward

        return total_reward, breakdown

    def _compute_safety_reward(self, response: str) -> float:
        """Compute safety reward"""
        if self.safety_classifier is not None:
            safety_score = self.safety_classifier.classify(response)
            return 1.0 if safety_score > self.safety_threshold else -2.0

        # Fallback: keyword-based check
        harmful_keywords = ["暴力", "色情", "歧视", "犯罪", "武器"]
        for kw in harmful_keywords:
            if kw in response:
                return -1.0
        return 1.0

    def _compute_coherence_reward(self, response: str) -> float:
        """Compute coherence reward based on response properties"""
        score = 0.0

        # Has proper sentence ending
        if any(response.endswith(p) for p in ["。", "！", "？", ".", "!"]):
            score += 0.3

        # Not too short
        if len(response) > 20:
            score += 0.2

        # Has structure (lists, etc)
        if any(marker in response for marker in ["：", ",", "；", "1.", "•"]):
            score += 0.3

        # Not too long
        if len(response) > 500:
            score -= 0.2

        return max(0.0, min(score, 1.0))

    def _compute_length_penalty(self, response: str) -> float:
        """Penalize excessively long responses"""
        optimal_length = 200
        length = len(response)

        if length > optimal_length * 2:
            return -0.1 * (length - optimal_length * 2) / 100
        return 0.0

    def _compute_repetition_penalty(self, response: str) -> float:
        """Penalize repetitive content"""
        words = response.split()
        if len(words) < 10:
            return 0.0

        # Check for repeated n-grams
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            return -0.2 * (1.0 - unique_ratio)
        return 0.0


class DialoguePPOTrainer:
    """PPO Trainer for Dialogue Systems"""

    def __init__(
        self,
        policy_model_path: str,
        ref_model_path: str,
        reward_model: Any,
        reward_signal: DialogueRewardSignal,
        output_dir: str = "./checkpoints/dialogue_rlhf",
        learning_rate: float = 1e-5,
        ppo_epochs: int = 4,
        batch_size: int = 8,
        mini_batch_size: int = 2,
        max_grad_norm: float = 1.0,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.1,
        entropy_coef: float = 0.01,
        kl_loss_coef: float = 0.1,
        max_seq_len: int = 4096,
    ):
        self.policy_model_path = policy_model_path
        self.ref_model_path = ref_model_path
        self.reward_model = reward_model
        self.reward_signal = reward_signal
        self.output_dir = output_dir

        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_loss_coef = kl_loss_coef

        self.max_seq_len = max_seq_len

        self.policy_model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None

    def setup(self):
        """Initialize models and tokenizer"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.policy_model_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load policy model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.policy_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Load reference model (same architecture, for KL penalty)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.ref_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.ref_model.eval()  # Reference model not updated

        # Setup reward signal with models
        self.reward_signal.reward_model = self.reward_model

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(), lr=self.learning_rate
        )

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        top_p: float = 0.9,
    ) -> str:
        """Generate response from policy model"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self.policy_model.device)

        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response (remove prompt part)
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = self._extract_response(full_text, prompt)

        return response

    def _extract_response(self, full_text: str, prompt: str) -> str:
        """Extract assistant response from generated text"""
        # Find where prompt ends and response begins
        if prompt in full_text:
            response = full_text[len(prompt) :].strip()
        else:
            response = full_text

        # Remove special tokens
        for token in ["<|im_end|>", "<|im_start|>", "assistant\n"]:
            response = response.replace(token, "")

        return response.strip()

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Compute rewards for prompt-response pairs"""
        rewards = []

        for prompt, response in zip(prompts, responses):
            # Get reward model score
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                self.policy_model.device
            )

            response_tokens = self.tokenizer(
                response, return_tensors="pt"
            ).input_ids.to(self.policy_model.device)

            with torch.no_grad():
                rm_reward = self.reward_model(
                    torch.cat([prompt_tokens, response_tokens], dim=-1),
                    torch.ones(1, prompt_tokens.shape[1] + response_tokens.shape[1]).to(
                        self.policy_model.device
                    ),
                ).item()

            # Compute total reward with multi-objective signal
            total_reward, _ = self.reward_signal.compute_total_reward(
                prompt, response, rm_reward
            )

            rewards.append(total_reward)

        return rewards

    def ppo_update(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform a PPO update step

        Args:
            prompts: List of prompts
            responses: List of responses
            rewards: Computed rewards
            logprobs: Log probabilities of responses under current policy
            advantages: Computed advantages
        """
        total_loss = 0.0
        policy_loss_sum = 0.0
        kl_loss_sum = 0.0
        entropy_sum = 0.0

        # Compute response logprobs under current policy
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            inputs = self.tokenizer(
                prompt + response,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            ).to(self.policy_model.device)

            # Forward pass
            outputs = self.policy_model(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )

            # Compute logprobs for response tokens
            response_logits = outputs.logits[:, :-1, :]
            response_tokens = inputs.input_ids[:, 1:]

            # Policy loss from PPO
            new_logprobs = F.log_softmax(response_logits, dim=-1)
            new_logprobs = torch.gather(
                new_logprobs, 2, response_tokens.unsqueeze(-1)
            ).squeeze(-1)

            # PPO policy loss
            ratio = torch.exp(new_logprobs.sum(-1) - logprobs[i])
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            policy_loss = -torch.min(
                ratio * advantages[i], clipped_ratio * advantages[i]
            )
            policy_loss_sum += policy_loss.item()

            # KL loss with reference model
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                )

            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(
                ref_logprobs, 2, response_tokens.unsqueeze(-1)
            ).squeeze(-1)

            kl_loss = F.kl_div(new_logprobs, ref_logprobs, reduction="batchmean")
            kl_loss_sum += kl_loss.item()

            # Entropy bonus
            entropy = -(new_logprobs * torch.exp(new_logprobs)).sum(-1).mean()
            entropy_sum += entropy.item()

            # Total loss for this sample
            sample_loss = (
                policy_loss + self.kl_loss_coef * kl_loss - self.entropy_coef * entropy
            )
            total_loss += sample_loss

        # Average and backward
        total_loss = total_loss / len(prompts)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss_sum / len(prompts),
            "kl_loss": kl_loss_sum / len(prompts),
            "entropy": entropy_sum / len(prompts),
        }

    def train_epoch(
        self, train_prompts: List[str], num_generations: int = 8
    ) -> Dict[str, float]:
        """Train one epoch"""
        epoch_stats = {
            "mean_reward": 0.0,
            "mean_generation_length": 0.0,
            "num_samples": 0,
        }

        all_prompts = []
        all_responses = []
        all_rewards = []

        # Generate responses for all prompts
        for prompt in tqdm(train_prompts, desc="Generating responses"):
            responses = []
            for _ in range(num_generations):
                response = self.generate_response(prompt)
                responses.append(response)

            all_prompts.extend([prompt] * num_generations)
            all_responses.extend(responses)

        # Compute rewards
        batch_size = self.batch_size
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i : i + batch_size]
            batch_responses = all_responses[i : i + batch_size]

            batch_rewards = self.compute_rewards(batch_prompts, batch_responses)
            all_rewards.extend(batch_rewards)

        # Compute advantages (simple version: reward - baseline)
        rewards_tensor = torch.tensor(all_rewards)
        reward_baseline = rewards_tensor.mean()
        advantages = rewards_tensor - reward_baseline

        # Normalize advantages
        if rewards_tensor.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update with multiple epochs
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(all_prompts))

            for i in range(0, len(indices), self.mini_batch_size):
                batch_indices = indices[i : i + self.mini_batch_size]

                batch_prompts = [all_prompts[idx] for idx in batch_indices]
                batch_responses = [all_responses[idx] for idx in batch_indices]
                batch_rewards = [all_rewards[idx] for idx in batch_indices]
                batch_advantages = advantages[batch_indices]

                # Estimate logprobs (simplified)
                logprobs = torch.zeros(len(batch_prompts))

                self.ppo_update(
                    batch_prompts,
                    batch_responses,
                    batch_rewards,
                    logprobs,
                    batch_advantages,
                )

        epoch_stats["mean_reward"] = rewards_tensor.mean().item()
        epoch_stats["num_samples"] = len(all_prompts)

        return epoch_stats


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer

    GRPO is a simplified alternative to PPO that uses relative ranking
    within a group of responses to compute advantages.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: Any,
        reward_signal: DialogueRewardSignal,
        num_generations: int = 8,
        learning_rate: float = 1e-5,
        kl_coef: float = 0.1,
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.reward_signal = reward_signal
        self.num_generations = num_generations
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )

    def compute_advantages(
        self, group_rewards: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relative advantages within a group

        A_i = r_i - mean(r_group)
        """
        rewards = torch.tensor(group_rewards, dtype=torch.float32)
        advantages = rewards - rewards.mean()

        # Normalize
        if rewards.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, rewards

    def compute_kl_divergence(
        self, prompt_tokens: torch.Tensor, response_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference model"""
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=prompt_tokens, attention_mask=torch.ones_like(prompt_tokens)
            )

        policy_outputs = self.policy(
            input_ids=prompt_tokens, attention_mask=torch.ones_like(prompt_tokens)
        )

        ref_logits = ref_outputs.logits[:, :-1, :]
        policy_logits = policy_outputs.logits[:, :-1, :]

        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)

        # KL divergence
        kl = F.kl_div(policy_logprobs, ref_logprobs, reduction="batchmean")

        return kl

    def update(
        self,
        prompts: List[str],
        groups_responses: List[List[str]],
        tokenizer: AutoTokenizer,
    ) -> Dict[str, float]:
        """
        GRPO update step

        For each prompt, we have multiple generated responses.
        We compute relative advantages within the group.
        """
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        total_reward = 0.0

        for prompt, responses in zip(prompts, groups_responses):
            if len(responses) < 2:
                continue

            # Compute rewards for all responses in group
            group_rewards = []
            for response in responses:
                # Get RM reward
                inputs = tokenizer(
                    prompt + response,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                ).to(self.policy.device)

                with torch.no_grad():
                    rm_reward = self.reward_model(
                        inputs.input_ids, inputs.attention_mask
                    ).item()

                # Get total reward
                total_r, _ = self.reward_signal.compute_total_reward(
                    prompt, response, rm_reward
                )
                group_rewards.append(total_r)

            # Compute advantages
            advantages, rewards = self.compute_advantages(group_rewards)

            # Policy gradient loss (weighted by advantages)
            # Since this is simplified, we use mean reward as proxy
            policy_loss = -rewards.mean()
            total_policy_loss += policy_loss.item()
            total_reward += rewards.mean().item()

            # KL loss
            full_tokens = tokenizer(
                prompt + responses[0],  # Use first response for KL
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.policy.device)

            kl_loss = self.compute_kl_divergence(
                full_tokens.input_ids, full_tokens.input_ids
            )
            total_kl_loss += kl_loss.item()

        # Combined loss
        total_loss = total_policy_loss / len(
            prompts
        ) + self.kl_coef * total_kl_loss / len(prompts)

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": total_policy_loss / len(prompts),
            "kl_loss": total_kl_loss / len(prompts),
            "mean_reward": total_reward / len(prompts),
        }


class DialogueRLHFPipeline:
    """Complete RLHF pipeline for dialogue"""

    def __init__(
        self,
        config: Dict[str, Any],
        reward_model: Any,
        safety_classifier: Optional[Any] = None,
    ):
        self.config = config

        # Initialize reward signal
        self.reward_signal = DialogueRewardSignal(
            rm_reward_weight=config.get("rm_reward_weight", 0.5),
            safety_weight=config.get("safety_weight", 0.3),
            coherence_weight=config.get("coherence_weight", 0.2),
            reward_model=reward_model,
            safety_classifier=safety_classifier,
        )

        # Initialize trainer
        use_grpo = config.get("algorithm", "ppo").lower() == "grpo"

        if use_grpo:
            self.trainer = None  # Would need to setup separately
        else:
            self.trainer = DialoguePPOTrainer(
                policy_model_path=config["sft_model_path"],
                ref_model_path=config["ref_model_path"],
                reward_model=reward_model,
                reward_signal=self.reward_signal,
                **config.get("ppo", {}),
            )

    def train(
        self,
        train_prompts: List[str],
        num_epochs: int = 3,
        save_checkpoint_every: int = 500,
    ):
        """Run the complete RLHF training"""
        self.trainer.setup()

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            epoch_stats = self.trainer.train_epoch(train_prompts)

            print(f"Mean Reward: {epoch_stats['mean_reward']:.4f}")
            print(f"Samples: {epoch_stats['num_samples']}")

            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}")

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        save_path = f"{self.config.get('output_dir', './checkpoints')}/{name}"
        self.trainer.policy_model.save_pretrained(save_path)
        self.trainer.tokenizer.save_pretrained(save_path)
        print(f"Checkpoint saved to {save_path}")


def main():
    """Example usage"""
    config = {
        "algorithm": "ppo",  # or "grpo"
        "sft_model_path": "./checkpoints/dialogue_sft",
        "ref_model_path": "./checkpoints/dialogue_sft",
        "output_dir": "./checkpoints/dialogue_rlhf",
        "ppo": {
            "learning_rate": 1e-5,
            "batch_size": 8,
            "mini_batch_size": 2,
            "ppo_epochs": 4,
            "clip_ratio": 0.2,
        },
        "rm_reward_weight": 0.5,
        "safety_weight": 0.3,
        "coherence_weight": 0.2,
    }

    print("Dialogue RLHF Module loaded")
    print("Use DialogueRLHFPipeline for complete training")


if __name__ == "__main__":
    main()
