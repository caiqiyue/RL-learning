import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, List
import logging
import os

from grpo_config import GRPOConfig

logger = logging.getLogger(__name__)


class GRPOTrainer:
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: GRPOConfig,
        reward_fn: Callable,
        tokenizer,
        data_collator: Optional[Callable] = None,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.data_collator = data_collator or self._default_collator

        self.device = next(policy_model.parameters()).device

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def step(self, batch: Dict) -> Dict:
        prompts = batch["prompt"]
        ground_truths = batch.get("ground_truth", [None] * len(prompts))

        responses, log_probs = self._group_sampling(prompts)
        rewards = self._compute_rewards(responses, ground_truths)
        advantages = self._compute_advantages(rewards)
        loss, info = self._compute_policy_loss(
            prompts, responses, log_probs, advantages
        )

        loss.backward()

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.max_grad_norm
            )

        self.optimizer.step()
        self.optimizer.zero_grad()

        flat_rewards = [r for group in rewards for r in group]
        info["reward_mean"] = sum(flat_rewards) / len(flat_rewards)

        return info

    def _group_sampling(self, prompts: List[str]) -> tuple:
        all_responses = []
        all_log_probs = []

        for prompt in prompts:
            prompt_responses = []
            prompt_log_probs = []

            inputs = self.tokenizer(
                [prompt] * self.config.group_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_response_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=self.config.group_size,
                )

            for i in range(self.config.group_size):
                response_ids = outputs.sequences[i][inputs["input_ids"].shape[1] :]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                prompt_responses.append(response)

                response_inputs = self.tokenizer(
                    response,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_response_length,
                ).to(self.device)

                with torch.no_grad():
                    output = self.policy_model(
                        input_ids=response_inputs["input_ids"],
                        attention_mask=response_inputs["attention_mask"],
                    )
                    logp = torch.log_softmax(output.logits, dim=-1)

                response_logp = (
                    logp[:-1]
                    .gather(2, response_inputs["input_ids"][1:].unsqueeze(-1))
                    .squeeze(-1)
                    .mean()
                )
                prompt_log_probs.append(response_logp)

            all_responses.append(prompt_responses)
            all_log_probs.append(torch.stack(prompt_log_probs))

        return all_responses, all_log_probs

    def _compute_rewards(
        self, responses: List[List[str]], ground_truths: List[str]
    ) -> List[List[float]]:
        all_rewards = []

        for prompt_responses, gt in zip(responses, ground_truths):
            prompt_rewards = []
            for response in prompt_responses:
                reward = self.reward_fn(response, gt)
                prompt_rewards.append(reward)
            all_rewards.append(prompt_rewards)

        return all_rewards

    def _compute_advantages(self, rewards: List[List[float]]) -> List[torch.Tensor]:
        advantages = []

        for group_rewards in rewards:
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            mean = rewards_tensor.mean()
            std = rewards_tensor.std()
            normalized = (rewards_tensor - mean) / (std + 1e-8)
            advantages.append(normalized)

        return advantages

    def _compute_policy_loss(
        self,
        prompts: List[str],
        responses: List[List[str]],
        log_probs: List[torch.Tensor],
        advantages: List[torch.Tensor],
    ) -> tuple:
        policy_losses = []
        kl_losses = []

        for i in range(len(prompts)):
            for j in range(len(responses[i])):
                log_prob = log_probs[i][j]
                advantage = advantages[i][j]

                ratio = torch.exp(log_prob)

                clipped_ratio = torch.clamp(
                    ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                )

                policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
                policy_losses.append(policy_loss)

                kl = ratio - torch.log(ratio + 1e-8) - 1
                kl_losses.append(kl)

        total_policy_loss = torch.stack(policy_losses).mean()
        total_kl_loss = self.config.kl_coef * torch.stack(kl_losses).mean()

        loss = total_policy_loss + total_kl_loss

        info = {
            "policy_loss": total_policy_loss.item(),
            "kl_loss": total_kl_loss.item(),
        }

        return loss, info

    def _default_collator(self, batch):
        return batch


def math_reward_function(response: str, ground_truth: str) -> float:
    import re

    def normalize_answer(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\$([^$]+)\$", r"\1", text)
        text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", text)
        return text

    def extract_final_answer(text: str) -> str:
        boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
        if boxed:
            return boxed.group(1).strip()

        final_line = re.search(r"####\s*(.+?)(?:\n|$)", text)
        if final_line:
            return final_line.group(1).strip()

        answer = re.search(
            r"(?:the answer is|answer:|therefore|thus)\s*[:=]?\s*(.+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        if answer:
            return answer.group(1).strip()

        lines = text.strip().split("\n")
        return lines[-1] if lines else ""

    extracted_answer = extract_final_answer(response)
    pred = normalize_answer(extracted_answer)
    gt = normalize_answer(ground_truth)

    try:
        if abs(float(pred) - float(gt)) < 1e-6:
            return 1.0
    except ValueError:
        if pred.strip() == gt.strip():
            return 1.0

    return 0.0


def train_grpo(
    policy_model: nn.Module,
    ref_model: nn.Module,
    train_loader: DataLoader,
    reward_fn: Callable,
    tokenizer,
    config: GRPOConfig,
    output_dir: str = "./output",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate),
        config=config,
        reward_fn=reward_fn,
        tokenizer=tokenizer,
    )

    global_step = 0
    history = {"loss": [], "reward": [], "kl": []}

    for episode in range(config.num_episodes):
        for batch in train_loader:
            info = trainer.step(batch)

            global_step += 1

            history["loss"].append(info.get("policy_loss", 0))
            history["reward"].append(info.get("reward_mean", 0))
            history["kl"].append(info.get("kl_loss", 0))

            if global_step % 10 == 0:
                logger.info(
                    f"Step {global_step} | "
                    f"Loss: {info.get('policy_loss', 0):.4f} | "
                    f"Reward: {info.get('reward_mean', 0):.4f}"
                )

        if (episode + 1) % 10 == 0:
            torch.save(
                {
                    "model_state_dict": policy_model.state_dict(),
                    "config": config,
                    "episode": episode,
                },
                f"{output_dir}/checkpoint_{episode}.pt",
            )
            logger.info(f"Saved checkpoint at episode {episode + 1}")

    return history


if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./output/grpo_model")
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = GRPOConfig(
        group_size=args.group_size,
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
    )

    print(f"Loading model: {args.model_name}")
    policy_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("Model loaded successfully")
    print(f"Output directory: {args.output_dir}")
