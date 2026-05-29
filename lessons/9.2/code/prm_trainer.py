"""
Process Reward Model (PRM) Trainer
用于训练过程奖励模型，对推理链的每一步进行评分
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class PRMTrainingSample:
    question: str
    steps: list[str]
    step_labels: list[int]
    final_answer_correct: bool


class ProcessRewardModel(nn.Module):
    """
    过程奖励模型：对推理的每一步预测一个 0~1 的分数
    分数越高表示该步骤越有可能导致正确答案
    """

    def __init__(self, encoder_model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        self.encoder = None  # HuggingFace encoder
        self.score_head = nn.Linear(768, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            step_scores: [batch_size, seq_len] 每一步的分数
        """
        # 实际使用 encoder 处理输入，然后接 score_head
        # 简化实现
        hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], 768)
        scores = torch.sigmoid(self.score_head(hidden))
        return scores.squeeze(-1)


def compute_prm_loss(
    model: ProcessRewardModel, samples: list[PRMTrainingSample]
) -> torch.Tensor:
    """
    计算 PRM 的训练损失

    正样本：通向正确答案的推理步骤 (label=1)
    负样本：通向错误答案的推理步骤 (label=0)

    损失函数：二分类交叉熵
    """
    total_loss = 0.0
    criterion = nn.BCELoss(reduction="sum")

    for sample in samples:
        step_scores = model(
            input_ids=sample.input_ids, attention_mask=sample.attention_mask
        )

        labels = torch.tensor(sample.step_labels, dtype=torch.float32)
        loss = criterion(step_scores, labels)
        total_loss += loss

    return total_loss / len(samples)


def train_prm(
    model: ProcessRewardModel,
    train_samples: list[PRMTrainingSample],
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-5,
):
    """
    PRM 训练循环

    Args:
        model: ProcessRewardModel 实例
        train_samples: 训练样本列表
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i : i + batch_size]

            optimizer.zero_grad()
            loss = compute_prm_loss(model, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def evaluate_prm(
    model: ProcessRewardModel, eval_samples: list[PRMTrainingSample]
) -> dict:
    """
    评估 PRM 模型

    返回：
        - step_accuracy: 每步预测的准确率
        - final_step_auc: 最后一步的 AUC（衡量整体性能）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in eval_samples:
            scores = model(
                input_ids=sample.input_ids, attention_mask=sample.attention_mask
            )

            for step_idx, (score, label) in enumerate(zip(scores, sample.step_labels)):
                pred = 1 if score > 0.5 else 0
                if pred == label:
                    correct += 1
                total += 1

    step_accuracy = correct / total if total > 0 else 0.0

    return {
        "step_accuracy": step_accuracy,
        "final_step_auc": 0.85,  # placeholder
    }


if __name__ == "__main__":
    model = ProcessRewardModel()

    samples = [
        PRMTrainingSample(
            question="求 1+2+...+100 的值",
            steps=["等差数列求和公式", "n*(n+1)/2", "100*101/2 = 5050"],
            step_labels=[1, 1, 1],
            final_answer_correct=True,
        )
    ]

    train_prm(model, samples, epochs=2)
    results = evaluate_prm(model, samples)
    print(f"Evaluation: {results}")
