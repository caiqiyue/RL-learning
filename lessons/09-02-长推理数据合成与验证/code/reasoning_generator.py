"""
Reasoning Generator
生成推理链并使用 PRM 对每步评分，筛选高质量推理
"""

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReasoningStep:
    content: str
    score: float
    is_valid: bool


@dataclass
class ReasoningChain:
    question: str
    steps: list[ReasoningStep]
    final_answer: str
    total_score: float


class StepScorer:
    """使用 PRM 对推理步骤进行评分"""

    def __init__(self, prm_model=None):
        self.prm_model = prm_model

    def score_step(
        self, question: str, previous_steps: list[str], current_step: str
    ) -> float:
        """
        对当前推理步骤打分

        Args:
            question: 问题
            previous_steps: 前面的推理步骤
            current_step: 当前步骤

        Returns:
            score: 0~1 之间的分数，表示该步骤对最终正确答案的贡献
        """
        if self.prm_model is None:
            return random.uniform(0.5, 1.0)

        # 实际使用 PRM 模型进行评分
        # 伪实现：返回模拟分数
        return 0.8


class ReasoningGenerator:
    """
    生成多样化推理链并评分筛选
    """

    def __init__(self, llm_model=None, step_scorer: Optional[StepScorer] = None):
        self.llm = llm_model
        self.step_scorer = step_scorer or StepScorer()

    def generate_reasoning_chain(
        self, question: str, max_steps: int = 10, temperature: float = 0.8
    ) -> ReasoningChain:
        """
        生成一条推理链

        Args:
            question: 输入问题
            max_steps: 最大推理步数
            temperature: 采样温度

        Returns:
            ReasoningChain: 包含所有步骤和评分的推理链
        """
        steps = []
        previous_steps = []

        for step_idx in range(max_steps):
            step_content = self._sample_next_step(question, previous_steps, temperature)

            if step_content is None:
                break

            score = self.step_scorer.score_step(question, previous_steps, step_content)
            is_valid = score > 0.5

            step = ReasoningStep(content=step_content, score=score, is_valid=is_valid)
            steps.append(step)
            previous_steps.append(step_content)

        final_answer = self._extract_final_answer(previous_steps)
        total_score = self._compute_chain_score(steps)

        return ReasoningChain(
            question=question,
            steps=steps,
            final_answer=final_answer,
            total_score=total_score,
        )

    def _sample_next_step(
        self, question: str, previous_steps: list[str], temperature: float
    ) -> Optional[str]:
        """使用 LLM 采样下一步推理"""
        if self.llm is None:
            return f"Step {len(previous_steps) + 1} reasoning..."

        # 实际使用 LLM 生成
        # 伪实现
        return None

    def _extract_final_answer(self, steps: list[str]) -> str:
        """从推理步骤中提取最终答案"""
        if not steps:
            return ""
        return steps[-1] if steps else ""

    def _compute_chain_score(self, steps: list[ReasoningStep]) -> float:
        """计算整条推理链的总分（加权平均）"""
        if not steps:
            return 0.0

        weights = [0.4, 0.3, 0.2, 0.1][: len(steps)]
        weights.reverse()

        scores = [s.score for s in steps]
        weighted_sum = sum(w * s for w, s in zip(weights, scores))
        return weighted_sum / sum(weights)


def best_of_n_sampling(
    generator: ReasoningGenerator, question: str, n: int = 8, max_steps: int = 10
) -> ReasoningChain:
    """
    Best-of-N 采样：用同一个问题生成 N 条推理链，选择最佳

    Args:
        generator: ReasoningGenerator 实例
        question: 问题
        n: 采样数量
        max_steps: 每条推理链的最大步数

    Returns:
        得分最高的推理链
    """
    chains = []

    for i in range(n):
        chain = generator.generate_reasoning_chain(
            question=question, max_steps=max_steps, temperature=random.uniform(0.6, 1.2)
        )
        chains.append(chain)

    best_chain = max(chains, key=lambda c: c.total_score)
    return best_chain


def filter_reasoning_chains(
    chains: list[ReasoningChain],
    min_avg_score: float = 0.6,
    min_steps: int = 2,
    max_steps: int = 20,
) -> list[ReasoningChain]:
    """
    筛选高质量推理链

    Args:
        chains: 推理链列表
        min_avg_score: 最低平均分阈值
        min_steps: 最少步数
        max_steps: 最多步数

    Returns:
        筛选后的推理链列表
    """
    filtered = []

    for chain in chains:
        if not chain.steps:
            continue

        avg_score = sum(s.score for s in chain.steps) / len(chain.steps)

        if avg_score >= min_avg_score and min_steps <= len(chain.steps) <= max_steps:
            filtered.append(chain)

    return filtered


def format_for_sft_training(chain: ReasoningChain) -> str:
    """
    将推理链格式化为 SFT 训练数据

    输出格式：
    <|user|>
    {question}
    <|assistant|>
    {reasoning_steps}
    {final_answer}
    <|end|>
    """
    reasoning_text = "\n".join(
        f"第 {i + 1} 步：{step.content} (score: {step.score:.2f})"
        for i, step in enumerate(chain.steps)
    )

    formatted = f"""<|user|>
{chain.question}

<|assistant|>
{reasoning_text}

最终答案：{chain.final_answer}

<|end|>"""

    return formatted


if __name__ == "__main__":
    scorer = StepScorer()
    generator = ReasoningGenerator(step_scorer=scorer)

    question = "求函数 f(x) = x² - 4x + 3 的最小值"
    chain = generator.generate_reasoning_chain(question, max_steps=5)

    print(f"Generated chain with {len(chain.steps)} steps")
    print(f"Total score: {chain.total_score:.3f}")
    print(format_for_sft_training(chain))
