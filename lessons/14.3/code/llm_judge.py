#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation Implementation

This module provides utilities for evaluating language model outputs
using a stronger LLM as the judge (e.g., GPT-4 evaluating GPT-3.5 outputs).
"""

import json
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from anthropic import Anthropic
    from openai import OpenAI
except ImportError:
    print("Error: anthropic or openai not installed. Run: pip install anthropic openai")


class JudgeModel(Enum):
    """Supported judge models."""

    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"


class EvaluationMode(Enum):
    """Evaluation mode for LLM-as-Judge."""

    SINGLE_RESPONSE = "single"  # Rate a single response
    PAIRWISE = "pairwise"  # Compare two responses


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    score: Optional[float] = None
    winner: Optional[str] = None  # For pairwise: "A", "B", or "tie"
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class EvalCase:
    """A single evaluation case."""

    question: str
    answer_a: str  # For pairwise, this is the first response
    answer_b: Optional[str] = None  # Optional for pairwise evaluation
    context: Optional[str] = None  # Additional context
    reference: Optional[str] = None  # Optional reference answer
    criteria: Optional[Dict[str, str]] = None  # Scoring criteria


class JudgeConfig:
    """Configuration for the judge model."""

    def __init__(
        self,
        model: JudgeModel = JudgeModel.GPT4,
        mode: EvaluationMode = EvaluationMode.SINGLE_RESPONSE,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens


class LLMasJudge:
    """
    LLM-as-Judge evaluator.

    Uses a stronger LLM to evaluate outputs from other models.
    Supports both single response scoring and pairwise comparison.
    """

    def __init__(
        self,
        config: Optional[JudgeConfig] = None,
        api_key: Optional[str] = None,
        judge_type: str = "openai",  # "openai" or "anthropic"
    ):
        self.config = config or JudgeConfig()
        self.judge_type = judge_type

        if judge_type == "openai":
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model_name = self.config.model.value
        elif judge_type == "anthropic":
            self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = self.config.model.value
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

    def _build_single_eval_prompt(
        self,
        eval_case: EvalCase,
        scoring_dimensions: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt for single response evaluation."""

        if scoring_dimensions is None:
            scoring_dimensions = ["准确性", "清晰性", "完整性", "有用性"]

        dimensions_text = "\n".join(f"- {dim}" for dim in scoring_dimensions)

        system_prompt = """你是一个专业的AI助手评估专家。
你的职责是客观评估AI助手的回答质量。
请根据提供的评分标准对回答进行评分。
始终按指定格式输出评分结果。"""

        user_prompt = f"""请评估以下AI助手的回答。

问题：{eval_case.question}
"""

        if eval_case.context:
            user_prompt += f"\n背景信息：{eval_case.context}\n"

        user_prompt += f"""
回答：
{eval_case.answer_a}
"""

        if eval_case.reference:
            user_prompt += f"\n参考答案：{eval_case.reference}\n"

        user_prompt += f"""
评分维度：
{dimensions_text}

请为每个维度打分（1-5分），然后给出总体评分。
输出格式：
"""

        if scoring_dimensions:
            for dim in scoring_dimensions:
                user_prompt += f"\n{dim}: [分数]"

        user_prompt += f"\n\n总体评分: [分数]\n"
        user_prompt += "理由: [简短解释评分依据]"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_pairwise_eval_prompt(
        self,
        eval_case: EvalCase,
        include_position_balancing: bool = True,
    ) -> List[Dict[str, str]]:
        """Build prompt for pairwise comparison."""

        system_prompt = """你是一个专业的AI助手评估专家。
你的职责是客观比较两个AI助手的回答，判断哪个更好。
仔细分析两个回答的优缺点，给出公正的判断。
如果两个回答质量相近，可以判定为平局。"""

        if include_position_balancing:
            position_note = "注意：回答A在左侧，回答B在右侧。\n"
        else:
            position_note = ""

        user_prompt = f"""请比较以下两个AI助手的回答，判断哪个更好。

{position_note}
问题：{eval_case.question}
"""

        if eval_case.context:
            user_prompt += f"\n背景信息：{eval_case.context}\n"

        user_prompt += f"""
回答A：
{eval_case.answer_a}

回答B：
{eval_case.answer_b}
"""

        user_prompt += """
请先分析两个回答的优缺点，然后给出最终判断。
输出格式：
理由: [分析两个回答的优缺点]
最终判断: A / B / 平局
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM API."""

        if self.judge_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content

        elif self.judge_type == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
            )
            return response.content[0].text

        raise ValueError(f"Unknown judge type: {self.judge_type}")

    def _parse_single_response(self, response: str) -> Dict[str, Any]:
        """Parse the raw LLM response for single evaluation."""

        result = {
            "scores": {},
            "overall_score": None,
            "reasoning": None,
        }

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if "总体评分" in key or "overall" in key.lower():
                    try:
                        result["overall_score"] = float(value)
                    except ValueError:
                        pass
                elif "理由" in key or "reason" in key.lower():
                    result["reasoning"] = value
                elif any(
                    dim in key for dim in ["准确性", "清晰性", "完整性", "有用性"]
                ):
                    try:
                        result["scores"][key] = float(value)
                    except ValueError:
                        pass

        return result

    def _parse_pairwise_response(self, response: str) -> Dict[str, Any]:
        """Parse the raw LLM response for pairwise evaluation."""

        result = {
            "winner": None,
            "reasoning": None,
        }

        lines = response.strip().split("\n")
        reasoning_parts = []

        for line in lines:
            line = line.strip()
            if "最终判断" in line or "winner" in line.lower():
                if (
                    "A" in line
                    and "平局" not in line
                    and "B" not in line.replace("A", "")
                ):
                    result["winner"] = "A"
                elif "B" in line and "平局" not in line:
                    result["winner"] = "B"
                elif "平局" in line:
                    result["winner"] = "tie"
            elif "理由" in line or "分析" in line:
                reasoning_parts.append(line.split(":", 1)[-1].strip())

        result["reasoning"] = " ".join(reasoning_parts)

        return result

    def evaluate_single(
        self,
        eval_case: EvalCase,
        scoring_dimensions: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single response.

        Args:
            eval_case: The evaluation case containing question and answer
            scoring_dimensions: Optional list of scoring dimensions

        Returns:
            EvaluationResult with scores and reasoning
        """
        messages = self._build_single_eval_prompt(eval_case, scoring_dimensions)
        raw_response = self._call_llm(messages)
        parsed = self._parse_single_response(raw_response)

        return EvaluationResult(
            score=parsed.get("overall_score"),
            reasoning=parsed.get("reasoning"),
            raw_response=raw_response,
        )

    def evaluate_pairwise(
        self,
        eval_case: EvalCase,
        swap_positions: bool = False,
    ) -> Tuple[EvaluationResult, EvaluationResult]:
        """
        Evaluate two responses in pairwise comparison.

        Runs the evaluation twice with swapped positions to mitigate position bias.

        Args:
            eval_case: The evaluation case containing question and both answers
            swap_positions: If True, swap A and B for the second evaluation

        Returns:
            Tuple of (first_result, second_result)
        """
        if swap_positions:
            swapped_case = EvalCase(
                question=eval_case.question,
                answer_a=eval_case.answer_b,
                answer_b=eval_case.answer_a,
                context=eval_case.context,
            )
        else:
            swapped_case = eval_case

        messages = self._build_pairwise_eval_prompt(swapped_case)
        raw_response = self._call_llm(messages)
        parsed = self._parse_pairwise_response(raw_response)

        first_result = EvaluationResult(
            winner=parsed.get("winner"),
            reasoning=parsed.get("reasoning"),
            raw_response=raw_response,
        )

        if swap_positions and first_result.winner:
            first_result.winner = "B" if first_result.winner == "A" else "A"

        if swap_positions:
            messages2 = self._build_pairwise_eval_prompt(eval_case)
            raw_response2 = self._call_llm(messages2)
            parsed2 = self._parse_pairwise_response(raw_response2)

            second_result = EvaluationResult(
                winner=parsed2.get("winner"),
                reasoning=parsed2.get("reasoning"),
                raw_response=raw_response2,
            )
        else:
            second_result = None

        return first_result, second_result

    def evaluate_pairwise_balanced(
        self,
        eval_case: EvalCase,
    ) -> EvaluationResult:
        """
        Evaluate pairwise with position balancing.

        Runs evaluation twice with swapped positions and combines results.
        This mitigates position bias in pairwise comparison.

        Args:
            eval_case: The evaluation case

        Returns:
            Combined EvaluationResult
        """
        result1, result2 = self.evaluate_pairwise(eval_case, swap_positions=True)

        win_a = 0
        win_b = 0
        ties = 0

        if result1.winner == "A":
            win_a += 1
        elif result1.winner == "B":
            win_b += 1
        else:
            ties += 1

        if result2 and result2.winner == "A":
            win_a += 1
        elif result2 and result2.winner == "B":
            win_b += 1
        elif result2:
            ties += 1

        if win_a > win_b:
            final_winner = "A"
        elif win_b > win_a:
            final_winner = "B"
        else:
            final_winner = "tie"

        return EvaluationResult(
            winner=final_winner,
            reasoning=f"Position-balanced evaluation: A won {win_a}, B won {win_b}, ties {ties}",
            raw_response=f"{result1.raw_response}\n\n---\n\n{result2.raw_response if result2 else 'N/A'}",
        )


class BatchEvaluator:
    """Batch evaluator for processing multiple evaluation cases."""

    def __init__(
        self,
        judge: LLMasJudge,
        verbose: bool = False,
    ):
        self.judge = judge
        self.verbose = verbose

    def evaluate_dataset(
        self,
        cases: List[EvalCase],
        mode: EvaluationMode = EvaluationMode.SINGLE_RESPONSE,
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of cases.

        Args:
            cases: List of evaluation cases
            mode: Evaluation mode

        Returns:
            List of evaluation results
        """
        results = []

        for i, case in enumerate(cases):
            if self.verbose:
                print(f"Evaluating case {i + 1}/{len(cases)}...")

            try:
                if mode == EvaluationMode.SINGLE_RESPONSE:
                    result = self.judge.evaluate_single(case)
                elif mode == EvaluationMode.PAIRWISE:
                    result = self.judge.evaluate_pairwise_balanced(case)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                results.append(result)

            except Exception as e:
                if self.verbose:
                    print(f"  Error: {e}")
                results.append(EvaluationResult(raw_response=str(e)))

        return results

    def aggregate_results(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """
        Aggregate evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with aggregated statistics
        """
        total = len(results)

        if not total:
            return {"error": "No results to aggregate"}

        scores = [r.score for r in results if r.score is not None]

        winners = [r.winner for r in results if r.winner is not None]
        win_a = sum(1 for w in winners if w == "A")
        win_b = sum(1 for w in winners if w == "B")
        ties = sum(1 for w in winners if w == "tie")

        return {
            "total_cases": total,
            "average_score": sum(scores) / len(scores) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "pairwise_stats": {
                "wins_a": win_a,
                "wins_b": win_b,
                "ties": ties,
                "win_rate_a": win_a / total if total > 0 else None,
                "win_rate_b": win_b / total if total > 0 else None,
            }
            if winners
            else None,
        }


def load_eval_cases_from_jsonl(file_path: str) -> List[EvalCase]:
    """Load evaluation cases from a JSONL file."""
    cases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            cases.append(
                EvalCase(
                    question=data.get("question", ""),
                    answer_a=data.get("answer_a", ""),
                    answer_b=data.get("answer_b"),
                    context=data.get("context"),
                    reference=data.get("reference"),
                )
            )
    return cases


def main():
    """Example usage of LLM-as-Judge."""

    judge = LLMasJudge(
        config=JudgeConfig(
            model=JudgeModel.GPT4,
            mode=EvaluationMode.SINGLE_RESPONSE,
        ),
        judge_type="openai",
    )

    test_case = EvalCase(
        question="解释什么是大语言模型",
        answer_a="大语言模型是一种使用深度学习技术训练的人工智能模型，能够理解和生成人类语言。",
        answer_b="大语言模型（LLM）是基于Transformer架构的大型神经网络，通过在海量文本数据上进行预训练来学习语言的统计规律。它们能够执行各种语言任务，如文本生成、翻译、问答等。",
    )

    print("Single Response Evaluation:")
    result = judge.evaluate_single(test_case)
    print(f"  Score: {result.score}")
    print(f"  Reasoning: {result.reasoning}")

    print("\nPairwise Evaluation (with position balancing):")
    pairwise_result = judge.evaluate_pairwise_balanced(test_case)
    print(f"  Winner: {pairwise_result.winner}")
    print(f"  Reasoning: {pairwise_result.reasoning}")


if __name__ == "__main__":
    main()
