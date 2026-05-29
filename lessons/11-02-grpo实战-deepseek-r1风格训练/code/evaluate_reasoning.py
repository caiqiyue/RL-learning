import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import re
import json
import argparse


def extract_final_answer(text: str) -> Optional[str]:
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
    return lines[-1] if lines else None


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", text)
    return text


def evaluate_math_response(
    response: str, ground_truth: str
) -> Tuple[bool, Optional[str]]:
    extracted = extract_final_answer(response)
    if extracted is None:
        return False, None

    pred = normalize_answer(extracted)
    gt = normalize_answer(ground_truth)

    try:
        correct = abs(float(pred) - float(gt)) < 1e-6
        return correct, extracted
    except ValueError:
        correct = pred.strip() == gt.strip()
        return correct, extracted


def extract_code(text: str) -> Optional[str]:
    code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]

    code_blocks = re.findall(r"```\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]

    def_match = re.search(r"(def\s+\w+.*?)(?=\n\n|\Z)", text, re.DOTALL)
    if def_match:
        return def_match.group(1)

    return None


def evaluate_code_response(
    response: str, test_cases: List[str], timeout: int = 10
) -> Tuple[float, Optional[str], Optional[str]]:
    import subprocess
    import tempfile

    code = extract_code(response)
    if code is None:
        return 0.0, None, "no_code_extracted"

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = f"{tmpdir}/solution.py"
        with open(code_path, "w") as f:
            f.write(code)

        passed = 0
        error_msg = None

        for i, test_case in enumerate(test_cases):
            try:
                result = subprocess.run(
                    ["python", "-c", test_case],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if result.returncode == 0:
                    passed += 1
                else:
                    error_msg = result.stderr

            except subprocess.TimeoutExpired:
                return 0.0, code, "timeout"
            except Exception as e:
                return 0.0, code, str(e)

        reward = passed / len(test_cases) if test_cases else 0.0
        return reward, code, error_msg


def evaluate_batch(
    model,
    tokenizer,
    problems: List[Dict],
    max_tokens: int = 2048,
    temperature: float = 0.9,
) -> Dict:
    results = []
    correct = 0
    total = len(problems)

    for item in problems:
        prompt = item["prompt"]
        ground_truth = item.get("ground_truth", "")
        test_cases = item.get("test_cases", [])
        task_type = item.get("type", "math")

        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

        if task_type == "math":
            is_correct, extracted = evaluate_math_response(response, ground_truth)
            results.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "extracted_answer": extracted,
                    "correct": is_correct,
                    "task_type": task_type,
                }
            )
            if is_correct:
                correct += 1

        elif task_type == "code":
            reward, code, error = evaluate_code_response(response, test_cases)
            results.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "extracted_code": code[:200] + "..."
                    if code and len(code) > 200
                    else code,
                    "reward": reward,
                    "error": error,
                    "task_type": task_type,
                }
            )
            correct += reward

        elif task_type == "logic":
            is_correct, extracted = evaluate_math_response(response, ground_truth)
            results.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "extracted_answer": extracted,
                    "correct": is_correct,
                    "task_type": task_type,
                }
            )
            if is_correct:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


class ReasoningBenchmark:
    def __init__(self, name: str):
        self.name = name
        self.results = []

    def add_result(self, result: Dict):
        self.results.append(result)

    def compute_metrics(self) -> Dict:
        if not self.results:
            return {"accuracy": 0.0, "num_examples": 0}

        correct = sum(1 for r in self.results if r.get("correct", False))
        total = len(self.results)

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "num_examples": total,
            "task_type": self.results[0].get("task_type", "unknown")
            if self.results
            else "unknown",
        }

    def print_summary(self):
        metrics = self.compute_metrics()
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {self.name}")
        print(f"{'=' * 60}")
        print(f"Total Examples: {metrics['num_examples']}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Task Type: {metrics['task_type']}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_examples", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    print("Model loaded successfully")

    sample_problems = [
        {
            "prompt": "A train travels 120 miles in 2 hours. At the same speed, how far would it travel in 5 hours?",
            "ground_truth": "300",
            "type": "math",
        },
        {
            "prompt": "If x + 3 = 7, what is the value of x?",
            "ground_truth": "4",
            "type": "math",
        },
        {
            "prompt": "John has 15 apples. He gives 4 to Mary and eats 2. How many apples does he have left?",
            "ground_truth": "9",
            "type": "math",
        },
        {
            "prompt": "A rectangle has a length of 8 and width of 5. What is its area?",
            "ground_truth": "40",
            "type": "math",
        },
        {
            "prompt": "Sarah bought 3 books at $12 each. She paid with a $50 bill. How much change did she get?",
            "ground_truth": "14",
            "type": "math",
        },
    ]

    benchmark = ReasoningBenchmark(args.benchmark)

    results = evaluate_batch(
        model, tokenizer, sample_problems, max_tokens=args.max_tokens
    )

    for r in results["results"]:
        benchmark.add_result(r)

    benchmark.print_summary()

    for i, r in enumerate(results["results"]):
        print(f"\nProblem {i + 1}:")
        print(f"Prompt: {r['prompt'][:80]}...")
        print(f"Correct: {r['correct']}")
        print(f"Extracted Answer: {r.get('extracted_answer', 'N/A')}")
