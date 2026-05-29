import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import argparse
import json


def generate_reasoning(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.9,
    top_p: float = 0.95,
    do_sample: bool = True,
    num_return_sequences: int = 1,
) -> List[Dict]:
    results = []

    for i in range(num_return_sequences):
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

        gen_tokens = response_ids.tolist()
        tokens_with_scores = list(zip(gen_tokens, [1.0] * len(gen_tokens)))

        results.append(
            {
                "response": response,
                "response_ids": response_ids.tolist(),
                "tokens_with_scores": tokens_with_scores,
            }
        )

    return results


def extract_final_answer(text: str) -> Optional[str]:
    import re

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


def analyze_reasoning_chain(response: str) -> Dict:
    lines = response.strip().split("\n")

    num_steps = len([l for l in lines if l.strip()])
    has_boxed = "\\boxed{" in response
    has_final_marker = "####" in response
    estimated_think_tokens = sum(len(l.split()) for l in lines) * 1.3

    return {
        "num_lines": num_steps,
        "has_boxed": has_boxed,
        "has_final_marker": has_final_marker,
        "estimated_think_tokens": int(estimated_think_tokens),
        "contains_reflection": any(
            word in response.lower()
            for word in ["however", "wait", "let me", "actually", "alternatively"]
        ),
    }


def batch_generate(
    model, tokenizer, prompts: List[str], **generation_kwargs
) -> List[List[Dict]]:
    all_results = []

    for prompt in prompts:
        results = generate_reasoning(model, tokenizer, prompt, **generation_kwargs)
        all_results.append(results)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    print("Model loaded successfully")

    sample_problems = [
        "A train travels 120 miles in 2 hours. At the same speed, how far would it travel in 5 hours?",
        "If x + 3 = 7, what is the value of x?",
        "John has 15 apples. He gives 4 to Mary and eats 2. How many apples does he have left?",
    ]

    for i, problem in enumerate(sample_problems):
        print(f"\n{'=' * 60}")
        print(f"Problem {i + 1}: {problem}")
        print("=" * 60)

        results = generate_reasoning(
            model,
            tokenizer,
            problem,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            num_return_sequences=args.num_samples,
        )

        for j, result in enumerate(results):
            print(f"\n--- Response {j + 1} ---")
            print(
                result["response"][:500] + "..."
                if len(result["response"]) > 500
                else result["response"]
            )

            analysis = analyze_reasoning_chain(result["response"])
            print(f"\nAnalysis: {analysis}")

            answer = extract_final_answer(result["response"])
            print(f"Extracted Answer: {answer}")
