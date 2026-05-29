import argparse
import gc
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdapterMerger:
    """
    Adapter merger with multiple strategies

    Supports:
    - average: Simple averaging of adapter weights
    - weighted: Weighted averaging based on quality scores
    - task_vector: Merge based on direction vectors
    """

    STRATEGIES = ["average", "weighted", "task_vector"]

    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def merge(
        self,
        adapter_paths: List[str],
        output_path: str,
        strategy: str = "average",
        weights: List[float] = None,
        quality_scores: Dict[str, float] = None,
    ) -> str:
        """
        Merge multiple adapters into a single model

        Args:
            adapter_paths: List of adapter paths to merge
            output_path: Output path for merged model
            strategy: Merge strategy (average, weighted, task_vector)
            weights: Custom weights for each adapter
            quality_scores: Quality scores per adapter (for weighted merge)

        Returns:
            Path to merged model
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"Starting {strategy} merge of {len(adapter_paths)} adapters")

        if strategy == "average":
            return self._merge_average(adapter_paths, output_path)
        elif strategy == "weighted":
            return self._merge_weighted(
                adapter_paths,
                output_path,
                weights=weights,
                quality_scores=quality_scores,
            )
        elif strategy == "task_vector":
            return self._merge_task_vector(adapter_paths, output_path)

    def _merge_average(
        self,
        adapter_paths: List[str],
        output_path: str,
    ) -> str:
        """Simple average merge of adapters"""
        logger.info("Merging with simple average strategy")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            load_in_4bit=False,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        base_model.eval()

        adapter_states = []
        for adapter_path in adapter_paths:
            adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
            adapter_states.append(
                {
                    name: param.clone()
                    for name, param in adapter_model.named_parameters()
                    if "lora_" in name
                }
            )

        num_adapters = len(adapter_paths)

        merged_state = {}
        for name in adapter_states[0]:
            merged_state[name] = (
                sum(states[name] for states in adapter_states) / num_adapters
            )

        for name, param in base_model.named_parameters():
            if name in merged_state:
                param.copy_(merged_state[name])

        os.makedirs(output_path, exist_ok=True)
        base_model.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(adapter_paths[0])
        tokenizer.save_pretrained(output_path)

        logger.info(f"Merged model saved to {output_path}")

        return output_path

    def _merge_weighted(
        self,
        adapter_paths: List[str],
        output_path: str,
        weights: List[float] = None,
        quality_scores: Dict[str, float] = None,
    ) -> str:
        """Weighted average merge based on quality scores"""
        logger.info("Merging with weighted strategy")

        if quality_scores and not weights:
            weights = [quality_scores.get(Path(p).name, 1.0) for p in adapter_paths]

        if weights is None:
            weights = [1.0 / len(adapter_paths)] * len(adapter_paths)

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        logger.info(f"Using weights: {weights}")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            load_in_4bit=False,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        base_model.eval()

        adapter_states = []
        for adapter_path in adapter_paths:
            adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
            adapter_states.append(
                {
                    name: param.clone()
                    for name, param in adapter_model.named_parameters()
                    if "lora_" in name
                }
            )

        merged_state = {}
        for name in adapter_states[0]:
            merged_state[name] = sum(
                w * states[name] for w, states in zip(weights, adapter_states)
            )

        for name, param in base_model.named_parameters():
            if name in merged_state:
                param.copy_(merged_state[name])

        os.makedirs(output_path, exist_ok=True)
        base_model.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(adapter_paths[0])
        tokenizer.save_pretrained(output_path)

        return output_path

    def _merge_task_vector(
        self,
        adapter_paths: List[str],
        output_path: str,
    ) -> str:
        """Task vector merge - accumulate direction vectors"""
        logger.info("Merging with Task Vector strategy")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            load_in_4bit=False,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        base_params = {
            name: param.clone() for name, param in base_model.named_parameters()
        }

        for adapter_path in adapter_paths:
            adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

            for name, param in adapter_model.named_parameters():
                if "lora_" in name:
                    base_name = name.replace("lora_A", "").replace("lora_B", "")
                    if base_name in base_params:
                        base_params[base_name].add_(param)

        for name, param in base_model.named_parameters():
            if name in base_params:
                param.copy_(base_params[name])

        os.makedirs(output_path, exist_ok=True)
        base_model.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(adapter_paths[0])
        tokenizer.save_pretrained(output_path)

        return output_path


class MultiModelEvaluator:
    """Evaluate models and adapters on multiple tasks"""

    def __init__(self, registry=None):
        self.registry = registry

    def evaluate(
        self,
        model_or_path,
        tokenizer_or_path,
        eval_tasks: List[Dict],
        batch_size: int = 4,
    ) -> Dict[str, dict]:
        """
        Evaluate model on multiple tasks

        Args:
            model_or_path: Model or path to model
            tokenizer_or_path: Tokenizer or path to tokenizer
            eval_tasks: List of task configurations
            batch_size: Evaluation batch size

        Returns:
            Dictionary of task results
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if isinstance(model_or_path, str):
            model = AutoModelForCausalLM.from_pretrained(
                model_or_path,
                load_in_4bit=True,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_path)
        else:
            model = model_or_path
            tokenizer = tokenizer_or_path

        results = {}

        for task in eval_tasks:
            task_name = task.get("name", "unknown")
            task_type = task.get("type", "generic")

            logger.info(f"Evaluating on task: {task_name}")

            try:
                if task_type == "perplexity":
                    metrics = self._eval_perplexity(model, tokenizer, task)
                elif task_type == "generation":
                    metrics = self._eval_generation(model, tokenizer, task)
                elif task_type == "classification":
                    metrics = self._eval_classification(model, tokenizer, task)
                else:
                    metrics = self._eval_generic(model, tokenizer, task)

                results[task_name] = metrics

            except Exception as e:
                logger.error(f"Task {task_name} failed: {e}")
                results[task_name] = {"error": str(e)}

        return results

    def _eval_perplexity(
        self,
        model,
        tokenizer,
        task: Dict,
    ) -> Dict:
        """Evaluate perplexity on text dataset"""
        import torch
        from tqdm import tqdm

        eval_dataset = task.get("dataset", [])
        max_length = task.get("max_length", 512)

        total_loss = 0.0
        num_batches = 0

        model.eval()
        with torch.no_grad():
            for i in range(0, len(eval_dataset), max_length):
                batch_texts = eval_dataset[i : i + max_length]

                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )

                outputs = model(**inputs)
                loss = outputs.loss

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
        }

    def _eval_generation(
        self,
        model,
        tokenizer,
        task: Dict,
    ) -> Dict:
        """Evaluate text generation quality"""
        prompts = task.get("prompts", [])
        max_new_tokens = task.get("max_new_tokens", 256)
        temperature = task.get("temperature", 0.7)

        generations = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            generations.append(generated)

        return {
            "generations": generations,
            "num_samples": len(generations),
        }

    def _eval_classification(
        self,
        model,
        tokenizer,
        task: Dict,
    ) -> Dict:
        """Evaluate classification accuracy"""
        import torch

        eval_dataset = task.get("dataset", [])
        labels = task.get("labels", [])

        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for item in eval_dataset:
                text = item.get("text", "")
                true_label = item.get("label", -1)

                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                outputs = model(**inputs)
                logits = outputs.logits

                pred = torch.argmax(logits, dim=-1).item()

                if pred == true_label:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    def _eval_generic(
        self,
        model,
        tokenizer,
        task: Dict,
    ) -> Dict:
        """Generic evaluation"""
        return {"status": "completed"}

    def compare_adapters(
        self,
        adapter_paths: List[str],
        eval_tasks: List[Dict],
        base_model_path: str = None,
    ) -> Dict:
        """
        Compare multiple adapters on evaluation tasks

        Args:
            adapter_paths: List of adapter paths
            eval_tasks: Evaluation tasks
            base_model_path: Optional base model path for loading

        Returns:
            Comparison results
        """
        all_results = {}

        for adapter_path in adapter_paths:
            adapter_name = Path(adapter_path).name
            logger.info(f"Evaluating adapter: {adapter_name}")

            try:
                model = PeftModel.from_pretrained(
                    AutoModelForCausalLM.from_pretrained(
                        base_model_path or adapter_path,
                        load_in_4bit=True,
                        device_map="auto",
                    ),
                    adapter_path,
                )

                tokenizer = AutoTokenizer.from_pretrained(adapter_path)

                results = self.evaluate(model, tokenizer, eval_tasks)

                all_results[adapter_name] = results

                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                logger.error(f"Failed to evaluate {adapter_name}: {e}")
                all_results[adapter_name] = {"error": str(e)}

        comparison = self._generate_comparison_table(all_results, eval_tasks)

        return comparison

    def _generate_comparison_table(
        self,
        results: Dict,
        tasks: List[Dict],
    ) -> Dict:
        """Generate comparison table across tasks and adapters"""
        table = {}

        for task in tasks:
            task_name = task.get("name", "unknown")
            task_results = {}

            for adapter_name, adapter_results in results.items():
                if "error" not in adapter_results.get(task_name, {}):
                    task_results[adapter_name] = adapter_results[task_name]

            if task_results:
                best_adapter = max(
                    task_results.keys(), key=lambda k: self._get_score(task_results[k])
                )
                table[task_name] = {
                    "results": task_results,
                    "best_adapter": best_adapter,
                    "best_score": self._get_score(task_results[best_adapter]),
                }

        return table

    def _get_score(self, result: Dict) -> float:
        """Extract score from result dictionary"""
        if "error" in result:
            return 0.0
        if "perplexity" in result:
            return 1.0 / result["perplexity"]
        if "accuracy" in result:
            return result["accuracy"]
        return result.get("score", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Merge adapters and evaluate models")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["merge", "eval", "compare"],
        help="Operation mode",
    )
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--adapters", type=str, nargs="+", help="Adapter paths")
    parser.add_argument("--output", type=str, default="./merged", help="Output path")
    parser.add_argument(
        "--strategy",
        type=str,
        default="average",
        choices=["average", "weighted", "task_vector"],
        help="Merge strategy",
    )
    parser.add_argument(
        "--registry", type=str, default=None, help="Adapter registry path"
    )
    parser.add_argument(
        "--eval_tasks", type=str, default=None, help="Evaluation tasks JSON file"
    )

    args = parser.parse_args()

    if args.mode == "merge":
        merger = AdapterMerger(args.base_model)
        output_path = merger.merge(
            adapter_paths=args.adapters,
            output_path=args.output,
            strategy=args.strategy,
        )
        logger.info(f"Merge completed: {output_path}")

    elif args.mode == "eval":
        evaluator = MultiModelEvaluator()

        eval_tasks = []
        if args.eval_tasks:
            import json

            with open(args.eval_tasks, "r") as f:
                eval_tasks = json.load(f)

        results = evaluator.evaluate(
            model_or_path=args.base_model,
            tokenizer_or_path=args.base_model,
            eval_tasks=eval_tasks,
        )

        logger.info(f"Evaluation results: {results}")

    elif args.mode == "compare":
        evaluator = MultiModelEvaluator()

        eval_tasks = []
        if args.eval_tasks:
            import json

            with open(args.eval_tasks, "r") as f:
                eval_tasks = json.load(f)

        comparison = evaluator.compare_adapters(
            adapter_paths=args.adapters,
            eval_tasks=eval_tasks,
            base_model_path=args.base_model,
        )

        logger.info(f"Comparison results: {comparison}")


if __name__ == "__main__":
    main()
