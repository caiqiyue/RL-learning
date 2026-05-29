import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AdapterInfo:
    """Adapter information metadata"""

    name: str
    model_name: str
    model_type: str
    adapter_path: str
    base_model_path: str
    created_at: str
    lora_rank: int
    lora_alpha: int
    task_type: str
    dataset: str
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class AdapterRegistry:
    """
    Registry for managing multiple model adapters

    Supports registration, querying, loading, and batch operations
    """

    def __init__(self, registry_path: str = "./adapter_registry.json"):
        self.registry_path = Path(registry_path)
        self.adapters: Dict[str, AdapterInfo] = {}
        self._load()

    def _load(self):
        """Load registry from file"""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                adapters_data = data.get("adapters", {})
                self.adapters = {
                    name: AdapterInfo(**info) for name, info in adapters_data.items()
                }

    def _save(self):
        """Save registry to file"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "adapters": {name: asdict(info) for name, info in self.adapters.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register(
        self,
        name: str,
        model_name: str,
        model_type: str,
        adapter_path: str,
        base_model_path: str,
        lora_rank: int,
        lora_alpha: int,
        task_type: str,
        dataset: str,
        metrics: dict = None,
        metadata: dict = None,
    ) -> AdapterInfo:
        """
        Register a new adapter

        Args:
            name: Unique adapter name
            model_name: Base model name
            model_type: Model type (llama, qwen2, chatglm, etc.)
            adapter_path: Path to adapter files
            base_model_path: Path to base model
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            task_type: Task type
            dataset: Training dataset
            metrics: Training metrics
            metadata: Additional metadata

        Returns:
            AdapterInfo object
        """
        from dataclasses import dataclass, field, asdict

        info = AdapterInfo(
            name=name,
            model_name=model_name,
            model_type=model_type,
            adapter_path=adapter_path,
            base_model_path=base_model_path,
            created_at=datetime.now().isoformat(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            task_type=task_type,
            dataset=dataset,
            metrics=metrics or {},
            metadata=metadata or {},
        )

        self.adapters[name] = info
        self._save()

        return info

    def unregister(self, name: str):
        """Unregister an adapter"""
        if name in self.adapters:
            del self.adapters[name]
            self._save()

    def get(self, name: str) -> Optional[AdapterInfo]:
        """Get adapter info by name"""
        return self.adapters.get(name)

    def list_adapters(
        self,
        model_type: str = None,
        task_type: str = None,
    ) -> List[AdapterInfo]:
        """
        List adapters with optional filters

        Args:
            model_type: Filter by model type
            task_type: Filter by task type

        Returns:
            List of AdapterInfo objects
        """
        results = list(self.adapters.values())

        if model_type:
            results = [a for a in results if a.model_type == model_type]

        if task_type:
            results = [a for a in results if a.task_type == task_type]

        return results

    def find_compatible(
        self,
        base_model_path: str,
        task_type: str = None,
    ) -> List[AdapterInfo]:
        """
        Find adapters compatible with a base model

        Args:
            base_model_path: Base model path
            task_type: Filter by task type

        Returns:
            List of compatible adapters
        """
        results = [
            a for a in self.adapters.values() if a.base_model_path == base_model_path
        ]

        if task_type:
            results = [a for a in results if a.task_type == task_type]

        return results

    def load_adapter_model(
        self,
        name: str,
        device: str = "cuda",
    ):
        """
        Load model with adapter

        Args:
            name: Adapter name
            device: Device to load model on

        Returns:
            (model, tokenizer) tuple
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        info = self.get(name)
        if not info:
            raise ValueError(f"Adapter not found: {name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            info.base_model_path,
            load_in_4bit=True,
            device_map=device,
        )

        model = PeftModel.from_pretrained(base_model, info.adapter_path)

        tokenizer = AutoTokenizer.from_pretrained(info.adapter_path)

        return model, tokenizer

    def export_config(self, name: str, output_path: str = None) -> dict:
        """
        Export adapter configuration

        Args:
            name: Adapter name
            output_path: Optional output file path

        Returns:
            Configuration dictionary
        """
        info = self.get(name)
        if not info:
            raise ValueError(f"Adapter not found: {name}")

        config = {
            "name": info.name,
            "model_name": info.model_name,
            "model_type": info.model_type,
            "base_model_path": info.base_model_path,
            "adapter_path": info.adapter_path,
            "lora_config": {
                "r": info.lora_rank,
                "alpha": info.lora_alpha,
            },
            "task_type": info.task_type,
            "dataset": info.dataset,
            "created_at": info.created_at,
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        return config

    def import_config(self, config_path: str, target_name: str = None):
        """
        Import adapter configuration

        Args:
            config_path: Path to configuration file
            target_name: Target name for the adapter (defaults to config name)
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        name = target_name or config["name"]

        self.register(
            name=name,
            model_name=config["model_name"],
            model_type=config["model_type"],
            adapter_path=config["adapter_path"],
            base_model_path=config["base_model_path"],
            lora_rank=config["lora_config"]["r"],
            lora_alpha=config["lora_config"]["alpha"],
            task_type=config["task_type"],
            dataset=config["dataset"],
        )

    def get_statistics(self) -> dict:
        """Get registry statistics"""
        stats = {
            "total_adapters": len(self.adapters),
            "by_model_type": {},
            "by_task_type": {},
            "created_timeline": [],
        }

        for adapter in self.adapters.values():
            model_type = adapter.model_type
            task_type = adapter.task_type

            stats["by_model_type"][model_type] = (
                stats["by_model_type"].get(model_type, 0) + 1
            )
            stats["by_task_type"][task_type] = (
                stats["by_task_type"].get(task_type, 0) + 1
            )
            stats["created_timeline"].append(adapter.created_at)

        return stats


class BatchAdapterOperations:
    """Batch operations on adapters"""

    def __init__(self, registry: AdapterRegistry):
        self.registry = registry

    def merge_all(
        self,
        output_dir: str,
        base_model_path: str,
        strategy: str = "average",
        weights: List[float] = None,
    ) -> Dict[str, str]:
        """
        Merge all compatible adapters

        Args:
            output_dir: Output directory
            base_model_path: Base model path
            strategy: Merge strategy (average, weighted, task_vector)
            weights: Weights for weighted merge

        Returns:
            Dictionary mapping adapter names to merged paths
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        os.makedirs(output_dir, exist_ok=True)
        merged_paths = {}

        compatible = self.registry.find_compatible(base_model_path)

        if not compatible:
            raise ValueError("No compatible adapters found")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            load_in_4bit=False,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        for adapter_info in compatible:
            adapter_model = PeftModel.from_pretrained(
                base_model,
                adapter_info.adapter_path,
            )

            merged_model = adapter_model.merge_and_unload()

            merged_path = os.path.join(output_dir, adapter_info.name)
            merged_model.save_pretrained(merged_path)

            tokenizer = AutoTokenizer.from_pretrained(adapter_info.adapter_path)
            tokenizer.save_pretrained(merged_path)

            merged_paths[adapter_info.name] = merged_path

        return merged_paths

    def batch_evaluate(
        self,
        base_model_path: str,
        adapters: List[str],
        eval_fn,
    ) -> Dict[str, dict]:
        """
        Batch evaluate multiple adapters

        Args:
            base_model_path: Base model path
            adapters: List of adapter names
            eval_fn: Evaluation function

        Returns:
            Dictionary mapping adapter names to metrics
        """
        import torch

        results = {}

        for adapter_name in adapters:
            info = self.registry.get(adapter_name)
            if not info or info.base_model_path != base_model_path:
                continue

            try:
                model, tokenizer = self.registry.load_adapter_model(adapter_name)
                metrics = eval_fn(model, tokenizer)
                results[adapter_name] = metrics
            except Exception as e:
                results[adapter_name] = {"error": str(e)}
            finally:
                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return results

    def batch_export(
        self,
        output_dir: str,
        adapters: List[str] = None,
    ) -> Dict[str, str]:
        """
        Batch export adapter configurations

        Args:
            output_dir: Output directory
            adapters: List of adapter names (None for all)

        Returns:
            Dictionary mapping adapter names to export paths
        """
        os.makedirs(output_dir, exist_ok=True)
        exported = {}

        if adapters is None:
            adapters = list(self.registry.adapters.keys())

        for name in adapters:
            info = self.registry.get(name)
            if not info:
                continue

            output_path = os.path.join(output_dir, f"{name}_config.json")
            self.registry.export_config(name, output_path)
            exported[name] = output_path

        return exported


from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


if __name__ == "__main__":
    registry = AdapterRegistry("./adapter_registry.json")

    print("Adapter Registry Commands:")
    print("  register(name, model_name, model_type, adapter_path, ...)")
    print("  get(name)")
    print("  list_adapters(model_type=None, task_type=None)")
    print("  find_compatible(base_model_path, task_type=None)")
    print("  load_adapter_model(name, device='cuda')")
    print("  export_config(name, output_path=None)")
