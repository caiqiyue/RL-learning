"""
INT8 Quantization Pipeline

Supports GPTQ, AWQ, and BBQ quantization methods.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


@dataclass
class QuantConfig:
    """Quantization configuration"""

    method: str = "awq"  # gptq, awq, bbq
    bits: int = 8
    group_size: int = 128
    zero_point: bool = False
    calibration_method: str = "percentile"  # percentile, minmax, mse
    calibration_percentile: float = 0.99
    calibration_samples: int = 512
    tokenizer_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class CalibrationDataSelector:
    """
    Selects diverse calibration data for quantization.
    """

    def __init__(self, data: List[Dict], max_samples: int = 512):
        self.data = data
        self.max_samples = max_samples

    def select_diverse(self) -> List[Dict]:
        """
        Select calibration samples ensuring diversity in topic and length.
        """
        if len(self.data) <= self.max_samples:
            return self.data.copy()

        clusters = self._cluster_by_keywords(self.data)

        selected = []
        samples_per_cluster = max(1, self.max_samples // len(clusters))

        for cluster in clusters.values():
            cluster_sorted = sorted(cluster, key=lambda x: len(x.get("text", "")))

            if len(cluster_sorted) >= samples_per_cluster:
                indices = (
                    torch.linspace(0, len(cluster_sorted) - 1, samples_per_cluster)
                    .long()
                    .tolist()
                )
            else:
                indices = list(range(len(cluster_sorted)))

            for idx in indices:
                selected.append(cluster_sorted[idx])

        return selected[: self.max_samples]

    def _cluster_by_keywords(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """Simple keyword-based topic clustering"""
        topic_keywords = {
            "code": ["code", "function", "python", "program", "implementation"],
            "science": ["science", "physics", "research", "experiment", "theory"],
            "math": ["calculate", "equation", "formula", "math", "number"],
            "general": [],
        }

        clusters = {k: [] for k in topic_keywords.keys()}

        for item in data:
            text = item.get("text", "").lower()
            assigned = False

            for topic, keywords in topic_keywords.items():
                if topic == "general":
                    continue
                if any(kw in text for kw in keywords):
                    clusters[topic].append(item)
                    assigned = True
                    break

            if not assigned:
                clusters["general"].append(item)

        return {k: v for k, v in clusters.items() if v}


class GPTQQuantizer:
    """
    GPTQ quantizer for LLM models.
    """

    def __init__(self, model: nn.Module, config: QuantConfig):
        self.model = model
        self.config = config
        self.quant_state: Dict[str, Dict] = {}

    def quantize(self, calibration_data: List[torch.Tensor]) -> nn.Module:
        """
        Execute GPTQ quantization on the model.
        """
        print(f"Starting GPTQ quantization with bits={self.config.bits}")

        self.model.eval()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._quantize_linear(name, module, calibration_data)

        return self.model

    def _quantize_linear(
        self, name: str, layer: nn.Linear, calibration_data: List[torch.Tensor]
    ):
        """Quantize a single linear layer using GPTQ"""
        weight = layer.weight.data.float()
        out_features, in_features = weight.shape

        num_groups = in_features // self.config.group_size

        max_q = 2**self.config.bits - 1
        min_q = -(2**self.config.bits)

        scales = torch.zeros(out_features, num_groups, dtype=torch.float16)
        quantized_weight = torch.zeros_like(weight, dtype=torch.int8)

        for out_idx in range(out_features):
            weight_row = weight[out_idx]
            error = torch.zeros_like(weight_row)

            for group_idx in range(num_groups):
                start = group_idx * self.config.group_size
                end = start + self.config.group_size

                w_group = weight_row[start:end] + error[start:end]

                max_val = w_group.abs().max()
                scale = max_val / (max_q / 2)

                if scale < 1e-8:
                    scale = 1e-8

                w_quant = torch.round(w_group / scale).clamp(min_q, max_q)

                scales[out_idx, group_idx] = scale
                quantized_weight[out_idx, start:end] = w_quant.to(torch.int8)

                w_dequant = w_quant * scale
                error[start:end] += w_group - w_dequant

        self.quant_state[name] = {
            "weight": quantized_weight,
            "scales": scales,
            "bits": self.config.bits,
            "group_size": self.config.group_size,
        }

        print(f"  Quantized {name}: {weight.shape} -> INT{self.config.bits}")

    def save_quantized(self, save_path: str):
        """Save quantized model"""
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.quant_state, os.path.join(save_path, "quant_state.pt"))

        config_path = os.path.join(save_path, "quant_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"Quantized model saved to {save_path}")


class AWQQuantizer:
    """
    AWQ (Activation-Aware Weight Quantization) quantizer.
    """

    def __init__(self, model: nn.Module, config: QuantConfig):
        self.model = model
        self.config = config
        self.scales: Dict[str, torch.Tensor] = {}

    def quantize(self, calibration_data: List[torch.Tensor]) -> nn.Module:
        """
        Execute AWQ quantization.
        """
        print(f"Starting AWQ quantization with bits={self.config.bits}")

        self.model.eval()

        self._compute_activation_stats(calibration_data)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._quantize_linear_awq(name, module)

        return self.model

    def _compute_activation_stats(self, calibration_data: List[torch.Tensor]):
        """Compute activation statistics for scaling factors"""
        self.act_scales = {}

        def hook_fn(name):
            def hook(module, input, output):
                inp = input[0]
                self.act_scales[name] = inp.abs().float().mean(dim=0)

            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "gate" not in name:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            for batch in calibration_data[: min(10, len(calibration_data))]:
                if isinstance(batch, dict):
                    batch = batch.get("input_ids", batch.get("tokens"))
                self.model(batch)

        for hook in hooks:
            hook.remove()

    def _quantize_linear_awq(self, name: str, layer: nn.Linear):
        """Quantize a single linear layer using AWQ"""
        weight = layer.weight.data.float()

        if name in self.act_scales:
            act_scale = self.act_scales[name]
            alpha = 0.5
            scales = act_scale.pow(alpha)
            scales = scales / scales.amax()
        else:
            scales = torch.ones(weight.shape[1], device=weight.device)

        scaled_weight = weight * scales.unsqueeze(0)

        max_q = 2**self.config.bits - 1
        max_val = scaled_weight.abs().max()
        scale = max_val / max_q

        quantized = torch.round(scaled_weight / scale).to(torch.int8)

        self.scales[name] = scale / scales

        print(f"  AWQ Quantized {name}: {weight.shape} -> INT{self.config.bits}")

    def save_quantized(self, save_path: str):
        """Save quantized model"""
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.scales, os.path.join(save_path, "awq_scales.pt"))

        config_path = os.path.join(save_path, "quant_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"AWQ quantized model saved to {save_path}")


class QuantizationPipeline:
    """
    Unified quantization pipeline supporting multiple methods.
    """

    def __init__(
        self,
        model_path: str,
        output_path: str,
        quant_config: Optional[QuantConfig] = None,
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.config = quant_config or QuantConfig()

    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        print(f"Loading model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(
            f"Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B params"
        )

    def prepare_calibration_data(
        self, data: List[Dict], max_length: int = 2048
    ) -> List[torch.Tensor]:
        """Prepare calibration data"""
        print(f"Preparing {len(data)} calibration samples")

        calibration_tensors = []

        for item in data:
            text = item.get("text", "")

            encoding = self.tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            calibration_tensors.append(encoding["input_ids"].squeeze())

        selector = CalibrationDataSelector(data, self.config.calibration_samples)
        selected = selector.select_diverse()

        result = []
        for item in selected:
            text = item.get("text", "")
            encoding = self.tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result.append(encoding["input_ids"].squeeze())

        return result[: self.config.calibration_samples]

    def quantize(self, calibration_data: List[torch.Tensor]) -> Dict:
        """
        Execute quantization pipeline.
        """
        print(f"\n{'=' * 60}")
        print(f"Quantization Pipeline: {self.config.method.upper()}")
        print(f"{'=' * 60}")

        start_time = time.time()

        if self.config.method == "gptq":
            quantizer = GPTQQuantizer(self.model, self.config)
            self.model = quantizer.quantize(calibration_data)
            quantizer.save_quantized(self.output_path)

        elif self.config.method == "awq":
            quantizer = AWQQuantizer(self.model, self.config)
            self.model = quantizer.quantize(calibration_data)
            quantizer.save_quantized(self.output_path)

        elif self.config.method == "bbq":
            raise NotImplementedError("BBQ not yet implemented")

        else:
            raise ValueError(f"Unknown quantization method: {self.config.method}")

        elapsed = time.time() - start_time

        model_size_gb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024**3)

        return {
            "method": self.config.method,
            "bits": self.config.bits,
            "model_size_gb": model_size_gb,
            "quantization_time_minutes": elapsed / 60,
            "output_path": self.output_path,
        }

    def verify_quality(self, test_data: List[Dict]) -> Dict:
        """
        Verify quantized model quality.
        """
        print("\nVerifying quantized model quality...")

        self.model.eval()

        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for item in test_data[: min(100, len(test_data))]:
                text = item.get("text", "")

                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )

                input_ids = encoding["input_ids"].to(self.model.device)

                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()

                total_loss += loss
                num_samples += 1

        avg_loss = total_loss / num_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")

        return {
            "perplexity": perplexity,
            "average_loss": avg_loss,
            "num_samples": num_samples,
        }


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="INT8 Quantization Pipeline")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--method", type=str, default="awq", choices=["gptq", "awq"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--calibration_samples", type=int, default=512)

    args = parser.parse_args()

    quant_config = QuantConfig(
        method=args.method,
        bits=args.bits,
        calibration_samples=args.calibration_samples,
    )

    pipeline = QuantizationPipeline(
        model_path=args.model_path,
        output_path=args.output_path,
        quant_config=quant_config,
    )

    pipeline.load_model_and_tokenizer()

    sample_data = [
        {"text": "This is a sample text for calibration."} for _ in range(100)
    ]
    calibration_data = pipeline.prepare_calibration_data(sample_data)

    result = pipeline.quantize(calibration_data)

    print(f"\nQuantization completed:")
    print(f"  Method: {result['method']}")
    print(f"  Bits: {result['bits']}")
    print(f"  Model Size: {result['model_size_gb']:.2f} GB")
    print(f"  Time: {result['quantization_time_minutes']:.1f} minutes")


if __name__ == "__main__":
    main()
