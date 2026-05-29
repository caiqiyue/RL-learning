#!/usr/bin/env python3
"""
Export Unsloth Model to HuggingFace Format
将Unsloth训练的LoRA模型导出为标准HuggingFace格式
支持多种保存方式：纯LoRA adapter、合并的16bit模型、合并的4bit模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import argparse


def export_lora_adapter(model_path, output_path, tokenizer_only=False):
    """
    导出为纯LoRA adapter格式
    保留LoRA权重和配置文件，不包含基础模型
    """
    print(f"\n{'=' * 50}")
    print(f"Exporting LoRA adapter to: {output_path}")
    print(f"Model path: {model_path}")

    os.makedirs(output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    print("Tokenizer saved.")

    if not tokenizer_only:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        peft_model = PeftModel.from_pretrained(base_model, model_path)
        peft_model.save_pretrained(output_path)

        print(f"LoRA adapter saved to: {output_path}")

        del base_model, peft_model
        torch.cuda.empty_cache()


def export_merged_16bit(model_path, output_path, base_model_name=None):
    """
    导出为合并的16bit模型（LoRA权重合并到基础模型）
    生成完整的FP16模型，可直接用于推理
    """
    print(f"\n{'=' * 50}")
    print(f"Exporting merged 16bit model to: {output_path}")
    print(f"Adapter path: {model_path}")

    os.makedirs(output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if base_model_name:
        base_model_path = base_model_name
    else:
        base_model_path = model_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    peft_model = PeftModel.from_pretrained(base_model, model_path)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged 16bit model saved to: {output_path}")
    print(
        f"Model size: {sum(p.numel() for p in merged_model.parameters()) / 1e9:.2f}B parameters"
    )

    del base_model, peft_model, merged_model
    torch.cuda.empty_cache()


def export_merged_4bit(model_path, output_path, base_model_name=None):
    """
    导出为合并的4bit模型（使用GPTQ量化）
    生成压缩后的模型，大幅减少模型大小
    """
    print(f"\n{'=' * 50}")
    print(f"Exporting merged 4bit model to: {output_path}")

    try:
        from auto_gptq import AutoGPTQ, BaseQuantizeConfig
    except ImportError:
        print("ERROR: auto-gptq not installed. Install with: pip install auto-gptq")
        return

    os.makedirs(output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if base_model_name:
        base_model_path = base_model_name
    else:
        base_model_path = model_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    peft_model = PeftModel.from_pretrained(base_model, model_path)
    merged_model = peft_model.merge_and_unload()

    quantization_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
    )

    print("Quantizing to 4bit (this may take a while)...")
    quantized_model = AutoGPTQ.quantize_model(
        merged_model,
        quantization_config,
    )

    quantized_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged 4bit model saved to: {output_path}")

    del base_model, peft_model, merged_model, quantized_model
    torch.cuda.empty_cache()


def export_unsloth_merged(model_path, output_path, save_method="merged_16bit"):
    """
    使用Unsloth的save_pretrained_merged导出模型

    Args:
        model_path: 原始模型路径或Unsloth adapter路径
        output_path: 输出目录
        save_method: 保存方式
            - "lora": 仅保存LoRA权重
            - "merged_16bit": 合并到FP16
            - "merged_4bit": 合并并量化到4bit
    """
    from unsloth import FastLanguageModel

    print(f"\n{'=' * 50}")
    print(f"Exporting using Unsloth method: {save_method}")
    print(f"From: {model_path}")
    print(f"To: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    if save_method == "lora":
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    else:
        model.save_pretrained_merged(output_path, tokenizer, save_method=save_method)

    print(f"Model exported to: {output_path}")

    del model, tokenizer
    torch.cuda.empty_cache()


def verify_exported_model(model_path):
    """
    验证导出的模型可以正常加载

    Returns:
        bool: 验证是否通过
    """
    print(f"\nVerifying exported model at: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        print(f"  Model loaded: {model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        input_ids = tokenizer("Hello, world!", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**input_ids, max_new_tokens=10)
        print(f"  Generation test passed!")

        del model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def push_to_huggingface(model_path, repo_id, token=None):
    """
    将导出的模型推送到HuggingFace Hub

    Args:
        model_path: 本地模型路径
        repo_id: HuggingFace仓库ID (如 "user/model-name")
        token: HuggingFace访问令牌
    """
    print(f"\n{'=' * 50}")
    print(f"Pushing model to HuggingFace Hub: {repo_id}")

    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi()

        if token:
            api.token = token

        try:
            create_repo(repo_id, exist_ok=True)
            print(f"Repository created/accessed: {repo_id}")
        except Exception as e:
            print(f"Warning: Could not create repo: {e}")

        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
        )

        print(f"Model pushed to: https://huggingface.co/{repo_id}")

    except ImportError:
        print(
            "ERROR: huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
    except Exception as e:
        print(f"Push failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Unsloth model to HuggingFace format"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to Unsloth model or adapter"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for exported model",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="merged_16bit",
        choices=["lora", "merged_16bit", "merged_4bit"],
        help="Export method",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model path (if different from adapter path)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify the exported model"
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Push to HuggingFace Hub (specify repo_id)",
    )
    parser.add_argument(
        "--hub_token", type=str, default=None, help="HuggingFace API token"
    )

    args = parser.parse_args()

    if args.method == "merged_4bit":
        print("Note: 4bit export requires auto-gptq package")
        print("Install with: pip install auto-gptq")

    try:
        if args.method == "lora":
            export_lora_adapter(args.model_path, args.output_path)
        elif args.method == "merged_16bit":
            if args.base_model:
                export_merged_16bit(args.model_path, args.output_path, args.base_model)
            else:
                export_merged_16bit(args.model_path, args.output_path)
        elif args.method == "merged_4bit":
            if args.base_model:
                export_merged_4bit(args.model_path, args.output_path, args.base_model)
            else:
                export_merged_4bit(args.model_path, args.output_path)

        if args.verify:
            if verify_exported_model(args.output_path):
                print("\nModel verification PASSED")
            else:
                print("\nModel verification FAILED")

        if args.push_to_hub:
            push_to_huggingface(args.output_path, args.push_to_hub, args.hub_token)

        print(f"\nExport completed successfully!")
        print(f"Output: {args.output_path}")

    except Exception as e:
        print(f"\nExport failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
