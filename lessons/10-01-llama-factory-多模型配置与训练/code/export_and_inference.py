#!/usr/bin/env python3
"""
LLaMA-Factory Export and Inference Script
LLaMA-Factory 模型导出与推理

功能：
1. 将训练好的 LoRA 权重合并到基座模型
2. 导出为 HuggingFace 格式
3. 使用合并后的模型进行推理
"""

import os
import json
import argparse
from typing import Optional, List, Dict


def export_model(
    model_name: str,
    checkpoint_dir: str,
    output_dir: str,
    merge_lora: bool = True,
    save_safetensors: bool = True,
    use_flash_attention: bool = True,
) -> None:
    """
    导出并合并 LoRA 模型

    Args:
        model_name: 模型名称（注册在 LLaMA-Factory 中的名字）
        checkpoint_dir: 训练 checkpoint 路径
        output_dir: 输出路径
        merge_lora: 是否合并 LoRA 权重
        save_safetensors: 是否使用 safetensors 格式
        use_flash_attention: 是否使用 Flash Attention
    """
    print("=" * 50)
    print("LLaMA-Factory Model Export")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Merge LoRA: {merge_lora}")
    print(f"Safetensors: {save_safetensors}")
    print("=" * 50)

    export_cmd = f"""
llamafactory-cli export \\
    --config examples/export.yaml \\
    --merge_lora {str(merge_lora).lower()} \\
    --output_dir {output_dir}
"""

    if save_safetensors:
        export_cmd += " --save_safetensors"

    print(f"Running: {export_cmd}")
    os.system(export_cmd)
    print("Export completed!")


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """
    加载模型和分词器

    Args:
        model_path: 模型路径
        device: 设备 ('cuda' or 'cpu')

    Returns:
        model, tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    print("Model loaded successfully!")
    return model, tokenizer


def format_prompt(prompt: str, tokenizer, system_prompt: Optional[str] = None) -> str:
    """
    格式化提示词为对话格式

    Args:
        prompt: 用户输入
        tokenizer: 分词器
        system_prompt: 系统提示词

    Returns:
        格式化后的文本
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return text


def inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
) -> str:
    """
    使用模型进行推理

    Args:
        model: 语言模型
        tokenizer: 分词器
        prompt: 输入提示词
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: nucleus 采样参数
        system_prompt: 系统提示词

    Returns:
        生成的文本
    """
    text = format_prompt(prompt, tokenizer, system_prompt)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|im_start|>" in response:
        response = response.split("<|im_start|>")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]

    return response.strip()


def batch_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """
    批量推理

    Args:
        model: 语言模型
        tokenizer: 分词器
        prompts: 提示词列表
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        system_prompt: 系统提示词

    Returns:
        生成文本列表
    """
    results = []

    for i, prompt in enumerate(prompts):
        print(f"Processing {i + 1}/{len(prompts)}...")
        response = inference(
            model, tokenizer, prompt, max_new_tokens, temperature, system_prompt
        )
        results.append(response)

    return results


def interactive_chat(model, tokenizer, system_prompt: str = "你是一个有帮助的AI助手。"):
    """
    交互式对话

    Args:
        model: 语言模型
        tokenizer: 分词器
        system_prompt: 系统提示词
    """
    print("\n" + "=" * 50)
    print("Interactive Chat (type 'quit' to exit)")
    print("=" * 50 + "\n")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=0.7, do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|im_start|>" in response:
            parts = response.split("<|im_start|>")
            response = parts[-1]

        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        assistant_msg = response.strip()
        messages.append({"role": "assistant", "content": assistant_msg})

        print(f"Assistant: {assistant_msg}\n")


def main():
    parser = argparse.ArgumentParser(description="LLaMA-Factory Export and Inference")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    export_parser = subparsers.add_parser("export", help="Export and merge LoRA model")
    export_parser.add_argument("--model_name", type=str, default="LLaMA-3-8B-Instruct")
    export_parser.add_argument("--checkpoint_dir", type=str, required=True)
    export_parser.add_argument("--output_dir", type=str, required=True)
    export_parser.add_argument("--merge_lora", type=bool, default=True)
    export_parser.add_argument("--save_safetensors", type=bool, default=True)

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--model_path", type=str, required=True, help="Path to merged model"
    )
    infer_parser.add_argument("--prompt", type=str, help="Single prompt for inference")
    infer_parser.add_argument(
        "--interactive", action="store_true", help="Interactive chat mode"
    )
    infer_parser.add_argument("--max_tokens", type=int, default=512)
    infer_parser.add_argument("--temperature", type=float, default=0.7)
    infer_parser.add_argument("--system", type=str, default="你是一个有帮助的AI助手。")

    args = parser.parse_args()

    if args.command == "export":
        export_model(
            model_name=args.model_name,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            merge_lora=args.merge_lora,
            save_safetensors=args.save_safetensors,
        )

    elif args.command == "infer":
        model, tokenizer = load_model_and_tokenizer(args.model_path)

        if args.interactive:
            interactive_chat(model, tokenizer, args.system)
        elif args.prompt:
            response = inference(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                system_prompt=args.system,
            )
            print("\n" + "=" * 50)
            print("Prompt:", args.prompt)
            print("=" * 50)
            print("Response:", response)
            print("=" * 50)
        else:
            print("Please provide --prompt or use --interactive")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
