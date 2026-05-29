import ollama
import sys


def demo_generate():
    print("=== 文本生成示例 ===")
    response = ollama.generate(
        model="llama3.2", prompt="用Python写一个计算斐波那契数列的函数"
    )
    print("生成的代码：")
    print(response["response"])
    print()


def demo_chat():
    print("=== 对话生成示例 ===")
    messages = [
        {"role": "system", "content": "你是一个专业的Python导师"},
        {"role": "user", "content": "解释什么是装饰器"},
        {"role": "assistant", "content": "装饰器是Python中一种强大的语法糖..."},
        {"role": "user", "content": "给我一个实际的使用例子"},
    ]
    response = ollama.chat(model="llama3.2", messages=messages)
    print("助手的回答：")
    print(response["message"]["content"])
    print()


def demo_streaming():
    print("=== 流式输出示例 ===")
    print("生成中...")
    stream = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "给我讲一个关于人工智能的短故事"}],
        stream=True,
    )

    print("\n故事：")
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n")


def demo_model_management():
    print("=== 模型管理示例 ===")

    print("\n列出所有已下载的模型：")
    models = ollama.list()
    for model in models["models"]:
        print(f"  - {model['name']} ({model['size'] / (1024**3):.2f} GB)")

    print("\n显示当前运行的模型：")
    ps = ollama.ps()
    print(f"  运行中的模型: {ps.get('models', [])}")


def demo_parameters():
    print("=== 参数调整示例 ===")

    messages = [{"role": "user", "content": "写一首关于大海的诗"}]

    print("\n--- 低温设置 (确定性) ---")
    response = ollama.chat(
        model="llama3.2", messages=messages, options={"temperature": 0.2}
    )
    print(response["message"]["content"])

    print("\n--- 高温设置 (创造性) ---")
    response = ollama.chat(
        model="llama3.2", messages=messages, options={"temperature": 1.2}
    )
    print(response["message"]["content"])


if __name__ == "__main__":
    print("Ollama API 使用示例\n")
    print("确保Ollama服务正在运行 (ollama serve)")
    print("确保 llama3.2 模型已下载 (ollama pull llama3.2)\n")

    try:
        demo_generate()
        demo_chat()
        demo_streaming()
        demo_model_management()
        demo_parameters()
    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保：")
        print("1. Ollama已安装并运行")
        print("2. llama3.2模型已下载")
        sys.exit(1)
