# 14.2 Ollama本地部署与API调用

## 课程概述

本课程介绍Ollama——一个专为本地运行大型语言模型而设计的推理引擎。学生将学习如何在个人电脑上部署、管理和使用开源大语言模型，实现完全离线的人工智能应用开发。

## 学习目标

- 理解Ollama的核心概念和架构
- 掌握Ollama的安装和基本配置
- 学会使用Modelfile创建自定义模型
- 熟练调用Ollama的REST API和Python SDK
- 能够将Ollama集成到LangChain和LlamaIndex中
- 了解模型量化和性能优化的技巧

## 前置知识

- Python编程基础
- 了解大语言模型的基本概念
- 基本的命令行操作经验

---

## 1. Ollama概述

### 1.1 什么是Ollama

Ollama是一个开源的本地大语言模型推理引擎，允许用户在本地机器上运行、配置和自定义开源LLM。它的设计理念是 simplicity first——让任何人都能轻松地在本地部署和使用强大的AI模型，而无需处理复杂的配置或云服务。

Ollama将模型权重、配置和工具链打包成一个独立的格式（Modelfile），用户可以通过简单的命令来管理和运行模型。与传统的本地部署方案相比，Ollama大大降低了技术门槛，同时保持了良好的性能和灵活性。

### 1.2 为什么选择Ollama

选择Ollama有以下几个核心优势：

**隐私保护**：所有数据都在本地处理，不会有任何信息离开你的机器。这对于处理敏感数据的企业应用或需要符合数据合规要求的场景特别重要。

**零API成本**：使用本地模型无需支付任何API调用费用。虽然需要投入硬件资源，但长期来看可以显著降低使用成本，特别是对于高频调用的应用场景。

**无网络依赖**：即使没有互联网连接，也能正常运行模型。这对于需要在离线环境中工作或对稳定性有严格要求的场景非常关键。

**简单易用**：通过几条简单的命令就能启动模型，复杂的配置都被隐藏在简洁的接口之下。

**模型多样性**：支持众多开源模型，包括Llama、DeepSeek、Mistral、Qwen、Phi、Gemma等热门模型，可以根据任务需求选择最合适的模型。

### 1.3 支持的模型

OllamaModel Library包含数百个预配置模型，涵盖多种架构和用途：

| 模型系列 | 代表模型 | 适用场景 |
|---------|---------|---------|
| Llama | llama3.2, llama3.1, llama3 | 通用对话、代码生成 |
| DeepSeek | deepseek-coder, deepseek-llm | 代码生成、数学推理 |
| Mistral | mistral, mixtral | 指令遵循、对话 |
| Qwen | qwen2.5, qwen2.5-coder | 中文对话、代码 |
| Phi | phi3.5, phi3 | 轻量级推理、移动端 |
| Gemma | gemma2, gemma2b | 文本生成、摘要 |

每个模型都有多个尺寸版本（如7B、8B、14B、70B参数），用户可以根据硬件条件选择合适的版本。

---

## 2. 安装与配置

### 2.1 下载安装

Ollama支持macOS、Linux和Windows三大平台。

**macOS安装**：
下载macOS版安装包后，按照标准应用安装流程完成安装。安装完成后，Ollama会在后台自动运行，你可以在菜单栏看到Ollama图标。

**Linux安装**：
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows（预览版）**：
直接从官网下载Windows安装包，双击运行安装程序。Windows版目前处于预览阶段，但功能已经相对完善。

**验证安装**：
安装完成后，打开终端或命令提示符，输入以下命令验证安装：

```bash
ollama --version
```

### 2.2 CLI基础命令

Ollama的命令行界面是最常用的交互方式。以下是核心命令：

**拉取模型**：
```bash
ollama pull llama3.2
```

首次运行模型时，Ollama会自动从模型库下载模型文件。模型文件较大（几GB到几十GB），下载时间取决于网络状况。

**运行模型**：
```bash
ollama run llama3.2
```

这会启动一个交互式对话界面，你可以直接输入问题与模型交流。输入 `/bye` 或 `exit` 退出对话。

**查看已下载的模型**：
```bash
ollama list
```

**查看正在运行的模型**：
```bash
ollama ps
```

**删除模型**：
```bash
ollama rm llama3.2
```

### 2.3 模型库与自定义模型

Ollama的模型库网址为 [ollama.com/library](https://ollama.com/library)，你可以在这里浏览所有可用的模型，了解每个模型的参数数量、尺寸和用途。

除了使用预配置模型，Ollama还支持通过Modelfile创建自定义模型。Modelfile是一种声明式的配置文件，允许你指定基础模型、参数配置、系统提示和模型权重等。这种灵活性使得用户可以根据特定需求定制模型的的行为。

---

## 3. 创建自定义模型

### 3.1 Modelfile语法

Modelfile是Ollama自定义模型的核心文件格式。它的语法简洁直观，类似于Dockerfile的设计理念。下面是一个基本的Modelfile示例：

```dockerfile
FROM llama3.2

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER context_window 8192

SYSTEM """
你是一位专业的中文技术文档写作助手。
请用简洁、专业的语言回答用户的问题。
"""
```

`FROM` 指令指定基础模型，可以是Ollama模型库中的任何模型，也可以是本地已下载的模型。

`PARAMETER` 指令用于设置模型的运行参数，如temperature、top_p、context_window等。这些参数会影响模型生成的风格和质量。

`SYSTEM` 指令用于设置系统提示词，这相当于给模型设定了一个固定的角色或行为准则。

### 3.2 导入GGUF格式

如果你有GGUF格式的模型权重文件（从Hugging Face等平台下载），可以通过Modelfile导入Ollama使用。首先确保GGUF文件在本地，然后创建Modelfile：

```dockerfile
FROM ./my-model.gguf

PARAMETER temperature 0.8
PARAMETER num_ctx 4096

SYSTEM """
你是一个专门用于代码审查的AI助手。
"""
```

将Modelfile和GGUF文件放在同一目录下，然后运行：

```bash
ollama create my-custom-model -f Modelfile
```

这会基于你的GGUF文件创建一个名为 `my-custom-model` 的新模型。

### 3.3 常用参数配置

以下是Modelfile中最常用的参数：

| 参数 | 说明 | 典型值 |
|------|------|--------|
| temperature | 控制输出的随机性（0-2） | 0.7 |
| top_p | 核采样阈值 | 0.9 |
| num_ctx | 上下文窗口大小 | 4096-8192 |
| top_k | 采样候选词数量 | 40 |
| repeat_penalty | 重复惩罚 | 1.1 |
| mirostat | Mirostat采样模式 | 0-2 |

调整这些参数可以显著改变模型的输出风格。例如，较低的温度会产生更确定性的回答，适合需要精确答案的任务；较高的温度则会产生更有创意的回答，适合需要多样性的任务。

### 3.4 创建你的第一个自定义模型

让我们从头创建一个针对中文对话优化的模型：

```dockerfile
FROM qwen2.5:7b

PARAMETER temperature 0.8
PARAMETER top_p 0.95
PARAMETER num_ctx 8192

SYSTEM """
你是一位友善、博学的中文助手。
- 回答总是用中文
- 如果不确定，明确说出来
- 给出解释时用简单的语言
- 适当使用例子帮助理解
"""

MESSAGE USER 你好，请介绍一下你自己
MESSAGE ASSISTANT 你好！我是一个AI助手，很高兴认识你。我可以帮你回答问题、解释概念、编写代码，或者只是聊聊天。有什么我可以帮你的吗？
```

创建完成后运行：

```bash
ollama create my-chatbot -f Modelfile
ollama run my-chatbot
```

---

## 4. API调用

### 4.1 REST API概览

Ollama在本地启动一个REST API服务器，默认地址是 `http://localhost:11434`。API采用OpenAI兼容的格式设计，因此可以无缝替换使用OpenAI API的代码。

Ollama服务器在首次调用时自动启动，你也可以手动启动：

```bash
ollama serve
```

### 4.2 生成接口（Completions）

生成接口用于续写文本或生成内容，是最基础的API：

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "解释什么是大语言模型",
  "stream": false
}'
```

响应格式如下：

```json
{
  "model": "llama3.2",
  "response": "大语言模型（Large Language Model）是一种...",
  "done": true,
  "context": [1, 2, 3, ...],
  "total_duration": 5123456789,
  "load_duration": 1234567890,
  "prompt_eval_count": 12
}
```

### 4.3 聊天接口（Chat Completions）

聊天接口更适合对话场景，支持多轮对话：

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
    {"role": "user", "content": "解释一下什么是Transformer架构"}
  ],
  "stream": false
}'
```

### 4.4 Python SDK使用

Ollama提供Python SDK简化API调用：

```python
import ollama

response = ollama.chat(
    model='llama3.2',
    messages=[
        {'role': 'user', 'content': '解释什么是RAG'},
    ]
)
print(response['message']['content'])
```

生成接口的使用方式：

```python
import ollama

response = ollama.generate(
    model='llama3.2',
    prompt='写一个Python函数来计算斐波那契数列'
)
print(response['response'])
```

### 4.5 流式响应

对于需要实时反馈的应用，Ollama支持流式响应：

```python
import ollama

stream = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': '给我讲一个故事'}],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
print()
```

REST API的流式调用：

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "写一首关于春天的诗"}],
  "stream": true
}'
```

---

## 5. 应用集成

### 5.1 LangChain集成

LangChain是当前最流行的LLM应用框架之一，Ollama提供了原生的LangChain集成支持。通过LangChain，你可以轻松构建复杂的AI应用，如RAG系统、代理系统等。

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

response = llm.invoke("解释什么是向量数据库")
print(response.content)
```

LangChain还支持流式输出、批量处理等高级功能：

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

# 批量处理
responses = llm.batch([
    "解释什么是机器学习",
    "解释什么是深度学习",
    "解释什么是强化学习"
])

for response in responses:
    print(f"问题: {response.content}")
```

### 5.2 LlamaIndex集成

LlamaIndex是另一个强大的数据索引和检索框架，特别适合构建RAG应用：

```python
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.2", request_timeout=120)

response = llm.complete("解释什么是知识图谱")
print(response.text)
```

LlamaIndex的对话功能：

```python
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.2")

# 对话式消息
messages = [
    {"role": "system", "content": "你是一个编程助手"},
    {"role": "user", "content": "Python中如何定义一个类？"}
]

response = llm.chat(messages)
print(response.text)
```

### 5.3 构建简单RAG管道

检索增强生成（RAG）是当前最流行的LLM应用架构之一。下面是一个使用Ollama和LlamaIndex构建的简单RAG系统：

整个流程包括以下步骤：

1. **文档加载与分块**：将原始文档分割成小的文本块
2. **向量嵌入**：将文本块转换为向量表示
3. **向量存储**：将嵌入存储到向量数据库
4. **检索**：根据用户查询找到最相关的文档块
5. **生成**：将检索结果和用户问题组合，调用LLM生成答案

这个架构的优势在于，模型可以基于实际的文档内容回答问题，而不是仅依赖训练数据中的知识。这对于构建领域知识库、企业文档问答等应用特别有用。

---

## 6. 性能优化

### 6.1 GPU内存需求

不同尺寸的模型对GPU内存有不同的要求。以下是参考数值（以FP16精度为例）：

| 模型规模 | 最小GPU内存 | 推荐GPU内存 |
|---------|-----------|-----------|
| 7B | 14GB | 16GB+ |
| 13B | 26GB | 32GB+ |
| 33B | 66GB | 80GB+ |
| 70B | 140GB | 160GB+ |

如果你没有足够的GPU内存，可以考虑使用量化模型来降低内存需求。

### 6.2 CPU回退方案

在没有GPU的情况下，Ollama也可以在CPU上运行模型，但速度会显著慢于GPU。以下是一些CPU运行的优化建议：

- 关闭GPU偏好，强制使用CPU：`OLLAMA_GPU_OVERRIDE=cpu`
- 减少模型尺寸，选择更小的模型
- 减少上下文窗口大小，降低内存占用
- 使用量化模型，减少计算量

### 6.3 模型量化

模型量化通过降低模型权重的精度来减少内存占用和计算需求。Ollama支持多种量化级别：

| 量化类型 | 内存占用 | 质量损失 | 推荐场景 |
|---------|---------|---------|---------|
| Q8_0 | 约75% | 极小 | 质量优先 |
| Q4_K_M | 约50% | 较小 | 平衡之选 |
| Q4_0 | 约50% | 适中 | 内存受限 |
| Q3_K_M | 约37% | 可见 | 极度受限 |

Ollama的默认模型通常已经过量化处理。如果需要使用特定量化版本，可以在模型库中查找或使用Modelfile指定。

### 6.4 性能监控

可以使用 `ollama ps` 命令监控当前运行模型的资源使用情况：

```bash
ollama ps
```

输出将显示每个模型的内存占用、GPU使用率和处理速度等信息。如果发现性能问题，可以据此调整模型配置或硬件资源。

---

## 总结

本课程涵盖了Ollama本地部署和API调用的核心知识。通过学习，你应该能够：

- 在本地机器上安装和配置Ollama
- 使用命令行工具管理模型
- 创建自定义模型满足特定需求
- 通过REST API和Python SDK调用模型
- 将Ollama集成到LangChain和LlamaIndex应用中
- 了解模型量化和性能优化的基本方法

Ollama代表了本地LLM运行的一种趋势——让强大的AI能力每个人都能触手可及。随着开源模型的不断进步和硬件成本的持续下降，本地AI应用将会变得越来越普及。

---

## 延伸阅读

- [Ollama官方文档](https://github.com/ollama/ollama) - 官方GitHub仓库，包含最新文档和更新
- [Ollama Model Library](https://ollama.com/library) - 官方模型库，浏览所有可用模型
- [OpenAI兼容API文档](https://github.com/ollama/ollama/blob/main/docs/api.md) - API详细说明
- [Modelfile参考](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) - Modelfile语法完整参考
- [LangChain Ollama集成](https://python.langchain.com/docs/integrations/chat/ollama) - LangChain官方集成指南
- [LlamaIndex Ollama集成](https://docs.llamaindex.ai/en/stable/examples/llm/ollama.html) - LlamaIndex官方集成指南

---

## 复习题

1. Ollama相比云端API服务有哪些主要优势？请列举至少三个。

2. Modelfile中的三个核心指令（FROM、PARAMETER、SYSTEM）分别起什么作用？

3. 解释模型量化的概念。为什么Q4_K_M量化比FP16精度更节省内存？

4. 如何使用Ollama Python SDK实现流式输出？请写出关键代码。

5. 在RAG架构中，检索（Retrieval）和生成（Generation）各自的职责是什么？它们如何协作？