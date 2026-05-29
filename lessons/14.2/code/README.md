# 14.2 Ollama本地部署 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：ollama, openai |
| `modelfile_example` | Modelfile示例：中文对话优化、系统提示、参数配置 |
| `api_client.py` | API客户端：生成/聊天/流式、管理端点、参数调整 |
| `simple_rag.py` | 简单RAG：文档加载、嵌入、检索、生成 |

## 运行方式

```bash
# 安装Ollama后
ollama run llama-2-7b
# 或使用Python API
python api_client.py --prompt "解释量子计算" --model llama-2-7b
python simple_rag.py --query "如何学习深度学习"
```

## 核心概念

- Ollama: 本地LLM推理引擎，隐私友好，无需API费用
- Modelfile: 定义模型配置、温度、top_p、系统提示
- REST API: OpenAI兼容接口，易于集成