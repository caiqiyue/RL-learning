import ollama
import os


class SimpleRAG:
    def __init__(self, model="llama3.2"):
        self.model = model
        self.documents = []
        self.embeddings = []
        print(f"初始化 SimpleRAG，使用模型: {model}")

    def load_documents(self, docs):
        self.documents = docs
        print(f"加载了 {len(docs)} 个文档")

    def ingest(self):
        print("正在生成文档嵌入...")
        for i, doc in enumerate(self.documents):
            response = ollama.embeddings(model="nomic-embed-text", prompt=doc)
            self.embeddings.append(response["embedding"])
            print(f"  文档 {i + 1}/{len(self.documents)} 已嵌入")
        print("嵌入完成\n")

    def retrieve(self, query, top_k=2):
        query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)[
            "embedding"
        ]

        scores = []
        for doc_embedding in self.embeddings:
            score = self._cosine_similarity(query_embedding, doc_embedding)
            scores.append(score)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [(self.documents[i], scores[i]) for i in top_indices]

    def _cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)

    def query(self, question):
        print(f"\n问题: {question}")
        print("-" * 50)

        relevant_docs = self.retrieve(question)
        print(f"检索到 {len(relevant_docs)} 个相关文档:")
        for i, (doc, score) in enumerate(relevant_docs):
            print(f"  [{i + 1}] (相似度: {score:.4f})")
            print(f"      {doc[:100]}...")

        context = "\n\n".join([doc for doc, _ in relevant_docs])

        prompt = f"""基于以下参考资料回答问题。如果资料中没有相关信息，请说明"资料中没有提供相关信息"。

参考资料：
{context}

问题：{question}

请给出准确、完整的回答："""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个基于给定资料回答问题的助手。必须仅使用提供的参考资料来回答，不要编造信息。",
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )

        print(f"\n回答:\n{response['message']['content']}")
        return response["message"]["content"]


def demo():
    print("=" * 60)
    print("Simple RAG 演示")
    print("=" * 60)
    print("\n需要先安装嵌入模型: ollama pull nomic-embed-text\n")

    docs = [
        "Ollama是一个开源的大语言模型推理引擎，允许用户在本地运行各种开源LLM，如Llama、DeepSeek、Mistral等。",
        "Ollama的主要优势包括：隐私性（数据本地处理）、零API成本、无网络依赖、简单易用。",
        "使用Modelfile可以自定义模型配置，包括基础模型选择、参数调整、系统提示设置等。",
        "Ollama提供REST API，默认端口11434，支持生成和聊天两种接口，兼容OpenAI API格式。",
        "支持的模型量化级别包括Q8_0、Q4_K_M、Q3_K_M等，可显著降低内存占用。",
    ]

    rag = SimpleRAG(model="llama3.2")
    rag.load_documents(docs)
    rag.ingest()

    questions = [
        "Ollama有哪些主要优势？",
        "如何使用Modelfile自定义模型？",
        "Ollama API的默认端口是什么？",
    ]

    for q in questions:
        rag.query(q)
        print()


if __name__ == "__main__":
    demo()
