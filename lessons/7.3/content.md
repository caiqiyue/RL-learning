# 7.3 高质量数据筛选与质量控制

## 课程概述

本节课聚焦于大模型微调数据工程中的核心环节——高质量数据筛选与质量控制。原始采集或合成产生的候选数据通常存在大量噪声，低质量数据不仅浪费训练资源，更会损害模型性能。本节课将系统讲解多阶段质量过滤pipeline、启发式规则过滤、模型评分筛选、采样策略以及质量控制闭环。

## 学习目标

- 理解多阶段质量过滤pipeline的设计原理与各阶段职责
- 掌握启发式规则过滤器的设计方法与阈值设置
- 学会使用模型评分（困惑度、奖励模型、DeBERTa分类器）进行质量筛选
- 掌握针对人工审核的三种采样策略
- 了解标注一致性指标与质量控制闭环的建立方法

## 前置知识

- 了解监督式微调（SFT）的基本原理
- 熟悉大模型推理的基本概念（困惑度、logits）
- 具备 Python 编程基础

---

## 1. 多阶段质量过滤Pipeline

我们采用**多阶段递进式过滤架构**，从粗糙到精细逐步筛选数据：

```
┌─────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  Raw    │───▶│  Heuristic       │───▶│  Model-based     │───▶│  Human Quality     │
│  Data   │    │  Filtering       │    │  Scoring         │    │  Verification      │
└─────────┘    └──────────────────┘    └──────────────────┘    └────────────────────┘
  原始数据        启发式规则过滤         模型评分筛选            人工质量验证
```

| 阶段 | 处理量级 | 通过率 | 筛选依据 |
|------|---------|--------|----------|
| 原始数据 | 100K-1M+ | 100% | - |
| 启发式过滤 | 10K-100K | 50-70% | 规则特征 |
| 模型评分 | 5K-50K | 30-50% | 分数阈值 |
| 人工验证 | 100-1000 | 80-95% | 人工判断 |

**设计原则**：
- **渐进式筛选**：每阶段只去除明确低质量样本，保留边界案例
- **防漏检设计**：可疑样本进入备选池由后续阶段重新评估
- **资源优化**：计算密集型任务放在流程后期

```python
class MultiStageFilter:
    def __init__(self, stages):
        self.stages = stages
    
    def filter(self, dataset):
        current = dataset
        for name, filter_fn in self.stages:
            before = len(current)
            current = filter_fn(current)
            print(f"[{name}] {before} → {len(current)} ({len(current)/before:.1%})")
        return current
```

---

## 2. 启发式规则过滤

启发式过滤是第一道防线，通过快速计算的规则特征剔除明显低质量样本。

### 2.1 长度过滤器

| 样本类型 | 最小长度 | 最大长度 |
|---------|---------|---------|
| 短指令（单问题） | 10 字符 | 500 字符 |
| 长指令（任务描述） | 50 字符 | 2000 字符 |
| 代码片段 | 50 字符 | 5000 字符 |

```python
class LengthFilter:
    def __init__(self, min_len=20, max_len=2000, field="response"):
        self.min_len, self.max_len, self.field = min_len, max_len, field
    
    def filter(self, dataset):
        return [item for item in dataset
                if self.min_len <= len(item.get(self.field, "").strip()) <= self.max_len]
```

### 2.2 重复检测器

N-gram重复率是检测重复内容的有效指标：

```python
from collections import Counter

def compute_ngram_repetition(text, n=3):
    words = list(jieba.cut(text))
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    repeated = sum(1 for count in Counter(ngrams).values() if count > 1)
    return repeated / len(ngrams)

class RepetitionFilter:
    def __init__(self, threshold=0.7, n=3):
        self.threshold, self.n = threshold, n
    
    def filter(self, dataset):
        filtered = []
        for item in dataset:
            rep_rate = compute_ngram_repetition(item.get("response", ""), self.n)
            if rep_rate < self.threshold:
                filtered.append(item)
            else:
                item["filter_reason"] = f"repetition:{rep_rate:.2f}"
        return filtered
```

### 2.3 特殊字符与代码检查

```python
class SpecialCharFilter:
    def __init__(self, max_code_ratio=0.5, max_special_ratio=0.3):
        self.max_code_ratio, self.max_special_ratio = max_code_ratio, max_special_ratio
    
    def compute_code_ratio(self, text):
        patterns = [r'```[\s\S]*?```', r'`[^`]+`', r'\b(def|class|import|return)\b']
        code_matches = sum(len(re.findall(p, text)) for p in patterns)
        return code_matches / max(len(text.split()), 1)
    
    def filter(self, dataset):
        return [item for item in dataset
                if self.compute_code_ratio(item.get("response", "")) <= self.max_code_ratio]
```

### 2.4 阈值汇总

| 过滤器 | 推荐阈值 | 典型通过率 |
|-------|---------|-----------|
| 长度过滤器 | 20-2000字符 | 70-85% |
| 重复检测器 | ngram_threshold=0.7 | 75-90% |
| 特殊字符 | max_code_ratio=0.5 | 80-95% |
| 代码完整性 | require_complete_blocks=True | 85-95% |

---

## 3. 模型评分筛选

经过启发式过滤后，数据量已大幅减少，可以引入计算成本更高的模型评分。

### 3.1 困惑度过滤

困惑度反映模型对文本的"惊讶程度"，高困惑度样本可能是语法错误或领域不匹配。

```python
class PerplexityFilter:
    def __init__(self, model_name, threshold=50):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.threshold = threshold
    
    def compute_perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            loss = self.model(**inputs, labels=inputs["input_ids"]).loss
        return torch.exp(loss).item()
    
    def filter(self, dataset):
        filtered = []
        for item in tqdm(dataset):
            try:
                ppl = self.compute_perplexity(item.get("response", ""))
                item["perplexity"] = ppl
                if ppl <= self.threshold:
                    filtered.append(item)
            except:
                pass
        return filtered
```

### 3.2 奖励模型评分

奖励模型在人类偏好数据上训练，可以判断样本质量：

```python
class RewardModelFilter:
    def __init__(self, reward_model, threshold=0.3):
        self.rm = reward_model
        self.threshold = threshold
    
    def filter(self, dataset):
        filtered = []
        for item in dataset:
            score = self.rm.predict(item.get("response", ""))
            item["reward_score"] = score
            if score >= self.threshold:
                filtered.append(item)
            else:
                item["filter_reason"] = f"low_reward:{score:.3f}"
        return filtered
```

### 3.3 DeBERTa质量分类器

```python
class DeBERTaQualityClassifier:
    def __init__(self, model_path, threshold=0.8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.threshold = threshold
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        probs = torch.softmax(self.model(**inputs).logits, dim=-1)
        return probs[0][1].item()
    
    def filter(self, dataset):
        filtered = []
        for item in tqdm(dataset):
            text = item.get("instruction", "") + " [SEP] " + item.get("response", "")
            score = self.predict(text)
            item["quality_score"] = score
            if score >= self.threshold:
                filtered.append(item)
        return filtered
```

### 3.4 阈值汇总

| 评分方法 | 推荐阈值 | 典型通过率 |
|---------|---------|-----------|
| 困惑度 | 25-50 | 60-80% |
| 奖励模型 | 0.3-0.5 | 50-70% |
| DeBERTa | 0.7-0.9 | 40-60% |

---

## 4. 人工审核采样策略

人工审核成本高，需采用智能采样策略最大化信息获取效率。

### 4.1 不确定性采样

优先审核模型"最拿不准"的样本（决策边界）：

```python
class UncertaintySampler:
    def __init__(self, range=(0.3, 0.7)):
        self.low, self.high = range
    
    def sample(self, scored_dataset, target_size):
        uncertain = [item for item in scored_dataset
                     if self.low <= item.get("quality_score", 0.5) <= self.high]
        
        if len(uncertain) >= target_size:
            return random.sample(uncertain, target_size)
        
        remaining = target_size - len(uncertain)
        other = [item for item in scored_dataset if item not in uncertain]
        return uncertain + random.sample(other, min(remaining, len(other)))
```

### 4.2 多样性采样

确保审核样本覆盖数据分布的各个角落：

```python
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class DiversitySampler:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
    
    def sample(self, dataset, target_size):
        texts = [item.get("instruction", "") + " " + item.get("response", "") for item in dataset]
        
        features = TfidfVectorizer(max_features=5000).fit_transform(texts)
        n_clusters = min(self.n_clusters, len(dataset) // 5)
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(features)
        
        sampled = []
        for cluster_id in range(n_clusters):
            cluster_items = [item for item, label in zip(dataset, labels) if label == cluster_id]
            items_per_cluster = max(1, target_size // n_clusters)
            
            if len(cluster_items) <= items_per_cluster:
                sampled.extend(cluster_items)
            else:
                cluster_items.sort(key=lambda x: abs(x.get("quality_score", 0.5) - 0.5))
                sampled.extend(cluster_items[:items_per_cluster])
        
        return sampled[:target_size]
```

### 4.3 对抗性采样

主动寻找模型的"弱点"和"失败模式"：

```python
class AdversarialSampler:
    def __init__(self, edge_threshold=0.15):
        self.edge_threshold = edge_threshold
    
    def sample(self, scored_dataset, target_size):
        adversarial = []
        
        for item in scored_dataset:
            score = item.get("quality_score", 0.5)
            
            if abs(score - 0.5) <= self.edge_threshold:
                item["selection_reason"] = "boundary"
                adversarial.append(item)
            elif score <= 0.2:
                item["selection_reason"] = "clearly_low"
                adversarial.append(item)
        
        if len(adversarial) >= target_size:
            return random.sample(adversarial, target_size)
        
        remaining = target_size - len(adversarial)
        boundary = [item for item in scored_dataset
                    if item.get("selection_reason") == "boundary" and item not in adversarial]
        adversarial.extend(random.sample(boundary, min(remaining, len(boundary))))
        
        return adversarial[:target_size]
```

### 4.4 策略对比

| 策略 | 适用场景 | 优势 | 劣势 |
|-----|---------|------|------|
| 不确定性采样 | 边界案例多 | 高信息密度 | 可能遗漏极端情况 |
| 多样性采样 | 分布广、数据类型多样 | 覆盖全面 | 可能重复审核相似样本 |
| 对抗性采样 | 安全关键、已知薄弱领域 | 针对性强 | 需要领域知识指导 |

---

## 5. 质量控制闭环

人工审核需要建立持续的质量控制闭环。

### 5.1 Cohen's Kappa一致性指标

```python
from sklearn.metrics import cohen_kappa_score

def compute_kappa(annotations_a, annotations_b):
    return cohen_kappa_score(annotations_a, annotations_b)

def interpret_kappa(kappa):
    if kappa >= 0.8: return "几乎完美一致"
    elif kappa >= 0.6: return "高度一致"
    elif kappa >= 0.4: return "中等一致"
    else: return "一致性极差，需重新设计标准"
```

**Kappa值解释**：

| Kappa值 | 含义 |
|--------|------|
| ≥ 0.8 | 几乎完美一致 |
| 0.6-0.8 | 高度一致 |
| 0.4-0.6 | 中等一致 |
| < 0.4 | 一致性极差，需重新设计标准 |

### 5.2 标注指南迭代流程

1. 分析不一致样本：找出多名标注者判断分歧最大的案例
2. 识别分歧原因：是标准描述模糊还是标注者理解差异
3. 更新标准：针对模糊点增加示例和边界说明
4. 重新标注：使用新标准对部分样本重新标注，验证改进效果

```python
class GuidelineRefiner:
    def analyze_disagreements(self, dataset, threshold=0.5):
        disagreements = []
        
        for item in dataset:
            labels = list(item.get("annotator_labels", {}).values())
            if len(labels) < 2:
                continue
            
            from collections import Counter
            counts = Counter(labels)
            total = sum(counts.values())
            
            entropy = -sum((c/total) * np.log2(c/total) for c in counts.values())
            max_entropy = np.log2(len(counts))
            purity = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
            
            if purity < threshold:
                disagreements.append({"item": item, "labels": labels, "purity": purity})
        
        return sorted(disagreements, key=lambda x: x["purity"])
```

### 5.3 错误分析周期

```python
class ErrorAnalysisCycle:
    def run_cycle(self, dataset, model, eval_benchmarks):
        results = {}
        
        # 1. 评估当前模型
        results["benchmark_scores"] = {
            name: model.evaluate(benchmark) for name, benchmark in eval_benchmarks.items()
        }
        
        # 2. 识别错误样本
        errors = []
        for item in dataset:
            predicted = model.predict(item["instruction"])
            expected = item.get("expected_output", "")
            
            if not self.is_correct(predicted, expected):
                item["error_type"] = self.classify_error(predicted, expected)
                errors.append(item)
        
        results["error_count"] = len(errors)
        results["error_patterns"] = Counter(e["error_type"] for e in errors)
        
        return results
    
    def is_correct(self, predicted, expected):
        from difflib import SequenceMatcher
        return SequenceMatcher(None, predicted, expected).ratio() > 0.8
    
    def classify_error(self, predicted, expected):
        if len(predicted) < len(expected) * 0.5:
            return "incomplete"
        elif len(predicted) > len(expected) * 1.5:
            return "verbose"
        return "wrong_content"
```

---

## 6. 数据持续更新策略

### 6.1 数据新鲜度评估

```python
class DataFreshnessMonitor:
    def __init__(self, staleness_threshold_days=90):
        self.threshold = staleness_threshold_days
    
    def assess_staleness(self, dataset, current_date):
        freshness = []
        
        for item in dataset:
            created_date = item.get("created_at", current_date)
            days_old = (current_date - created_date).days
            
            score = 0.0 if days_old > self.threshold else 1 - (days_old / self.threshold)
            freshness.append({"item_id": item.get("id"), "days_old": days_old, "score": score})
        
        return sorted(freshness, key=lambda x: x["score"])
```

### 6.2 增量数据融合

无需全量重训练，通过增量融合实现知识更新：

```python
class IncrementalDataFusion:
    def __init__(self, fusion_ratio=0.2):
        self.fusion_ratio = fusion_ratio
    
    def fuse_new_data(self, existing_dataset, new_samples):
        target_new_count = int(len(existing_dataset) * self.fusion_ratio / (1 - self.fusion_ratio))
        
        if len(new_samples) <= target_new_count:
            return existing_dataset + new_samples
        
        new_samples.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        selected = new_samples[:target_new_count]
        
        return existing_dataset + selected
```

---

## 总结

本节课我们学习了高质量数据筛选与质量控制的核心技术：

1. **多阶段过滤Pipeline**：通过启发式规则 → 模型评分 → 人工审核的递进架构实现高效筛选。渐进式筛选确保不因早期过度过滤而损失有效数据。

2. **启发式规则过滤**：长度过滤、重复检测、特殊字符比率检查、代码完整性验证构成第一道防线，以极低计算成本快速去除明显低质量样本。

3. **模型评分筛选**：困惑度、奖励模型、DeBERTa质量分类器三个层次逐步精筛，利用语义层面的质量信号识别难以通过规则判断的问题样本。

4. **人工审核采样策略**：不确定性采样聚焦决策边界，多样性采样确保覆盖各类数据，对抗性采样主动寻找模型弱点。

5. **质量控制闭环**：通过Cohen's Kappa一致性指标监控标注质量，标注指南迭代机制及时修正标准模糊，规律性的错误分析周期持续优化数据和模型。

---

## 延伸阅读

- Taori et al. "Self-Instruct: Aligning Language Models with Self-Generated Instructions" (2023)
- Wang et al. "Super-NaturalInstructions: Generalization via Declarative Instructions" (2022)
- Zhou et al. "Large Language Models are Human-Level Prompt Engineers" (2023)
- Ouyang et al. "Training language models to follow instructions with human feedback" (InstructGPT)

---

## 复习题

1. 描述多阶段质量过滤pipeline的设计原理，解释为什么采用渐进式筛选而非一步到位的过滤策略。

2. 对比启发式规则过滤与模型评分筛选的优缺点，说明两种方法在不同阶段的配合方式。

3. 设计一个针对中文对话数据进行质量筛选的完整pipeline，包括至少三种启发式过滤器和两种模型评分方法。

4. 解释Cohen's Kappa相比普通一致率在评估标注质量时的优势，并说明当Kappa值较低时应该如何改进。

5. 讨论数据持续更新的挑战，以及如何在数据新鲜度和训练效率之间取得平衡。