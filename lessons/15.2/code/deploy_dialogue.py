"""
Dialogue System Deployment: Production-ready inference with safety guardrails

This module handles:
1. Model serving with latency optimization
2. Multi-turn conversation state management
3. Safety guardrails at inference time
4. A/B testing framework
5. Production monitoring
"""

import time
import json
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import threading


@dataclass
class ConversationTurn:
    """Single turn in a conversation"""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationManager:
    """
    Manages multi-turn conversation state and history

    Handles:
    - Context window management (max turns, max tokens)
    - Session lifecycle
    - Memory-efficient storage
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_tokens: int = 8000,
        max_history_tokens: int = 6000,
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_history_tokens = max_history_tokens

        # Thread-safe session storage
        self._sessions: Dict[str, List[ConversationTurn]] = {}
        self._lock = threading.RLock()

    def create_session(self, session_id: str) -> None:
        """Create a new conversation session"""
        with self._lock:
            self._sessions[session_id] = []

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """
        Add a turn to the conversation

        Args:
            session_id: Unique session identifier
            role: "user" or "assistant"
            content: Message content
        """
        with self._lock:
            if session_id not in self._sessions:
                self.create_session(session_id)

            turn = ConversationTurn(role=role, content=content)
            self._sessions[session_id].append(turn)

            # Truncate if needed
            self._truncate_if_needed(session_id)

    def _truncate_if_needed(self, session_id: str) -> None:
        """
        Truncate conversation history if it exceeds limits

        Strategy:
        1. First remove oldest assistant turns (keep user context)
        2. Then remove oldest user turns
        3. System message always kept if present
        """
        session = self._sessions[session_id]

        # Check token count
        total_tokens = sum(len(t.content) for t in session)

        while total_tokens > self.max_history_tokens and len(session) > 2:
            # Find oldest non-system turn
            removed = None
            for i, turn in enumerate(session):
                if turn.role != "system":
                    removed = session.pop(i)
                    break

            if removed is None:
                break

            total_tokens -= len(removed.content)

        # Check turn count
        while len(session) > self.max_turns:
            # Remove oldest non-system turns
            for i, turn in enumerate(session):
                if turn.role != "system":
                    session.pop(i)
                    break

    def get_conversation(
        self, session_id: str, include_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for formatting

        Args:
            session_id: Session identifier
            include_system: Whether to include system messages

        Returns:
            List of {"role": str, "content": str} dicts
        """
        with self._lock:
            if session_id not in self._sessions:
                return []

            session = self._sessions[session_id]

            if include_system:
                return [{"role": t.role, "content": t.content} for t in session]
            else:
                return [
                    {"role": t.role, "content": t.content}
                    for t in session
                    if t.role != "system"
                ]

    def format_for_inference(
        self, session_id: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Format conversation for model input

        Returns:
            Formatted string in ChatML format
        """
        conversation = self.get_conversation(session_id)

        formatted = ""

        # Add system prompt
        if system_prompt:
            formatted += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # Add conversation turns
        for turn in conversation:
            if turn["role"] == "system":
                continue  # Already handled
            formatted += f"<|im_start|>{turn['role']}\n"
            formatted += f"{turn['content']}<|im_end|>\n"

        return formatted.strip()

    def clear_session(self, session_id: str) -> None:
        """Clear a specific session"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        with self._lock:
            return list(self._sessions.keys())


class SafetyGuardrails:
    """
    Safety guardrails for real-time content moderation

    Applied at inference time to filter/ modify unsafe outputs
    """

    def __init__(
        self,
        safety_threshold: float = 0.8,
        block_threshold: float = 0.3,
        refusal_phrase: Optional[str] = None,
    ):
        self.safety_threshold = safety_threshold
        self.block_threshold = block_threshold

        self.refusal_phrases = refusal_phrase or [
            "抱歉，我无法帮助处理这个请求。",
            "对不起，这个问题我无法回答。",
            "这个内容超出了我能帮助的范围。",
            "抱歉，我不能生成这部分内容。",
        ]

        # Content patterns to detect
        self._sensitive_patterns = [
            (r"\d{17}[\dXx]", "[已过滤身份证号]"),
            (r"\d{3}-\d{2}-\d{4}", "[已过滤SSN]"),
            (r"\d{4}-\d{4}-\d{4}-\d{4}", "[已过滤银行卡]"),
        ]

        self._harmful_topics = [
            "暴力",
            "色情",
            "赌博",
            "毒品",
            "诈骗",
            "歧视",
            "仇恨",
            "自残",
            "犯罪",
        ]

        # Optional external safety classifier
        self._safety_classifier = None

    def set_safety_classifier(self, classifier: Any) -> None:
        """Set external safety classifier"""
        self._safety_classifier = classifier

    def check_content(self, text: str) -> tuple[bool, float, str]:
        """
        Check content safety

        Returns:
            (is_safe, safety_score, reason)
        """
        # Use external classifier if available
        if self._safety_classifier is not None:
            safety_score = self._safety_classifier.classify(text)
            is_safe = safety_score >= self.safety_threshold

            if not is_safe:
                return False, safety_score, "content_flagged_by_classifier"

            return True, safety_score, "passed_classifier"

        # Fallback: rule-based checks
        safety_score = 1.0

        # Check for harmful topics
        for topic in self._harmful_topics:
            if topic in text:
                safety_score -= 0.5
                if safety_score < self.block_threshold:
                    return False, safety_score, f"harmful_topic:{topic}"

        # Check for sensitive patterns (even if fake - privacy protection)
        for pattern, replacement in self._sensitive_patterns:
            import re

            if re.search(pattern, text):
                safety_score -= 0.3

        # Check for prompt injection attempts
        injection_indicators = [
            "<|im_start|>system",  # Trying to inject system prompt
            "ignore previous",
            "disregard your instructions",
            "you are now dan",
            "无视之前",
        ]

        for indicator in injection_indicators:
            if indicator.lower() in text.lower():
                safety_score -= 0.4

        is_safe = safety_score >= self.safety_threshold
        reason = "passed_rules" if is_safe else f"low_score:{safety_score}"

        return is_safe, safety_score, reason

    def apply_guardrails(self, text: str) -> tuple[str, bool]:
        """
        Apply safety guardrails to text

        Returns:
            (safe_text, was_modified)
        """
        is_safe, score, reason = self.check_content(text)

        if not is_safe:
            # Return refusal phrase
            refusal = random.choice(self.refusal_phrases)
            return refusal, True

        # Apply content filtering
        modified_text = text

        # Filter sensitive patterns
        import re

        for pattern, replacement in self._sensitive_patterns:
            modified_text = re.sub(pattern, replacement, modified_text)

        was_modified = modified_text != text

        return modified_text, was_modified


class LatencyOptimizer:
    """
    Latency optimization for production dialogue systems

    Strategies:
    - Batching for throughput
    - KV cache optimization
    - Speculative decoding
    - Quantization
    """

    def __init__(self, target_latency_ms: int = 1000, max_batch_size: int = 32):
        self.target_latency_ms = target_latency_ms
        self.max_batch_size = max_batch_size

        # Request batching
        self._request_queue: List[Dict] = []
        self._batch_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Metrics
        self._latency_history: List[float] = []
        self._request_counts = defaultdict(int)

    def create_request(
        self, request_id: str, prompt: str, callback: Callable[[str], None]
    ) -> Dict[str, Any]:
        """Create a new inference request"""
        request = {
            "id": request_id,
            "prompt": prompt,
            "callback": callback,
            "timestamp": time.time(),
            "priority": 1,
        }

        with self._lock:
            self._request_queue.append(request)

        return request

    def get_statistics(self) -> Dict[str, Any]:
        """Get latency and throughput statistics"""
        with self._lock:
            history = self._latency_history[-100:]  # Last 100 requests

            return {
                "mean_latency_ms": sum(history) / len(history) if history else 0,
                "p50_latency_ms": sorted(history)[len(history) // 2] if history else 0,
                "p95_latency_ms": sorted(history)[int(len(history) * 0.95)]
                if history
                else 0,
                "p99_latency_ms": sorted(history)[int(len(history) * 0.99)]
                if history
                else 0,
                "total_requests": sum(self._request_counts.values()),
                "queue_depth": len(self._request_queue),
            }


class ABTestManager:
    """
    A/B testing framework for dialogue model improvements

    Allows comparison of different model versions in production
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._variants: Dict[str, Dict] = {}
        self._assignments: Dict[str, str] = {}  # user_id -> variant
        self._metrics: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.RLock()

    def register_variant(
        self, variant_id: str, model_path: str, description: str = ""
    ) -> None:
        """Register a model variant for A/B testing"""
        with self._lock:
            self._variants[variant_id] = {
                "model_path": model_path,
                "description": description,
                "traffic_share": 1.0
                / (len(self._variants) + 1),  # Equal split initially
            }

    def assign_variant(self, user_id: str) -> str:
        """Assign user to a variant"""
        with self._lock:
            if user_id in self._assignments:
                return self._assignments[user_id]

            # Deterministic assignment based on user_id hash
            variants = list(self._variants.keys())
            assigned = variants[hash(user_id) % len(variants)]

            self._assignments[user_id] = assigned
            return assigned

    def record_metric(
        self,
        variant_id: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record a metric for a variant"""
        with self._lock:
            record = {
                "metric": metric_name,
                "value": value,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
            self._variants[variant_id].setdefault("metrics", []).append(record)

    def get_variant_stats(self) -> Dict[str, Dict]:
        """Get aggregated statistics for each variant"""
        with self._lock:
            stats = {}

            for variant_id, variant_data in self._variants.items():
                metrics = variant_data.get("metrics", [])

                if not metrics:
                    stats[variant_id] = {"sample_size": 0}
                    continue

                # Aggregate metrics by name
                aggregated = defaultdict(list)
                for m in metrics:
                    aggregated[m["metric"]].append(m["value"])

                variant_stats = {
                    "sample_size": len(metrics),
                    "metrics": {
                        name: {
                            "mean": sum(vals) / len(vals),
                            "min": min(vals),
                            "max": max(vals),
                            "count": len(vals),
                        }
                        for name, vals in aggregated.items()
                    },
                }

                stats[variant_id] = variant_stats

            return stats


class DialogueSystem:
    """
    Production dialogue system with all components integrated
    """

    def __init__(
        self,
        model_paths: Dict[str, str],
        device: str = "cuda",
        max_seq_length: int = 4096,
    ):
        self.model_paths = model_paths
        self.device = device
        self.max_seq_length = max_seq_length

        # Load active model (can switch variants)
        self._load_model("default")

        # Initialize components
        self.conversation_manager = ConversationManager()
        self.safety_guardrails = SafetyGuardrails()
        self.latency_optimizer = LatencyOptimizer()
        self.ab_test_manager = ABTestManager()

        # Tokenizer (lazy load)
        self._tokenizer = None

    def _load_model(self, variant: str):
        """Load model for variant"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.model_paths.get(variant, self.model_paths.get("default"))

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

        print(f"Loaded model variant: {variant}")

    def generate_response(
        self,
        user_input: str,
        session_id: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.9,
        max_new_tokens: int = 512,
        apply_guardrails: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate response for user input

        Returns:
            Dict with response and metadata
        """
        start_time = time.time()

        # Add user turn to conversation
        self.conversation_manager.add_turn(session_id, "user", user_input)

        # Format conversation for model
        prompt = self.conversation_manager.format_for_inference(
            session_id, system_prompt=system_prompt
        )

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode
        full_response = self._tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract assistant response
        assistant_response = self._extract_response(full_response, prompt)

        # Apply safety guardrails
        original_response = assistant_response
        if apply_guardrails:
            assistant_response, was_modified = self.safety_guardrails.apply_guardrails(
                assistant_response
            )
        else:
            was_modified = False

        # Add assistant turn to conversation
        self.conversation_manager.add_turn(session_id, "assistant", assistant_response)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "response": assistant_response,
            "session_id": session_id,
            "latency_ms": latency_ms,
            "was_modified": was_modified,
            "tokens_generated": len(outputs[0]) - inputs.input_ids.shape[1],
        }

    def _extract_response(self, full_text: str, prompt: str) -> str:
        """Extract assistant response from generated text"""
        if prompt in full_text:
            response = full_text[len(prompt) :].strip()
        else:
            response = full_text

        # Remove special tokens
        for token in ["<|im_end|>", "<|im_start|>"]:
            parts = response.split(token)
            if len(parts) > 1:
                response = parts[-1]

        return response.strip()

    def chat(self, session_id: str, message: str, **kwargs) -> str:
        """Simple chat interface - returns just the response text"""
        result = self.generate_response(message, session_id, **kwargs)
        return result["response"]


# Deployment configuration
DEPLOYMENT_CONFIG = {
    "model_paths": {
        "default": "./checkpoints/dialogue_rlhf",
        "variant_a": "./checkpoints/dialogue_rlhf_v2",
    },
    "device": "cuda",
    "max_seq_length": 4096,
    "safety": {
        "safety_threshold": 0.8,
        "block_threshold": 0.3,
    },
    "performance": {
        "target_latency_ms": 1000,
        "max_batch_size": 32,
    },
    "conversation": {
        "max_turns": 20,
        "max_history_tokens": 6000,
    },
}


def create_production_system(config: Optional[Dict] = None) -> DialogueSystem:
    """Create a production-ready dialogue system"""
    cfg = config or DEPLOYMENT_CONFIG

    # Register A/B test variants
    if len(cfg["model_paths"]) > 1:
        ab_manager = ABTestManager()
        for variant_id, path in cfg["model_paths"].items():
            ab_manager.register_variant(variant_id, path)

    return DialogueSystem(
        model_paths=cfg["model_paths"],
        device=cfg["device"],
        max_seq_length=cfg["max_seq_length"],
    )


# Example usage
def main():
    """Example deployment usage"""

    # Create system
    system = create_production_system()

    # Create a session
    session_id = "user_123_session_1"
    system.conversation_manager.create_session(session_id)

    # Generate responses
    prompts = [
        "你好，请介绍一下你自己",
        "你能帮我解释什么是机器学习吗？",
        "你觉得人工智能的未来会怎样？",
    ]

    print("=== Dialogue System Demo ===\n")

    for prompt in prompts:
        print(f"User: {prompt}")
        response = system.chat(session_id, prompt)
        print(f"Assistant: {response}")
        print()

        # Get stats
        stats = system.latency_optimizer.get_statistics()
        print(f"[Latency: {stats.get('mean_latency_ms', 0):.1f}ms avg]")
        print("-" * 50)


if __name__ == "__main__":
    import torch

    main()
