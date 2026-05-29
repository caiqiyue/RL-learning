"""
Inference Monitoring Script

Monitors vLLM/Triton inference performance with metrics collection,
alerting, and drift detection.
"""

import os
import sys
import time
import json
import queue
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict

import requests

try:
    import psutil
    import GPUtil
except ImportError:
    psutil = None
    GPUtil = None


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency tracking metrics"""

    timestamps: deque = None
    latencies: deque = None

    def __post_init__(self):
        self.timestamps = deque(maxlen=10000)
        self.latencies = deque(maxlen=10000)

    def record(self, timestamp: float, latency_ms: float):
        self.timestamps.append(timestamp)
        self.latencies.append(latency_ms)

    def get_percentiles(self) -> Dict[str, float]:
        if not self.latencies:
            return {}

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        return {
            "count": n,
            "mean_ms": sum(sorted_latencies) / n,
            "p50_ms": sorted_latencies[int(n * 0.50)],
            "p90_ms": sorted_latencies[int(n * 0.90)],
            "p95_ms": sorted_latencies[int(n * 0.95)],
            "p99_ms": sorted_latencies[int(n * 0.99)],
            "max_ms": max(sorted_latencies),
            "min_ms": min(sorted_latencies),
        }


@dataclass
class ThroughputMetrics:
    """Throughput tracking metrics"""

    timestamps: deque = None
    token_counts: deque = None

    def __post_init__(self):
        self.timestamps = deque(maxlen=10000)
        self.token_counts = deque(maxlen=10000)
        self.start_time = time.time()
        self.total_tokens = 0

    def record(self, timestamp: float, num_tokens: int):
        self.timestamps.append(timestamp)
        self.token_counts.append(num_tokens)
        self.total_tokens += num_tokens

    def get_throughput(self) -> Dict[str, float]:
        if not self.timestamps:
            return {}

        elapsed = time.time() - self.start_time
        current_throughput = self.token_counts[-1] / elapsed if elapsed > 0 else 0

        return {
            "total_tokens": self.total_tokens,
            "uptime_seconds": elapsed,
            "current_tokens_per_sec": current_throughput,
            "avg_tokens_per_sec": self.total_tokens / elapsed if elapsed > 0 else 0,
        }


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_utilization: float = 0.0
    gpu_temperature: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ErrorMetrics:
    """Error tracking metrics"""

    total_errors: int = 0
    error_types: Dict[str, int] = None

    def __post_init__(self):
        self.error_types = defaultdict(int)

    def record_error(self, error_type: str):
        self.total_errors += 1
        self.error_types[error_type] += 1

    def get_error_rate(self, total_requests: int) -> float:
        if total_requests == 0:
            return 0.0
        return self.total_errors / total_requests


class PrometheusExporter:
    """Export metrics in Prometheus format"""

    def __init__(self, metrics_port: int = 9090):
        self.metrics_port = metrics_port
        self.metrics_queue = queue.Queue()
        self.server_thread: Optional[threading.Thread] = None

    def start(self):
        """Start metrics server"""
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        logger.info(f"Prometheus metrics server started on port {self.metrics_port}")

    def _run_server(self):
        """Run metrics HTTP server"""
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()

                    metrics_data = (
                        self.server.metrics_queue.get()
                        if not self.server.metrics_queue.empty()
                        else ""
                    )
                    self.wfile.write(metrics_data.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        MetricsHandler.server = self

        server = HTTPServer(("0.0.0.0", self.metrics_port), MetricsHandler)
        server.serve_forever()

    def export(self, metrics: Dict):
        """Export metrics in Prometheus format"""
        lines = []

        lines.append(f"# HELP inference_requests_total Total inference requests")
        lines.append(f"# TYPE inference_requests_total counter")
        lines.append(f"inference_requests_total {metrics.get('total_requests', 0)}")

        lines.append(f"# HELP inference_tokens_total Total tokens processed")
        lines.append(f"# TYPE inference_tokens_total counter")
        lines.append(f"inference_tokens_total {metrics.get('total_tokens', 0)}")

        lines.append(f"# HELP inference_throughput_tokens_per_sec Token throughput")
        lines.append(f"# TYPE inference_throughput_tokens_per_sec gauge")
        lines.append(
            f"inference_throughput_tokens_per_sec {metrics.get('throughput_tokens_per_sec', 0):.2f}"
        )

        latency = metrics.get("latency", {})
        lines.append(f"# HELP inference_latency_ms Request latency in milliseconds")
        lines.append(f"# TYPE inference_latency_ms summary")
        lines.append(
            f'inference_latency_ms{{quantile="0.5"}} {latency.get("p50_ms", 0):.2f}'
        )
        lines.append(
            f'inference_latency_ms{{quantile="0.9"}} {latency.get("p90_ms", 0):.2f}'
        )
        lines.append(
            f'inference_latency_ms{{quantile="0.99"}} {latency.get("p99_ms", 0):.2f}'
        )

        lines.append(f"# HELP inference_errors_total Total inference errors")
        lines.append(f"# TYPE inference_errors_total counter")
        lines.append(f"inference_errors_total {metrics.get('errors', 0)}")

        self.metrics_queue.put("\n".join(lines))


class AlertManager:
    """Manage and trigger alerts based on metrics thresholds"""

    def __init__(self):
        self.alert_rules = [
            {
                "name": "high_latency",
                "metric": "latency.p99_ms",
                "threshold": 5000,
                "severity": "warning",
                "comparison": ">",
                "message": "P99 latency exceeds 5000ms",
            },
            {
                "name": "critical_latency",
                "metric": "latency.p99_ms",
                "threshold": 10000,
                "severity": "critical",
                "comparison": ">",
                "message": "P99 latency exceeds 10000ms",
            },
            {
                "name": "high_error_rate",
                "metric": "error_rate",
                "threshold": 0.01,
                "severity": "warning",
                "comparison": ">",
                "message": "Error rate exceeds 1%",
            },
            {
                "name": "critical_error_rate",
                "metric": "error_rate",
                "threshold": 0.05,
                "severity": "critical",
                "comparison": ">",
                "message": "Error rate exceeds 5%",
            },
            {
                "name": "high_gpu_memory",
                "metric": "resources.gpu_memory_utilization",
                "threshold": 95,
                "severity": "warning",
                "comparison": ">",
                "message": "GPU memory utilization exceeds 95%",
            },
            {
                "name": "low_throughput",
                "metric": "throughput_tokens_per_sec",
                "threshold": 100,
                "severity": "warning",
                "comparison": "<",
                "message": "Throughput below 100 tokens/sec",
            },
        ]

        self.active_alerts: List[Dict] = []
        self.alert_history: List[Dict] = []

    def evaluate(self, metrics: Dict) -> List[Dict]:
        """Evaluate metrics against alert rules"""
        triggered_alerts = []

        for rule in self.alert_rules:
            metric_path = rule["metric"].split(".")
            value = metrics

            for key in metric_path:
                if isinstance(value, dict):
                    value = value.get(key, 0)
                else:
                    value = 0
                    break

            threshold = rule["threshold"]
            comparison = rule["comparison"]

            triggered = False
            if comparison == ">" and value > threshold:
                triggered = True
            elif comparison == "<" and value < threshold:
                triggered = True
            elif comparison == ">=" and value >= threshold:
                triggered = True
            elif comparison == "<=" and value <= threshold:
                triggered = True

            if triggered:
                alert = {
                    "name": rule["name"],
                    "severity": rule["severity"],
                    "message": rule["message"],
                    "value": value,
                    "threshold": threshold,
                    "timestamp": datetime.now().isoformat(),
                }
                triggered_alerts.append(alert)

                if alert not in self.active_alerts:
                    self.active_alerts.append(alert)
                    self.alert_history.append(alert)
                    logger.warning(
                        f"ALERT [{rule['severity'].upper()}]: {rule['message']} (value={value:.2f})"
                    )

        self.active_alerts = triggered_alerts

        return triggered_alerts


class DriftDetector:
    """
    Detect distribution drift in model inputs/outputs.
    """

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.05,
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_stats: Optional[Dict] = None
        self.current_window: List[Dict] = []

    def set_reference(self, reference_data: List[Dict]):
        """Set reference distribution for comparison"""
        self.reference_stats = self._compute_stats(reference_data)
        logger.info(f"Reference distribution set with {len(reference_data)} samples")

    def add_sample(self, sample: Dict):
        """Add sample to current window"""
        self.current_window.append(sample)

        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)

    def detect_drift(self) -> Optional[Dict]:
        """Detect drift between reference and current distribution"""
        if (
            self.reference_stats is None
            or len(self.current_window) < self.window_size // 10
        ):
            return None

        current_stats = self._compute_stats(self.current_window)

        drift_scores = {}
        for key in self.reference_stats:
            if key in current_stats:
                ref_val = self.reference_stats[key]
                cur_val = current_stats[key]

                if ref_val != 0:
                    drift = abs(cur_val - ref_val) / abs(ref_val)
                    drift_scores[key] = drift

        max_drift = max(drift_scores.values()) if drift_scores else 0
        max_key = max(drift_scores, key=drift_scores.get) if drift_scores else None

        drift_detected = max_drift > self.drift_threshold

        return {
            "drift_detected": drift_detected,
            "max_drift": max_drift,
            "max_drift_key": max_key,
            "drift_threshold": self.drift_threshold,
            "drift_scores": drift_scores,
        }

    def _compute_stats(self, data: List[Dict]) -> Dict:
        """Compute statistics for sample data"""
        if not data:
            return {}

        lengths = [len(str(d.get("prompt", ""))) for d in data]

        return {
            "mean_length": sum(lengths) / len(lengths),
            "std_length": self._std(lengths),
            "sample_count": len(data),
        }

    def _std(self, values: List[float]) -> float:
        """Compute standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance**0.5


class InferenceMonitor:
    """
    Main inference monitoring class.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        poll_interval: int = 5,
        enable_prometheus: bool = True,
        prometheus_port: int = 9090,
    ):
        self.api_url = api_url
        self.poll_interval = poll_interval
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

        self.latency_metrics = LatencyMetrics()
        self.throughput_metrics = ThroughputMetrics()
        self.error_metrics = ErrorMetrics()
        self.resource_metrics = ResourceMetrics()

        self.alert_manager = AlertManager()
        self.drift_detector = DriftDetector()

        self.prometheus_exporter = None
        if enable_prometheus:
            self.prometheus_exporter = PrometheusExporter(prometheus_port)

        self.uptime_start = time.time()

    def start(self):
        """Start monitoring"""
        self.running = True

        if self.prometheus_exporter:
            self.prometheus_exporter.start()

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Inference monitor started")

    def stop(self):
        """Stop monitoring"""
        self.running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        logger.info("Inference monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._fetch_and_record_metrics()
                self._check_alerts()
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(self.poll_interval)

    def _fetch_and_record_metrics(self):
        """Fetch metrics from API and record"""
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=5)
            response.raise_for_status()
            stats = response.json()

            current_time = time.time()

            self.latency_metrics.record(
                current_time, stats.get("latency", {}).get("mean_ms", 0)
            )

            self.throughput_metrics.record(current_time, stats.get("total_tokens", 0))

            self.resource_metrics = self._fetch_resource_metrics()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch metrics: {e}")
            self.error_metrics.record_error("fetch_error")

    def _fetch_resource_metrics(self) -> ResourceMetrics:
        """Fetch current resource utilization"""
        metrics = ResourceMetrics()

        if psutil:
            metrics.cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            metrics.memory_percent = mem.percent
            metrics.memory_used_gb = mem.used / (1024**3)
            metrics.memory_total_gb = mem.total / (1024**3)

        if GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.gpu_utilization = gpu.load * 100
                    metrics.gpu_memory_used_gb = gpu.memoryUsed / 1024
                    metrics.gpu_memory_total_gb = gpu.memoryTotal / 1024
                    metrics.gpu_memory_utilization = gpu.memoryUtil * 100
                    metrics.gpu_temperature = gpu.temperature
            except Exception as e:
                logger.warning(f"Failed to fetch GPU metrics: {e}")

        return metrics

    def _check_alerts(self):
        """Check for alert conditions"""
        metrics = self.get_full_report()
        alerts = self.alert_manager.evaluate(metrics)

        for alert in alerts:
            self._handle_alert(alert)

    def _handle_alert(self, alert: Dict):
        """Handle triggered alert"""
        if alert["severity"] == "critical":
            logger.error(f"CRITICAL ALERT: {alert['message']}")
        elif alert["severity"] == "warning":
            logger.warning(f"WARNING: {alert['message']}")

    def get_full_report(self) -> Dict:
        """Get complete monitoring report"""
        latency = self.latency_metrics.get_percentiles()
        throughput = self.throughput_metrics.get_throughput()

        total_requests = latency.get("count", 0) + self.error_metrics.total_errors

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.uptime_start,
            "total_requests": total_requests,
            "total_tokens": throughput.get("total_tokens", 0),
            "throughput_tokens_per_sec": throughput.get("avg_tokens_per_sec", 0),
            "latency": latency,
            "first_token_latency": {
                "mean_ms": 0,
                "p99_ms": 0,
            },
            "resources": self.resource_metrics.to_dict(),
            "errors": {
                "total_errors": self.error_metrics.total_errors,
                "error_rate": self.error_metrics.get_error_rate(total_requests),
                "error_types": dict(self.error_metrics.error_types),
            },
        }

    def record_custom_sample(self, sample: Dict):
        """Record a custom sample (for drift detection)"""
        self.drift_detector.add_sample(sample)

    def set_reference_distribution(self, reference_data: List[Dict]):
        """Set reference distribution for drift detection"""
        self.drift_detector.set_reference(reference_data)

    def get_drift_status(self) -> Optional[Dict]:
        """Get current drift status"""
        return self.drift_detector.detect_drift()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        report = self.get_full_report()

        lines = []

        lines.append(f"# HELP vllm_uptime_seconds Time since monitor started")
        lines.append(f"# TYPE vllm_uptime_seconds counter")
        lines.append(f"vllm_uptime_seconds {report['uptime_seconds']:.2f}")

        lines.append(f"# HELP vllm_requests_total Total inference requests")
        lines.append(f"# TYPE vllm_requests_total counter")
        lines.append(f"vllm_requests_total {report['total_requests']}")

        lines.append(f"# HELP vllm_tokens_total Total tokens")
        lines.append(f"# TYPE vllm_tokens_total counter")
        lines.append(f"vllm_tokens_total {report['total_tokens']}")

        lines.append(f"# HELP vllm_throughput_tokens_per_sec Token throughput")
        lines.append(f"# TYPE vllm_throughput_tokens_per_sec gauge")
        lines.append(
            f"vllm_throughput_tokens_per_sec {report['throughput_tokens_per_sec']:.2f}"
        )

        latency = report.get("latency", {})
        lines.append(f"# HELP vllm_latency_ms Request latency")
        lines.append(f"# TYPE vllm_latency_ms summary")
        lines.append(
            f'vllm_latency_ms{{quantile="0.5"}} {latency.get("p50_ms", 0):.2f}'
        )
        lines.append(
            f'vllm_latency_ms{{quantile="0.9"}} {latency.get("p90_ms", 0):.2f}'
        )
        lines.append(
            f'vllm_latency_ms{{quantile="0.99"}} {latency.get("p99_ms", 0):.2f}'
        )

        resources = report.get("resources", {})
        lines.append(f"# HELP vllm_gpu_utilization GPU utilization")
        lines.append(f"# TYPE vllm_gpu_utilization gauge")
        lines.append(f"vllm_gpu_utilization {resources.get('gpu_utilization', 0):.2f}")

        lines.append(f"# HELP vllm_gpu_memory_utilization GPU memory utilization")
        lines.append(f"# TYPE vllm_gpu_memory_utilization gauge")
        lines.append(
            f"vllm_gpu_memory_utilization {resources.get('gpu_memory_utilization', 0):.2f}"
        )

        lines.append(f"# HELP vllm_errors_total Total errors")
        lines.append(f"# TYPE vllm_errors_total counter")
        lines.append(f"vllm_errors_total {report['errors']['total_errors']}")

        return "\n".join(lines)


def main():
    """CLI interface for monitoring"""
    import argparse

    parser = argparse.ArgumentParser(description="Inference Monitor")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000")
    parser.add_argument("--poll_interval", type=int, default=5)
    parser.add_argument("--prometheus_port", type=int, default=9090)
    parser.add_argument("--duration", type=int, default=0, help="0 for infinite")

    args = parser.parse_args()

    monitor = InferenceMonitor(
        api_url=args.api_url,
        poll_interval=args.poll_interval,
        prometheus_port=args.prometheus_port,
    )

    monitor.start()

    print("\n" + "=" * 60)
    print("Inference Monitor Running")
    print("=" * 60)
    print(f"API URL: {args.api_url}")
    print(f"Poll Interval: {args.poll_interval}s")
    print(f"Prometheus Port: {args.prometheus_port}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                report = monitor.get_full_report()
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
                print(f"  Requests: {report['total_requests']}")
                print(
                    f"  Throughput: {report['throughput_tokens_per_sec']:.1f} tokens/sec"
                )
                print(f"  Latency P99: {report['latency'].get('p99_ms', 0):.0f}ms")
                print(f"  Error Rate: {report['errors']['error_rate'] * 100:.2f}%")
                print(
                    f"  GPU Memory: {report['resources'].get('gpu_memory_utilization', 0):.1f}%"
                )
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
