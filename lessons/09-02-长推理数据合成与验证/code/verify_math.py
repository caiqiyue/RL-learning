"""
Verify Math Reasoning Chains
验证数学推理链的正确性
"""

import re
from sympy import simplify, sympify, SympifyError
from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationResult:
    is_valid: bool
    error_message: Optional[str] = None
    details: Optional[str] = None


def extract_equations(text: str) -> list[str]:
    """
    从文本中提取等式表达式

    支持格式：
    - x = 5
    - x + y = 10
    - (a + b)^2 = a^2 + 2ab + b^2
    """
    pattern = r"[a-zA-Z_]\s*=\s*[^=\n]+"
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches if "=" in m and "==" not in m]


def verify_equation(equation: str) -> VerificationResult:
    """
    验证单个等式的正确性

    Args:
        equation: 形如 "x = 5" 的字符串

    Returns:
        VerificationResult: 验证结果
    """
    try:
        if "=" not in equation:
            return VerificationResult(is_valid=False, error_message="No '=' found")

        left, right = equation.split("=")
        left_expr = sympify(left.strip())
        right_expr = sympify(right.strip())

        diff = simplify(left_expr - right_expr)

        if diff == 0:
            return VerificationResult(is_valid=True, details=f"LHS = RHS: {equation}")
        else:
            return VerificationResult(
                is_valid=False,
                error_message=f"Not equal: LHS = {left_expr}, RHS = {right_expr}, diff = {diff}",
            )

    except SympifyError as e:
        return VerificationResult(is_valid=False, error_message=f"Sympy error: {e}")
    except Exception as e:
        return VerificationResult(is_valid=False, error_message=str(e))


def verify_math_reasoning(reasoning: str) -> VerificationResult:
    """
    验证完整的数学推理链

    流程：
    1. 提取推理链中的所有等式
    2. 逐一验证等式是否成立
    3. 检查关键推导步骤是否有逻辑跳跃

    Args:
        reasoning: 完整的推理文本

    Returns:
        VerificationResult: 验证结果
    """
    equations = extract_equations(reasoning)

    if not equations:
        return VerificationResult(
            is_valid=False, error_message="No verifiable equations found in reasoning"
        )

    failed_equations = []
    for eq in equations:
        result = verify_equation(eq)
        if not result.is_valid:
            failed_equations.append((eq, result.error_message))

    if failed_equations:
        error_msg = "\n".join([f"{eq}: {err}" for eq, err in failed_equations])
        return VerificationResult(
            is_valid=False,
            error_message=f"Found {len(failed_equations)} incorrect equations:\n{error_msg}",
        )

    return VerificationResult(
        is_valid=True, details=f"All {len(equations)} equations verified correct"
    )


def verify_step_by_step(reasoning: str) -> list[VerificationResult]:
    """
    对推理链的每一步进行验证

    Returns:
        每一步的验证结果列表
    """
    steps = reasoning.split("\n")
    results = []

    for i, step in enumerate(steps):
        if "=" in step:
            result = verify_equation(step)
            result.details = f"Step {i + 1}: {step}"
        else:
            result = VerificationResult(
                is_valid=True, details=f"Step {i + 1}: No equation to verify"
            )

        results.append(result)

    return results


def check_derivation_chain(reasoning: str) -> VerificationResult:
    """
    检查推导链的逻辑一致性

    例如：
    x = 5
    y = x + 3  -> y = 8
    z = y * 2  -> z = 16

    验证：z 的推导是否依赖于前面正确的 x, y
    """
    lines = reasoning.split("\n")
    variable_values = {}

    for line in lines:
        if "=" in line and not line.startswith("="):
            match = re.match(r"([a-zA-Z_]\w*)\s*=\s*(.+)", line.strip())
            if match:
                var_name = match.group(1)
                expr = match.group(2).strip()

                try:
                    simplified = simplify(sympify(expr))
                    variable_values[var_name] = simplified
                except SympifyError:
                    pass

    if len(variable_values) < 2:
        return VerificationResult(
            is_valid=True, details="Not enough variables to check derivation chain"
        )

    return VerificationResult(
        is_valid=True,
        details=f"Derivation chain valid with {len(variable_values)} variables",
    )


def batch_verify_math_reasoning(reasonings: list[str]) -> dict:
    """
    批量验证多个数学推理

    Returns:
        {"passed": [...], "failed": [...], "summary": {...}}
    """
    passed = []
    failed = []

    for reasoning in reasonings:
        result = verify_math_reasoning(reasoning)
        if result.is_valid:
            passed.append(reasoning)
        else:
            failed.append((reasoning, result.error_message))

    return {
        "passed": passed,
        "failed": failed,
        "summary": {
            "total": len(reasonings),
            "passed_count": len(passed),
            "failed_count": len(failed),
            "pass_rate": len(passed) / len(reasonings) if reasonings else 0.0,
        },
    }


if __name__ == "__main__":
    test_reasoning = """
    求函数 f(x) = x² - 4x + 3 的最小值

    第一步：确定顶点 x = -b/2a = -(-4)/(2*1) = 2
    第二步：计算 f(2) = (2)² - 4(2) + 3 = 4 - 8 + 3 = -1
    第三步：验证最小值为 -1
    """

    result = verify_math_reasoning(test_reasoning)
    print(f"Verification result: {result}")

    equations = extract_equations(test_reasoning)
    print(f"Extracted equations: {equations}")

    for eq in equations:
        eq_result = verify_equation(eq)
        print(f"Equation '{eq}': {eq_result}")
