"""
logic_fault_report.py — 임베딩 불일치 + 마르코프 경로 결합 판정
================================================================

두 신호를 결합해서 어떤 종류의 이탈인지 판정.
비용 0, 로컬 실행.

정직한 한계:
  - 이건 신호 분류이지 환각 확정 판정이 아님
  - CRITICAL/FATAL이어도 사실 정확할 수 있음
  - SAFE여도 표면 자연스러운 사실 오류는 못 잡음
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional


def generate_logic_fault_report(
    mismatch_rate: float,
    threshold: float,
    path_analysis: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    임베딩 불일치율 + 마르코프 경로 분석을 결합하여
    신호 유형과 설명을 반환.

    args:
        mismatch_rate: v4 임베딩 불일치율 (0~100)
        threshold: 불일치율 임계값 τ
        path_analysis: state_path.analyze_response_path() 결과 (없으면 None)

    returns:
        fault_detected: 이상 신호 감지 여부
        error_type: NORMAL / CONTEXT_SHIFT / DOMAIN_DEVIATION / COMBINED_SIGNAL
        severity: SAFE / INFO / WARNING / CRITICAL
        explanation: 한국어 설명
    """
    # 경로 분석 데이터 추출
    first_jump = None
    jump_tokens: List[str] = []
    inside_ratio = 1.0

    if path_analysis and "error" not in path_analysis:
        first_jump = path_analysis.get("first_jump_position", -1)
        if first_jump == -1:
            first_jump = None
        jump_tokens = path_analysis.get("jump_tokens", [])
        inside_ratio = path_analysis.get("inside_ratio", 1.0)

    is_mismatch_fault = mismatch_rate >= threshold
    is_markov_fault   = first_jump is not None

    # 기본값: 정상
    report: Dict[str, Any] = {
        "fault_detected": False,
        "error_type": "NORMAL",
        "severity": "SAFE",
        "explanation": (
            "두 신호 모두 허용 범위 안입니다. "
            "다만 표면이 자연스러운 사실 오류는 이 도구로 잡히지 않습니다."
        ),
        "mismatch_rate": mismatch_rate,
        "first_jump": first_jump,
        "jump_tokens": jump_tokens,
        "inside_ratio": inside_ratio,
    }

    if not (is_mismatch_fault or is_markov_fault):
        return report

    report["fault_detected"] = True

    # ── 시나리오 1: 임베딩만 이탈 ──────────────────────────────
    # 단어 표면은 도메인 안인데 두 관점 응답이 의미상 크게 다름
    if is_mismatch_fault and not is_markov_fault:
        report["error_type"] = "CONTEXT_SHIFT"
        report["severity"]   = "CRITICAL"
        report["explanation"] = (
            f"[임베딩 신호] 단어 표현은 학습 도메인과 유사하지만, "
            f"두 관점(자유/제어) 응답의 의미 거리가 {mismatch_rate:.1f}% "
            f"(임계값 {threshold}%)로 크게 벌어졌습니다. "
            f"같은 도메인 어휘로 서로 다른 결론을 내렸을 가능성이 있습니다. "
            f"답변 내용을 직접 확인하세요."
        )

    # ── 시나리오 2: 마르코프만 이탈 ───────────────────────────
    # 두 관점 응답은 일치하지만 학습 코퍼스 표현과 다른 어휘 등장
    elif is_markov_fault and not is_mismatch_fault:
        token_str = ", ".join(f"'{t}'" for t in jump_tokens[:5])
        report["error_type"] = "DOMAIN_DEVIATION"
        report["severity"]   = "WARNING"
        report["explanation"] = (
            f"[마르코프 신호] 두 관점 응답은 일치하지만, "
            f"위치 {first_jump}번째 부근에서 학습 코퍼스 밖 표현이 등장했습니다. "
            f"해당 단어: {token_str if token_str else '(확인 필요)'}. "
            f"동의어·어미 변화일 수도 있고, 도메인 밖 표현일 수도 있습니다. "
            f"직접 확인이 필요합니다."
        )

    # ── 시나리오 3: 두 신호 동시 이탈 ────────────────────────
    elif is_mismatch_fault and is_markov_fault:
        token_str = ", ".join(f"'{t}'" for t in jump_tokens[:5])
        report["error_type"] = "COMBINED_SIGNAL"
        report["severity"]   = "CRITICAL"
        report["explanation"] = (
            f"[두 신호 동시 감지] 임베딩 불일치율 {mismatch_rate:.1f}% "
            f"(임계값 {threshold}%)와 마르코프 경로 이탈(위치 {first_jump}번째, "
            f"단어: {token_str if token_str else '(확인 필요)'})이 "
            f"동시에 감지됐습니다. "
            f"두 신호가 겹칠 때 실제 이탈 가능성이 높습니다. "
            f"이 도구는 신호 분류이며, 최종 판단은 사람이 해야 합니다."
        )

    return report


def format_report_for_ui(report: Dict[str, Any]) -> Dict[str, str]:
    """UI 표시용 포맷 변환."""
    severity_map = {
        "SAFE":     ("✅", "정상"),
        "INFO":     ("🟢", "참고"),
        "WARNING":  ("🟡", "주의"),
        "CRITICAL": ("🔴", "경고"),
    }
    icon, label = severity_map.get(report["severity"], ("⚪", "알 수 없음"))
    return {
        "icon": icon,
        "label": label,
        "error_type": report["error_type"],
        "explanation": report["explanation"],
    }


# ── 자체 검증 ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 정상
    r = generate_logic_fault_report(5.0, 15.0, None)
    print(f"[정상] {r['severity']}: {r['explanation'][:60]}")

    # 임베딩만 이탈
    r = generate_logic_fault_report(25.0, 15.0, None)
    print(f"[임베딩] {r['severity']}: {r['explanation'][:60]}")

    # 마르코프만 이탈
    r = generate_logic_fault_report(5.0, 15.0, {
        "first_jump_position": 3,
        "jump_tokens": ["양자역학", "광합성"],
        "inside_ratio": 0.7,
    })
    print(f"[마르코프] {r['severity']}: {r['explanation'][:60]}")

    # 둘 다
    r = generate_logic_fault_report(30.0, 15.0, {
        "first_jump_position": 2,
        "jump_tokens": ["블록체인"],
        "inside_ratio": 0.2,
    })
    print(f"[둘 다] {r['severity']}: {r['explanation'][:60]}")
