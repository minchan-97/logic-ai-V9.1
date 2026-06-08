"""
markov_guardrail.py — 도메인 자연스러움 가드레일 (v5 옵션)
==========================================================

오늘 본인이 만든 발상의 모듈화:
  "마르코프의 한계(학습 데이터 갇힘)를 가드레일에서 장점(도메인 신호기)으로"

핵심:
  - 사용자가 자기 도메인 코퍼스 텍스트를 업로드
  - 마르코프 n-gram 모델 학습 (보통 n=2)
  - LLM 응답에 마르코프 logP 점수 매김
  - 점수가 너무 낮으면 "이 응답은 학습된 도메인 흐름에서 벗어남" 신호

정직한 한계 (UI에도 표시):
  - 이건 환각 탐지기가 아니라 "도메인 자연스러움 점수기"
  - "강아지 다리 6개"처럼 표면 자연스러운 사실 오류는 못 잡음
  - 학습 코퍼스가 작으면 가짜 negative 많음 (자연스러운 응답을 거부)
  - LLM의 일반 어휘가 자기 도메인 코퍼스 어휘와 다르면 거의 다 -10점

v4와 보완:
  - v4 mismatch: LLM 자체 헷갈림 신호
  - 마르코프 점수: 도메인 밖 표현 신호
  - 둘은 다른 종류의 신호 (실험 5에서 확인)
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class GuardrailConfig:
    """마르코프 가드레일 설정."""
    n: int = 2                      # n-gram 크기
    unknown_penalty: float = -10.0  # 학습 안 된 context 패널티
    warning_threshold: float = -8.0  # 이 값보다 logP 낮으면 경고
    min_corpus_tokens: int = 100   # 코퍼스가 이보다 작으면 학습 안 함
    enabled: bool = False           # 기본 OFF


@dataclass
class GuardrailState:
    """학습된 마르코프 모델 + 메타."""
    model: Dict = field(default_factory=dict)
    vocab: List[str] = field(default_factory=list)
    corpus_tokens: int = 0
    is_trained: bool = False
    corpus_name: str = ""


try:
    from korean_tokenizer import tokenize as _ko_tokenize
    _KO_AVAILABLE = True
except Exception:
    _KO_AVAILABLE = False

def tokenize_text(text: str) -> List[str]:
    """한국어 형태소 분석 기반 토큰화 (조사/어미 분리)."""
    if _KO_AVAILABLE:
        return _ko_tokenize(text.replace("\n", " "))
    return [t for t in text.replace("\n", " ").split() if t]


def train_guardrail(text: str, cfg: GuardrailConfig) -> GuardrailState:
    """텍스트로 마르코프 가드레일 학습.
    
    args:
        text: 도메인 코퍼스 텍스트 (충분히 큰)
        cfg: 설정
    
    returns:
        학습된 GuardrailState
    """
    tokens = tokenize_text(text)
    if len(tokens) < cfg.min_corpus_tokens:
        raise ValueError(
            f"코퍼스가 너무 작습니다 ({len(tokens)} 토큰). "
            f"최소 {cfg.min_corpus_tokens} 토큰 필요."
        )
    
    n = cfg.n
    model: Dict[Tuple, Counter] = defaultdict(Counter)
    for i in range(len(tokens) - n):
        context = tuple(tokens[i:i + n])
        next_tok = tokens[i + n]
        model[context][next_tok] += 1
    
    state = GuardrailState(
        model=dict(model),
        vocab=sorted(set(tokens)),
        corpus_tokens=len(tokens),
        is_trained=True,
    )
    return state


def score_response(response: str, state: GuardrailState,
                    cfg: GuardrailConfig) -> Dict:
    """LLM 응답에 마르코프 점수 매김.
    
    returns:
        - avg_logp: 평균 log 확률 (높을수록 자연스러움)
        - log_likelihood: 전체 log P
        - coverage: 학습된 transition 비율
        - unknown_count: 학습 안 된 context 수
        - is_warning: warning_threshold 초과 여부
        - per_token: 위치별 logP 리스트 (시각화용)
    """
    if not state.is_trained:
        return {"error": "가드레일이 학습되지 않음"}
    
    n = cfg.n
    tokens = tokenize_text(response)
    
    if len(tokens) <= n:
        return {
            "avg_logp": float("nan"),
            "warning": True,
            "reason": f"응답이 너무 짧음 (n={n} 미만)"
        }
    
    per_token = []
    unknown = 0
    valid = 0
    
    for i in range(n, len(tokens)):
        context = tuple(tokens[i - n:i])
        next_tok = tokens[i]
        
        if context not in state.model:
            per_token.append({"token": next_tok, "logp": cfg.unknown_penalty,
                              "status": "unknown_context"})
            unknown += 1
            continue
        
        next_counts = state.model[context]
        total = sum(next_counts.values())
        count = next_counts.get(next_tok, 0)
        
        if count == 0:
            # context는 봤지만 next_tok 안 나옴
            prob = 1.0 / (total + len(next_counts) + 10)
            per_token.append({"token": next_tok, "logp": float(np.log(prob)),
                              "status": "unseen_transition"})
            unknown += 1
        else:
            prob = count / total
            per_token.append({"token": next_tok, "logp": float(np.log(prob)),
                              "status": "ok"})
            valid += 1
    
    logps = [p["logp"] for p in per_token]
    avg_logp = float(np.mean(logps)) if logps else float("nan")
    total_logp = float(np.sum(logps)) if logps else float("nan")
    coverage = valid / max(len(per_token), 1)
    is_warning = avg_logp < cfg.warning_threshold
    
    return {
        "avg_logp": avg_logp,
        "log_likelihood": total_logp,
        "coverage": coverage,
        "unknown_count": unknown,
        "valid_count": valid,
        "is_warning": is_warning,
        "warning_threshold": cfg.warning_threshold,
        "per_token": per_token,
        "n_tokens": len(tokens),
    }


def score_response_jm(response: str, state: "GuardrailState",
                       cfg: "GuardrailConfig",
                       lambdas: tuple = (0.1, 0.6, 0.3)) -> Dict:
    """Jelinek-Mercer Smoothing 버전 마르코프 점수.

    기존 score_response는 n-gram context가 코퍼스에 없으면 즉시 -10 패널티.
    이 버전은 여러 차수(unigram, bigram, trigram)를 가중 결합해서
    조사·어미 변화 같은 표면 차이에 덜 민감하게 만듦.

    lambdas: (λ1=unigram 비중, λ2=bigram 비중, λ3=trigram 비중)
    λ1 + λ2 + λ3 = 1.0
    """
    if not state.is_trained:
        return {"error": "가드레일이 학습되지 않음"}

    tokens = tokenize_text(response)
    if len(tokens) < 2:
        return {"avg_logp": float("nan"), "warning": True, "reason": "응답 너무 짧음"}

    # 각 차수별 모델 학습 (학습된 모델에서 역산)
    # unigram: 단어 자체 확률
    unigram_counts = Counter(tokenize_text(
        " ".join(" ".join(k) for k in state.model.keys())
    ))
    total_unigram = sum(unigram_counts.values()) + 1

    l1, l2, l3 = lambdas
    eps = 1e-10  # 라플라스 스무딩 하한

    per_token = []

    for i in range(1, len(tokens)):
        tok = tokens[i]

        # λ1: unigram P(w)
        p1 = (unigram_counts.get(tok, 0) + eps) / total_unigram

        # λ2: bigram P(w | w-1)
        ctx2 = (tokens[i-1],)
        # bigram context는 n=1 마르코프로 근사
        bigram_hits = sum(
            v.get(tok, 0)
            for k, v in state.model.items()
            if len(k) == cfg.n and k[-1] == tokens[i-1]
        )
        bigram_total = sum(
            sum(v.values())
            for k, v in state.model.items()
            if len(k) == cfg.n and k[-1] == tokens[i-1]
        )
        p2 = (bigram_hits + eps) / (bigram_total + eps * len(state.vocab) + 1)

        # λ3: trigram P(w | w-2, w-1) — 원래 마르코프 n-gram
        p3 = eps
        if i >= 2:
            ctx3 = tuple(tokens[i - cfg.n:i])
            if ctx3 in state.model:
                nc = state.model[ctx3]
                total = sum(nc.values())
                p3 = (nc.get(tok, 0) + eps) / (total + eps)
            else:
                p3 = eps

        # JM 혼합
        p_jm = l1 * p1 + l2 * p2 + l3 * p3
        per_token.append({
            "token": tok,
            "logp": float(np.log(p_jm + eps)),
            "p1": p1, "p2": p2, "p3": p3,
        })

    if not per_token:
        return {"avg_logp": float("nan"), "warning": True}

    avg_logp = float(np.mean([p["logp"] for p in per_token]))
    is_warning = avg_logp < cfg.warning_threshold

    return {
        "avg_logp": avg_logp,
        "method": "jelinek-mercer",
        "lambdas": lambdas,
        "is_warning": is_warning,
        "warning_threshold": cfg.warning_threshold,
        "per_token": per_token,
        "n_tokens": len(per_token),
        "coverage": sum(1 for p in per_token if p["p3"] > eps * 2) / max(len(per_token), 1),
    }
    """점수를 한국어 설명으로."""
    if avg_logp >= -10.0:
        return "✅ 도메인 안 (자연스러운 표현)"
    elif avg_logp >= -12.0:
        return "🟡 경계 (일부 표현이 도메인 밖)"
    elif avg_logp >= warning_threshold:
        return "🟠 주의 (도메인 밖 표현 다수)"
    else:
        return "🔴 이탈 (도메인 완전 이탈)"


# ----------------------------------------------------------------
