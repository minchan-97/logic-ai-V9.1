"""
matrix_guardrail_engine.py — GasCode MatrixGuardrail 엔진
=========================================================
skip-gram 임베딩 + JM Smoothing 결합 로컬 검증 엔진
+ 한국어 규칙 기반 형태소 분석기 결합 (조사/어미 오탐 감소)
GPU 없음, CPU만, numpy만 사용
"""
import numpy as np
from collections import defaultdict, Counter

try:
    from korean_tokenizer import tokenize, tokenize_dual
    _KO_TOKENIZER = True
except Exception:
    _KO_TOKENIZER = False
    def tokenize(text): return text.strip().split()
    def tokenize_dual(text): return [(w, w) for w in text.strip().split()]


class MatrixGuardrailEngine:
    def __init__(self, lambda_1=0.6, lambda_2=0.3, lambda_3=0.1, alpha=0.01):
        self.l1 = lambda_1; self.l2 = lambda_2; self.l3 = lambda_3
        self.alpha = alpha
        self.unigram_counts = Counter()
        self.bigram_counts  = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.total_tokens = 0
        self.word2idx = {}; self.vocab_size = 0
        self.word_embeddings = None
        self.is_trained = False
        self.corpus_name = ""

    def train(self, corpus_text, embedding_dim=32, epochs=20, window=2, lr=0.05):
        tokens = tokenize(corpus_text)  # 형태소 분석 적용
        self.total_tokens = len(tokens)
        for i, t in enumerate(tokens):
            self.unigram_counts[t] += 1
            if i >= 1: self.bigram_counts[tokens[i-1]][t] += 1
            if i >= 2: self.trigram_counts[(tokens[i-2], tokens[i-1])][t] += 1

        words = list(Counter(tokens).keys())
        self.word2idx   = {w: i for i, w in enumerate(words)}
        self.vocab_size = len(words)

        rng   = np.random.default_rng(42)
        W_in  = (rng.random((self.vocab_size, embedding_dim)) - 0.5) / embedding_dim
        W_out = (rng.random((self.vocab_size, embedding_dim)) - 0.5) / embedding_dim

        pairs = []
        for i, w in enumerate(tokens):
            if w not in self.word2idx: continue
            c = self.word2idx[w]
            for j in range(max(0, i-window), min(len(tokens), i+window+1)):
                if j != i and tokens[j] in self.word2idx:
                    pairs.append((c, self.word2idx[tokens[j]]))

        def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        for _ in range(epochs):
            rng.shuffle(pairs)
            for center, ctx in pairs:
                negs = rng.integers(0, self.vocab_size, size=5)
                vc = W_in[center]; vp = W_out[ctx]
                pp = sig(np.dot(vc, vp))
                gc = (pp - 1.0) * vp
                W_out[ctx] -= lr * (pp - 1.0) * vc
                for ng in negs:
                    vn = W_out[ng]; pn = sig(np.dot(vc, vn))
                    gc += pn * vn; W_out[ng] -= lr * pn * vc
                W_in[center] -= lr * gc

        norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-12
        self.word_embeddings = W_in / norms
        self.is_trained = True

    def get_vec(self, word):
        if word in self.word2idx: return self.word_embeddings[self.word2idx[word]]
        return None

    def score_jm(self, tokens):
        total_lp = 0.0; K = len(tokens)
        if K < 3: return 0.0, []
        per = []
        for t in range(2, K):
            wc, wp, wpp = tokens[t], tokens[t-1], tokens[t-2]
            p1 = (self.unigram_counts[wc] + self.alpha) / (self.total_tokens + self.alpha * self.vocab_size)
            cp = self.unigram_counts[wp]
            p2 = (self.bigram_counts[wp][wc] / cp) if cp > 0 else 0.0
            cpp = self.trigram_counts[(wpp, wp)][wc]
            p3 = (cpp / self.bigram_counts[wpp][wp]) if self.bigram_counts[wpp][wp] > 0 else 0.0
            p_jm = self.l1 * p3 + self.l2 * p2 + self.l3 * p1
            lp = float(np.log(p_jm + 1e-12))
            total_lp += lp
            in_graph = (p2 > 0 or p3 > 0)
            # 토큰별 이상 여부: logp가 매우 낮을 때만 표시 (오탐 줄이기)
            is_outlier = lp < -12.0 and not in_graph
            per.append({"token": wc, "logp": lp, "in_graph": in_graph, "outlier": is_outlier})
        return total_lp / (K - 1), per

    def score_mismatch(self, tokens):
        vecs = [self.get_vec(w) for w in tokens if self.get_vec(w) is not None]
        if len(vecs) < 2: return 0.5
        sims = [float(np.dot(vecs[i], vecs[i+1])) for i in range(len(vecs)-1)]
        return float((1.0 - np.mean(sims)) / 2.0)

    def evaluate(self, text, logp_thr=-8.0, mis_thr=0.55):
        import time
        t0 = time.perf_counter()
        token_pairs = tokenize_dual(text)  # (원본, 정규화) 쌍
        tokens = [norm for _, norm in token_pairs]
        if len(tokens) < 3:
            return {"status": "SKIP", "avg_logp": 0.0, "mismatch": 0.0, "elapsed_ms": 0.0, "per_token": []}
        avg_logp, per = self.score_jm(tokens)
        # per_token에 원본 토큰 추가
        raw_tokens = [raw for raw, _ in token_pairs]
        for i, p in enumerate(per):
            p["raw_token"] = raw_tokens[i + 2] if i + 2 < len(raw_tokens) else p["token"]
        mismatch = self.score_mismatch(tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        mf = avg_logp < logp_thr; ef = mismatch > mis_thr
        if   not mf and not ef: status = "PASS"
        elif mf  and not ef:    status = "WARNING"
        elif not mf and ef:     status = "CRITICAL"
        else:                   status = "FATAL"

        return {
            "status": status, "avg_logp": avg_logp,
            "mismatch": mismatch, "elapsed_ms": elapsed_ms,
            "per_token": per,
        }
