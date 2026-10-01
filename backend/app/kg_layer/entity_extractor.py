# -*- coding: utf-8 -*-
"""
实体识别与实体链接
==================
* 训练/推理模型：**BERT-BiLSTM-CRF**（PPT 指定）
* 降级策略：模型权重缺失时，使用「医疗词典 + 规则 + 模糊匹配」实现，
  保证系统在任何环境下均可完成实体识别与链接。

BERT-BiLSTM-CRF 结构（`BertBiLstmCrf`）：
    BERT 编码 → BiLSTM 上下文建模 → CRF 转移约束解码
    * BERT 提供字级上下文表示（解决医疗长术语切分歧义）
    * BiLSTM 捕捉实体跨度内的顺序依赖
    * CRF 保证标签序列合法（B- 后必须跟 I- 或 O），显著提升实体边界准确率
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.config import settings
from app.kg_layer.memory_store import MemoryGraphStore
from app.schemas import ENTITY_TYPE_META
from app.utils.logger import get_logger
from app.utils.text import similarity

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = object  # type: ignore


#: BIO 标签体系（7 类实体 × (B, I) + O）
ENTITY_LABELS: List[str] = [
    "O",
    "B-Disease", "I-Disease",
    "B-Symptom", "I-Symptom",
    "B-Drug", "I-Drug",
    "B-Department", "I-Department",
    "B-Treatment", "I-Treatment",
    "B-Check", "I-Check",
    "B-Population", "I-Population",
]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(ENTITY_LABELS)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}


@dataclass
class EntityMention:
    """识别出的实体提及"""

    text: str
    type: str
    start: int
    end: int
    score: float = 1.0
    method: str = "dictionary"
    kg_id: str = ""
    kg_name: str = ""

    @property
    def label(self) -> str:
        return ENTITY_TYPE_META.get(self.type, {}).get("label", self.type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "type": self.type, "label": self.label,
            "start": self.start, "end": self.end, "score": round(self.score, 4),
            "method": self.method, "kg_id": self.kg_id, "kg_name": self.kg_name,
        }


# =============================================================================
#  一、BERT-BiLSTM-CRF 模型定义
# =============================================================================
if HAS_TORCH:

    class CrfLayer(nn.Module):
        """
        条件随机场层（CRF）
        ------------------
        学习标签转移矩阵，在解码时用 Viterbi 求全局最优标签序列，
        避免 BiLSTM 逐帧独立打分产生的非法标签组合。
        """

        def __init__(self, num_tags: int) -> None:
            super().__init__()
            self.num_tags = num_tags
            #: 转移分数 transitions[i, j] = 从 j 转移到 i 的分数
            self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
            #: 起始 / 终止转移
            self.start_transitions = nn.Parameter(torch.randn(num_tags))
            self.end_transitions = nn.Parameter(torch.randn(num_tags))

        def forward(self, emissions: "torch.Tensor", tags: "torch.Tensor",
                    mask: "torch.Tensor") -> "torch.Tensor":
            """负对数似然损失（batch 平均）"""
            return -self._log_likelihood(emissions, tags, mask).mean()

        def _log_likelihood(self, emissions, tags, mask):
            B, L, T = emissions.shape
            score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
            for t in range(1, L):
                m = mask[:, t]
                emit = emissions[:, t].gather(1, tags[:, t : t + 1]).squeeze(1)
                trans = self.transitions[tags[:, t], tags[:, t - 1]]
                score = score + (emit + trans) * m
            last_idx = mask.long().sum(dim=1) - 1
            score = score + self.end_transitions[tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)]
            return score - self._log_partition(emissions, mask)

        def _log_partition(self, emissions, mask):
            B, L, T = emissions.shape
            # 前向算法（log-sum-exp）
            alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
            for t in range(1, L):
                m = mask[:, t].unsqueeze(1)
                emit = emissions[:, t].unsqueeze(1)                 # (B,1,T)
                trans = self.transitions.unsqueeze(0)                # (1,T,T)
                new_alpha = torch.logsumexp(alpha.unsqueeze(2) + trans, dim=1) + emit.squeeze(1)
                alpha = new_alpha * m + alpha * (1 - m)
            alpha = alpha + self.end_transitions.unsqueeze(0)
            return torch.logsumexp(alpha, dim=1)

        @torch.no_grad()
        def decode(self, emissions: "torch.Tensor", mask: "torch.Tensor") -> List[List[int]]:
            """Viterbi 解码"""
            B, L, T = emissions.shape
            paths: List[List[int]] = []
            for b in range(B):
                length = int(mask[b].sum().item())
                if length == 0:
                    paths.append([])
                    continue
                score = self.start_transitions + emissions[b, 0]
                history = []
                for t in range(1, length):
                    nxt = score.unsqueeze(1) + self.transitions
                    best_score, best_idx = nxt.max(dim=0)
                    history.append(best_idx)
                    score = best_score + emissions[b, t]
                score = score + self.end_transitions
                best_last = int(score.argmax().item())
                best_path = [best_last]
                for hist in reversed(history):
                    best_last = int(hist[best_last].item())
                    best_path.append(best_last)
                paths.append(list(reversed(best_path)))
            return paths

    class BertBiLstmCrf(nn.Module):
        """
        BERT-BiLSTM-CRF 医疗实体识别模型（PPT 指定模型）

        forward 返回 emissions，配合 CrfLayer 计算损失 / Viterbi 解码。
        """

        def __init__(
            self,
            bert_path: str = "bert-base-chinese",
            num_labels: int = len(ENTITY_LABELS),
            lstm_hidden: int = 256,
            lstm_layers: int = 1,
            dropout: float = 0.3,
            freeze_bert: bool = False,
        ) -> None:
            super().__init__()
            from transformers import AutoModel, AutoConfig  # 延迟导入

            self.num_labels = num_labels
            self.bert = AutoModel.from_pretrained(bert_path)
            hidden = AutoConfig.from_pretrained(bert_path).hidden_size
            if freeze_bert:
                for p in self.bert.parameters():
                    p.requires_grad = False

            self.dropout = nn.Dropout(dropout)
            self.bilstm = nn.LSTM(
                hidden, lstm_hidden, num_layers=lstm_layers,
                batch_first=True, bidirectional=True,
            )
            self.classifier = nn.Linear(lstm_hidden * 2, num_labels)
            self.crf = CrfLayer(num_labels)
            self._init_weights()

        def _init_weights(self) -> None:
            for name, param in self.bilstm.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.constant_(self.classifier.bias, 0.0)

        def emissions(self, input_ids, attention_mask, token_type_ids=None):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            seq = self.dropout(out.last_hidden_state)
            lstm_out, _ = self.bilstm(seq)
            lstm_out = self.dropout(lstm_out)
            return self.classifier(lstm_out)                      # (B, L, T)

        def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
            emissions = self.emissions(input_ids, attention_mask, token_type_ids)
            if labels is not None:
                return self.crf(emissions, labels, attention_mask.float())
            return emissions

        @torch.no_grad()
        def predict(self, input_ids, attention_mask, token_type_ids=None) -> List[List[int]]:
            self.eval()
            emissions = self.emissions(input_ids, attention_mask, token_type_ids)
            return self.crf.decode(emissions, attention_mask.float())


# =============================================================================
#  二、词典 + 规则实体识别（降级实现，始终可用）
# =============================================================================
class DictionaryEntityRecognizer:
    """
    基于知识图谱节点词典的实体识别
    ------------------------------
    利用 `MemoryGraphStore` 中全部实体名 + 别名构建 Trie 式词典，
    对输入文本做**最长优先双向匹配**，并按实体类型加权打分。

    这是医疗领域非常有效的实用方案（医疗实体边界清晰、术语封闭），
    同时为 BERT-BiLSTM-CRF 提供高置信度的预标注（远程监督）。
    """

    def __init__(self, store: Optional[MemoryGraphStore] = None) -> None:
        self.store = store or MemoryGraphStore.instance()
        self._build_lexicon()

    def _build_lexicon(self) -> None:
        self.lexicon: Dict[str, Tuple[str, str]] = {}   # 表面形式 -> (类型, 标准名)
        by_len: Dict[int, List[str]] = {}
        for nid, n in self.store.nodes.items():
            name = n["name"]
            if len(name) < 2:
                continue
            self.lexicon.setdefault(name, (n["type"], name))
            alias = n["properties"].get("alias") or []
            if isinstance(alias, str):
                alias = [alias]
            for a in alias:
                a = str(a).strip()
                if len(a) >= 2 and re.search(r"[\u4e00-\u9fff]", a):
                    self.lexicon.setdefault(a, (n["type"], name))
        for w in self.lexicon:
            by_len.setdefault(len(w), []).append(w)
        self._by_len = by_len
        self._max_len = max(by_len) if by_len else 1
        logger.info("实体词典构建完成：表面形式 %d 条，最长 %d 字", len(self.lexicon), self._max_len)

    # ------------------------------------------------------------------
    def recognize(self, text: str, types: Optional[Iterable[str]] = None) -> List[EntityMention]:
        text = text or ""
        n = len(text)
        if n == 0:
            return []
        type_set = set(types) if types else None
        taken = [False] * n
        mentions: List[EntityMention] = []

        # 最长优先扫描
        for i in range(n):
            if taken[i]:
                continue
            for L in range(min(self._max_len, n - i), 1, -1):
                span = text[i : i + L]
                hit = self.lexicon.get(span)
                if not hit:
                    continue
                etype, canon = hit
                if type_set and etype not in type_set:
                    continue
                if any(taken[i : i + L]):
                    continue
                for k in range(i, i + L):
                    taken[k] = True
                mentions.append(EntityMention(
                    text=span, type=etype, start=i, end=i + L,
                    score=0.95 if span == canon else 0.88,
                    method="dictionary", kg_name=canon,
                    kg_id=self.store.index.get((etype, canon), ""),
                ))
                break
        return mentions

    # ------------------------------------------------------------------
    def fuzzy_link(self, text: str, min_score: float = 0.62) -> Optional[EntityMention]:
        """把未命中的短文本模糊链接到最相似的知识图谱实体"""
        text = (text or "").strip()
        if len(text) < 2:
            return None
        best_name, best_type, best_score = "", "", 0.0
        for surface, (etype, canon) in self.lexicon.items():
            s = similarity(text, surface)
            if s > best_score:
                best_name, best_type, best_score = canon, etype, s
        if best_score >= min_score:
            return EntityMention(
                text=text, type=best_type, start=0, end=len(text),
                score=best_score, method="fuzzy", kg_name=best_name,
                kg_id=self.store.index.get((best_type, best_name), ""),
            )
        return None


# =============================================================================
#  三、统一门面
# =============================================================================
class EntityExtractor:
    """
    实体识别 + 实体链接统一入口
    --------------------------
    优先级：BERT-BiLSTM-CRF 权重 > 词典 + 模糊匹配
    并额外提供 `link_to_kg()` 完成「提及 → 图谱节点」的实体链接。
    """

    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu") -> None:
        self.device = device
        self.store = MemoryGraphStore.instance()
        self.dictionary = DictionaryEntityRecognizer(self.store)
        self.model = None
        self.tokenizer = None
        self.backend = "dictionary"

        path = Path(model_path or settings.abspath(settings.NER_MODEL_PATH))
        if HAS_TORCH and path.exists():
            self._load_torch(path)
        else:
            reason = "未安装 PyTorch" if not HAS_TORCH else f"未找到权重目录 {path}"
            logger.info("BERT-BiLSTM-CRF 权重不可用（%s），实体识别降级为词典实现", reason)

    # ------------------------------------------------------------------
    def _load_torch(self, path: Path) -> None:
        try:
            from transformers import AutoTokenizer  # noqa: F401

            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            model = BertBiLstmCrf(bert_path=str(path), num_labels=len(ENTITY_LABELS))
            state = torch.load(path / "model.pt", map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device).eval()
            self.model = model
            self.backend = "bert_bilstm_crf"
            logger.info("BERT-BiLSTM-CRF 实体识别模型加载成功：%s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载 BERT-BiLSTM-CRF 权重失败：%s，降级为词典实现", exc)
            self.model = None
            self.backend = "dictionary"

    # ------------------------------------------------------------------
    def recognize(self, text: str, types: Optional[Iterable[str]] = None) -> List[EntityMention]:
        if self.backend == "bert_bilstm_crf" and self.model is not None:
            try:
                return self._recognize_with_model(text, types)
            except Exception as exc:  # noqa: BLE001
                logger.warning("模型推理失败，回落词典实现：%s", exc)
        return self.dictionary.recognize(text, types)

    # ------------------------------------------------------------------
    def _recognize_with_model(self, text: str, types: Optional[Iterable[str]]) -> List[EntityMention]:
        """
        真实 BERT-BiLSTM-CRF 推理
        --------------------------
        流程：tokenizer → model.predict → BIO 标签 → 合并 span → 实体提及
        说明：由于 BERT 采用字级中文切分，token 与字符基本一一对应，
              这里按 word_ids 映射回原文字符位置（兼容 subword）。
        """
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=256, return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        pred = self.model.predict(enc["input_ids"], enc["attention_mask"],
                                  enc.get("token_type_ids"))[0]

        mentions: List[EntityMention] = []
        cur: Optional[Dict[str, Any]] = None
        for idx, tag_id in enumerate(pred):
            tag = ID2LABEL.get(tag_id, "O")
            s, e = offsets[idx] if idx < len(offsets) else (0, 0)
            if s == e:      # special token
                continue
            if tag.startswith("B-"):
                if cur:
                    mentions.append(self._finalize(cur, text))
                cur = {"type": tag[2:], "start": s, "end": e}
            elif tag.startswith("I-") and cur and cur["type"] == tag[2:]:
                cur["end"] = e
            else:
                if cur:
                    mentions.append(self._finalize(cur, text))
                    cur = None
        if cur:
            mentions.append(self._finalize(cur, text))

        type_set = set(types) if types else None
        return [m for m in mentions if not type_set or m.type in type_set]

    @staticmethod
    def _finalize(cur: Dict[str, Any], text: str) -> EntityMention:
        return EntityMention(
            text=text[cur["start"] : cur["end"]], type=cur["type"],
            start=cur["start"], end=cur["end"], score=0.92, method="bert_bilstm_crf",
        )

    # ------------------------------------------------------------------
    def link_to_kg(self, mentions: Sequence[EntityMention]) -> List[EntityMention]:
        """
        实体链接（Entity Linking）：把识别出的提及对齐到知识图谱节点
          * 精确名匹配 → 别名匹配 → 模糊匹配
          * 结果写回 kg_id / kg_name
        """
        out: List[EntityMention] = []
        for m in mentions:
            if m.kg_id:
                out.append(m)
                continue
            nid = self.store.resolve(m.text, m.type) or self.store.resolve(m.text)
            if nid:
                node = self.store.nodes[nid]
                m.kg_id = nid
                m.kg_name = node["name"]
                m.type = node["type"]
                m.label = ENTITY_TYPE_META.get(node["type"], {}).get("label", node["type"])
                if node["name"] != m.text:
                    m.score = min(m.score, 0.9)
                    m.method = m.method + "+alias" if "alias" not in m.method else m.method
                out.append(m)
                continue
            fuzzy = self.dictionary.fuzzy_link(m.text)
            if fuzzy:
                m.kg_id = fuzzy.kg_id
                m.kg_name = fuzzy.kg_name
                m.type = fuzzy.type
                m.score = round(m.score * fuzzy.score, 4)
                m.method = "fuzzy"
                out.append(m)
        return out

    # ------------------------------------------------------------------
    def analyze(self, text: str) -> Dict[str, Any]:
        """识别 + 链接一体化（供问答服务调用）"""
        raw = self.recognize(text)
        linked = self.link_to_kg(raw)
        return {
            "backend": self.backend,
            "mentions": [m.to_dict() for m in linked],
            "diseases": [m.kg_name for m in linked if m.type == "Disease"],
            "symptoms": [m.kg_name for m in linked if m.type == "Symptom"],
            "departments": [m.kg_name for m in linked if m.type == "Department"],
            "drugs": [m.kg_name for m in linked if m.type == "Drug"],
            "checks": [m.kg_name for m in linked if m.type == "Check"],
            "treatments": [m.kg_name for m in linked if m.type == "Treatment"],
        }

    # ------------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "has_torch": HAS_TORCH,
            "label_scheme": "BIO / 7 类实体",
            "entity_labels": ENTITY_LABELS,
            "lexicon_size": len(self.dictionary.lexicon),
        }


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor
