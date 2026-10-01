# -*- coding: utf-8 -*-
"""
BERT-BiLSTM-CRF 医疗实体识别模型训练脚本
=========================================
模型结构（见 `app/kg_layer/entity_extractor.py::BertBiLstmCrf`）：
    BERT 编码 → BiLSTM 上下文建模 → CRF 转移约束解码

为什么医疗实体识别需要 CRF：
    BiLSTM 逐帧独立分类会产生非法标签序列（如 O 后面跟 I-Disease 而没有 B-Disease）。
    医疗实体的边界识别尤其重要——"慢性阻塞性肺疾病"若被切成"阻塞性肺疾病"，
    实体链接就会失败。CRF 的标签转移矩阵能从根本上消除这类错误。

训练数据格式（JSONL，每行一条）：
    {"text": "患者因突发性呼吸困难伴胸痛就诊。",
     "entities": [{"text": "突发性呼吸困难", "type": "Symptom", "start": 4, "end": 11},
                  {"text": "胸痛", "type": "Symptom", "start": 12, "end": 14}]}

自动构建远程监督数据：以疾病种子库的实体名做**字符串匹配标注**（词典远程监督），
这是医疗 NER 最实用的冷启动方案。

用法
----
    python scripts/train_ner_bert.py --build-data --epochs 8 --batch-size 16
    python scripts/train_ner_bert.py --bert models/bert-base-chinese --epochs 10
    python scripts/train_ner_bert.py --predict "我最近头晕还恶心，血压有点高"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.kg_layer.entity_extractor import (  # noqa: E402
    ENTITY_LABELS,
    ID2LABEL,
    LABEL2ID,
    HAS_TORCH,
)
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("train_ner")

#: 从句子模板（与 PCNN 训练脚本共用句式风格，保证数据一致性）
SENTENCE_TEMPLATES = [
    "患者{head}，主要表现为{symptom}，建议行{check}。",
    "既往有{head}病史，近期出现{symptom}，予以{treatment}。",
    "因{symptom}就诊，检查提示{check}，考虑{head}。",
    "{head}患者常见{symptom}，可应用{drug}治疗。",
    "门诊以{head}收入{department}，完善{check}。",
    "患者主诉{symptom}，既往{head}病史多年，长期服用{drug}。",
    "{head}合并{symptom}时应警惕，建议至{department}就诊。",
]


# =============================================================================
#  一、远程监督数据构建
# =============================================================================
def build_data(out_dir: Path, per_disease: int = 8, dev_ratio: float = 0.15) -> Tuple[Path, Path]:
    """基于疾病种子库构建带 BIO 标注的训练数据（词典远程监督）"""
    seed = settings.seed_dir / "diseases.json"
    if not seed.exists():
        logger.error("未找到疾病种子文件：%s", seed)
        raise SystemExit(1)
    with seed.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    diseases = payload.get("diseases", payload if isinstance(payload, list) else [])
    if not diseases:
        raise SystemExit("疾病种子库为空")

    samples: List[Dict[str, Any]] = []
    for d in diseases:
        name = d.get("name")
        if not name:
            continue
        symptoms = d.get("symptoms") or []
        checks = d.get("checks") or []
        drugs = d.get("drugs") or []
        treatments = d.get("treatments") or []
        dept = d.get("department") or d.get("category2") or d.get("category1")
        for i in range(per_disease):
            tpl = SENTENCE_TEMPLATES[i % len(SENTENCE_TEMPLATES)]
            text = tpl.format(
                head=name,
                symptom=random.choice(symptoms) if symptoms else "相应症状",
                check=random.choice(checks) if checks else "常规检查",
                drug=random.choice(drugs) if drugs else "相应药物",
                treatment=random.choice(treatments) if treatments else "对症治疗",
                department=dept or "内科",
            )
            ents = _label_by_dictionary(text, name, d)
            if ents:
                samples.append({"text": text, "entities": ents})

    if not samples:
        raise SystemExit("未能生成任何 NER 训练样本")
    random.shuffle(samples)
    n_dev = max(1, int(len(samples) * dev_ratio))
    dev, train = samples[:n_dev], samples[n_dev:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_p, dev_p = out_dir / "ner_train.jsonl", out_dir / "ner_dev.jsonl"
    for path, rows in ((train_p, train), (dev_p, dev)):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("NER 数据构建完成：训练 %d 条，验证 %d 条 → %s", len(train), len(dev), out_dir)
    stats: Dict[str, int] = {}
    for s in samples:
        for e in s["entities"]:
            stats[e["type"]] = stats.get(e["type"], 0) + 1
    logger.info("实体类型分布：%s", stats)
    return train_p, dev_p


def _label_by_dictionary(text: str, head: str, disease: Dict[str, Any]) -> List[Dict[str, Any]]:
    """用词典在句子中标注实体（远程监督）"""
    ents: List[Dict[str, Any]] = []
    candidates: List[Tuple[str, str]] = [(head, "Disease")]
    candidates += [(s, "Symptom") for s in (disease.get("symptoms") or [])]
    candidates += [(c, "Check") for c in (disease.get("checks") or [])]
    candidates += [(d, "Drug") for d in (disease.get("drugs") or [])]
    candidates += [(t, "Treatment") for t in (disease.get("treatments") or [])]
    for dept in filter(None, [disease.get("department"), disease.get("category2"),
                              disease.get("category1")]):
        candidates.append((dept, "Department"))
    pop = disease.get("population")
    if pop:
        candidates.append((pop, "Population"))

    taken = [False] * len(text)
    # 长实体优先，避免"肺疾病"覆盖"慢性阻塞性肺疾病"
    for surface, etype in sorted(candidates, key=lambda kv: -len(kv[0])):
        if not surface or len(surface) < 2:
            continue
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx < 0:
                break
            end = idx + len(surface)
            if not any(taken[idx:end]):
                for k in range(idx, end):
                    taken[k] = True
                ents.append({"text": surface, "type": etype, "start": idx, "end": end})
            start = idx + 1
    ents.sort(key=lambda e: e["start"])
    return ents


# =============================================================================
#  二、BIO 标签转换
# =============================================================================
def to_bio(text: str, entities: Sequence[Dict[str, Any]]) -> List[int]:
    """把实体列表转为逐字 BIO 标签 ID 序列"""
    labels = ["O"] * len(text)
    for e in entities:
        s, en, etype = int(e["start"]), int(e["end"]), e["type"]
        if s < 0 or en > len(text) or s >= en:
            continue
        labels[s] = f"B-{etype}"
        for i in range(s + 1, en):
            labels[i] = f"I-{etype}"
    return [LABEL2ID.get(l, 0) for l in labels]


def from_bio(text: str, label_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """BIO 标签序列还原为实体列表"""
    ents: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for i, lid in enumerate(label_ids):
        tag = ID2LABEL.get(int(lid), "O")
        if tag.startswith("B-"):
            if cur:
                ents.append(cur)
            cur = {"text": "", "type": tag[2:], "start": i, "end": i + 1}
        elif tag.startswith("I-") and cur and cur["type"] == tag[2:]:
            cur["end"] = i + 1
        else:
            if cur:
                ents.append(cur)
                cur = None
    if cur:
        ents.append(cur)
    for e in ents:
        e["text"] = text[e["start"]: e["end"]]
    return ents


# =============================================================================
#  三、训练
# =============================================================================
def train(args: argparse.Namespace) -> Dict[str, Any]:
    if not HAS_TORCH:
        logger.error("未安装 PyTorch，无法训练。请执行：pip install torch==2.2.1 transformers==4.39.1")
        logger.error("提示：不训练模型时系统仍可运行（实体识别自动降级为词典实现）")
        raise SystemExit(2)

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer

    from app.kg_layer.entity_extractor import BertBiLstmCrf

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_p, dev_p = Path(args.train), Path(args.dev)
    if args.build_data or not train_p.exists():
        logger.info("构建远程监督 NER 训练数据…")
        train_p, dev_p = build_data(settings.corpus_dir / "ner", per_disease=args.per_disease)

    train_rows = _read_jsonl(train_p)
    dev_rows = _read_jsonl(dev_p) if dev_p.exists() else train_rows[:100]
    logger.info("训练集 %d 条，验证集 %d 条", len(train_rows), len(dev_rows))

    bert_path = args.bert
    if not Path(bert_path).exists():
        logger.warning("本地 BERT 路径不存在：%s", bert_path)
        logger.warning("将尝试从 HuggingFace 下载（如网络受限，请先执行 README 第 6.1 节的下载命令）")
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    class _DS(Dataset):
        def __init__(self, rows: Sequence[Dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int) -> Dict[str, Any]:
            r = self.rows[i]
            text = r["text"][: args.max_len]
            labels = to_bio(text, r.get("entities") or [])[: args.max_len]
            enc = tokenizer(text, truncation=True, max_length=args.max_len,
                            padding="max_length", return_tensors=None)
            # 中文 BERT 为字级切分，token 与字符基本一一对应；
            # 对 [CLS]/[SEP]/padding 位置把标签设为 -100（忽略）
            ids = enc["input_ids"]
            bio = [-100] * len(ids)
            for j in range(len(labels)):
                if j + 1 < len(ids) - 1:      # 跳过 [CLS]
                    bio[j + 1] = labels[j]
            return {
                "input_ids": ids,
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0] * len(ids)),
                "labels": bio,
            }

    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        keys = batch[0].keys()
        return {k: torch.tensor([b[k] for b in batch], dtype=torch.long) for k in keys}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = DataLoader(_DS(train_rows), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    dev_loader = DataLoader(_DS(dev_rows), batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    model = BertBiLstmCrf(
        bert_path=bert_path, num_labels=len(ENTITY_LABELS),
        lstm_hidden=args.lstm_hidden, dropout=args.dropout,
        freeze_bert=args.freeze_bert,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型：BERT-BiLSTM-CRF（%d 标签），可训练参数 %s",
                len(ENTITY_LABELS), f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss = time.time(), 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            loss = model(batch["input_ids"], batch["attention_mask"],
                         labels=labels, token_type_ids=batch.get("token_type_ids"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        metrics = evaluate_ner(model, dev_loader, dev_rows, tokenizer, device, args.max_len)
        history.append({"epoch": epoch,
                        "loss": round(total_loss / max(len(train_loader), 1), 4),
                        **metrics, "seconds": round(time.time() - t0, 1)})
        logger.info("Epoch %2d/%d | loss %.4f | 实体级 P %.4f R %.4f F1 %.4f | %.1fs",
                    epoch, args.epochs, history[-1]["loss"], metrics["precision"],
                    metrics["recall"], metrics["f1"], history[-1]["seconds"])

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), out_dir / "model.pt")
            tokenizer.save_pretrained(str(out_dir))
            logger.info("  ↑ 保存最佳模型（F1=%.4f）", best_f1)

    result = {
        "model": "BERT-BiLSTM-CRF",
        "bert": bert_path,
        "labels": ENTITY_LABELS,
        "best_f1": round(best_f1, 4),
        "train_size": len(train_rows),
        "dev_size": len(dev_rows),
        "params": n_params,
        "history": history,
        "output": str(out_dir),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("训练完成，最佳实体级 F1 = %.4f", best_f1)
    return result


def evaluate_ner(model, loader, rows, tokenizer, device, max_len: int) -> Dict[str, Any]:
    """实体级（严格匹配）精确率 / 召回率 / F1"""
    import torch

    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            tt = batch.get("token_type_ids")
            tt = tt.to(device) if tt is not None else None
            preds = model.predict(ids, am, tt)
            for bi, path in enumerate(preds):
                text = rows[bi]["text"][:max_len]
                gold = {(e["start"], e["end"], e["type"])
                        for e in (rows[bi].get("entities") or []) if e["end"] <= len(text)}
                # 预测标签回映到字符位置（+1 因跳过 [CLS]）
                char_ids = [0] * len(text)
                for j, lid in enumerate(path):
                    ci = j - 1
                    if 0 <= ci < len(text):
                        char_ids[ci] = lid
                pred = {(e["start"], e["end"], e["type"]) for e in from_bio(text, char_ids)}
                tp += len(gold & pred)
                fp += len(pred - gold)
                fn += len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


# =============================================================================
#  四、推理
# =============================================================================
def predict(text: str, model_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载模型并对文本做实体识别"""
    if not HAS_TORCH:
        logger.error("未安装 PyTorch，改用词典实现（app.kg_layer.entity_extractor）")
        from app.kg_layer.entity_extractor import get_entity_extractor

        return [m.to_dict() for m in get_entity_extractor().recognize(text)]

    import torch
    from transformers import AutoTokenizer

    from app.kg_layer.entity_extractor import BertBiLstmCrf

    path = Path(model_dir or settings.abspath(settings.NER_MODEL_PATH))
    if not path.exists():
        logger.error("模型目录不存在：%s（请先训练）", path)
        return []

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = BertBiLstmCrf(bert_path=str(path), num_labels=len(ENTITY_LABELS))
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    model.eval()

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        paths = model.predict(enc["input_ids"], enc["attention_mask"],
                              enc.get("token_type_ids"))
    char_ids = [0] * len(text)
    for j, lid in enumerate(paths[0]):
        ci = j - 1
        if 0 <= ci < len(text):
            char_ids[ci] = lid
    return from_bio(text, char_ids)


# =============================================================================
#  五、工具
# =============================================================================
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description="BERT-BiLSTM-CRF 医疗实体识别训练",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--build-data", action="store_true", help="从疾病种子库构建远程监督训练数据")
    p.add_argument("--per-disease", type=int, default=8, help="每病生成句数")
    p.add_argument("--train", type=str, default=str(settings.corpus_dir / "ner" / "ner_train.jsonl"))
    p.add_argument("--dev", type=str, default=str(settings.corpus_dir / "ner" / "ner_dev.jsonl"))
    p.add_argument("--bert", type=str, default=str(settings.abspath(settings.BERT_BASE_PATH)),
                   help="BERT 底座路径（默认 models/bert-base-chinese）")
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--freeze-bert", action="store_true", help="冻结 BERT（小数据防过拟合）")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output-dir", type=str, default=str(settings.abspath(settings.NER_MODEL_PATH)))
    p.add_argument("--predict", type=str, default="", help="对给定文本做实体识别推理")
    args = p.parse_args()

    logger.info("=" * 72)
    logger.info("  BERT-BiLSTM-CRF 医疗实体识别训练")
    logger.info("  PyTorch 可用：%s", "是" if HAS_TORCH else "否（将自动降级为词典实现）")
    logger.info("=" * 72)

    if args.predict:
        ents = predict(args.predict, args.output_dir)
        logger.info("文本：%s", args.predict)
        for e in ents:
            logger.info("  「%s」 → %s [%d, %d)", e["text"], e["type"], e["start"], e["end"])
        if not ents:
            logger.info("  未识别到实体（可先用 --build-data 训练模型）")
        return 0

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
