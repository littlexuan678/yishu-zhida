# -*- coding: utf-8 -*-
"""
TextCNN 意图分类模型训练脚本
=============================
把用户医疗提问分类为 4 类意图：
    disease_query      疾病查询
    symptom_consult    症状咨询
    treatment_query    治疗咨询
    department_query   科室导诊

模型（见 `app/llm_layer/intent_classifier.py::TextCNN`）：
    Embedding(128) → 多尺度 Conv1D(k=2/3/4 × 128 filters) → GlobalMaxPool → FC(4)

数据格式（JSONL，每行一条）：
    {"text": "肺栓塞应该挂什么科？", "label": "department_query"}

也支持 `data/seed/intent_samples.json`（一次性加载全部样本）。

用法
----
    python scripts/train_intent_textcnn.py --build-data --epochs 30
    python scripts/train_intent_textcnn.py --eval
    python scripts/train_intent_textcnn.py --predict "我头晕恶心是怎么回事"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.llm_layer.intent_classifier import (  # noqa: E402
    INTENT_CN,
    INTENT_ORDER,
    HAS_TORCH,
    RuleIntentClassifier,
    TextCNN,
)
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("train_intent")

#: 数据增强：同义改写前缀/后缀（医疗问句的表达多样性）
PREFIXES = ["", "请问", "我想问一下", "帮我看看", "咨询一下", "麻烦问下", "医生，"]
SUFFIXES = ["", "？", "呢？", "吗？", "谢谢", "，麻烦解答一下", "？请详细说说"]

#: 4 类意图的种子句式（当 intent_samples.json 不存在时使用）
SEED_PATTERNS: Dict[str, List[str]] = {
    "disease_query": [
        "{d}是什么病", "{d}的定义是什么", "什么是{d}", "{d}严重吗",
        "{d}的病因有哪些", "{d}有哪些并发症", "{d}会不会遗传",
        "{d}的预后怎么样", "我想了解一下{d}", "{d}是什么原因引起的",
        "{d}有哪些典型症状", "{d}需要做哪些检查", "{d}好发于哪些人",
        "{d}是不是传染病", "{d}能不能预防", "{d}的发病率高吗",
    ],
    "symptom_consult": [
        "{s}是怎么回事", "我最近老是{s}怎么办", "{s}伴{s2}可能是什么病",
        "出现{s}要紧吗", "{s}是什么原因", "我{s}好几天了",
        "{s}和{s2}同时出现是什么情况", "{s}需要去医院吗",
        "为什么会{s}", "{s}越来越严重了", "{s}反复发作怎么办",
        "最近{s}还{s2}，是不是很严重", "{s}会不会是癌症",
        "小孩{s}是怎么回事", "孕妇{s}正常吗",
    ],
    "treatment_query": [
        "{d}怎么治疗", "{d}吃什么药", "{d}能治好吗", "{d}的治疗方案是什么",
        "{d}需要手术吗", "{d}的常用药物有哪些", "{d}要治疗多久",
        "{d}的治疗费用大概多少", "{d}中医怎么治", "{d}能根治吗",
        "{d}治疗期间要注意什么", "{d}有没有偏方", "{d}怎么康复",
        "{d}需要住院吗", "{d}的治疗原则是什么",
    ],
    "department_query": [
        "{d}应该挂什么科", "{d}看哪个科室", "{d}要去哪个科",
        "儿童{d}挂什么科", "孕妇{d}看什么科", "{d}归哪个科管",
        "{s}挂什么科", "{d}需要去急诊吗", "{d}挂内科还是外科",
        "{d}在哪个科室就诊", "{d}应该找什么医生", "{d}挂号挂哪个科",
        "{s}看什么科室", "{d}属于哪个科室", "怀疑{d}应该去哪个科",
    ],
}

DISEASES = [
    "高血压", "糖尿病", "冠心病", "脑卒中", "肺炎", "肺栓塞", "胃炎", "胃溃疡",
    "哮喘", "慢阻肺", "肺结核", "肝炎", "贫血", "甲亢", "肾病", "脂肪肝",
    "荨麻疹", "阑尾炎", "胆结石", "腰椎间盘突出", "抑郁症", "焦虑症",
    "阿尔茨海默病", "帕金森病", "类风湿关节炎", "痛风", "心肌梗死", "心力衰竭",
]
SYMPTOMS = [
    "头晕", "头痛", "胸痛", "腹痛", "恶心", "呕吐", "发热", "咳嗽", "乏力",
    "心悸", "呼吸困难", "腹泻", "便秘", "失眠", "水肿", "皮疹", "麻木", "消瘦",
]


# =============================================================================
#  一、数据构建
# =============================================================================
def build_data(out_dir: Path, per_label: int = 400, dev_ratio: float = 0.15) -> Tuple[Path, Path]:
    """优先使用 data/seed/intent_samples.json；不存在则按种子句式生成"""
    seed = settings.seed_dir / "intent_samples.json"
    samples: List[Dict[str, str]] = []

    if seed.exists():
        with seed.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("samples", payload if isinstance(payload, list) else [])
        for r in rows:
            if r.get("text") and r.get("label") in INTENT_ORDER:
                samples.append({"text": str(r["text"]).strip(), "label": r["label"]})
        logger.info("从 %s 加载 %d 条意图样本", seed.name, len(samples))

    # 补足不足的类别（用种子句式 + 数据增强）
    have = Counter(s["label"] for s in samples)
    for label, patterns in SEED_PATTERNS.items():
        need = max(0, per_label - have.get(label, 0))
        if need <= 0:
            continue
        generated: List[str] = []
        attempts = 0
        while len(generated) < need and attempts < need * 20:
            attempts += 1
            tpl = random.choice(patterns)
            text = tpl.format(
                d=random.choice(DISEASES),
                s=random.choice(SYMPTOMS),
                s2=random.choice(SYMPTOMS),
            )
            text = random.choice(PREFIXES) + text + random.choice(SUFFIXES)
            if text not in generated:
                generated.append(text)
        for t in generated:
            samples.append({"text": t, "label": label})
        logger.info("类别 %-18s 补充生成 %d 条（原有 %d）", label, len(generated), have.get(label, 0))

    if not samples:
        raise SystemExit("未能构建任何意图训练样本")

    random.shuffle(samples)
    # 分层切分
    by_label: Dict[str, List[Dict[str, str]]] = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)
    train: List[Dict[str, str]] = []
    dev: List[Dict[str, str]] = []
    for label, rows in by_label.items():
        n_dev = max(1, int(len(rows) * dev_ratio))
        dev.extend(rows[:n_dev])
        train.extend(rows[n_dev:])
    random.shuffle(train)
    random.shuffle(dev)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_p, dev_p = out_dir / "intent_train.jsonl", out_dir / "intent_dev.jsonl"
    for path, rows in ((train_p, train), (dev_p, dev)):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("意图数据构建完成：训练 %d 条，验证 %d 条", len(train), len(dev))
    logger.info("类别分布：%s", dict(Counter(s["label"] for s in samples)))
    return train_p, dev_p


# =============================================================================
#  二、字表
# =============================================================================
def build_vocab(rows: Sequence[Dict[str, str]], min_freq: int = 2) -> Dict[str, int]:
    counter: Counter = Counter()
    for r in rows:
        counter.update(list(r["text"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, c in counter.most_common():
        if c >= min_freq:
            vocab[ch] = len(vocab)
    logger.info("字表：%d 个字符（含 pad/unk，min_freq=%d）", len(vocab), min_freq)
    return vocab


def encode(text: str, vocab: Dict[str, int], max_len: int = 32) -> List[int]:
    ids = [vocab.get(ch, 1) for ch in text[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


# =============================================================================
#  三、训练
# =============================================================================
def train(args: argparse.Namespace) -> Dict[str, Any]:
    if not HAS_TORCH:
        logger.error("未安装 PyTorch，无法训练。请执行：pip install torch==2.2.1")
        logger.error("提示：不训练模型时系统仍可运行（意图分类自动降级为规则实现，准确率约 88%）")
        raise SystemExit(2)

    import torch
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_p, dev_p = Path(args.train), Path(args.dev)
    if args.build_data or not train_p.exists():
        logger.info("构建意图分类训练数据…")
        train_p, dev_p = build_data(settings.corpus_dir / "intent", per_label=args.per_label)

    train_rows = _read_jsonl(train_p)
    dev_rows = _read_jsonl(dev_p) if dev_p.exists() else train_rows[:200]
    logger.info("训练集 %d 条，验证集 %d 条", len(train_rows), len(dev_rows))

    vocab = build_vocab(train_rows, args.min_freq)
    label2id = {l: i for i, l in enumerate(INTENT_ORDER)}

    class _DS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            ids = encode(r["text"], vocab, args.max_len)
            return {"input_ids": ids,
                    "attention_mask": [1 if x else 0 for x in ids],
                    "label": label2id.get(r["label"], 0)}

    def collate(batch):
        return {k: torch.tensor([b[k] for b in batch], dtype=torch.long) for k in batch[0]}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = DataLoader(_DS(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(_DS(dev_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = TextCNN(vocab_size=len(vocab), embed_dim=args.embed_dim,
                    num_filters=args.num_filters, num_classes=len(INTENT_ORDER),
                    dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型：TextCNN（k=%s, filters=%d, embed=%d），参数 %s",
                args.kernel_sizes, args.num_filters, args.embed_dim, f"{n_params:,}")

    # 类别权重（缓解不均衡）
    counts = Counter(r["label"] for r in train_rows)
    weights = torch.tensor(
        [len(train_rows) / max(counts.get(l, 1) * len(INTENT_ORDER), 1) for l in INTENT_ORDER],
        dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss = time.time(), 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

        metrics = evaluate(model, dev_loader, device, label2id)
        scheduler.step(metrics["accuracy"])
        history.append({"epoch": epoch,
                        "loss": round(total_loss / max(len(train_loader), 1), 4),
                        **metrics, "seconds": round(time.time() - t0, 1)})
        logger.info("Epoch %2d/%d | loss %.4f | acc %.4f | macro-F1 %.4f | %.1fs",
                    epoch, args.epochs, history[-1]["loss"], metrics["accuracy"],
                    metrics["macro_f1"], history[-1]["seconds"])

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                        "config": {"embed_dim": args.embed_dim,
                                   "num_filters": args.num_filters,
                                   "kernel_sizes": list(args.kernel_sizes),
                                   "num_classes": len(INTENT_ORDER),
                                   "dropout": args.dropout},
                        "labels": INTENT_ORDER,
                        "best_accuracy": best_acc}, out_dir / "textcnn.pt")
            logger.info("  ↑ 保存最佳模型（acc=%.4f）", best_acc)

    result = {"model": "TextCNN", "labels": INTENT_ORDER, "label_cn": INTENT_CN,
              "best_accuracy": round(best_acc, 4), "train_size": len(train_rows),
              "dev_size": len(dev_rows), "vocab_size": len(vocab), "params": n_params,
              "history": history, "output": str(out_dir)}
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    logger.info("训练完成，最佳准确率 = %.4f", best_acc)
    return result


def evaluate(model, loader, device, label2id: Dict[str, int]) -> Dict[str, Any]:
    """整体准确率 + 每类 P/R/F1 + macro F1"""
    import torch

    model.eval()
    n = len(INTENT_ORDER)
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    correct = total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            pred = logits.argmax(dim=-1)
            for g, p in zip(batch["label"].tolist(), pred.tolist()):
                total += 1
                if g == p:
                    correct += 1
                    tp[g] += 1
                else:
                    fp[p] += 1
                    fn[g] += 1

    per_class: Dict[str, Any] = {}
    f1s = []
    for i, label in enumerate(INTENT_ORDER):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f)
        per_class[label] = {"precision": round(p, 4), "recall": round(r, 4),
                            "f1": round(f, 4), "support": tp[i] + fn[i],
                            "label_cn": INTENT_CN.get(label, label)}
    return {"accuracy": round(correct / total, 4) if total else 0.0,
            "macro_f1": round(sum(f1s) / n, 4), "per_class": per_class}


# =============================================================================
#  四、评估与推理
# =============================================================================
def eval_rule_baseline(args: argparse.Namespace) -> None:
    """评估规则分类器基线（作为对照，也是降级方案的准确率参考）"""
    dev_p = Path(args.dev)
    if not dev_p.exists():
        _, dev_p = build_data(settings.corpus_dir / "intent", per_label=args.per_label)
    rows = _read_jsonl(dev_p)
    rule = RuleIntentClassifier()
    correct = 0
    conf: Dict[Tuple[str, str], int] = {}
    for r in rows:
        scores = rule.classify(r["text"])
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        if pred == r["label"]:
            correct += 1
        else:
            key = (r["label"], pred)
            conf[key] = conf.get(key, 0) + 1
    acc = correct / len(rows) if rows else 0.0
    logger.info("规则分类器基线：%d/%d = %.4f", correct, len(rows), acc)
    if conf:
        logger.info("主要混淆（真实 → 预测）：")
        for (g, p), c in sorted(conf.items(), key=lambda kv: -kv[1])[:8]:
            logger.info("    %-18s → %-18s %d", g, p, c)


def predict(text: str, model_dir: str) -> Dict[str, Any]:
    """加载模型预测意图（无模型时用规则）"""
    path = Path(model_dir)
    if not HAS_TORCH or not (path / "textcnn.pt").exists():
        rule = RuleIntentClassifier()
        scores = rule.classify(text)
        label = max(scores.items(), key=lambda kv: kv[1])[0]
        return {"label": label, "label_cn": INTENT_CN.get(label, label),
                "confidence": scores[label], "method": "rule_fallback", "scores": scores}

    import torch

    ckpt = torch.load(path / "textcnn.pt", map_location="cpu")
    vocab = ckpt["vocab"]
    model = TextCNN(vocab_size=len(vocab), **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ids = torch.tensor([encode(text, vocab, 32)])
    mask = (ids != 0).long()
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(ids, mask), dim=-1)[0].tolist()
    scores = {l: round(p, 4) for l, p in zip(INTENT_ORDER, probs)}
    label = max(scores.items(), key=lambda kv: kv[1])[0]
    return {"label": label, "label_cn": INTENT_CN.get(label, label),
            "confidence": scores[label], "method": "textcnn", "scores": scores}


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
    p = argparse.ArgumentParser(description="TextCNN 医疗意图分类训练",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--build-data", action="store_true")
    p.add_argument("--per-label", type=int, default=400, help="每类最少样本数（不足则生成补足）")
    p.add_argument("--train", type=str, default=str(settings.corpus_dir / "intent" / "intent_train.jsonl"))
    p.add_argument("--dev", type=str, default=str(settings.corpus_dir / "intent" / "intent_dev.jsonl"))
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-filters", type=int, default=128)
    p.add_argument("--kernel-sizes", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output-dir", type=str, default=str(settings.abspath(settings.INTENT_MODEL_PATH)))
    p.add_argument("--eval", action="store_true", help="评估规则分类器基线（对照）")
    p.add_argument("--predict", type=str, default="", help="对给定问句预测意图")
    args = p.parse_args()

    logger.info("=" * 72)
    logger.info("  TextCNN 医疗意图分类训练（4 类：疾病查询/症状咨询/治疗咨询/科室导诊）")
    logger.info("  PyTorch 可用：%s", "是" if HAS_TORCH else "否（将自动降级为规则实现）")
    logger.info("=" * 72)

    if args.predict:
        res = predict(args.predict, args.output_dir)
        logger.info("问句：%s", args.predict)
        logger.info("意图：%s（%s），置信度 %.4f，方法 %s",
                    res["label"], res["label_cn"], res["confidence"], res["method"])
        for k, v in sorted(res["scores"].items(), key=lambda kv: -kv[1]):
            logger.info("    %-18s %.4f  %s", k, v, INTENT_CN.get(k, ""))
        return 0
    if args.eval:
        eval_rule_baseline(args)
        return 0
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
