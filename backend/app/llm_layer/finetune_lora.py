# -*- coding: utf-8 -*-
"""
Llama 3 医疗领域指令微调（LoRA / QLoRA）
=========================================
PPT 要求：「利用高质量医疗问答对，针对 Llama 3 / Mistral 等开源大模型进行
         指令微调（Instruction-tuning），使其快速适配医疗领域的专业术语与问答逻辑。」

为什么用 LoRA 而不是全参微调：
  * Llama 3 8B 全参微调需 ≥8×A100-80G；LoRA 只训练 ~0.1% 参数，单张 24G 显卡可行
  * LoRA 权重仅 ~40MB，便于版本管理与多场景切换（医疗问答 / 导诊 / 风险解读）
  * 基座模型保持只读，**知识不会被"写进"参数**，与 RAG 的可溯源定位不冲突:
    RAG 负责"事实正确与来源可追溯"，LoRA 负责"医疗术语与回答格式规范"

训练数据格式（JSONL，Alpaca 风格）：
    {"instruction": "肺栓塞应该挂什么科？",
     "input": "",
     "output": "根据知识库，肺栓塞应就诊于呼吸内科 [KG-1]。...",
     "system": "你是智愈医典的 AI 医生助手..."}          # system 可选

数据构建建议
------------
用本系统的 RAG 引擎在**高质量种子问答**上生成答案，再经医学专家校验，
即「self-instruct + 人工审核」流程（见 `scripts/build_sft_data.py` 的说明）。

用法
----
    # 1) 标准 LoRA 微调（需 16G+ 显存）
    python -m app.llm_layer.finetune_lora \
        --base_model models/Meta-Llama-3-8B-Instruct \
        --data_path data/seed/medical_qa_sft.jsonl \
        --output_dir models/llama3-medical-lora \
        --lora_r 16 --lora_alpha 32 --num_epochs 3

    # 2) QLoRA（4-bit 量化，单张 12G 显卡可跑）
    python -m app.llm_layer.finetune_lora --load_in_4bit --lora_r 8

    # 3) 仅做数据校验（不训练，检查数据格式与长度分布）
    python -m app.llm_layer.finetune_lora --data_path data/seed/medical_qa_sft.jsonl --validate_only

    # 4) 合并 LoRA 到基座（导出可独立部署的模型）
    python -m app.llm_layer.finetune_lora --merge_only \
        --base_model models/Meta-Llama-3-8B-Instruct \
        --output_dir models/llama3-medical-merged
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("finetune_lora")


#: Llama 3 官方对话模板（ChatML 变体，务必与推理端保持一致，否则效果显著下降）
LLAMA3_CHAT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
)

#: 医疗微调默认 system prompt（约束输出格式与安全边界）
DEFAULT_SYSTEM = (
    "你是\"智愈医典\"医疗知识问答系统的 AI 医生助手。\n"
    "回答必须严格基于给定的知识图谱事实，不得编造；\n"
    "每条医学陈述后标注引用编号（如 [KG-1]）；\n"
    "不得给出具体用药剂量，不得给出确定性诊断；\n"
    "输出末尾必须包含：本回答由 AI 生成，仅供参考，不能替代执业医师诊断。"
)

#: LoRA 默认挂载的目标模块（Llama 3 全注意力 + MLP 投影层）
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# =============================================================================
#  一、数据准备
# =============================================================================
def load_sft_data(path: Path) -> List[Dict[str, Any]]:
    """加载并校验 SFT 数据"""
    if not path.exists():
        logger.error("❌ SFT 数据文件不存在：%s", path)
        logger.error("   请准备 medical_qa_sft.jsonl，每行形如：")
        logger.error('   {"instruction": "肺栓塞应该挂什么科？", "input": "", "output": "..."}')
        raise SystemExit(1)

    rows: List[Dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("第 %d 行 JSON 解析失败：%s", i, exc)
                bad += 1
                continue
            # 兼容两种字段命名
            instr = obj.get("instruction") or obj.get("question") or obj.get("prompt")
            out = obj.get("output") or obj.get("answer") or obj.get("response")
            if not instr or not out:
                bad += 1
                continue
            rows.append({
                "system": obj.get("system") or DEFAULT_SYSTEM,
                "instruction": str(instr).strip(),
                "input": str(obj.get("input") or "").strip(),
                "output": str(out).strip(),
            })
    logger.info("SFT 数据加载完成：有效 %d 条，跳过 %d 条", len(rows), bad)
    if not rows:
        raise SystemExit("没有有效的训练样本")
    return rows


def validate_sft_data(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """统计长度分布与安全检查，帮助发现数据质量问题"""
    lens = sorted(len(r["instruction"]) + len(r["output"]) for r in rows)
    n = len(lens)

    def pct(p: float) -> int:
        return lens[min(n - 1, int(n * p))]

    risky = [r for r in rows if any(
        k in r["output"] for k in ("mg", "毫克", "每日", "一天", "片/次", "tid", "bid")
    )]
    no_cite = [r for r in rows if "[KG-" not in r["output"]]
    no_disclaimer = [r for r in rows if "不能替代执业医师" not in r["output"]]

    stats = {
        "samples": n,
        "char_total": {"min": lens[0], "p25": pct(0.25), "median": pct(0.5),
                       "p75": pct(0.75), "p95": pct(0.95), "max": lens[-1]},
        "outputs_with_dose_words": len(risky),
        "outputs_without_kg_citation": len(no_cite),
        "outputs_without_disclaimer": len(no_disclaimer),
    }
    logger.info("=" * 66)
    logger.info("  SFT 数据质量校验")
    logger.info("  样本数            : %d", stats["samples"])
    logger.info("  字符长度          : min=%d p50=%d p95=%d max=%d",
                lens[0], pct(0.5), pct(0.95), lens[-1])
    logger.info("  含剂量表述        : %d  （应接近 0，医疗微调必须规避剂量）", stats["outputs_with_dose_words"])
    logger.info("  缺少 [KG-n] 引用  : %d  （建议 <5%%）", stats["outputs_without_kg_citation"])
    logger.info("  缺少免责声明      : %d  （建议 0）", stats["outputs_without_disclaimer"])
    logger.info("=" * 66)
    return stats


def format_prompts(rows: List[Dict[str, Any]], tokenizer: Any = None) -> List[Dict[str, str]]:
    """渲染为 Llama 3 对话格式的 text 字段"""
    out: List[Dict[str, str]] = []
    for r in rows:
        user = r["instruction"]
        if r.get("input"):
            user = f"{user}\n\n补充信息：{r['input']}"
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": r["system"]},
                 {"role": "user", "content": user},
                 {"role": "assistant", "content": r["output"]}],
                tokenize=False, add_generation_prompt=False,
            )
        else:
            text = LLAMA3_CHAT_TEMPLATE.format(
                system=r["system"], user=user, assistant=r["output"])
        out.append({"text": text})
    return out


# =============================================================================
#  二、训练
# =============================================================================
def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            DataCollatorForLanguageModeling, Trainer, TrainingArguments,
        )
    except ImportError as exc:
        logger.error("缺少微调依赖：%s", exc)
        logger.error("请执行：pip install torch transformers datasets peft accelerate bitsandbytes")
        logger.error("")
        logger.error("说明：本模块是**可选的进阶能力**。不安装这些依赖时，")
        logger.error("      系统仍可通过 RAG（prompt 工程）获得高质量医疗问答，")
        logger.error("      微调仅用于进一步规范输出风格与术语一致性。")
        raise SystemExit(2)

    rows = load_sft_data(Path(args.data_path))
    validate_sft_data(rows)

    logger.info("加载分词器：%s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- 量化配置（QLoRA）----
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("启用 QLoRA（4-bit NF4 量化）")

    logger.info("加载基座模型：%s（dtype=%s）", args.base_model, args.dtype)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto" if not args.cpu else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # ---- LoRA 配置 ----
    target_modules = args.target_modules or DEFAULT_TARGET_MODULES
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters() if hasattr(model, "get_nb_trainable_parameters") else (0, 0)
    if total:
        logger.info("LoRA 已挂载：可训练参数 %s / 总参数 %s（%.3f%%）",
                    f"{trainable:,}", f"{total:,}", trainable / total * 100)
    logger.info("目标模块：%s", target_modules)

    # ---- 数据集 ----
    formatted = format_prompts(rows, tokenizer)

    def tokenize_fn(example: Dict[str, str]) -> Dict[str, Any]:
        out = tokenizer(
            example["text"], truncation=True,
            max_length=args.max_seq_length, padding=False,
        )
        out["labels"] = list(out["input_ids"])
        return out

    dataset = Dataset.from_list(formatted).map(
        tokenize_fn, remove_columns=["text"], desc="tokenizing")
    split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed) \
        if args.val_ratio > 0 else {"train": dataset, "test": None}
    logger.info("数据集：训练 %d 条，验证 %d 条",
                len(split["train"]), len(split["test"]) if split["test"] else 0)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- 训练参数 ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if split["test"] else "no",
        save_total_limit=2,
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
    )

    logger.info("=" * 66)
    logger.info("  开始 LoRA 医疗指令微调")
    logger.info("  基座=%s", args.base_model)
    logger.info("  r=%d alpha=%d dropout=%.2f lr=%s epochs=%d",
                args.lora_r, args.lora_alpha, args.lora_dropout, args.learning_rate, args.num_epochs)
    logger.info("=" * 66)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练元信息
    meta = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "train_samples": len(split["train"]),
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "chat_template": "llama3",
    }
    (Path(args.output_dir) / "finetune_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("")
    logger.info("✅ 微调完成，LoRA 适配器已保存：%s", args.output_dir)
    logger.info("")
    logger.info("启用方式（.env）：")
    logger.info("  LLM_LORA_PATH=%s", args.output_dir)
    logger.info("")
    logger.info("推理端加载示例：")
    logger.info("  from peft import PeftModel")
    logger.info("  model = PeftModel.from_pretrained(base_model, '%s')", args.output_dir)


# =============================================================================
#  三、合并 LoRA 到基座
# =============================================================================
def merge_lora(args: argparse.Namespace) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        logger.error("缺少依赖：%s（pip install torch transformers peft）", exc)
        raise SystemExit(2)

    lora_path = args.lora_path or settings.LLM_LORA_PATH
    if not lora_path or not Path(lora_path).exists():
        logger.error("找不到 LoRA 适配器：%s", lora_path)
        raise SystemExit(1)

    logger.info("加载基座：%s", args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    logger.info("挂载 LoRA：%s", lora_path)
    model = PeftModel.from_pretrained(base, lora_path)
    logger.info("合并权重中…")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("✅ 合并完成：%s", args.output_dir)
    logger.info("   该目录可直接用 vLLM / Ollama 加载部署：")
    logger.info("   python -m vllm.entrypoints.openai.api_server --model %s --port 8001", args.output_dir)


# =============================================================================
#  四、CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Llama 3 医疗领域指令微调（LoRA / QLoRA）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    # 模型与数据
    p.add_argument("--base_model", type=str, default="models/Meta-Llama-3-8B-Instruct",
                   help="Llama 3 基座模型路径或 HuggingFace 名称")
    p.add_argument("--data_path", type=str,
                   default=str(settings.seed_dir / "medical_qa_sft.jsonl"),
                   help="SFT 数据 JSONL 路径")
    p.add_argument("--output_dir", type=str, default="models/llama3-medical-lora",
                   help="输出目录（LoRA 适配器或合并模型）")
    p.add_argument("--lora_path", type=str, default="", help="合并模式下指定 LoRA 路径")
    # LoRA 超参
    p.add_argument("--lora_r", type=int, default=16, help="LoRA 秩（8/16/32；越大容量越强）")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA 缩放（经验值 = 2×r）")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, nargs="*", default=None,
                   help=f"LoRA 目标模块，默认 {DEFAULT_TARGET_MODULES}")
    # 训练超参
    p.add_argument("--num_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # 显存优化
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA：4-bit 量化（单张 12G 显卡可跑）")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--cpu", action="store_true", help="强制 CPU（仅供极慢的调试）")
    # 模式
    p.add_argument("--validate_only", action="store_true", help="仅校验数据，不训练")
    p.add_argument("--merge_only", action="store_true", help="仅合并 LoRA 到基座")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logger.info("=" * 66)
    logger.info("  Llama 3 医疗指令微调（LoRA / QLoRA）")
    logger.info("  定位：RAG 保证事实与溯源，LoRA 规范术语与输出风格")
    logger.info("=" * 66)

    if args.validate_only:
        rows = load_sft_data(Path(args.data_path))
        validate_sft_data(rows)
        logger.info("数据校验完成；如需训练请去掉 --validate_only")
        return 0

    if args.merge_only:
        merge_lora(args)
        return 0

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
