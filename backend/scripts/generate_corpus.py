# -*- coding: utf-8 -*-
"""
演示语料生成器
==============
用途：在没有网络/未运行爬虫的环境中，**合成一份符合 PubMed 结构规范的医学语料**，
      用于演示完整的「语料 → 清洗 → 脱敏 → 实体识别 → 关系抽取 → 写图」流水线。

生成内容：
  * `data/processed/pubmed.jsonl`      合成 PubMed 文献摘要（含真实结构字段）
  * `data/processed/med_site.jsonl`    合成结构化疾病条目

⚠️ 重要说明
-----------
本脚本生成的是**演示用合成语料**（基于疾病种子库的字段组合生成自然语言句子），
**不是真实文献**。其 `pmid` 字段为演示占位符，不得作为真实文献引用。

真实采集请使用：
    scrapy runspider app/data_layer/crawler/spiders/pubmed_spider.py -a query="hypertension"

用法
----
    python scripts/generate_corpus.py                  # 生成演示语料
    python scripts/generate_corpus.py --per-disease 12 # 每病生成 12 条文献
    python scripts/generate_corpus.py --with-noise     # 加入广告/重复噪声以演示清洗效果
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("generate_corpus")

#: 合成摘要的句式模板（模拟真实文献摘要结构：背景/方法/结果/结论）
ABSTRACT_TEMPLATES = [
    "背景：{disease}是临床常见的{disease_kind}，其发病机制尚未完全阐明。"
    "目的：探讨{disease}患者的临床特征与{dimension}的关系。"
    "方法：回顾性分析{year}年收治的{n}例{disease}患者资料。"
    "结果：患者主要临床表现为{symptoms}。影像学与实验室检查提示{checks}。"
    "治疗方面以{treatments}为主，部分患者联合使用{drugs}。"
    "结论：{disease}应早期识别并及时干预，规范治疗可改善预后。{conclusion}",

    "目的：分析{disease}的流行病学特征及易感人群。"
    "方法：纳入{disease}确诊患者{n}例，收集人口学与临床资料。"
    "结果：本组患者平均年龄约{age}岁，{population}占比较高；"
    "常见症状为{symptoms}；并发症以{complications}最为多见。"
    "结论：{population}为{disease}的重点防治对象，建议定期筛查。",

    "摘要：{disease}的诊断需结合临床表现与辅助检查。"
    "本研究对{n}例疑似{disease}患者行{checks}检查。"
    "结果显示，{checks_main}对{disease}的诊断具有较好价值。"
    "治疗方案包括{treatments}，常用药物有{drugs}。"
    "随访结果显示，规范治疗组预后优于对照组。"
    "关键词：{disease}；诊断；治疗；预后。",
]

#: 不同一级分类对应的疾病性质描述
DISEASE_KIND = {
    "内科": "慢性非传染性疾病",
    "外科": "需外科干预的疾病",
    "妇产科": "妇产科疾病",
    "儿科": "儿童常见疾病",
    "皮肤科": "皮肤疾病",
    "眼科": "眼部疾病",
    "耳鼻咽喉科": "耳鼻咽喉疾病",
    "口腔科": "口腔疾病",
    "传染科": "感染性疾病",
    "精神心理科": "精神心理疾病",
    "肿瘤科": "肿瘤性疾病",
    "急诊科": "急危重症",
}

#: 噪声样本（用于演示清洗与去重管道的效果）
NOISE_SAMPLES = [
    "{disease}的最新治疗方法！加微信 xxx123 免费领取专家方案，限时优惠！",
    "{disease}怎么治？点击购买我们的特效产品，扫码关注公众号获取更多。",
    "{disease} {disease} {disease} 广告 推广 代理加盟 微商",
    "【专家在线】{disease}咨询热线：13800138000，患者张某某，身份证 110101196503151234。",
    "版权所有 Copyright 2024 京ICP备12345678号 本站内容仅供参考 {disease}",
]


def _pick(items: List[str], n: int, default: str = "临床观察") -> str:
    if not items:
        return default
    n = min(n, len(items))
    return "、".join(random.sample(items, n))


def load_diseases() -> List[Dict[str, Any]]:
    p = settings.seed_dir / "diseases.json"
    if not p.exists():
        logger.error("❌ 未找到疾病种子文件：%s", p)
        logger.error("   请先准备 data/seed/diseases.json（疾病结构化种子库）")
        return []
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("diseases", payload if isinstance(payload, list) else [])


def gen_pubmed_docs(diseases: List[Dict[str, Any]], per_disease: int) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    base_year = 2025
    for di, d in enumerate(diseases):
        name = d.get("name", "")
        for i in range(per_disease):
            tpl = random.choice(ABSTRACT_TEMPLATES)
            symptoms = d.get("symptoms") or ["相应症状"]
            checks = d.get("checks") or ["常规检查"]
            treatments = d.get("treatments") or (["药物治疗"] if d.get("treatment") else ["对症治疗"])
            drugs = d.get("drugs") or ["相应药物"]
            complications = d.get("complications") or ["相关并发症"]
            abstract = tpl.format(
                disease=name,
                disease_kind=DISEASE_KIND.get(d.get("category1", ""), "临床疾病"),
                dimension=random.choice(["病程", "年龄", "并发症发生率", "治疗方案", "预后"]),
                year=base_year - random.randint(0, 4),
                n=random.randint(48, 860),
                age=f"{random.randint(35, 72)}.{random.randint(0, 9)}",
                symptoms=_pick(symptoms, min(4, len(symptoms))),
                checks=_pick(checks, min(3, len(checks))),
                checks_main=(checks[0] if checks else "常规检查"),
                treatments=_pick(treatments, min(2, len(treatments))),
                drugs=_pick(drugs, min(3, len(drugs))),
                complications=_pick(complications, min(2, len(complications))),
                population=d.get("population") or "中老年人群",
                conclusion=random.choice([
                    "本研究为临床实践提供了参考依据。",
                    "提示应加强该病的早期筛查与规范管理。",
                    "联合治疗方案显示出更好的临床获益。",
                    "长期随访对于改善预后具有重要意义。",
                ]),
            )
            # 演示占位 PMID（真实采集请用 pubmed_spider）
            pmid = str(30000000 + di * 100 + i)
            docs.append({
                "id": f"PMID{pmid}",
                "pmid": pmid,
                "doi": f"10.1000/zhiyu.demo.{di:04d}.{i:03d}",
                "title": f"{name}的临床特征与治疗分析（演示语料 {i + 1}）",
                "abstract": abstract,
                "keywords": [name] + (d.get("symptoms") or [])[:3],
                "mesh_terms": [name, d.get("category2") or d.get("category1") or "内科"],
                "journal": random.choice([
                    "中华内科杂志", "中华医学杂志", "中华心血管病杂志",
                    "中华结核和呼吸杂志", "中华消化杂志", "中华神经科杂志",
                    "中国实用内科杂志", "中华传染病杂志",
                ]),
                "year": base_year - random.randint(0, 4),
                "authors": [f"作者{random.choice('ABCDEFGH')}{random.randint(1, 99)}"],
                "language": "zh",
                "source": "pubmed",
                "source_name": "PubMed（演示合成）",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "authority": "B",
                "retrieved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "is_demo": True,
            })
    return docs


def gen_disease_items(diseases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合成"结构化疾病条目"（模拟医学站点抓取结果）"""
    out: List[Dict[str, Any]] = []
    for d in diseases:
        out.append({
            "disease_name": d.get("name"),
            "alias": d.get("alias") or [],
            "category1": d.get("category1"),
            "category2": d.get("category2"),
            "definition": d.get("definition"),
            "cause": d.get("cause"),
            "symptoms": d.get("symptoms") or [],
            "diagnosis": d.get("diagnosis"),
            "checks": d.get("checks") or [],
            "treatments": d.get("treatments") or [],
            "drugs": d.get("drugs") or [],
            "department": d.get("department"),
            "prognosis": d.get("prognosis"),
            "population": d.get("population"),
            "complications": d.get("complications") or [],
            "differential": d.get("differential") or [],
            "is_infectious": bool(d.get("is_infectious")),
            "source": "website",
            "url": f"https://www.example-medical-wiki.org/disease/{d.get('disease_id', '')}",
            "authority": "A",
            "retrieved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "is_demo": True,
        })
    return out


def gen_noise(diseases: List[Dict[str, Any]], n: int = 20) -> List[Dict[str, Any]]:
    """生成噪声样本，用于验证清洗管道的过滤能力"""
    out: List[Dict[str, Any]] = []
    for i in range(n):
        d = random.choice(diseases) if diseases else {"name": "示例疾病"}
        text = random.choice(NOISE_SAMPLES).format(disease=d.get("name", "该病"))
        out.append({
            "id": f"NOISE{i:04d}",
            "title": f"广告-{i}",
            "abstract": text,
            "source": "website",
            "url": f"https://spam.example.com/{i}",
            "authority": "C",
            "is_noise": True,
        })
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成演示用医学语料（非真实文献）")
    parser.add_argument("--per-disease", type=int, default=8, help="每个疾病生成的文献数")
    parser.add_argument("--with-noise", action="store_true", help="加入噪声样本（演示清洗效果）")
    parser.add_argument("--out-dir", type=str, default="", help="输出目录（默认 data/processed）")
    args = parser.parse_args()

    diseases = load_diseases()
    if not diseases:
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else settings.corpus_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("  演示语料生成（⚠️ 合成数据，非真实文献，不可作为引用来源）")
    logger.info("  疾病数：%d，每病文献数：%d", len(diseases), args.per_disease)
    logger.info("=" * 72)

    pubmed = gen_pubmed_docs(diseases, args.per_disease)
    write_jsonl(out_dir / "pubmed.jsonl", pubmed)
    logger.info("✔ data/processed/pubmed.jsonl   %d 条", len(pubmed))

    items = gen_disease_items(diseases)
    write_jsonl(out_dir / "med_site.jsonl", items)
    logger.info("✔ data/processed/med_site.jsonl %d 条", len(items))

    if args.with_noise:
        noise = gen_noise(diseases, 20)
        write_jsonl(out_dir / "noise.jsonl", noise)
        logger.info("✔ data/processed/noise.jsonl    %d 条（噪声，用于演示清洗管道）", len(noise))

    # 同时输出一份合并语料（供 load_kg.py --corpus 直接使用）
    merged = pubmed + items + (gen_noise(diseases, 20) if args.with_noise else [])
    write_jsonl(out_dir / "corpus.jsonl", merged)
    logger.info("✔ data/processed/corpus.jsonl   %d 条（合并语料）", len(merged))

    logger.info("")
    logger.info("下一步：")
    logger.info("  1) 验证清洗与脱敏：")
    logger.info("     python -c \"from app.data_layer.preprocess import Preprocessor;"
                "import json;pp=Preprocessor();"
                "rows=pp.load_jsonl('data/processed/corpus.jsonl');"
                "docs=pp.process_batch(rows);print(pp.info())\"")
    logger.info("  2) 导入图谱：")
    logger.info("     python scripts/load_kg.py --corpus data/processed/corpus.jsonl --extract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
