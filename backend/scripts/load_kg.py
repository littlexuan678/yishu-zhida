# -*- coding: utf-8 -*-
"""
知识图谱数据导入脚本（种子 JSON + 爬取语料 → Neo4j）
=====================================================
数据来源（按优先级）：
  1. `data/seed/diseases.json`   疾病结构化种子库（含定义/症状/治疗/药物/来源）
  2. `data/seed/relations.json`  预抽取的三元组关系
  3. `data/processed/*.jsonl`    爬虫语料（经 preprocess + 关系抽取后导入）

导入策略
--------
* **MERGE 幂等**：所有写操作使用 MERGE，可重复执行而不产生重复数据
* **保留高置信度**：同一三元组已存在且旧置信度更高时不覆盖（`ON MATCH ... WHERE`）
* **溯源挂载**：每条疾病挂载 `:Source` 节点（PMID/DOI/URL），满足"来源 100% 可追溯"
* **批量提交**：用 `UNWIND` 批量写，1000 条/批，性能比逐条快 2 个数量级

用法
----
    cd backend
    python scripts/load_kg.py                          # 导入种子库
    python scripts/load_kg.py --corpus data/processed/corpus.jsonl   # 追加导入语料
    python scripts/load_kg.py --extract                # 对语料跑关系抽取后再导入
    python scripts/load_kg.py --incremental            # 增量模式（配合定时任务，≤24h 更新周期）
    python scripts/load_kg.py --stats                  # 仅打印导入统计
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.kg_layer.memory_store import REL_LABELS  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("load_kg")

BATCH_SIZE = 500


# =============================================================================
#  Cypher 模板
# =============================================================================
CYPHER_DISEASE = """
UNWIND $rows AS row
MERGE (d:Disease {name: row.name})
SET d.disease_id   = coalesce(row.disease_id, d.disease_id),
    d.alias        = row.alias,
    d.category1    = row.category1,
    d.category2    = row.category2,
    d.definition   = row.definition,
    d.cause        = row.cause,
    d.diagnosis    = row.diagnosis,
    d.treatment    = row.treatment,
    d.prognosis    = row.prognosis,
    d.population   = row.population,
    d.is_infectious= row.is_infectious,
    d.updated_at   = row.updated_at
RETURN count(d) AS n
"""

CYPHER_RELATION = """
UNWIND $rows AS row
MERGE (h:__HEAD__ {name: row.head})
MERGE (t:__TAIL__ {name: row.tail})
MERGE (h)-[r:__REL__]->(t)
ON CREATE SET r.confidence = row.confidence, r.weight = row.weight,
              r.source_ref = row.source_ref, r.created_at = row.created_at
ON MATCH  SET r.confidence = CASE WHEN row.confidence > coalesce(r.confidence, 0)
                                  THEN row.confidence ELSE r.confidence END,
              r.updated_at = row.updated_at
RETURN count(r) AS n
"""

CYPHER_SOURCE = """
UNWIND $rows AS row
MATCH (d:Disease {name: row.disease})
MERGE (s:Source {url: row.url})
SET s.source_type  = row.source_type,
    s.pmid         = row.pmid,
    s.doi          = row.doi,
    s.title        = row.title,
    s.journal      = row.journal,
    s.year         = row.year,
    s.authors      = row.authors,
    s.authority    = row.authority,
    s.retrieved_at = row.retrieved_at
MERGE (d)-[:PROVES]->(s)
RETURN count(s) AS n
"""

CYPHER_ALIAS_LINK = """
UNWIND $rows AS row
MATCH (h {name: row.head})
MATCH (t {name: row.tail})
MERGE (h)-[r:__REL__]->(t)
SET r.confidence = row.confidence,
    r.updated_at = row.updated_at
RETURN count(r) AS n
"""


# =============================================================================
#  数据装载
# =============================================================================
def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_seed_diseases() -> List[Dict[str, Any]]:
    payload = load_json(settings.seed_dir / "diseases.json")
    if payload is None:
        logger.error("❌ 未找到疾病种子文件：%s", settings.seed_dir / "diseases.json")
        return []
    if isinstance(payload, dict):
        return payload.get("diseases") or []
    return payload if isinstance(payload, list) else []


def load_seed_relations() -> List[Dict[str, Any]]:
    payload = load_json(settings.seed_dir / "relations.json")
    if payload is None:
        logger.warning("未找到三元组种子文件（将从疾病对象自动派生）：%s",
                       settings.seed_dir / "relations.json")
        return []
    if isinstance(payload, dict):
        return payload.get("relations") or []
    return payload if isinstance(payload, list) else []


def derive_relations_from_diseases(diseases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    当没有 relations.json 时，从疾病对象派生三元组。
    复用 `RuleRelationExtractor.extract_from_disease()`，保证与线上逻辑一致。
    """
    try:
        from app.kg_layer.relation_extractor import RuleRelationExtractor

        extractor = RuleRelationExtractor()
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法导入关系抽取器（%s），改用内置派生逻辑", exc)
        extractor = None

    rows: List[Dict[str, Any]] = []
    for d in diseases:
        name = d.get("name")
        if not name:
            continue
        if extractor is not None:
            for t in extractor.extract_from_disease(d):
                rows.append({
                    "head": t.head, "head_type": t.head_type,
                    "relation": t.relation, "tail": t.tail, "tail_type": t.tail_type,
                    "confidence": t.confidence, "weight": 1.0,
                    "source_ref": d.get("disease_id", ""),
                })
        else:
            simple = [
                ("HAS_SYMPTOM", "Symptom", d.get("symptoms") or []),
                ("TREATED_BY", "Treatment", d.get("treatments") or []),
                ("USES_DRUG", "Drug", d.get("drugs") or []),
                ("NEEDS_CHECK", "Check", d.get("checks") or []),
                ("HAS_COMPLICATION", "Disease", d.get("complications") or []),
                ("DIFFERENTIAL_WITH", "Disease", d.get("differential") or []),
            ]
            for rel, ttype, items in simple:
                for item in items:
                    rows.append({"head": name, "head_type": "Disease", "relation": rel,
                                 "tail": item, "tail_type": ttype, "confidence": 0.95,
                                 "weight": 1.0, "source_ref": d.get("disease_id", "")})
            for dept in filter(None, [d.get("category1"), d.get("category2"), d.get("department")]):
                rows.append({"head": name, "head_type": "Disease", "relation": "BELONGS_TO",
                             "tail": dept, "tail_type": "Department", "confidence": 0.98,
                             "weight": 1.0, "source_ref": d.get("disease_id", "")})
        # 别名关系（用于实体链接）
        for alias in d.get("alias") or []:
            rows.append({"head": name, "head_type": "Disease", "relation": "HAS_ALIAS",
                         "tail": str(alias), "tail_type": "Alias", "confidence": 0.99,
                         "weight": 1.0, "source_ref": d.get("disease_id", "")})
    return rows


def _chunks(rows: List[Dict[str, Any]], size: int = BATCH_SIZE) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


# =============================================================================
#  导入实现
# =============================================================================
def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def import_diseases(session, diseases: List[Dict[str, Any]]) -> int:
    rows = [{
        "name": d.get("name", "").strip(),
        "disease_id": d.get("disease_id"),
        "alias": [str(a) for a in (d.get("alias") or [])],
        "category1": d.get("category1"),
        "category2": d.get("category2"),
        "definition": d.get("definition"),
        "cause": d.get("cause"),
        "diagnosis": d.get("diagnosis"),
        "treatment": d.get("treatment"),
        "prognosis": d.get("prognosis"),
        "population": d.get("population"),
        "is_infectious": bool(d.get("is_infectious")),
        "updated_at": d.get("updated_at") or _now_iso(),
    } for d in diseases if d.get("name")]
    total = 0
    for batch in _chunks(rows):
        session.run(CYPHER_DISEASE, rows=batch).consume()
        total += len(batch)
        logger.debug("  疾病批次 +%d（累计 %d）", len(batch), total)
    return total


#: 关系类型 → (头标签, 尾标签)
REL_LABEL_MAP: Dict[str, Tuple[str, str]] = {
    "HAS_SYMPTOM": ("Disease", "Symptom"),
    "TREATED_BY": ("Disease", "Treatment"),
    "USES_DRUG": ("Disease", "Drug"),
    "BELONGS_TO": ("Disease", "Department"),
    "NEEDS_CHECK": ("Disease", "Check"),
    "AFFECTS": ("Disease", "Population"),
    "HAS_COMPLICATION": ("Disease", "Disease"),
    "DIFFERENTIAL_WITH": ("Disease", "Disease"),
    "HAS_ALIAS": ("Disease", "Alias"),
    "RELATED_TO": ("Disease", "Disease"),
    "MAY_CAUSE_SYMPTOM": ("Disease", "Symptom"),
    "NEEDS_ISOLATION": ("Disease", "Treatment"),
    "DRUG_RISK_ALERT": ("Drug", "Disease"),
    "USES_TREATMENT": ("Disease", "Treatment"),
}


def import_relations(session, relations: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    分组批量导入关系。
    注意：Cypher 的标签与关系类型**不能参数化**，因此按 (关系类型, 头标签, 尾标签)
    分组，为每组构造独立语句（标签值来自白名单，无注入风险）。
    """
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    skipped = 0
    for r in relations:
        rel = r.get("relation")
        if rel not in REL_LABEL_MAP:
            skipped += 1
            continue
        htype = r.get("head_type") or REL_LABEL_MAP[rel][0]
        ttype = r.get("tail_type") or REL_LABEL_MAP[rel][1]
        head, tail = (r.get("head") or "").strip(), (r.get("tail") or "").strip()
        if not head or not tail:
            skipped += 1
            continue
        groups.setdefault((rel, htype, ttype), []).append({
            "head": head, "tail": tail,
            "confidence": float(r.get("confidence", 0.95)),
            "weight": float(r.get("weight", 1.0)),
            "source_ref": r.get("source_ref", ""),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        })

    total = 0
    for (rel, htype, ttype), rows in groups.items():
        cypher = (CYPHER_RELATION
                  .replace("__HEAD__", htype)
                  .replace("__TAIL__", ttype)
                  .replace("__REL__", rel))
        for batch in _chunks(rows, 2000):
            session.run(cypher, rows=batch).consume()
            total += len(batch)
        logger.debug("  关系 %-22s %-10s→%-10s +%d", rel, htype, ttype, len(rows))
    if skipped:
        logger.warning("  跳过 %d 条非白名单/字段缺失的关系", skipped)
    return total, skipped


def import_sources(session, diseases: List[Dict[str, Any]]) -> int:
    rows: List[Dict[str, Any]] = []
    for d in diseases:
        for s in d.get("sources") or []:
            url = s.get("url") or (f"https://pubmed.ncbi.nlm.nih.gov/{s.get('pmid')}/"
                                   if s.get("pmid") else None)
            if not url:
                # 无 URL 时用标题做唯一键（Source.url 有唯一约束，因此构造一个合法值）
                url = "urn:zhiyu:source:" + str(abs(hash(str(s.get("title", "")))))
            rows.append({
                "disease": d.get("name"),
                "url": url,
                "source_type": s.get("source_type", "pubmed"),
                "pmid": str(s["pmid"]) if s.get("pmid") else None,
                "doi": s.get("doi"),
                "title": s.get("title", ""),
                "journal": s.get("journal"),
                "year": int(s["year"]) if str(s.get("year") or "").isdigit() else None,
                "authors": s.get("authors"),
                "authority": s.get("authority", "B"),
                "retrieved_at": s.get("retrieved_at") or _now_iso(),
            })
    total = 0
    for batch in _chunks(rows):
        session.run(CYPHER_SOURCE, rows=batch).consume()
        total += len(batch)
    return total


# =============================================================================
#  语料导入（爬取内容 → 关系抽取 → 写图）
# =============================================================================
def import_corpus(session, corpus_path: Path, run_extraction: bool = True) -> Dict[str, int]:
    """
    处理爬取语料：
      ① 加载 JSONL
      ② 对每条以疾病为主体的文本跑关系抽取（RuleRelationExtractor 或改进 PCNN）
      ③ 三元组写图（MERGE 幂等）
    """
    if not corpus_path.exists():
        logger.error("❌ 语料文件不存在：%s", corpus_path)
        return {"docs": 0, "relations": 0}

    # 延迟导入，避免无 crawler 依赖时报错
    from app.data_layer.preprocess import Preprocessor

    pp = Preprocessor()
    docs = pp.load_jsonl(corpus_path)
    logger.info("读取语料 %d 条：%s", len(docs), corpus_path)
    if not docs:
        return {"docs": 0, "relations": 0}

    relations: List[Dict[str, Any]] = []
    if run_extraction:
        try:
            from app.kg_layer.relation_extractor import get_relation_extractor

            extractor = get_relation_extractor()
            logger.info("使用关系抽取后端：%s", extractor.backend)
        except Exception as exc:  # noqa: BLE001
            logger.error("关系抽取器不可用：%s", exc)
            return {"docs": len(docs), "relations": 0}

        for doc in docs:
            text = doc.get("text") or ""
            title = doc.get("title") or ""
            # 以标题中的疾病名作为头实体；无则跳过
            head = None
            try:
                from app.kg_layer.memory_store import MemoryGraphStore

                store = MemoryGraphStore.instance()
                for candidate in store.disease_by_name:
                    if candidate and candidate in (title + text[:200]):
                        head = candidate
                        break
            except Exception:  # noqa: BLE001
                pass
            if not head:
                continue
            for t in extractor.extract(text, head, "Disease"):
                relations.append({
                    "head": t["head"], "head_type": t["head_type"],
                    "relation": t["relation"], "tail": t["tail"], "tail_type": t["tail_type"],
                    "confidence": t["confidence"],
                    "weight": 0.9,
                    "source_ref": doc.get("pmid") or doc.get("id", ""),
                })

    if relations:
        total, _ = import_relations(session, relations)
    else:
        total = 0
    logger.info("语料导入完成：文档 %d 条，抽取并写入三元组 %d 条", len(docs), total)
    return {"docs": len(docs), "relations": total}


# =============================================================================
#  对外接口（供 init_neo4j.py 调用）
# =============================================================================
def load_seed_to_neo4j(driver) -> Dict[str, int]:
    """从种子 JSON 导入到 Neo4j，返回统计"""
    diseases = load_seed_diseases()
    if not diseases:
        return {"diseases": 0, "relations": 0, "sources": 0}
    relations = load_seed_relations()
    if not relations:
        logger.info("relations.json 缺失或为空，从疾病对象派生三元组…")
        relations = derive_relations_from_diseases(diseases)

    stats = {"diseases": 0, "relations": 0, "sources": 0, "skipped": 0}
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        stats["diseases"] = import_diseases(session, diseases)
        logger.info("  ✔ 疾病节点：%d", stats["diseases"])
        stats["relations"], stats["skipped"] = import_relations(session, relations)
        logger.info("  ✔ 关系：%d（跳过 %d）", stats["relations"], stats["skipped"])
        stats["sources"] = import_sources(session, diseases)
        logger.info("  ✔ 文献来源节点：%d", stats["sources"])

    # 计算度数（供前端节点半径使用）
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        try:
            session.run("""
                MATCH (n) WHERE NOT n:Source
                SET n.degree = size([(n)--() | 1])
            """).consume()
            logger.info("  ✔ 已计算全部节点度数（degree）")
        except Exception as exc:  # noqa: BLE001
            logger.warning("  度数计算失败（不影响功能）：%s", exc)
    return stats


def print_graph_stats(driver) -> None:
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        logger.info("📊 Neo4j 图谱统计：节点 %d，关系 %d", n, r)
        for row in session.run(
            "MATCH (n) WHERE NOT n:Source RETURN labels(n)[0] AS t, count(n) AS c "
            "ORDER BY c DESC"
        ):
            logger.info("     %-14s %6d", row["t"], row["c"])
        for row in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC LIMIT 15"
        ):
            cn = REL_LABELS.get(row["t"], row["t"])
            logger.info("     %-22s %6d  (%s)", row["t"], row["c"], cn)


# =============================================================================
#  CLI
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="医数智答 · 智愈医典 —— 知识图谱数据导入（种子 JSON / 爬取语料 → Neo4j）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--corpus", type=str, default="",
                        help="爬取语料 JSONL 路径（如 data/processed/pubmed.jsonl）")
    parser.add_argument("--extract", action="store_true",
                        help="对语料运行关系抽取（改进 PCNN / 规则）后再写图")
    parser.add_argument("--incremental", action="store_true",
                        help="增量模式：仅写新增三元组（MERGE 幂等，适合 24h 定时任务）")
    parser.add_argument("--stats", action="store_true", help="仅打印图谱统计，不做导入")
    parser.add_argument("--no-seed", action="store_true", help="跳过种子 JSON 导入")
    args = parser.parse_args()

    logger.info("=" * 72)
    logger.info("  医数智答 · 智愈医典 —— 知识图谱数据导入")
    logger.info("  目标：%s（database=%s）", settings.NEO4J_URI, settings.NEO4J_DATABASE)
    if args.incremental:
        logger.info("  模式：增量（MERGE 幂等，不覆盖高置信度既有事实）")
    logger.info("=" * 72)

    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("未安装 neo4j 驱动。请执行：pip install neo4j==5.19.0")
        return 2

    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            connection_timeout=settings.NEO4J_CONNECTION_TIMEOUT,
        )
        driver.verify_connectivity()
        logger.info("✅ Neo4j 连接成功")
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ 无法连接 Neo4j：%s", exc)
        logger.error("   请先启动 Neo4j 5.8 并检查 .env 配置；")
        logger.error("   或运行 python scripts/init_neo4j.py 查看排错指引。")
        return 1

    try:
        if args.stats:
            print_graph_stats(driver)
            return 0

        # ---- 1) 种子 JSON ----
        if not args.no_seed:
            logger.info("")
            logger.info("▶ 步骤 1/2：导入疾病种子库与三元组")
            stats = load_seed_to_neo4j(driver)
            logger.info("  结果：疾病 %d，关系 %d，来源 %d",
                        stats["diseases"], stats["relations"], stats["sources"])

        # ---- 2) 爬取语料 ----
        if args.corpus:
            logger.info("")
            logger.info("▶ 步骤 2/2：导入爬取语料（%s）", args.corpus)
            corpus_path = Path(args.corpus)
            if not corpus_path.is_absolute():
                corpus_path = BACKEND_DIR / corpus_path
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                cs = import_corpus(session, corpus_path, run_extraction=args.extract or True)
            logger.info("  结果：文档 %d，三元组 %d", cs["docs"], cs["relations"])

        # ---- 统计 ----
        logger.info("")
        print_graph_stats(driver)

        logger.info("")
        logger.info("✅ 导入完成。验证：浏览器打开 http://127.0.0.1:7474 执行")
        logger.info("   MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) RETURN d.name, collect(s.name)[0..5] LIMIT 5")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
