# -*- coding: utf-8 -*-
"""
Neo4j 5.8 图谱初始化脚本
========================
执行内容：
  1. 连接 Neo4j（连接失败时给出清晰的排错指引）
  2. 依次执行 `scripts/cypher/` 下的：
       01_constraints.cypher   唯一性约束
       02_indexes.cypher       属性索引 + 全文索引
  3. 打印执行结果与图谱现状统计

用法
----
    cd backend
    python scripts/init_neo4j.py                  # 建约束与索引
    python scripts/init_neo4j.py --with-seed      # 同时导入演示种子数据
    python scripts/init_neo4j.py --drop-all       # ⚠️ 清空整库后重建（危险操作，需二次确认）
    python scripts/init_neo4j.py --check          # 仅检查连接与现状

Docker 快速启动 Neo4j 5.8：
    docker run -d --name zhiyu-neo4j \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/medical123 \
      -e NEO4J_PLUGINS='["apoc","graph-data-science"]' \
      neo4j:5.8-community
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

# 允许以 `python scripts/init_neo4j.py` 方式运行（把 backend/ 加入 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("init_neo4j")

CYPHER_DIR = Path(__file__).resolve().parent / "cypher"


# =============================================================================
#  Cypher 脚本解析器
# =============================================================================
def parse_cypher_file(path: Path) -> List[Tuple[int, str]]:
    """
    解析 .cypher 文件为语句列表。

    * 去除 `//` 行注释（但保留字符串内的内容）
    * 按 `;` 切分语句
    * 返回 [(行号, 语句)]
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    statements: List[Tuple[int, str]] = []
    buf: List[str] = []
    start_line = 1

    for i, raw in enumerate(lines, 1):
        line = raw
        # 去掉行注释（简单处理：不在引号内的 //）
        in_single = False
        out_chars = []
        j = 0
        while j < len(line):
            ch = line[j]
            if ch == "'":
                in_single = not in_single
            if not in_single and ch == "/" and j + 1 < len(line) and line[j + 1] == "/":
                break
            out_chars.append(ch)
            j += 1
        line = "".join(out_chars)

        if line.strip():
            if not buf:
                start_line = i
            buf.append(line)

        # 按分号切分（可能一行多条）
        while ";" in "\n".join(buf):
            joined = "\n".join(buf)
            idx = joined.index(";")
            stmt = joined[:idx].strip()
            if stmt:
                statements.append((start_line, stmt))
            rest = joined[idx + 1 :]
            buf = [rest] if rest.strip() else []

    tail = "\n".join(buf).strip()
    if tail:
        statements.append((start_line, tail))
    return statements


# =============================================================================
#  连接与执行
# =============================================================================
def get_driver():
    """创建 Neo4j 驱动（失败时抛出带排错指引的异常）"""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("未安装 neo4j 驱动。请执行：pip install neo4j==5.19.0")
        raise SystemExit(2)

    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            connection_timeout=settings.NEO4J_CONNECTION_TIMEOUT,
        )
        driver.verify_connectivity()
        logger.info("✅ Neo4j 连接成功：%s", settings.NEO4J_URI)
        return driver
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ 无法连接 Neo4j（%s）：%s", settings.NEO4J_URI, exc)
        logger.error("")
        logger.error("排错指引：")
        logger.error("  1) 确认 Neo4j 5.8 已启动：docker ps | grep neo4j")
        logger.error("     或本地服务：neo4j console")
        logger.error("  2) 确认 .env 中的 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD 正确")
        logger.error("     （Neo4j 5.x 默认数据库名为 neo4j，默认用户为 neo4j）")
        logger.error("  3) 浏览器打开 http://127.0.0.1:7474 确认服务可访问")
        logger.error("  4) 快速启动：")
        logger.error("     docker run -d --name zhiyu-neo4j -p 7474:7474 -p 7687:7687 \\")
        logger.error("       -e NEO4J_AUTH=neo4j/medical123 neo4j:5.8-community")
        logger.error("")
        logger.error("提示：后端服务无需 Neo4j 也能运行（自动降级到内存图存储），")
        logger.error("      本脚本仅用于初始化真实图数据库。")
        raise SystemExit(1)


def run_statements(driver, statements: List[Tuple[int, str]], label: str) -> Tuple[int, int]:
    """执行语句列表，返回 (成功数, 失败数)"""
    ok = fail = 0
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        for lineno, stmt in statements:
            preview = re.sub(r"\s+", " ", stmt)[:90]
            try:
                result = session.run(stmt)
                summary = result.consume()
                counters = summary.counters
                detail = []
                if counters.constraints_added:
                    detail.append(f"约束 +{counters.constraints_added}")
                if counters.indexes_added:
                    detail.append(f"索引 +{counters.indexes_added}")
                if counters.nodes_created:
                    detail.append(f"节点 +{counters.nodes_created}")
                if counters.relationships_created:
                    detail.append(f"关系 +{counters.relationships_created}")
                if counters.properties_set:
                    detail.append(f"属性 +{counters.properties_set}")
                logger.info("  ✔ [%s:%d] %s %s", label, lineno, preview,
                            ("(" + ", ".join(detail) + ")") if detail else "")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("  ✘ [%s:%d] %s\n      错误：%s", label, lineno, preview, exc)
                fail += 1
    return ok, fail


# =============================================================================
#  现状统计
# =============================================================================
def print_status(driver) -> None:
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        try:
            n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            r = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            logger.info("📊 图谱现状：节点 %d 个，关系 %d 条", n, r)

            rows = session.run(
                "MATCH (n) WHERE NOT n:Source "
                "RETURN labels(n)[0] AS t, count(n) AS c ORDER BY c DESC"
            )
            for row in rows:
                logger.info("     - %-14s %d", row["t"], row["c"])

            cons = session.run("SHOW CONSTRAINTS YIELD name RETURN count(*) AS c").single()["c"]
            idx = session.run("SHOW INDEXES YIELD name RETURN count(*) AS c").single()["c"]
            logger.info("     - 约束 %d 个，索引 %d 个", cons, idx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("统计失败（可能库为空）：%s", exc)


def drop_all(driver) -> None:
    """⚠️ 清空整库（危险操作）"""
    logger.warning("⚠️  即将清空 Neo4j 全部数据（DROP 所有节点、关系、约束、索引）")
    answer = input("请输入 yes 确认清空：").strip().lower()
    if answer != "yes":
        logger.info("已取消清空操作。")
        return
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("已删除全部节点与关系")
        for row in session.run("SHOW CONSTRAINTS YIELD name"):
            try:
                session.run(f"DROP CONSTRAINT {row['name']} IF EXISTS")
            except Exception:  # noqa: BLE001
                pass
        for row in session.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'"):
            try:
                session.run(f"DROP INDEX {row['name']} IF EXISTS")
            except Exception:  # noqa: BLE001
                pass
        logger.info("已删除全部约束与索引")


# =============================================================================
#  主流程
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="医数智答 · 智愈医典 —— Neo4j 5.8 图谱初始化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--with-seed", action="store_true",
                        help="同时导入 scripts/cypher/03_seed_diseases.cypher 演示种子数据")
    parser.add_argument("--load-json", action="store_true",
                        help="从 data/seed/*.json 导入完整疾病库（等价于 python scripts/load_kg.py）")
    parser.add_argument("--drop-all", action="store_true", help="⚠️ 清空整库后重建（需二次确认）")
    parser.add_argument("--check", action="store_true", help="仅检查连接与图谱现状，不执行任何变更")
    parser.add_argument("--only", type=str, default="",
                        help="仅执行指定脚本，如 --only 02（按文件名前缀匹配）")
    args = parser.parse_args()

    logger.info("=" * 72)
    logger.info("  医数智答 · 智愈医典 —— Neo4j 图谱初始化")
    logger.info("  目标数据库：%s（database=%s）", settings.NEO4J_URI, settings.NEO4J_DATABASE)
    logger.info("=" * 72)

    driver = get_driver()

    try:
        if args.check:
            print_status(driver)
            return 0

        if args.drop_all:
            drop_all(driver)

        # ---- 收集待执行脚本 ----
        scripts = sorted(CYPHER_DIR.glob("*.cypher"))
        if args.only:
            scripts = [p for p in scripts if p.name.startswith(args.only)]
        if not args.with_seed:
            # 默认不导入 03，避免与 --load-json 重复
            scripts = [p for p in scripts if not p.name.startswith("03")]
        if not scripts:
            logger.warning("未找到待执行的 Cypher 脚本（目录：%s）", CYPHER_DIR)

        total_ok = total_fail = 0
        for path in scripts:
            stmts = parse_cypher_file(path)
            logger.info("")
            logger.info("▶ 执行 %s（%d 条语句）", path.name, len(stmts))
            ok, fail = run_statements(driver, stmts, path.stem)
            total_ok += ok
            total_fail += fail

        logger.info("")
        logger.info("=" * 72)
        logger.info("  初始化完成：成功 %d 条，失败 %d 条", total_ok, total_fail)
        logger.info("=" * 72)

        if args.load_json or args.with_seed:
            logger.info("")
            logger.info("▶ 导入种子数据（JSON → Neo4j）…")
            try:
                from load_kg import load_seed_to_neo4j

                stats = load_seed_to_neo4j(driver)
                logger.info("  导入完成：疾病 %d，实体 %d，关系 %d",
                            stats.get("diseases", 0), stats.get("nodes", 0), stats.get("relations", 0))
            except Exception as exc:  # noqa: BLE001
                logger.error("种子数据导入失败：%s", exc)
                logger.error("请改为手动执行：python scripts/load_kg.py")

        logger.info("")
        print_status(driver)

        logger.info("")
        logger.info("下一步：")
        logger.info("  1) 验证图谱：浏览器打开 http://127.0.0.1:7474")
        logger.info("     执行：MATCH (n) RETURN n LIMIT 50")
        logger.info("  2) 启动后端：uvicorn main:app --reload --port 8000")
        logger.info("  3) 打开接口文档：http://127.0.0.1:8000/docs")
        return 0 if total_fail == 0 else 3

    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
