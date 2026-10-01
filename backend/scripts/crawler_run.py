# -*- coding: utf-8 -*-
"""
增量采集调度脚本（知识更新周期 ≤24h）
=====================================
PPT 指标：「知识更新周期 ≤24h」。

本脚本把「采集 → 清洗 → 脱敏 → 关系抽取 → 写图」串成一条可定时执行的流水线。

用法
----
    # 增量采集近 1 天的 PubMed 新文献
    python scripts/crawler_run.py --since 1d

    # 采集指定主题（MeSH 检索）
    python scripts/crawler_run.py --query "hypertension[MeSH Terms]" --retmax 200

    # 全流程：采集 + 清洗 + 抽取 + 写图
    python scripts/crawler_run.py --since 7d --load

    # 只跑离线演示语料（无需网络），用于演示完整流水线
    python scripts/crawler_run.py --offline --load

    # 安装为每日定时任务（打印 crontab / schtasks 命令）
    python scripts/crawler_run.py --print-schedule

合规说明
--------
* 遵守目标站点 robots.txt，单站点并发 ≤4，下载延迟 ≥1s
* 仅采集公开的学术文献元数据与摘要，**不采集任何个人健康信息（PHI）**
* PubMed 官方 E-utilities 接口，需提供 tool 与 email 标识
* 所有采集内容在落盘前完成 PII 脱敏与敏感字段加密
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("crawler_run")


# =============================================================================
#  时间窗口解析
# =============================================================================
def parse_since(value: str) -> int:
    """把 '1d' / '12h' / '7d' / '30m' 解析为天数（向上取整）"""
    v = (value or "").strip().lower()
    if not v:
        return 1
    try:
        if v.endswith("d"):
            return max(1, int(float(v[:-1])))
        if v.endswith("h"):
            return max(1, int(-(-float(v[:-1]) // 24)))     # 向上取整到天
        if v.endswith("m"):
            return 1
        return max(1, int(float(v)))
    except ValueError:
        logger.warning("无法解析时间窗口 '%s'，回退为 1 天", value)
        return 1


# =============================================================================
#  采集阶段
# =============================================================================
def run_spider(
    spider: str,
    args: List[str],
    output: Path,
    timeout: int = 1800,
) -> bool:
    """
    以子进程方式运行 Scrapy spider。

    使用 `scrapy runspider` 而非 `scrapy crawl`，避免依赖 scrapy.cfg 项目配置，
    便于在任意目录直接执行。
    """
    if spider == "pubmed":
        script = BACKEND_DIR / "app" / "data_layer" / "crawler" / "spiders" / "pubmed_spider.py"
    else:
        script = BACKEND_DIR / "app" / "data_layer" / "crawler" / "spiders" / "med_site_spider.py"

    if not script.exists():
        logger.error("❌ 未找到 spider 文件：%s", script)
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scrapy", "runspider", str(script),
        "-O", str(output),        # -O 覆盖写（增量采集每轮独立文件）
        "-s", f"LOG_LEVEL={settings.LOG_LEVEL}",
    ]
    for a in args:
        cmd += ["-a", a]

    logger.info("执行：%s", " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(BACKEND_DIR),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("❌ 采集超时（%ds），已终止", timeout)
        return False
    except FileNotFoundError:
        logger.error("❌ 未安装 Scrapy。请执行：pip install Scrapy==2.11.1")
        return False

    elapsed = time.time() - t0
    if proc.returncode != 0:
        logger.error("❌ 采集失败（exit=%d），stderr 尾部：\n%s",
                     proc.returncode, (proc.stderr or "")[-1500:])
        return False

    lines = (proc.stdout or "").strip().split("\n")
    logger.info("✅ 采集完成，用时 %.1fs", elapsed)
    for line in lines[-8:]:
        logger.info("   %s", line.strip())

    if output.exists():
        n = sum(1 for _ in output.open("r", encoding="utf-8"))
        logger.info("   输出文件：%s（%d 条记录）", output, n)
        return n > 0
    logger.warning("   未生成输出文件：%s", output)
    return False


# =============================================================================
#  清洗阶段
# =============================================================================
def run_preprocess(raw_files: List[Path], out_file: Path) -> int:
    """清洗 + 去重 + 脱敏，输出标准语料 JSONL"""
    from app.data_layer.preprocess import Preprocessor

    pp = Preprocessor()
    raws = []
    for f in raw_files:
        if f.exists():
            raws.extend(pp.load_jsonl(f))

    if not raws:
        logger.warning("没有待清洗的原始数据")
        return 0

    logger.info("读取原始记录 %d 条，开始清洗（去噪 / 去重 / 术语归一化 / 脱敏）…", len(raws))
    docs = pp.process_batch(raws, apply_privacy=True)
    pp.save_jsonl(docs, out_file)

    st = pp.info()["stats"]
    logger.info("清洗完成：输入 %d → 输出 %d（去重 %d，过短 %d，无医学术语 %d，广告 %d）",
                st.get("documents_in", 0), st.get("documents_out", 0),
                st.get("duplicates", 0), st.get("too_short", 0),
                st.get("no_medical_term", 0), st.get("ads", 0))
    return len(docs)


# =============================================================================
#  写图阶段
# =============================================================================
def run_load(corpus: Path) -> bool:
    """调用 load_kg.py 把语料抽取为三元组并写入 Neo4j"""
    script = BACKEND_DIR / "scripts" / "load_kg.py"
    if not script.exists():
        logger.error("❌ 未找到 load_kg.py")
        return False

    cmd = [sys.executable, str(script), "--corpus", str(corpus), "--extract", "--no-seed"]
    logger.info("执行：%s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=str(BACKEND_DIR),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        logger.error("❌ 写图超时")
        return False

    tail = (proc.stdout or "").strip().split("\n")[-12:]
    for line in tail:
        logger.info("   %s", line.strip())
    if proc.returncode != 0:
        logger.error("❌ 写图失败（exit=%d）", proc.returncode)
        logger.error((proc.stderr or "")[-1200:])
        return False
    logger.info("✅ 图谱增量更新完成")
    return True


# =============================================================================
#  定时任务
# =============================================================================
def print_schedule() -> None:
    """打印把本脚本安装为每日任务所需的命令（满足 ≤24h 更新周期）"""
    here = Path(__file__).resolve()
    py = sys.executable
    logger.info("=" * 74)
    logger.info("  知识更新定时任务（周期 ≤24h）")
    logger.info("=" * 74)
    logger.info("")
    logger.info("【Linux / macOS —— crontab】")
    logger.info("  执行 `crontab -e` 后加入以下行（每日 03:00 增量采集并写图）：")
    logger.info("")
    logger.info("  0 3 * * * cd %s && %s %s --since 1d --load >> %s/crawler_cron.log 2>&1",
                BACKEND_DIR, py, here, settings.log_dir)
    logger.info("")
    logger.info("【Windows —— 任务计划程序 schtasks】")
    logger.info("  以管理员身份运行 PowerShell：")
    logger.info("")
    logger.info('  schtasks /Create /SC DAILY /ST 03:00 /TN "ZhiyuMedicalKGUpdate" ^')
    logger.info('    /TR "\\"%s\\" \\"%s\\" --since 1d --load" /F', py, here)
    logger.info("")
    logger.info("【容器化 —— K8s CronJob 片段】")
    logger.info("  apiVersion: batch/v1")
    logger.info("  kind: CronJob")
    logger.info("  spec:")
    logger.info('    schedule: "0 3 * * *"')
    logger.info("    jobTemplate:")
    logger.info("      spec:")
    logger.info("        template:")
    logger.info("          spec:")
    logger.info("            containers:")
    logger.info("            - name: kg-update")
    logger.info("              image: zhiyu-backend:1.0.0")
    logger.info("              command: [\"python\", \"scripts/crawler_run.py\", \"--since\", \"1d\", \"--load\"]")
    logger.info("")
    logger.info("【验证方式】")
    logger.info("  在 Neo4j Browser 中对比更新前后的统计：")
    logger.info("    MATCH (n) RETURN count(n) AS 节点数;")
    logger.info("    MATCH ()-[r]->() RETURN count(r) AS 关系数;")
    logger.info("  或调用接口：GET /api/v1/graph/stats")
    logger.info("=" * 74)


# =============================================================================
#  CLI
# =============================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="知识增量采集与更新流水线（PPT 指标：更新周期 ≤24h）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--since", type=str, default="1d",
                   help="增量时间窗口，如 1d / 12h / 7d（默认 1d，满足 ≤24h 指标）")
    p.add_argument("--query", type=str, default="",
                   help="PubMed 检索式（留空则按内置 16 个 MeSH 主题词采集）")
    p.add_argument("--retmax", type=int, default=200, help="单次最多取回条数")
    p.add_argument("--site", type=str, default="",
                   help="医学站点 key（见 med_site_spider.SITES），留空则采集全部站点")
    p.add_argument("--offline", action="store_true",
                   help="离线模式：用 generate_corpus.py 生成演示语料，不发起网络请求")
    p.add_argument("--load", action="store_true", help="清洗后继续写图（Neo4j）")
    p.add_argument("--no-crawl", action="store_true", help="跳过采集，直接清洗已有原始数据")
    p.add_argument("--print-schedule", action="store_true", help="打印定时任务配置（不执行采集）")
    p.add_argument("--timeout", type=int, default=1800, help="单阶段超时秒数")
    args = p.parse_args()

    if args.print_schedule:
        print_schedule()
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = settings.corpus_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 74)
    logger.info("  知识增量采集与更新流水线")
    logger.info("  时间窗口：%s（%d 天）", args.since, parse_since(args.since))
    logger.info("  模式：%s", "离线演示语料" if args.offline else "在线采集")
    logger.info("  时间：%s", datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    logger.info("=" * 74)

    t_start = time.time()
    raw_files: List[Path] = []

    # ---------- 阶段 1：采集 ----------
    if args.offline:
        logger.info("")
        logger.info("▶ 阶段 1/3：生成离线演示语料")
        gen = BACKEND_DIR / "scripts" / "generate_corpus.py"
        try:
            proc = subprocess.run(
                [sys.executable, str(gen), "--per-disease", "4", "--with-noise"],
                cwd=str(BACKEND_DIR), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=args.timeout,
            )
            for line in (proc.stdout or "").strip().split("\n")[-6:]:
                logger.info("   %s", line.strip())
            raw_files = [settings.corpus_dir / "corpus.jsonl"]
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ 离线语料生成失败：%s", exc)
            return 1
    elif not args.no_crawl:
        logger.info("")
        logger.info("▶ 阶段 1/3：PubMed 文献采集（E-utilities API）")
        pubmed_out = raw_dir / f"pubmed_{stamp}.jsonl"
        spider_args = [f"days={parse_since(args.since)}", f"retmax={args.retmax}"]
        if args.query:
            spider_args.append(f"query={args.query}")
        ok = run_spider("pubmed", spider_args, pubmed_out, timeout=args.timeout)
        if ok:
            raw_files.append(pubmed_out)
        else:
            logger.warning("PubMed 采集未获得数据（可能是网络受限或该时间窗无新文献）")

        logger.info("")
        logger.info("▶ 阶段 1b：权威医学站点采集（结构化疾病条目）")
        site_out = raw_dir / f"med_site_{stamp}.jsonl"
        site_args = [f"limit={settings.CRAWLER_MAX_PAGES}"]
        if args.site:
            site_args.append(f"site={args.site}")
        if run_spider("med_site", site_args, site_out, timeout=args.timeout):
            raw_files.append(site_out)
        else:
            logger.warning("医学站点采集未获得数据（站点配置需按目标源更新）")
    else:
        # 复用已有原始文件
        raw_files = sorted(raw_dir.glob("*.jsonl"))
        logger.info("跳过采集，复用已有原始文件 %d 个", len(raw_files))

    if not raw_files:
        logger.warning("")
        logger.warning("⚠️  未获得任何原始数据。")
        logger.warning("   可尝试：python scripts/crawler_run.py --offline --load")
        logger.warning("   （离线模式生成演示语料，用于演示完整流水线）")
        return 1

    # ---------- 阶段 2：清洗 ----------
    logger.info("")
    logger.info("▶ 阶段 2/3：清洗 / 去重 / 术语归一化 / PII 脱敏")
    corpus = settings.corpus_dir / "corpus_incremental.jsonl"
    n_docs = run_preprocess(raw_files, corpus)
    if n_docs == 0:
        logger.warning("⚠️  清洗后无有效语料（可能全被去重或质量过滤）")
        return 1

    # ---------- 阶段 3：写图 ----------
    if args.load:
        logger.info("")
        logger.info("▶ 阶段 3/3：关系抽取 + 写入知识图谱")
        ok = run_load(corpus)
        if not ok:
            logger.warning("写图未成功（Neo4j 可能未启动）。语料已保存：%s", corpus)
            logger.warning("可稍后手动执行：python scripts/load_kg.py --corpus %s --extract", corpus)
    else:
        logger.info("")
        logger.info("⏭  跳过写图（未指定 --load）。语料已保存：%s", corpus)
        logger.info("   手动写图：python scripts/load_kg.py --corpus %s --extract", corpus)

    logger.info("")
    logger.info("=" * 74)
    logger.info("  ✅ 流水线完成，总用时 %.1fs", time.time() - t_start)
    logger.info("  语料文件：%s（%d 条）", corpus, n_docs)
    logger.info("  下一步：验证图谱 GET /api/v1/graph/stats")
    logger.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
