# -*- coding: utf-8 -*-
"""
医数智答 · 智愈医典 —— Hugging Face Spaces 一键部署脚本
=========================================================

用途
----
把本仓库（根级一体化 Dockerfile 模式）发布为 Hugging Face Space（Docker SDK），
获得公开可访问的在线演示地址：https://<owner>-<space>.hf.space

前置条件
--------
1. 注册 Hugging Face 账号：https://huggingface.co/join
2. 创建 Access Token（角色选 write）：https://huggingface.co/settings/tokens
3. pip install huggingface_hub

用法
----
    set HF_TOKEN=hf_xxxxxxxxxxxxxxxx          # Windows cmd
    $env:HF_TOKEN="hf_xxxxxxxxxxxxxxxx"       # PowerShell
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx       # Linux/macOS

    python tools/deploy_hf_space.py                       # 默认空间名 yishu-zhida
    python tools/deploy_hf_space.py --name my-demo        # 自定义空间名
    python tools/deploy_hf_space.py --private             # 私有空间

说明
----
* 脚本会复制仓库文件到临时目录（剔除 .git / node_modules / 密钥等），
  并生成带 Spaces frontmatter 的 README.md，再通过 HTTP API 上传——
  不改动本地 git 仓库，也不需要配置 git 凭据。
* 重复执行即增量更新部署（upload_folder 只上传变更文件）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# 仓库根目录（本文件位于 tools/ 下）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 上传时剔除的路径（与 .dockerignore 对齐；backend/data/seed 必须保留）
EXCLUDE_DIRS = {
    ".git", ".github", "node_modules", "dist", "__pycache__",
    ".venv", ".venv-backend", "venv", "logs", "processed", "models",
    "screenshots",
}
EXCLUDE_FILES = {".env", ".DS_Store", "Thumbs.db"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}

SPACE_README = """---
title: 医数智答·智愈医典
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 医数智答 · 智愈医典 —— 医疗大数据知识问答系统

> 基于「医疗大数据 + 知识图谱 + RAG 大模型」的专业医疗知识问答平台
> 参赛赛道：大模型与智能体应用赛道

## 四大核心功能

| 功能 | 说明 |
|------|------|
| 🕸️ 知识图谱可视化 | 医学知识的全景地图，实体-关系交互探索 |
| 🔍 智能疾病查询 | 精准权威的掌上医典 |
| 💬 智能问答交互 | 可解释的 AI 医生（答案带 [KG-x] 溯源编号） |
| 📊 多维数据分析 | 健康风险预测与预警 |

## 说明

* 本 Space 以**全离线演示模式**运行：内存知识图谱 + 医疗 RAG 模板引擎，
  无需 GPU 与外部 API Key 即可体验完整功能。
* 源代码与完整文档（架构 / API / 部署）：https://github.com/littlexuan678/yishu-zhida
* 接口文档：进入应用后访问 `/docs`（FastAPI Swagger UI）。

> ⚠️ **免责声明**：本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。
> 出现急危重症表现请立即拨打 120 或前往急诊。
"""


def should_exclude(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    for part in rel.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.is_file():
        if path.name in EXCLUDE_FILES:
            return True
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            return True
    return False


def stage_repo(staging: Path) -> int:
    """把仓库文件复制到临时目录（剔除排除项），返回文件数"""
    count = 0
    for src in REPO_ROOT.rglob("*"):
        if should_exclude(src, REPO_ROOT):
            continue
        rel = src.relative_to(REPO_ROOT)
        dst = staging / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="部署医数智答到 Hugging Face Spaces")
    parser.add_argument("--name", default="yishu-zhida", help="Space 名称（默认 yishu-zhida）")
    parser.add_argument("--owner", default="", help="HF 用户名/组织名（默认取 token 所属账号）")
    parser.add_argument("--private", action="store_true", help="创建私有空间")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="HF Access Token（或设 HF_TOKEN 环境变量）")
    args = parser.parse_args()

    if not args.token:
        print("❌ 未提供 HF Token。请设置 HF_TOKEN 环境变量或使用 --token 参数。")
        print("   创建 Token（write 角色）：https://huggingface.co/settings/tokens")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ 缺少 huggingface_hub：pip install huggingface_hub")
        return 1

    api = HfApi(token=args.token)

    # 确认账号身份
    me = api.whoami()
    owner = args.owner or me["name"]
    repo_id = f"{owner}/{args.name}"
    print(f"✅ 已登录 Hugging Face：{me['name']}")
    print(f"📦 目标 Space：{repo_id}（{'私有' if args.private else '公开'}）")

    # 创建 Space（Docker SDK；已存在则复用）
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    print("✅ Space 已创建（或已存在）")

    # 暂存仓库文件 + 生成 Space 专用 README
    with tempfile.TemporaryDirectory(prefix="hf-space-") as tmp:
        staging = Path(tmp)
        n = stage_repo(staging)
        (staging / "README.md").write_text(SPACE_README, encoding="utf-8")
        print(f"✅ 已暂存 {n + 1} 个文件（含 Space README）")

        print("⏫ 正在上传（首次约需几分钟，视网络而定）…")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(staging),
            commit_message="deploy: 医数智答·智愈医典 一体化演示（Docker SDK）",
        )

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    app_url = f"https://{owner}-{args.name}.hf.space"
    print()
    print("🎉 上传完成！HF 正在后台构建 Docker 镜像（首次约 5–15 分钟）。")
    print(f"   Space 主页（看构建日志）：{space_url}")
    print(f"   演示地址（构建完成后可访问）：{app_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
