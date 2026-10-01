# 贡献指南 · Contributing

感谢你对「医数智答 · 智愈医典」的关注！这是一个面向 **大模型与智能体应用赛道** 的比赛项目，
欢迎提出 Issue 与 Pull Request。

## 开发环境

| 组件 | 版本 |
|------|------|
| Python | 3.9+（开发用 3.12 验证） |
| Node.js | ≥ 18 |
| Neo4j | 5.8 社区版（可选，缺失时自动降级为内存图） |

## 快速开始

```bash
# 后端（轻量演示模式，无需 GPU / 大模型权重）
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-lite.txt
cp ../.env.example .env
uvicorn main:app --reload --port 8000

# 前端（新终端）
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173
```

> **说明**：系统采用「有权重则加载、无权重则规则降级」设计，即使不下载任何模型、
> 不启动 Neo4j、不接入大模型，也能完整运行四大核心功能用于演示与联调。

## 分支与提交

- 主分支：`main`（受保护），功能开发请基于 `feat/<name>` 分支。
- 提交信息建议遵循 [Conventional Commits](https://www.conventionalcommits.org/)：
  `feat: ...` / `fix: ...` / `docs: ...` / `refactor: ...` / `test: ...` / `chore: ...`

## 代码规范

- **后端**：遵循 PEP 8，公共函数与模块需带中文 docstring 说明职责；新增接口请在
  `backend/app/schemas.py` 中定义 Pydantic 契约，并在 `docs/API.md` 中登记。
- **前端**：Vue3 `<script setup>` + Composition API；组件置于 `src/components/`，
  页面置于 `src/views/`；统一走 `src/api/` 封装发起请求，不直接散落调用 axios。

## 目录约定（四层架构）

```
backend/app/
├── data_layer/   # ① 数据层：爬虫 / 清洗 / 脱敏加密
├── kg_layer/     # ② 知识图谱层：实体识别 / 关系抽取 / Neo4j / 推理
├── llm_layer/    # ③ LLM 增强层：意图分类 / 实体链接 / RAG / 提示工程
└── app_layer/    # ④ 应用层：FastAPI 路由与业务服务
```

## 测试

```bash
cd backend
pytest -q                 # 单元 / 冒烟测试
python ../tools/verify_full_stack.py   # 端到端联通验证（需后端已启动）
```

## ⚠️ 重要合规约定

- **不得提交任何真实患者数据 / 可识别个人身份的信息**；
- `.env`、模型权重、爬取语料已在 `.gitignore` 中排除，请勿强制提交；
- 所有 AI 输出必须经过 `llm_layer/rag/hallucination_guard.py` 校验；
- 任何面向用户的页面必须保留免责声明（`DisclaimerBar.vue`）。

## 报告问题

提交 Issue 时请附带：运行环境（OS/Python/Node 版本）、复现步骤、
`backend/logs/zhiyu.log` 中的相关日志片段（注意脱敏）。
