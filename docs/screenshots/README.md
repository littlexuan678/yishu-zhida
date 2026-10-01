# 运行效果截图

本目录存放「医数智答 · 智愈医典」各功能页面的运行截图，用于 README 效果预览与答辩展示。

## 建议截取的页面（应用启动后访问 http://127.0.0.1:5173 ）

| 文件名 | 页面 | 说明 |
|--------|------|------|
| `01-home.png` | 主页 | 核心功能与技术特色总览 |
| `02-graph.png` | 知识图谱可视化 | D3.js 力导向图，疾病-症状-药物-科室关联 |
| `03-disease.png` | 智能疾病查询 | 卡片式结构化疾病信息 |
| `04-chat.png` | 智能问答 | 可解释 AI 医生，含 [KG-n] 溯源角标与推理链路 |
| `05-analytics.png` | 数据分析与健康预警 | 多维看板 + 疾病风险预测 |
| `06-swagger.png` | 后端接口文档 | http://127.0.0.1:8000/docs |

## 启动方式

```bash
# 终端 1：后端
cd backend && uvicorn main:app --reload --port 8000

# 终端 2：前端
cd frontend && npm run dev
```

截好后将 PNG 放入本目录，并在主 `README.md` 的「效果预览」一节中以
`docs/screenshots/0X-xxx.png` 的相对路径引用即可。
