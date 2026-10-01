# -*- coding: utf-8 -*-
"""
多维数据分析服务
================
为「数据分析中心」页面提供：
  * 看板概览指标（医疗实体总数 / 疾病分类数 / 传染性疾病 / 治疗周期类型）
  * 各类节点数量统计（环形饼图）
  * 一级分类下的疾病数量（柱状图）
  * 传染性疾病比例（环形图）
  * 症状-疾病关联热力（附加图表）

同时用 **PyEcharts** 生成服务端渲染的 HTML 片段（PPT 指定），
并返回 ECharts option JSON 供前端直接使用（前端为 ECharts，二者数据同源）。
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.kg_layer.memory_store import MemoryGraphStore
from app.schemas import (
    AnalyticsOverviewResponse,
    ChartResponse,
    OverviewMetric,
    PieSeriesItem,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from pyecharts import options as opts
    from pyecharts.charts import Bar, Line, Pie

    HAS_PYECHARTS = True
except Exception:  # pragma: no cover
    HAS_PYECHARTS = False
    opts = None  # type: ignore
    Pie = Bar = Line = None  # type: ignore

#: 图表配色（与前端主题一致）
PALETTE = ["#409EFF", "#F56C6C", "#31c48d", "#E6A23C", "#b37feb",
           "#00BCD4", "#F2C037", "#909399", "#5B8FF9", "#FF9845"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class AnalyticsService:
    """数据分析业务服务"""

    def __init__(self, store: Optional[MemoryGraphStore] = None) -> None:
        self.store = store or MemoryGraphStore.instance()

    # ==================================================================
    #  1) 概览指标
    # ==================================================================
    def overview(self) -> AnalyticsOverviewResponse:
        stats = self.store.stats()
        diseases = list(self.store.diseases.values())
        infectious = [d for d in diseases if d.get("is_infectious")]
        cat1 = {d.get("category1") for d in diseases if d.get("category1")}
        # 治疗周期类型：按 treatments 的组合模式归类（演示口径）
        treatment_patterns = {tuple(sorted(d.get("treatments") or [])) for d in diseases}
        # 治疗费用档位（如种子数据含该字段则统计，否则用治疗方式数）
        fee_levels = {d.get("fee_level") for d in diseases if d.get("fee_level")}

        metrics = [
            OverviewMetric(key="total_entities", label="医疗实体总数",
                           value=stats["total_nodes"], icon="DataLine"),
            OverviewMetric(key="disease_categories", label="疾病分类数",
                           value=len(cat1), icon="Grid"),
            OverviewMetric(key="infectious", label="传染性疾病",
                           value=len(infectious), icon="Warning"),
            OverviewMetric(key="treatment_types", label="治疗周期类型",
                           value=len(fee_levels) or max(len(treatment_patterns), 1),
                           icon="Timer"),
        ]
        return AnalyticsOverviewResponse(
            metrics=metrics, generated_at=_now_iso(), data_source=stats.get("data_source", "memory")
        )

    # ==================================================================
    #  2) 各类节点数量统计（饼图）
    # ==================================================================
    def node_type_pie(self) -> ChartResponse:
        stats = self.store.stats()
        raw: List[Tuple[str, int]] = [
            (t["label"], t["count"]) for t in stats["entity_types"] if t["count"] > 0
        ]
        total = sum(v for _, v in raw) or 1
        series = [
            PieSeriesItem(name=n, value=v, color=PALETTE[i % len(PALETTE)])
            for i, (n, v) in enumerate(raw)
        ]
        insight = (
            f"知识图谱共 {total} 个实体节点，其中"
            + "、".join(f"{n} {v / total * 100:.2f}%" for n, v in raw[:3])
            + "，构成以疾病为中心、症状与治疗方案为支撑的知识网络。"
        )
        option = {
            "title": {"text": "各类节点数量统计", "left": "center", "top": 4,
                      "textStyle": {"fontSize": 14, "color": "#303133"}},
            "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c} 个（{d}%）"},
            "legend": {"orient": "vertical", "left": "left", "top": "middle",
                       "itemWidth": 10, "itemHeight": 10,
                       "textStyle": {"fontSize": 12}},
            "series": [{
                "name": "节点类型", "type": "pie", "radius": ["42%", "68%"],
                "center": ["60%", "55%"],
                "avoidLabelOverlap": True,
                "itemStyle": {"borderRadius": 4, "borderColor": "#fff", "borderWidth": 2},
                "label": {"show": True, "formatter": "{b}\n{d}%", "fontSize": 11, "color": "#606266"},
                "labelLine": {"length": 10, "length2": 8},
                "data": [{"name": s.name, "value": s.value,
                          "itemStyle": {"color": s.color}} for s in series],
            }],
        }
        return ChartResponse(
            chart_type="pie", title="各类节点数量统计", series=series, unit="个",
            echarts_option=option, pyecharts_html=self._py_pie(raw, "各类节点数量统计", donut=True),
            insight=insight,
        )

    # ==================================================================
    #  3) 一级分类下的疾病数量（柱状图）
    # ==================================================================
    def category_bar(self) -> ChartResponse:
        dist: Dict[str, int] = {}
        for d in self.store.diseases.values():
            c = d.get("category1") or "未分类"
            dist[c] = dist.get(c, 0) + 1
        ordered = sorted(dist.items(), key=lambda kv: -kv[1])
        cats = [k for k, _ in ordered]
        vals = [v for _, v in ordered]
        top = ordered[0] if ordered else ("", 0)
        insight = (
            f"共 {len(cats)} 个一级分类，{top[0]}类疾病数量最多（{top[1]} 个），"
            f"占全部疾病的 {top[1] / max(sum(vals), 1) * 100:.1f}%。"
        )
        option = {
            "title": {"text": "一级分类下的疾病数量", "left": "center", "top": 4,
                      "textStyle": {"fontSize": 14, "color": "#303133"}},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"left": "12%", "right": "6%", "top": 50, "bottom": 40},
            "xAxis": {"type": "category", "data": cats,
                      "axisLabel": {"rotate": len(cats) > 6 and 30 or 0, "fontSize": 11,
                                    "color": "#606266"},
                      "axisLine": {"lineStyle": {"color": "#dcdfe6"}}},
            "yAxis": {"type": "value", "name": "疾病数量",
                      "splitLine": {"lineStyle": {"color": "#f0f2f5"}},
                      "axisLabel": {"color": "#909399"}},
            "series": [{
                "name": "疾病数量", "type": "bar", "barWidth": "46%",
                "data": vals,
                "itemStyle": {
                    "borderRadius": [6, 6, 0, 0],
                    "color": {
                        "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#38bdf8"},
                            {"offset": 1, "color": "#2b8fe8"},
                        ],
                    },
                },
                "label": {"show": True, "position": "top", "fontSize": 11, "color": "#606266"},
            }],
        }
        return ChartResponse(
            chart_type="bar", title="一级分类下的疾病数量", unit="个",
            categories=cats, values=[float(v) for v in vals], x_axis=cats,
            series=[PieSeriesItem(name=c, value=float(v), color=PALETTE[i % len(PALETTE)])
                    for i, (c, v) in enumerate(ordered)],
            echarts_option=option,
            pyecharts_html=self._py_bar(cats, vals, "一级分类下的疾病数量"),
            insight=insight,
        )

    # ==================================================================
    #  4) 传染性疾病比例（环形图）
    # ==================================================================
    def infectious_gauge(self) -> ChartResponse:
        diseases = list(self.store.diseases.values())
        total = len(diseases) or 1
        yes = sum(1 for d in diseases if d.get("is_infectious"))
        no = total - yes
        raw = [("否", no), ("是", yes), ("是否传染", total)]
        series = [
            PieSeriesItem(name="否", value=float(no), color="#409EFF"),
            PieSeriesItem(name="是", value=float(yes), color="#F56C6C"),
            PieSeriesItem(name="是否传染", value=float(total), color="#31c48d"),
        ]
        insight = (
            f"在 {total} 种疾病中，传染性疾病 {yes} 种（{yes / total * 100:.2f}%），"
            f"非传染性疾病 {no} 种（{no / total * 100:.2f}%）。"
            f"传染病需重点关注防控与报告要求。"
        )
        option = {
            "title": {"text": "传染性疾病比例", "left": "center", "top": 4,
                      "textStyle": {"fontSize": 14, "color": "#303133"}},
            "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c} 种（{d}%）"},
            "legend": {"orient": "vertical", "left": "left", "top": "middle",
                       "itemWidth": 10, "itemHeight": 10, "textStyle": {"fontSize": 12}},
            "series": [{
                "name": "传染性疾病比例", "type": "pie", "radius": ["45%", "70%"],
                "center": ["58%", "55%"],
                "itemStyle": {"borderRadius": 4, "borderColor": "#fff", "borderWidth": 2},
                "label": {"show": True, "formatter": "{b}\n{d}%", "fontSize": 11, "color": "#606266"},
                "labelLine": {"length": 8, "length2": 6},
                "data": [{"name": s.name, "value": s.value,
                          "itemStyle": {"color": s.color}} for s in series],
            }],
        }
        return ChartResponse(
            chart_type="donut", title="传染性疾病比例", series=series, unit="种",
            echarts_option=option,
            pyecharts_html=self._py_pie(raw, "传染性疾病比例", donut=True),
            insight=insight,
        )

    # ==================================================================
    #  5) 症状 TopN（附加图表）
    # ==================================================================
    def symptom_top(self, topn: int = 12) -> ChartResponse:
        counts: Dict[str, int] = {}
        for e in self.store.edges:
            if e["rel"] != "HAS_SYMPTOM":
                continue
            tail = self.store.nodes.get(e["target"])
            if tail:
                counts[tail["name"]] = counts.get(tail["name"], 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:topn]
        cats = [k for k, _ in ordered]
        vals = [v for _, v in ordered]
        option = {
            "title": {"text": f"高频症状 Top{topn}", "left": "center", "top": 4,
                      "textStyle": {"fontSize": 14, "color": "#303133"}},
            "tooltip": {"trigger": "axis"},
            "grid": {"left": "14%", "right": "8%", "top": 46, "bottom": 30},
            "xAxis": {"type": "value", "splitLine": {"lineStyle": {"color": "#f0f2f5"}}},
            "yAxis": {"type": "category", "data": cats[::-1],
                      "axisLabel": {"fontSize": 11, "color": "#606266"}},
            "series": [{
                "name": "关联疾病数", "type": "bar", "data": vals[::-1],
                "itemStyle": {"borderRadius": [0, 6, 6, 0],
                              "color": {"type": "linear", "x": 0, "y": 0, "x2": 1, "y2": 0,
                                        "colorStops": [{"offset": 0, "color": "#7dd3fc"},
                                                       {"offset": 1, "color": "#2b8fe8"}]}},
                "label": {"show": True, "position": "right", "fontSize": 11, "color": "#606266"},
            }],
        }
        insight = (
            f"「{cats[0]}」是图谱中关联疾病最多的症状（{vals[0]} 种疾病），"
            "反映其在临床鉴别中的高区分价值。" if ordered else "图谱中暂无症状数据。"
        )
        return ChartResponse(
            chart_type="bar", title=f"高频症状 Top{topn}", unit="种",
            categories=cats, values=[float(v) for v in vals], x_axis=cats,
            echarts_option=option, pyecharts_html="", insight=insight,
        )

    # ==================================================================
    #  6) 科室疾病分布（附加图表）
    # ==================================================================
    def department_distribution(self, topn: int = 10) -> ChartResponse:
        counts: Dict[str, int] = {}
        for e in self.store.edges:
            if e["rel"] != "BELONGS_TO":
                continue
            tail = self.store.nodes.get(e["target"])
            if tail and tail["type"] == "Department":
                counts[tail["name"]] = counts.get(tail["name"], 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:topn]
        series = [
            PieSeriesItem(name=k, value=float(v), color=PALETTE[i % len(PALETTE)])
            for i, (k, v) in enumerate(ordered)
        ]
        option = {
            "title": {"text": "科室疾病分布", "left": "center", "top": 4,
                      "textStyle": {"fontSize": 14, "color": "#303133"}},
            "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c} 条关联（{d}%）"},
            "legend": {"type": "scroll", "bottom": 0, "textStyle": {"fontSize": 11}},
            "series": [{
                "name": "科室关联数", "type": "pie", "radius": ["38%", "62%"],
                "center": ["50%", "52%"],
                "itemStyle": {"borderRadius": 4, "borderColor": "#fff", "borderWidth": 2},
                "label": {"show": True, "formatter": "{b} {d}%", "fontSize": 10,
                          "color": "#606266"},
                "data": [{"name": s.name, "value": s.value,
                          "itemStyle": {"color": s.color}} for s in series],
            }],
        }
        return ChartResponse(
            chart_type="pie", title="科室疾病分布", series=series, unit="条",
            echarts_option=option, pyecharts_html=self._py_pie(
                [(k, int(v)) for k, v in ordered], "科室疾病分布", donut=False),
            insight=f"覆盖 {len(counts)} 个科室，{ordered[0][0] if ordered else ''} 关联疾病最多。",
        )

    # ==================================================================
    #  PyEcharts 服务端渲染
    # ==================================================================
    @staticmethod
    def _py_pie(data: List[Tuple[str, int]], title: str, donut: bool = True) -> str:
        if not HAS_PYECHARTS:
            return ""
        try:
            chart = Pie(init_opts=opts.InitOpts(
                width="100%", height="320px", bg_color="transparent"))
            chart.add(
                series_name=title,
                data_pair=[(n, v) for n, v in data],
                radius=["42%", "68%"] if donut else ["0%", "62%"],
                center=["58%", "55%"],
                label_opts=opts.LabelOpts(formatter="{b}\n{d}%", font_size=11),
            )
            chart.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=title, pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(font_size=14, color="#303133")),
                legend_opts=opts.LegendOpts(orient="vertical", pos_left="left", pos_top="middle"),
                tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}<br/>{c}（{d}%）"),
                color=PALETTE,
            )
            return chart.render_embed()
        except Exception as exc:  # noqa: BLE001
            logger.warning("PyEcharts 渲染失败：%s", exc)
            return ""

    @staticmethod
    def _py_bar(cats: List[str], vals: List[float], title: str) -> str:
        if not HAS_PYECHARTS:
            return ""
        try:
            chart = Bar(init_opts=opts.InitOpts(
                width="100%", height="320px", bg_color="transparent"))
            chart.add_xaxis(cats)
            chart.add_yaxis(
                series_name="疾病数量", y_axis=vals,
                label_opts=opts.LabelOpts(is_show=True, position="top", font_size=11),
                itemstyle_opts=opts.ItemStyleOpts(color="#409EFF"),
            )
            chart.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=title, pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(font_size=14, color="#303133")),
                xaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(rotate=30 if len(cats) > 6 else 0)),
                yaxis_opts=opts.AxisOpts(name="疾病数量"),
                legend_opts=opts.LegendOpts(is_show=False),
                color=PALETTE,
            )
            return chart.render_embed()
        except Exception as exc:  # noqa: BLE001
            logger.warning("PyEcharts 渲染失败：%s", exc)
            return ""

    # ==================================================================
    def info(self) -> Dict[str, Any]:
        return {
            "pyecharts_available": HAS_PYECHARTS,
            "charts": ["overview", "node_type_pie", "category_bar",
                       "infectious_gauge", "symptom_top", "department_distribution"],
        }


# ---------------------------------------------------------------------------
_service: Optional[AnalyticsService] = None


def get_analytics_service() -> AnalyticsService:
    global _service
    if _service is None:
        _service = AnalyticsService()
    return _service
