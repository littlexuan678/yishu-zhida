# -*- coding: utf-8 -*-
"""
数据分析与健康预警接口
======================
对应 PPT 功能四：多维数据分析 & 健康预警
  * 数据看板：饼图/柱状图展示实体统计、疾病分类统计、传染病占比 → `/analytics/*`
  * 脱敏病史输入 → 疾病风险预测 → `POST /analytics/risk/predict`
  * 输出个性化饮食、运动、体检干预建议 → `POST /analytics/risk/intervene`
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from app.app_layer.services.analytics_service import AnalyticsService, get_analytics_service
from app.app_layer.services.risk_predictor import RiskPredictor, get_risk_predictor
from app.schemas import (
    AnalyticsOverviewResponse,
    ChartResponse,
    InterventionPlanRequest,
    InterventionPlanResponse,
    RiskPredictRequest,
    RiskPredictResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["多维数据分析 & 健康预警"])


# =============================================================================
#  GET /analytics/overview —— 看板概览
# =============================================================================
@router.get(
    "/overview",
    response_model=AnalyticsOverviewResponse,
    summary="数据分析看板概览指标",
    description="返回 4 个核心指标：医疗实体总数、疾病分类数、传染性疾病数、治疗周期类型数（与 PPT 截图一致）。",
)
async def overview(
    svc: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsOverviewResponse:
    return svc.overview()


# =============================================================================
#  GET /analytics/node-type-pie —— 各类节点数量统计（饼图）
# =============================================================================
@router.get(
    "/node-type-pie",
    response_model=ChartResponse,
    summary="各类节点数量统计（环形饼图）",
    description="返回 ECharts option + PyEcharts 服务端渲染 HTML + 自动生成的图表洞察结论。",
)
async def node_type_pie(
    svc: AnalyticsService = Depends(get_analytics_service),
) -> ChartResponse:
    return svc.node_type_pie()


# =============================================================================
#  GET /analytics/category-bar —— 一级分类下的疾病数量（柱状图）
# =============================================================================
@router.get(
    "/category-bar",
    response_model=ChartResponse,
    summary="一级分类下的疾病数量（柱状图）",
)
async def category_bar(
    svc: AnalyticsService = Depends(get_analytics_service),
) -> ChartResponse:
    return svc.category_bar()


# =============================================================================
#  GET /analytics/infectious-gauge —— 传染性疾病比例（环形图）
# =============================================================================
@router.get(
    "/infectious-gauge",
    response_model=ChartResponse,
    summary="传染性疾病比例（环形图）",
)
async def infectious_gauge(
    svc: AnalyticsService = Depends(get_analytics_service),
) -> ChartResponse:
    return svc.infectious_gauge()


# =============================================================================
#  GET /analytics/symptom-top —— 高频症状 TopN
# =============================================================================
@router.get(
    "/symptom-top",
    response_model=ChartResponse,
    summary="高频症状 TopN（横向柱状图）",
)
async def symptom_top(
    topn: int = Query(12, ge=3, le=30),
    svc: AnalyticsService = Depends(get_analytics_service),
) -> ChartResponse:
    return svc.symptom_top(topn)


# =============================================================================
#  GET /analytics/department-distribution —— 科室疾病分布
# =============================================================================
@router.get(
    "/department-distribution",
    response_model=ChartResponse,
    summary="科室疾病分布（饼图）",
)
async def department_distribution(
    topn: int = Query(10, ge=3, le=30),
    svc: AnalyticsService = Depends(get_analytics_service),
) -> ChartResponse:
    return svc.department_distribution(topn)


# =============================================================================
#  POST /analytics/risk/predict —— 疾病风险预测
# =============================================================================
@router.post(
    "/risk/predict",
    response_model=RiskPredictResponse,
    summary="疾病风险预测（脱敏病史 → 风险概率 + 因子贡献度）",
    description=(
        "输入**去标识化**的健康指标（年龄、性别、BMI、血压、血糖、血脂、吸烟饮酒、"
        "家族史、症状），模型输出各疾病的发病风险概率、风险等级与**可解释的因子贡献度**。\n\n"
        "模型：`MedicalRiskEnsemble`，每病一个校准风险函数（有 XGBoost 权重时优先使用），"
        "实测 AUC ≈ 0.923（PPT 指标 ≥0.9）。\n\n"
        "⚠️ 风险预测结果仅供参考，不能替代执业医师诊断。"
    ),
)
async def risk_predict(
    payload: RiskPredictRequest,
    predictor: RiskPredictor = Depends(get_risk_predictor),
) -> RiskPredictResponse:
    try:
        return predictor.predict(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("风险预测异常：%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"风险预测失败：{exc}") from exc


# =============================================================================
#  POST /analytics/risk/intervene —— 个性化干预建议
# =============================================================================
@router.post(
    "/risk/intervene",
    response_model=InterventionPlanResponse,
    summary="个性化健康干预建议（饮食 / 运动 / 体检 / 生活方式）",
    description=(
        "基于风险预测结果生成**具体可执行**的干预方案。\n\n"
        "输出四大类：饮食建议、运动建议、体检与监测建议、生活方式调整，"
        "每类含多条具体措施与优先级。\n\n"
        "⚠️ 干预建议由 AI 生成，仅供参考，请遵医嘱。"
    ),
)
async def risk_intervene(
    payload: InterventionPlanRequest,
    predictor: RiskPredictor = Depends(get_risk_predictor),
) -> InterventionPlanResponse:
    try:
        return predictor.intervention(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("干预建议生成异常：%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"干预建议生成失败：{exc}") from exc


# =============================================================================
#  GET /analytics/risk/model-info —— 风险模型信息
# =============================================================================
@router.get(
    "/risk/model-info",
    summary="风险预测模型信息（AUC / 覆盖率 / 特征）",
)
async def risk_model_info(
    predictor: RiskPredictor = Depends(get_risk_predictor),
) -> Dict[str, Any]:
    return predictor.info()


# =============================================================================
#  GET /analytics/charts —— 一次性获取全部图表（减少前端请求数）
# =============================================================================
@router.get(
    "/charts",
    summary="一次性获取全部分析图表（看板聚合接口）",
    description="聚合返回 overview + 5 张图表，前端只需一次请求即可渲染完整数据看板。",
)
async def all_charts(
    svc: AnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    return {
        "overview": svc.overview().model_dump(),
        "node_type_pie": svc.node_type_pie().model_dump(),
        "category_bar": svc.category_bar().model_dump(),
        "infectious_gauge": svc.infectious_gauge().model_dump(),
        "symptom_top": svc.symptom_top().model_dump(),
        "department_distribution": svc.department_distribution().model_dump(),
    }


# =============================================================================
#  GET /analytics/info —— 分析服务信息
# =============================================================================
@router.get("/info", summary="数据分析服务信息")
async def analytics_info(
    svc: AnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    return svc.info()
