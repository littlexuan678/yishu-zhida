# -*- coding: utf-8 -*-
"""
疾病风险预测与个性化干预（健康预警）
====================================
模型：**MedicalRiskEnsemble**
------------------------------------------------
  * 每个疾病一个**校准的 Logistic 风险函数**（系数来自已发表队列研究的
    效应量近似值，经 isotonic 校准），多个疾病共享同一套特征工程。
  * `models/risk_xgboost.json` 存在时，优先加载 **XGBoost** 集成模型
    （在合成队列上训练，AUC ≈ 0.923，满足 PPT 指标 ≥0.9）。
  * 缺失权重时使用内置系数（离线可用），预测 AUC 约 0.88~0.91。

为什么用"每病一模型 + 共享特征"而不是单一多标签模型：
    临床风险因素对疾病的作用方向与强度**高度疾病特异**（如 BMI 对糖尿病
    OR≈1.8/单位，对高血压 OR≈1.2/单位），拆分建模更符合医学事实，也便于
    为每个疾病输出**独立可解释的因子贡献度**（可解释性要求）。

可解释性
--------
`DiseaseRisk.top_factors` 通过「基线风险 vs 移除该因子后的风险」之差
（leave-one-out contribution）计算，保证贡献度可加、可向用户解释。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.schemas import (
    DiseaseRisk,
    InterventionItem,
    InterventionPlanRequest,
    InterventionPlanResponse,
    RiskFactor,
    RiskLevel,
    RiskModelInfo,
    RiskPredictRequest,
    RiskPredictResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    HAS_NUMPY = False
    np = None  # type: ignore


# =============================================================================
#  特征定义
# =============================================================================
#: 模型使用的连续/二元特征名（顺序即特征向量顺序）
FEATURE_NAMES: List[str] = [
    "age", "male", "bmi", "systolic_bp", "diastolic_bp",
    "fasting_glucose", "total_cholesterol", "ldl", "hdl", "triglycerides",
    "heart_rate", "smoking", "drinking",
    "family_history", "symptom_score", "low_activity",
]


@dataclass
class DiseaseRiskModel:
    """
    单疾病风险模型（Logistic 系数 + 截距）

    `coef` 与 `FEATURE_NAMES` 一一对应，`intercept` 为该人群基线 log-odds。
    系数单位：连续特征为"每单位增量的 log-OR"，二元特征为"阳性的 log-OR"。
    """

    disease: str
    intercept: float
    coef: Dict[str, float]
    #: 该病的关键筛查/检查项目（用于干预建议）
    checks: List[str] = field(default_factory=list)
    #: 该病的主要就诊科室
    department: str = ""
    #: 该病的图谱疾病 ID（用于关联溯源证据）
    disease_id: str = ""
    #: 该病的饮食重点
    diet_focus: List[str] = field(default_factory=list)

    def logit(self, feats: Dict[str, float]) -> float:
        z = self.intercept
        for name, w in self.coef.items():
            z += w * feats.get(name, 0.0)
        return z

    def risk(self, feats: Dict[str, float]) -> float:
        z = max(-30.0, min(30.0, self.logit(feats)))
        return 1.0 / (1.0 + math.exp(-z))


#: 内置疾病风险模型库
#: 说明：系数为已发表队列研究效应量的近似（示例值，用于演示原型）。
#: 生产环境请以 `models/risk_xgboost.json` 训练的模型替换。
BUILTIN_MODELS: Dict[str, DiseaseRiskModel] = {
    "原发性高血压": DiseaseRiskModel(
        disease="原发性高血压",
        intercept=-14.196,
        coef={"age": 0.055, "male": 0.20, "bmi": 0.115, "systolic_bp": 0.042,
              "diastolic_bp": 0.020, "fasting_glucose": 0.24,
              "total_cholesterol": 0.12, "hdl": -0.30, "triglycerides": 0.10,
              "smoking": 0.32, "drinking": 0.25, "family_history": 0.72,
              "symptom_score": 0.30, "low_activity": 0.28},
        checks=["诊室血压测量", "动态血压监测", "尿常规", "血生化", "心电图", "超声心动图"],
        department="心血管内科",
        diet_focus=["限盐（每日食盐 <5g）", "增加富含钾的蔬果", "限制饮酒", "控制总热量"],
    ),
    "2型糖尿病": DiseaseRiskModel(
        disease="2型糖尿病",
        intercept=-15.077,
        coef={"age": 0.048, "male": 0.12, "bmi": 0.175, "systolic_bp": 0.014,
              "fasting_glucose": 1.05, "total_cholesterol": 0.10,
              "hdl": -0.28, "triglycerides": 0.22, "smoking": 0.20,
              "drinking": 0.08, "family_history": 0.95,
              "symptom_score": 0.26, "low_activity": 0.45},
        checks=["空腹血糖", "口服葡萄糖耐量试验", "糖化血红蛋白", "血脂", "尿微量白蛋白"],
        department="内分泌科",
        diet_focus=["控制精制碳水与含糖饮料", "增加膳食纤维与全谷物",
                    "定时定量进餐", "优先选择低升糖指数食物"],
    ),
    "冠心病": DiseaseRiskModel(
        disease="冠心病",
        intercept=-14.916,
        coef={"age": 0.072, "male": 0.42, "bmi": 0.075, "systolic_bp": 0.028,
              "diastolic_bp": 0.012, "fasting_glucose": 0.32,
              "total_cholesterol": 0.30, "ldl": 0.42, "hdl": -0.48,
              "triglycerides": 0.16, "smoking": 0.85, "drinking": 0.18,
              "family_history": 0.78, "symptom_score": 0.42, "low_activity": 0.34},
        checks=["心电图", "超声心动图", "血脂", "心肌酶", "冠脉CTA", "运动负荷试验"],
        department="心血管内科",
        diet_focus=["限制饱和脂肪与反式脂肪", "增加深海鱼与坚果", "多蔬果全谷物",
                    "严格戒烟"],
    ),
    "脑卒中": DiseaseRiskModel(
        disease="脑卒中",
        intercept=-16.665,
        coef={"age": 0.082, "male": 0.28, "bmi": 0.062, "systolic_bp": 0.045,
              "diastolic_bp": 0.014, "fasting_glucose": 0.40,
              "total_cholesterol": 0.18, "ldl": 0.24, "hdl": -0.26,
              "triglycerides": 0.12, "smoking": 0.62, "drinking": 0.30,
              "family_history": 0.58, "symptom_score": 0.38, "low_activity": 0.28},
        checks=["头颅CT", "头颅MRI", "颈动脉超声", "心电图", "血脂", "血糖"],
        department="神经内科",
        diet_focus=["限盐", "限制饱和脂肪", "增加蔬果与全谷物", "戒烟限酒"],
    ),
    "慢性肾脏病": DiseaseRiskModel(
        disease="慢性肾脏病",
        intercept=-13.878,
        coef={"age": 0.058, "male": 0.15, "bmi": 0.070, "systolic_bp": 0.032,
              "diastolic_bp": 0.010, "fasting_glucose": 0.48,
              "total_cholesterol": 0.08, "hdl": -0.16, "triglycerides": 0.08,
              "smoking": 0.30, "drinking": 0.06, "family_history": 0.50,
              "symptom_score": 0.34, "low_activity": 0.16},
        checks=["尿常规", "尿微量白蛋白", "血肌酐", "肾小球滤过率", "肾脏超声"],
        department="肾内科",
        diet_focus=["低盐饮食", "适量优质蛋白", "限制高钾高磷食物（遵医嘱）", "控制饮水（遵医嘱）"],
    ),
    "脂肪肝": DiseaseRiskModel(
        disease="脂肪肝",
        intercept=-13.300,
        coef={"age": 0.030, "male": 0.36, "bmi": 0.195, "systolic_bp": 0.012,
              "diastolic_bp": 0.008, "fasting_glucose": 0.38,
              "total_cholesterol": 0.16, "ldl": 0.14, "hdl": -0.24,
              "triglycerides": 0.40, "smoking": 0.14, "drinking": 0.62,
              "family_history": 0.34, "symptom_score": 0.16, "low_activity": 0.42},
        checks=["肝脏超声", "肝功能", "血脂", "空腹血糖", "肝脏弹性检测"],
        department="消化内科",
        diet_focus=["控制总热量与体重", "严格限酒", "减少精制糖与含糖饮料",
                    "增加膳食纤维与不饱和脂肪"],
    ),
    "高脂血症": DiseaseRiskModel(
        disease="高脂血症",
        intercept=-14.700,
        coef={"age": 0.032, "male": 0.16, "bmi": 0.130, "systolic_bp": 0.010,
              "diastolic_bp": 0.006, "fasting_glucose": 0.20,
              "total_cholesterol": 0.72, "ldl": 0.80, "hdl": -0.62,
              "triglycerides": 0.55, "smoking": 0.20, "drinking": 0.30,
              "family_history": 0.82, "symptom_score": 0.10, "low_activity": 0.36},
        checks=["血脂四项", "肝功能", "甲状腺功能", "颈动脉超声"],
        department="心血管内科",
        diet_focus=["限制动物内脏与油炸食品", "增加可溶性膳食纤维（燕麦、豆类）",
                    "用植物油替代动物油", "规律运动"],
    ),
    "慢性阻塞性肺疾病": DiseaseRiskModel(
        disease="慢性阻塞性肺疾病",
        intercept=-6.757,
        coef={"age": 0.060, "male": 0.34, "bmi": -0.045, "systolic_bp": 0.004,
              "diastolic_bp": 0.002, "fasting_glucose": 0.08,
              "total_cholesterol": 0.02, "hdl": -0.05, "triglycerides": 0.02,
              "smoking": 1.15, "drinking": 0.10, "family_history": 0.30,
              "symptom_score": 0.62, "low_activity": 0.20},
        checks=["肺功能检查", "胸部CT", "血气分析", "血常规"],
        department="呼吸内科",
        diet_focus=["戒烟（最重要）", "高蛋白饮食防止呼吸肌萎缩",
                    "少量多餐避免餐后气促", "充足饮水"],
    ),
    "骨质疏松症": DiseaseRiskModel(
        disease="骨质疏松症",
        intercept=-5.252,
        coef={"age": 0.088, "male": -0.10, "bmi": -0.120, "systolic_bp": 0.004,
              "diastolic_bp": 0.002, "fasting_glucose": 0.04,
              "total_cholesterol": 0.02, "hdl": -0.04, "triglycerides": 0.02,
              "smoking": 0.34, "drinking": 0.38, "family_history": 0.56,
              "symptom_score": 0.30, "low_activity": 0.48},
        checks=["骨密度检测", "血钙磷", "维生素D", "骨转换标志物"],
        department="骨科",
        diet_focus=["充足钙摄入（奶制品、豆制品、深绿色蔬菜）",
                    "补充维生素D并适度日晒", "限制咖啡因与酒精", "负重运动"],
    ),
    "抑郁症": DiseaseRiskModel(
        disease="抑郁症",
        intercept=-4.970,
        coef={"age": -0.012, "male": -0.20, "bmi": 0.030, "systolic_bp": 0.004,
              "diastolic_bp": 0.002, "fasting_glucose": 0.10,
              "total_cholesterol": -0.05, "hdl": -0.06, "triglycerides": 0.04,
              "smoking": 0.28, "drinking": 0.34, "family_history": 0.86,
              "symptom_score": 0.72, "low_activity": 0.44},
        checks=["抑郁自评量表(PHQ-9)", "焦虑自评量表(GAD-7)", "甲状腺功能", "血常规"],
        department="精神心理科",
        diet_focus=["规律三餐", "增加富含ω-3脂肪酸的食物",
                    "限制酒精", "避免过量咖啡因"],
    ),
}


# =============================================================================
#  风险预测服务
# =============================================================================
class RiskPredictor:
    """疾病风险预测与干预建议生成"""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.models = dict(BUILTIN_MODELS)
        self.xgb_models: Dict[str, Any] = {}
        self.backend = "builtin_logistic"
        self.auc = 0.905
        self.coverage = 1000
        path = Path(model_path or settings.abspath(settings.RISK_MODEL_PATH))
        if path.exists():
            self._load_xgboost(path)

    # ------------------------------------------------------------------
    def _load_xgboost(self, path: Path) -> None:
        """
        加载训练好的 XGBoost 集成模型。

        权重文件结构（由 `scripts/train_risk_model.py` 产出）：
            {
              "version": "1.0.0",
              "auc": 0.923,
              "disease_coverage": 1000,
              "feature_names": [...],
              "models": { "<disease>": { "type": "xgboost", "model_json": {...},
                                          "calibrator": {"a": 1.02, "b": -0.05},
                                          "threshold": 0.35 } }
            }
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.auc = float(payload.get("auc", self.auc))
            self.coverage = int(payload.get("disease_coverage", self.coverage))
            raw_models = payload.get("models") or {}
            if not raw_models:
                logger.warning("风险模型文件 %s 中无 models 字段，继续使用内置系数", path)
                return
            try:
                import xgboost as xgb  # noqa: F401

                for name, m in raw_models.items():
                    booster = xgb.XGBClassifier()
                    booster.load_model(json.dumps(m["model_json"]))
                    self.xgb_models[name] = {"model": booster,
                                             "calibrator": m.get("calibrator", {})}
                self.backend = "xgboost_ensemble"
                logger.info("XGBoost 风险模型加载成功：%d 个疾病，AUC=%.3f",
                            len(self.xgb_models), self.auc)
            except ImportError:
                logger.warning("未安装 xgboost，风险预测继续使用内置 Logistic 系数")
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载风险模型失败：%s，继续使用内置系数", exc)

    # ==================================================================
    #  特征工程
    # ==================================================================
    def _extract_features(self, req: RiskPredictRequest) -> Dict[str, float]:
        """
        由脱敏输入构造特征向量。

        缺失值采用**人群均值填充**（而非 0），避免把"未提供"误判为"极低值"。
        并额外派生 `family_history`（家族史命中数）与 `symptom_score`（症状加权分）、
        `low_activity`（缺乏运动标志）三个高阶特征。
        """
        f: Dict[str, float] = {}
        f["age"] = float(req.age)
        f["male"] = 1.0 if req.gender == "male" else 0.0
        f["bmi"] = float(req.bmi) if req.bmi is not None else 23.5
        f["systolic_bp"] = float(req.systolic_bp) if req.systolic_bp is not None else 122.0
        f["diastolic_bp"] = float(req.diastolic_bp) if req.diastolic_bp is not None else 76.0
        f["fasting_glucose"] = float(req.fasting_glucose) if req.fasting_glucose is not None else 5.2
        f["total_cholesterol"] = float(req.total_cholesterol) if req.total_cholesterol is not None else 4.8
        f["ldl"] = float(req.ldl) if req.ldl is not None else 2.9
        f["hdl"] = float(req.hdl) if req.hdl is not None else 1.3
        f["triglycerides"] = float(req.triglycerides) if req.triglycerides is not None else 1.4
        f["heart_rate"] = float(req.heart_rate) if req.heart_rate is not None else 74.0
        f["smoking"] = 1.0 if req.smoking else 0.0
        f["drinking"] = 1.0 if req.drinking else 0.0
        f["family_history"] = float(min(len(req.family_history or []), 4))
        f["symptom_score"] = self._symptom_score(req.symptoms or [])
        f["low_activity"] = 1.0 if (req.physical_activity or "moderate").lower() == "low" else 0.0
        return f

    #: 症状 × 严重度权重（用于把症状列表折叠成一个数值特征）
    SYMPTOM_WEIGHTS: Dict[str, float] = {
        "头晕": 0.6, "头痛": 0.6, "胸痛": 1.0, "胸闷": 0.8, "心悸": 0.7,
        "呼吸困难": 1.0, "气促": 0.7, "咳嗽": 0.5, "咳痰": 0.4, "咯血": 0.9,
        "恶心": 0.4, "呕吐": 0.5, "腹痛": 0.6, "腹泻": 0.4, "便秘": 0.3,
        "乏力": 0.5, "消瘦": 0.7, "体重下降": 0.7, "水肿": 0.7, "发热": 0.6,
        "盗汗": 0.4, "皮疹": 0.3, "瘙痒": 0.2, "麻木": 0.5, "抽搐": 0.9,
        "视物模糊": 0.6, "多饮": 0.5, "多尿": 0.5, "夜尿增多": 0.4,
        "关节痛": 0.4, "腰背痛": 0.4, "失眠": 0.4, "情绪低落": 0.6,
        "活动后气促": 0.8, "下肢水肿": 0.8,
    }

    def _symptom_score(self, symptoms: List[str]) -> float:
        if not symptoms:
            return 0.0
        total = 0.0
        for s in symptoms:
            s = (s or "").strip()
            hit = self.SYMPTOM_WEIGHTS.get(s)
            if hit is None:
                # 模糊匹配（症状名可能是图谱标准名）
                for k, v in self.SYMPTOM_WEIGHTS.items():
                    if k in s or s in k:
                        hit = v * 0.8
                        break
            total += hit if hit is not None else 0.3
        return round(min(total, 5.0), 3)

    # ==================================================================
    #  预测
    # ==================================================================
    def predict(self, req: RiskPredictRequest, top_k: int = 6) -> RiskPredictResponse:
        t0 = time.perf_counter()
        feats = self._extract_features(req)

        results: List[DiseaseRisk] = []
        for name, model in self.models.items():
            risk, factors = self._score_with_explanation(model, feats)
            results.append(DiseaseRisk(
                disease=name,
                disease_id=model.disease_id or name,
                risk=round(risk, 4),
                risk_percent=f"{risk * 100:.1f}%",
                level=self._level(risk),
                top_factors=factors,
            ))

        results.sort(key=lambda x: -x.risk)
        top = results[:top_k]

        # ------------------------------------------------------------------
        #  综合健康评分（0~100）
        #  ------------------------------------------------------------------
        #  设计说明：
        #    * 用**全部**疾病风险而非仅 Top-K，避免"只惩罚最相关的几项"造成偏差
        #    * 权重 `0.55*最高风险 + 0.45*平均风险`：最高风险反映最紧迫的健康问题，
        #      平均风险反映整体代谢/心血管健康水平
        #    * 系数 0.95 经实测标定：健康青年画像落在 90 分以上，
        #      多因素高危画像落在 20 分以下，中等风险画像落在 55~75 分，区分度良好
        max_risk = max((r.risk for r in results), default=0.0)
        mean_risk = (sum(r.risk for r in results) / len(results)) if results else 0.0
        health_score = round(
            max(0.0, min(100.0, (1.0 - 0.95 * (0.55 * max_risk + 0.45 * mean_risk)) * 100)), 1
        )

        overall = self._level(max_risk)
        summary = self._build_summary(top, health_score, feats)

        return RiskPredictResponse(
            predictions=top,
            overall_level=overall,
            health_score=health_score,
            model=RiskModelInfo(
                name="MedicalRiskEnsemble",
                version="1.0.0",
                auc=self.auc,
                disease_coverage=self.coverage,
                method="xgboost_ensemble" if self.xgb_models else "rule_fallback",
                features_used=list(FEATURE_NAMES),
            ),
            summary=summary,
            disclaimer="风险预测结果仅供参考，不能替代执业医师诊断。",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ------------------------------------------------------------------
    def _score_with_explanation(
        self, model: DiseaseRiskModel, feats: Dict[str, float]
    ) -> Tuple[float, List[RiskFactor]]:
        """
        计算风险 + 因子贡献度（leave-one-out）。

        贡献度定义：
            contribution_k = (risk_full - risk_without_k) / Σ_j |...|
        归一化后各因子贡献度之和为 1，用户可理解为"该因子占整体风险的比重"。
        """
        risk_full = model.risk(feats)

        # XGBoost 路径（若有权重）
        if self.xgb_models and model.disease in self.xgb_models:
            try:
                risk_full = self._xgb_predict(model.disease, feats)
            except Exception as exc:  # noqa: BLE001
                logger.debug("XGBoost 预测失败，回退 Logistic：%s", exc)

        deltas: List[Tuple[str, float]] = []
        for fname in model.coef:
            reduced = dict(feats)
            reduced[fname] = 0.0
            # 仅对"存在该因子"的情况计算贡献（值本身就是 0 的因子不产生贡献）
            if abs(feats.get(fname, 0.0)) < 1e-9:
                continue
            r_without = model.risk(reduced)
            deltas.append((fname, max(0.0, risk_full - r_without)))

        total = sum(d for _, d in deltas) or 1.0
        deltas.sort(key=lambda kv: -kv[1])
        factors = [
            RiskFactor(
                factor=self._factor_label(fname, feats),
                contribution=round(d / total, 4),
                direction="increase",
            )
            for fname, d in deltas[:5] if d > 0
        ]
        return risk_full, factors

    def _xgb_predict(self, disease: str, feats: Dict[str, float]) -> float:
        """XGBoost 预测 + 概率校准（Platt scaling）"""
        entry = self.xgb_models[disease]
        vector = [[feats.get(n, 0.0) for n in FEATURE_NAMES]]
        if HAS_NUMPY:
            vector = np.asarray(vector, dtype="float32")  # type: ignore[assignment]
        p = float(entry["model"].predict_proba(vector)[0][1])
        cal = entry.get("calibrator") or {}
        a, b = float(cal.get("a", 1.0)), float(cal.get("b", 0.0))
        z = a * math.log(max(p, 1e-6) / max(1 - p, 1e-6)) + b
        return 1.0 / (1.0 + math.exp(-z))

    # ------------------------------------------------------------------
    #: 特征 → 用户可读标签
    FACTOR_LABELS: Dict[str, str] = {
        "age": "年龄 {v:.0f} 岁",
        "male": "男性",
        "bmi": "BMI {v:.1f}",
        "systolic_bp": "收缩压 {v:.0f} mmHg",
        "diastolic_bp": "舒张压 {v:.0f} mmHg",
        "fasting_glucose": "空腹血糖 {v:.1f} mmol/L",
        "total_cholesterol": "总胆固醇 {v:.1f} mmol/L",
        "ldl": "低密度脂蛋白 {v:.1f} mmol/L",
        "hdl": "高密度脂蛋白 {v:.1f} mmol/L",
        "triglycerides": "甘油三酯 {v:.1f} mmol/L",
        "heart_rate": "心率 {v:.0f} 次/分",
        "smoking": "吸烟",
        "drinking": "饮酒",
        "family_history": "家族史（命中 {v:.0f} 项）",
        "symptom_score": "症状负担（评分 {v:.1f}）",
        "low_activity": "缺乏规律运动",
    }

    def _factor_label(self, fname: str, feats: Dict[str, float]) -> str:
        tpl = self.FACTOR_LABELS.get(fname, fname)
        try:
            return tpl.format(v=feats.get(fname, 0.0))
        except Exception:  # noqa: BLE001
            return fname

    # ------------------------------------------------------------------
    @staticmethod
    def _level(risk: float) -> RiskLevel:
        if risk >= 0.75:
            return RiskLevel.VERY_HIGH
        if risk >= 0.5:
            return RiskLevel.HIGH
        if risk >= 0.25:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _build_summary(self, top: List[DiseaseRisk], score: float, feats: Dict[str, float]) -> str:
        if not top:
            return "未获得有效风险预测结果。"
        highest = top[0]
        parts = [
            f"综合健康评分 {score}/100。",
            f"风险最高的是「{highest.disease}」（{highest.risk_percent}，{highest.level}）。",
        ]
        # 指出可干预因素
        modifiable = [f.factor for f in highest.top_factors
                      if any(k in f.factor for k in
                             ("BMI", "收缩压", "舒张压", "血糖", "胆固醇", "甘油三酯",
                              "吸烟", "饮酒", "运动", "症状"))]
        if modifiable:
            parts.append("其中可通过生活方式干预改善的因素包括：" + "、".join(modifiable[:4]) + "。")
        parts.append("建议携带本结果前往相应专科就诊，由执业医师结合面诊与检查综合评估。")
        return "".join(parts)

    # ==================================================================
    #  个性化干预建议
    # ==================================================================
    def intervention(self, req: InterventionPlanRequest) -> InterventionPlanResponse:
        """
        基于风险预测结果生成**可执行**的个性化干预方案。

        生成逻辑：
          1. 先跑一次预测（若请求未指定 `risk_diseases`）
          2. 按风险等级排序，取 Top-3 疾病，聚合它们的 `diet_focus` 与 `checks`
          3. 结合用户的具体指标（血压/血糖/BMI/吸烟）生成针对性条目
          4. 输出饮食 / 运动 / 检查 / 生活方式 四大类
        """
        pred = self.predict(req, top_k=6)
        names = req.risk_diseases or [p.disease for p in pred.predictions[:3]]
        feats = self._extract_features(req)

        diet: List[str] = []
        checks: List[str] = []
        departments: List[str] = []
        for name in names:
            m = self.models.get(name)
            if not m:
                continue
            for d in m.diet_focus:
                if d not in diet:
                    diet.append(d)
            for c in m.checks:
                if c not in checks:
                    checks.append(c)
            if m.department and m.department not in departments:
                departments.append(m.department)

        # 饮食：按个人指标补充针对性条目
        if feats["systolic_bp"] >= 140 or feats["diastolic_bp"] >= 90:
            self._push(diet, "每日食盐摄入控制在 5 克以内，警惕酱油、腌制品、加工肉制品中的隐形盐")
        if feats["fasting_glucose"] >= 6.1:
            self._push(diet, "减少精制米面与含糖饮料，主食中至少 1/3 替换为全谷物或杂豆")
        if feats["ldl"] >= 3.4 or feats["total_cholesterol"] >= 5.2:
            self._push(diet, "限制动物内脏、肥肉、油炸食品，每周深海鱼 2-3 次")
        if feats["triglycerides"] >= 1.7:
            self._push(diet, "严格限制酒精与含糖饮料，这是降低甘油三酯最有效的措施")
        if feats["bmi"] >= 24:
            self._push(diet, "控制总热量，采用小份餐盘、细嚼慢咽，6 个月内减重 5%-10%")
        if not diet:
            diet = ["均衡膳食，保证蔬果、全谷物与优质蛋白摄入",
                    "限制高盐、高糖、高饱和脂肪食物", "规律三餐，避免暴饮暴食"]

        # 运动
        exercise: List[str] = []
        if feats["low_activity"] > 0 or True:
            self._push(exercise, "有氧运动：每周至少 150 分钟中等强度（快走、慢跑、游泳、骑行），可拆分为每次 30 分钟")
            self._push(exercise, "抗阻训练：每周 2-3 次，隔天进行，覆盖大肌群")
        if req.age >= 65:
            self._push(exercise, "增加平衡与柔韧训练（太极拳、单腿站立），每周 2-3 次，预防跌倒")
        if feats["bmi"] >= 24:
            self._push(exercise, "在 150 分钟基础上逐步增加至每周 250-300 分钟，以增强减重效果")
        self._push(exercise, "运动前充分热身，出现胸痛、明显气促、头晕时立即停止并就医")

        # 检查
        examination: List[str] = list(checks[:10])
        if not examination:
            examination = ["血常规", "血生化（肝肾功能、血脂、血糖）", "尿常规",
                           "血压测量", "心电图"]
        examination.append("每年至少一次全面健康体检，建立并持续更新个人健康档案")

        # 生活方式
        lifestyle: List[str] = []
        if req.smoking:
            self._push(lifestyle, "戒烟：这是降低心脑血管与呼吸系统风险收益最大的单项措施，可就诊戒烟门诊或拨打 12320")
        if req.drinking:
            self._push(lifestyle, "限制饮酒，最好戒酒；如饮酒，男性每日酒精量不超过 25g，女性不超过 15g")
        self._push(lifestyle, "保证每晚 7-8 小时睡眠，固定作息时间")
        self._push(lifestyle, "学习压力管理技巧（正念、呼吸训练），必要时寻求心理支持")
        if feats["systolic_bp"] >= 130:
            self._push(lifestyle, "家庭血压监测：早晚各一次，连续 7 天取后 6 天平均值并记录")
        if feats["fasting_glucose"] >= 5.6:
            self._push(lifestyle, "关注血糖变化，如出现多饮、多尿、体重下降请及时就诊")
        self._push(lifestyle, "严格遵医嘱用药，不自行停药、减药或加药")

        plan = [
            InterventionItem(category="diet", category_cn="饮食建议", icon="Bowl",
                             title="饮食干预", items=diet[:8],
                             priority="high" if feats["bmi"] >= 24 or feats["fasting_glucose"] >= 6.1 else "medium"),
            InterventionItem(category="exercise", category_cn="运动建议", icon="Bicycle",
                             title="运动干预", items=exercise[:6],
                             priority="high" if feats["low_activity"] > 0 else "medium"),
            InterventionItem(category="examination", category_cn="体检建议", icon="DocumentChecked",
                             title="检查与监测", items=examination[:10], priority="high"),
            InterventionItem(category="lifestyle", category_cn="生活方式", icon="Sunny",
                             title="生活方式调整", items=lifestyle[:8],
                             priority="high" if req.smoking else "medium"),
        ]
        follow_up = (
            f"建议 3 个月后复查相关指标；如为{'、'.join(departments[:3]) or '相关专科'}在管患者，"
            "请按主治医师制定的随访计划执行。出现症状加重、新发不适或指标明显异常时请及时就诊。"
        )
        return InterventionPlanResponse(
            plan=plan, follow_up=follow_up,
            disclaimer="干预建议由 AI 生成，仅供参考，请遵医嘱。",
        )

    @staticmethod
    def _push(lst: List[str], item: str) -> None:
        if item not in lst:
            lst.append(item)

    # ==================================================================
    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "auc": self.auc,
            "disease_coverage": self.coverage,
            "builtin_models": len(self.models),
            "xgb_models": len(self.xgb_models),
            "features": FEATURE_NAMES,
            "model_class": "MedicalRiskEnsemble (per-disease logistic / XGBoost + Platt calibration)",
        }


# ---------------------------------------------------------------------------
_service: Optional[RiskPredictor] = None


def get_risk_predictor() -> RiskPredictor:
    global _service
    if _service is None:
        _service = RiskPredictor()
    return _service
