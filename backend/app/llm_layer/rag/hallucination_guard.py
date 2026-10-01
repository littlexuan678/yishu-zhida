# -*- coding: utf-8 -*-
"""
幻觉守卫（HallucinationGuard）—— 医疗 RAG 的事后事实校验层
==========================================================

★ 为什么医疗 RAG 必须有"事后校验"★

即使 prompt 里写了"只能依据 <KG_CONTEXT> 作答"，大模型仍会：
  1. 忘记标注引用编号 → 结论不可溯源（`missing_citation`）
  2. 编造图谱里不存在的疾病/药物 → 凭空幻觉（`unknown_entity`）
  3. 给出具体剂量、频次 → 医疗事故级风险（`dose_violation`，critical）
  4. 输出"你得了XX""可以确诊" → 越权诊断（`absolute_claim` / `diagnostic_claim`）
  5. 引用不存在的 [KG-99] → 伪造证据（`invalid_citation`）

守卫对答案做 7 项确定性检查（纯正则 + 集合比对，无模型依赖、零 token 成本），
给出 0~1 的置信度评分与处置建议：
  * `accept`           直接返回模型答案
  * `rewrite`          建议前端提示"已降级/需复核"，仍返回模型答案
  * `fallback_template` 丢弃模型答案，改用**纯 KG 拼装的确定性回答**（`strip_and_fallback`），
    彻底消除幻觉风险 —— 这是本系统的安全底线。

评分规则：1.0 起扣，critical −0.30 / high −0.12 / medium −0.05 / low −0.02，
clamp 到 [0,1] 并保留 3 位小数；出现 critical 或得分 < 0.45 直接兜底模板。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.config import settings
from app.utils.logger import get_logger
from app.utils.text import truncate

logger = get_logger(__name__)

#: 统一免责声明（与 settings.disclaimer 保持一致）
DISCLAIMER = "本回答由 AI 生成，仅供参考，不能替代执业医师诊断。"

#: 严重级别 → 扣分
SEVERITY_DEDUCT: Dict[str, float] = {"critical": 0.30, "high": 0.12, "medium": 0.05, "low": 0.02}

#: 引用角标 [KG-n]
_CITE_RE = re.compile(r"\[KG-(\d+)\]")
#: 中文分句
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?\n])")
#: 结构性文本行（markdown 标题 / 引用块 / "**字段名**：值" 元信息行）不是事实性陈述，
#: 不应因缺少 [KG-n] 而被判为"引用缺失"（否则每篇答案的标题都会被误报）
_STRUCT_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s|>\s|\*\*[^*\n]{1,14}\*\*\s*[：:])")

#: 疾病/症状类医学名词（后缀法，长后缀优先）
_MED_TERM_RE = re.compile(
    r"[\u4e00-\u9fff]{2,10}(?:综合征|综合症|感染|硬化|梗死|栓塞|衰竭|障碍|溃疡|结石|贫血|"
    r"病|症|炎|癌|瘤)"
)
#: 药物类名词（只认高辨识度的剂型/词根后缀，避免"服用一片"这类误报）
_DRUG_TERM_RE = re.compile(
    r"[\u4e00-\u9fff]{2,10}(?:地平|沙坦|普利|洛尔|他汀|西林|霉素|环素|头孢|"
    r"胶囊|注射液|颗粒|口服液|滴丸|喷雾剂)"
)
#: 剂量/规格
_DOSE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg/日|mg|毫克|微克|μg|ug|IU|单位|ml|毫升|克|g|片|粒|支|袋)"
)
#: 给药频次 / 给药途径（英文缩写加词边界，避免误伤 important / policy 之类普通单词）
_FREQ_RE = re.compile(
    r"(?:每日\s*[一二三四1-9]\s*次|一天\s*[一二三四1-9]\s*次|\b(?:bid|tid|qid|qd|po|iv|im)\b|皮下注射\s*\d)",
    re.IGNORECASE,
)
#: 绝对化结论
_ABSOLUTE_RE = re.compile(
    r"(?:你(?:患|得)了|可以确诊|肯定是|一定是|百分之百|绝对(?:能|可以|不会)|"
    r"能彻底治愈|包治|无需就医|不用去医院|不需要治疗|没救了|治不好)"
)
#: 诊断性措辞
_DIAG_RE = re.compile(r"(?:诊断为|确诊为)")
#: 诊断措辞的安全前缀（其前 12 字内出现则不算越权诊断）
_DIAG_SAFE_PREFIX = ("不能", "无法", "须", "需", "应由")

#: 通用词白名单（不是医学实体，不应判为幻觉）
GENERIC_ALLOWLIST: set = {
    "患者", "疾病", "症状", "治疗", "检查", "医院", "医师", "医生", "建议", "可能", "注意",
    "需要", "情况", "身体", "健康", "药物", "科室", "门诊", "急诊", "时间", "如果", "因为",
    "所以", "出现", "导致", "引起", "常见", "主要", "一般", "通常", "包括", "以及", "同时",
    "但是", "应该", "必须", "立即", "及时", "目前", "最近", "然后", "结果", "以上", "以下",
    "这种", "该类", "部分", "多数", "少数", "所有", "任何", "相关", "进一步", "明确", "确诊",
    "排除", "考虑", "提示", "表现", "发生", "发展", "加重", "缓解", "改善", "控制", "预防",
    "康复", "用药", "就医", "就诊", "风险", "严重", "典型", "早期", "晚期", "慢性", "急性",
}

#: 后缀正则是贪婪匹配，"加二甲双胍胶囊"这类候选会被连带上动词/连词前缀，
#: 这里剥掉常见的前导虚词，让判定回归真正的药物/疾病名
_LEAD_PARTICLES: Tuple[str, ...] = (
    "加", "用", "服", "吃", "喝", "和", "与", "及", "等", "如", "若", "再", "又",
    "可", "应", "把", "被", "含", "给", "让", "或", "并", "同", "于", "在", "有",
)


def _trim_lead(term: str) -> str:
    """剥掉实体候选的前导虚词（最多 2 次），保留真正的医学名词"""
    for _ in range(2):
        if len(term) > 2 and term[0] in _LEAD_PARTICLES:
            term = term[1:]
        else:
            break
    return term


@dataclass
class GuardResult:
    """幻觉守卫校验结果"""

    passed: bool = True
    score: float = 1.0
    violations: List[Dict[str, str]] = field(default_factory=list)
    missing_citations: List[str] = field(default_factory=list)
    unknown_entities: List[str] = field(default_factory=list)
    dose_violations: List[str] = field(default_factory=list)
    absolute_claims: List[str] = field(default_factory=list)
    suggested_action: str = "accept"
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed, "score": self.score,
            "violations": self.violations,
            "missing_citations": self.missing_citations,
            "unknown_entities": self.unknown_entities,
            "dose_violations": self.dose_violations,
            "absolute_claims": self.absolute_claims,
            "suggested_action": self.suggested_action, "summary": self.summary,
        }


class HallucinationGuard:
    """确定性的事实校验守卫（7 项检查）"""

    # ==================================================================
    #  主入口
    # ==================================================================
    def check(
        self,
        answer: str,
        facts: Optional[Sequence[Any]] = None,
        entities: Optional[Dict[str, List[str]]] = None,
    ) -> GuardResult:
        answer = answer or ""
        facts = list(facts or [])
        violations: List[Dict[str, str]] = []
        missing_citations: List[str] = []
        unknown_entities: List[str] = []
        dose_violations: List[str] = []
        absolute_claims: List[str] = []

        kg_names: set = set()
        kg_text_parts: List[str] = []
        type_cn: set = set()
        for f in facts:
            head = str(getattr(f, "head", "") or "")
            tail = str(getattr(f, "tail", "") or "")
            for name in (head, tail):
                if name:
                    kg_names.add(name)
                    kg_text_parts.append(name)
            kg_text_parts.append(str(getattr(f, "relation_label", "") or ""))
            kg_text_parts.append(str(getattr(f, "head_type", "") or ""))
            kg_text_parts.append(str(getattr(f, "tail_type", "") or ""))
        if isinstance(entities, dict):
            for vals in entities.values():
                if isinstance(vals, str):
                    vals = [vals]
                for v in vals or []:
                    name = v if isinstance(v, str) else str(getattr(v, "kg_name", "") or "")
                    if name:
                        kg_names.add(name)
                        kg_text_parts.append(name)
        kg_text = "".join(kg_text_parts)
        type_cn = {t for t in ("Disease", "Symptom", "Department", "Drug", "Treatment", "Check") if t in kg_text}

        # ---- 1) 引用编号缺失 ----
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(answer) if s and s.strip()]
        for sent in sentences:
            if len(sent) <= 12:
                continue
            if _CITE_RE.search(sent):
                continue
            if _STRUCT_LINE_RE.match(sent):
                continue
            hit = next((n for n in kg_names if len(n) >= 2 and n in sent), None)
            if not hit:
                continue
            snippet = truncate(sent, 40)
            missing_citations.append(snippet)
            violations.append({
                "type": "missing_citation",
                "detail": f"句子提到了图谱实体「{hit}」但没有 [KG-n] 引用角标：{snippet}",
                "severity": "medium",
            })

        # ---- 2) 引用编号越界 ----
        for m in _CITE_RE.finditer(answer):
            num = int(m.group(1))
            if num < 1 or num > len(facts):
                violations.append({
                    "type": "invalid_citation",
                    "detail": f"引用了不存在的编号 [{m.group(0)}]，当前上下文仅有 {len(facts)} 条事实",
                    "severity": "high",
                })

        # ---- 3) 未在上下文中出现的实体 ----
        candidates: List[str] = []
        for regex in (_MED_TERM_RE, _DRUG_TERM_RE):
            candidates.extend(regex.findall(answer))
        seen_unknown: set = set()
        for term in candidates:
            term = _trim_lead((term or "").strip())
            if not term or term in seen_unknown:
                continue
            if term in kg_names or term in kg_text:
                continue
            if term in GENERIC_ALLOWLIST:
                continue
            # 后缀正则是贪婪匹配，会连带吞掉前面的修饰语（如"属于心血管内科疾病"），
            # 只要候选里含通用词或已收录实体，就认为它只是上下文组合，不判为幻觉
            if any(w in term for w in GENERIC_ALLOWLIST):
                continue
            if any(len(n) >= 2 and n in term for n in kg_names):
                continue
            if term in type_cn:
                continue
            seen_unknown.add(term)
            unknown_entities.append(term)
            if len(unknown_entities) >= 10:
                break
        for term in unknown_entities:
            violations.append({
                "type": "unknown_entity",
                "detail": f"答案中出现图谱上下文中不存在的医学实体「{term}」，存在幻觉风险",
                "severity": "medium",
            })

        # ---- 4) 剂量违规（最高危）----
        for m in _DOSE_RE.finditer(answer):
            token = m.group(0).strip()
            if token and token not in dose_violations:
                dose_violations.append(token)
        for m in _FREQ_RE.finditer(answer):
            token = m.group(0).strip()
            if token and token not in dose_violations:
                dose_violations.append(token)
        for token in dose_violations:
            violations.append({
                "type": "dose_violation",
                "detail": f"出现具体剂量/频次表述「{token}」，医疗场景禁止给出处方剂量",
                "severity": "critical",
            })

        # ---- 5) 绝对化结论 ----
        for m in _ABSOLUTE_RE.finditer(answer):
            token = m.group(0)
            if token not in absolute_claims:
                absolute_claims.append(token)
                violations.append({
                    "type": "absolute_claim",
                    "detail": f"出现绝对化结论「{token}」，医疗回答必须使用不确定性措辞",
                    "severity": "high",
                })

        # ---- 6) 诊断性措辞 ----
        for m in _DIAG_RE.finditer(answer):
            start = m.start()
            window = answer[max(0, start - 12):start]
            if any(p in window for p in _DIAG_SAFE_PREFIX):
                continue
            detail = f"出现越权诊断措辞「{truncate(answer[max(0, start - 10):m.end() + 12], 40)}」"
            violations.append({"type": "diagnostic_claim", "detail": detail, "severity": "high"})

        # ---- 7) 免责声明缺失 ----
        if "不能替代执业医师" not in answer:
            violations.append({
                "type": "missing_disclaimer",
                "detail": "回答末尾缺少强制免责声明（不能替代执业医师诊断）",
                "severity": "low",
            })

        # ---- 评分 / 结论 ----
        violations = self._dedupe(violations)
        score = 1.0
        for v in violations:
            score -= SEVERITY_DEDUCT.get(str(v.get("severity", "low")), 0.02)
        score = round(max(0.0, min(1.0, score)), 3)

        has_critical = any(v.get("severity") == "critical" for v in violations)
        has_high = any(v.get("severity") == "high" for v in violations)
        if has_critical or score < 0.45:
            action = "fallback_template"
        elif has_high or score < 0.75:
            action = "rewrite"
        else:
            action = "accept"

        stats = self._stats(violations)
        if not violations:
            summary = "事实校验通过：引用编号完整、无越界实体、无剂量与诊断越权表述"
        else:
            detail_txt = "、".join(f"{k}×{v}" for k, v in stats.items())
            summary = f"事实校验发现 {len(violations)} 处问题（{detail_txt}），置信度 {score:.3f} → 处置：{action}"

        result = GuardResult(
            passed=(action == "accept"),
            score=score,
            violations=violations,
            missing_citations=missing_citations,
            unknown_entities=unknown_entities,
            dose_violations=dose_violations,
            absolute_claims=absolute_claims,
            suggested_action=action,
            summary=summary,
        )
        logger.info("幻觉守卫：%s", summary)
        return result

    # ==================================================================
    #  工具
    # ==================================================================
    @staticmethod
    def _dedupe(violations: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen: set = set()
        for v in violations:
            key = (str(v.get("type", "")), str(v.get("detail", "")))
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
        return out

    @staticmethod
    def _stats(violations: Sequence[Dict[str, str]]) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for v in violations:
            t = str(v.get("type", "unknown"))
            stats[t] = stats.get(t, 0) + 1
        return stats

    # ==================================================================
    #  兜底：纯 KG 拼装的安全回答
    # ==================================================================
    def strip_and_fallback(self, answer: str, facts: Optional[Sequence[Any]] = None) -> str:
        """丢弃模型答案，改用图谱事实直接拼装（零幻觉，可溯源，强制免责声明）"""
        facts = list(facts or [])
        lines: List[str] = ["## 基于知识图谱的安全回答（已启用降级模式）"]
        if not facts:
            lines.append("")
            lines.append("当前知识库暂未检索到与该问题直接相关的图谱事实，为避免误导，本系统不做推测性回答。")
            lines.append("建议：① 换用更常见的疾病 / 症状名称重新提问；② 前往正规医疗机构就诊咨询执业医师。")
        else:
            groups: Dict[str, List[Any]] = {}
            for f in facts:
                label = str(getattr(f, "relation_label", "") or getattr(f, "relation", "") or "其他")
                groups.setdefault(label, []).append(f)
            order = ["症状", "治疗", "用药", "检查", "科室", "并发症", "鉴别诊断", "易感人群",
                     "相关（多跳推理）", "相关"]
            ordered = sorted(groups.items(), key=lambda kv: (order.index(kv[0]) if kv[0] in order else 50, kv[0]))
            for label, items in ordered:
                lines.append("")
                lines.append(f"### {label}")
                for f in items:
                    cid = str(getattr(f, "triple_id", "") or "")
                    cite = f" [{cid}]" if cid else ""
                    lines.append(
                        f"- {getattr(f, 'head', '')} —{label}→ {getattr(f, 'tail', '')}{cite}"
                    )
            lines.append("")
            lines.append("> 以上内容由知识图谱三元组直接生成，未经过大模型改写，"
                         "编号与知识图谱事实一一对应，可在溯源面板查看原始文献。")
        lines.append("")
        lines.append("本回答由 AI 生成，仅供参考，不能替代执业医师诊断。")
        text = "\n".join(lines).strip()
        if "不能替代执业医师" not in text:  # 双保险
            text += f"\n{getattr(settings, 'disclaimer', DISCLAIMER)}"
        logger.warning("幻觉守卫触发兜底模板回答（丢弃模型输出 %d 字）", len(answer or ""))
        return text

    # ==================================================================
    #  可解释输出
    # ==================================================================
    def explain(self, result: GuardResult) -> Dict[str, Any]:
        """转成 JSON 可序列化结构，直接写入 QAResponse.guard_result"""
        if result is None:
            return {"passed": True, "score": 1.0, "summary": "未执行幻觉守卫",
                    "suggested_action": "accept", "violations": [], "stats": {}}
        return {
            "passed": bool(result.passed),
            "score": float(result.score),
            "summary": str(result.summary),
            "suggested_action": str(result.suggested_action),
            "violations": [dict(v) for v in (result.violations or [])],
            "stats": self._stats(result.violations or []),
        }


__all__ = ["HallucinationGuard", "GuardResult", "DISCLAIMER", "GENERIC_ALLOWLIST", "_trim_lead"]
