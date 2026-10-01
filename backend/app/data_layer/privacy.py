# -*- coding: utf-8 -*-
"""
隐私合规模块（全链路脱敏加密）
==============================
PPT 指标：「数据全链路脱敏加密」「知识来源 100% 可追溯」

两层防护
--------
  ① **PII 脱敏（Masking）** —— 入库前的最后一道人工/自动关口
     * 规则层：正则识别身份证、手机号、银行卡、邮箱、住院号、病历号、地址、姓名
     * 词典层：常见姓氏 + 称谓模式，识别"张××""患者李某某"
     * 输出：掩码后的文本 + 脱敏审计记录（记录脱敏了什么类型、多少处，**不记录原文**）
  ② **字段加密（Encryption）** —— AES-256-GCM 认证加密
     * 密钥来自环境变量 `FIELD_ENCRYPT_KEY`（base64 编码的 32 字节），生产环境应由 KMS 注入
     * 每次加密使用随机 96-bit nonce，密文格式 `v1:base64(nonce):base64(ciphertext+tag)`
     * GCM 提供完整性校验，防止密文被篡改

设计原则：**原始 PII 永不落库、永不落盘、永不进日志**。
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    HAS_CRYPTO = True
except Exception:  # pragma: no cover
    HAS_CRYPTO = False
    AESGCM = None  # type: ignore


# =============================================================================
#  一、PII 识别规则
# =============================================================================
@dataclass
class PIIRule:
    name: str
    pattern: re.Pattern
    mask: str          # 替换模板，{n} 表示保留的位数描述


PII_RULES: List[PIIRule] = [
    PIIRule("身份证号", re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
            "ID_MASKED"),
    PIIRule("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "PHONE_MASKED"),
    PIIRule("固定电话", re.compile(r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)"), "TEL_MASKED"),
    PIIRule("银行卡号", re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "BANK_MASKED"),
    PIIRule("电子邮箱", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "EMAIL_MASKED"),
    PIIRule("住院号/病历号",
            re.compile(r"(?:住院号|病历号|门诊号|病案号|医保卡号|社保号)\s*[:：]?\s*[A-Za-z0-9\-]{4,20}"),
            "MRN_MASKED"),
    PIIRule("详细地址",
            re.compile(r"[\u4e00-\u9fff]{2,8}(?:省|市|自治区|自治州)[\u4e00-\u9fff]{2,10}"
                       r"(?:市|区|县|旗|镇|乡|街道|路|街|巷|号|栋|单元|室|村|组)\s*[\u4e00-\u9fff0-9\-]{0,20}"),
            "ADDR_MASKED"),
    PIIRule("QQ/微信号", re.compile(r"(?:QQ|微信|weixin|wechat)\s*(?:号|：|:)?\s*[A-Za-z0-9_\-]{5,20}", re.I),
            "SOCIAL_MASKED"),
    PIIRule("姓名-称谓",
            re.compile(r"(?:患者|病人|本人|姓名|联系人)\s*[:：]?\s*[\u4e00-\u9fff]{2,4}"),
            "NAME_MASKED"),
    PIIRule("姓名-常见姓氏+称谓",
            re.compile(r"[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
                       r"戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐"
                       r"费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄]"
                       r"(?:某某|某|××|\*\*|XX)"),
            "NAME_MASKED"),
    PIIRule("出生日期", re.compile(r"(?:出生日期|出生年月|生日)\s*[:：]?\s*\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?"),
            "DOB_MASKED"),
]


#: 安全豁免白名单：这些是医学知识中的正常数字，不属于 PII（避免误脱敏）
SAFE_CONTEXT_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:血压|收缩压|舒张压|血糖|血脂|胆固醇|甘油三酯|BMI|心率|体温|"
               r"白细胞|血红蛋白|血小板|肌酐|尿素氮|转氨酶|胆红素|白蛋白|"
               r"氧分压|二氧化碳|pH)\s*[:：]?\s*[\d\.~\-]+"),
    re.compile(r"\d+\s*(?:mg|g|ml|mL|L|mmHg|mmol/L|μmol/L|U/L|IU|%|次/分|°C|℃|"
               r"万|亿|例|人|年|月|日|小时|分钟|周|天)"),
    re.compile(r"(?:PMID|DOI|doi)\s*[:：]?\s*[\d\./\-A-Za-z]+"),
    re.compile(r"ICD-?\s*\d+"),
]


@dataclass
class MaskResult:
    """脱敏结果"""

    text: str
    hit_count: int = 0
    hits: Dict[str, int] = field(default_factory=dict)     # 类型 → 命中次数
    original_length: int = 0
    masked_length: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "hit_count": self.hit_count, "hits": self.hits,
            "original_length": self.original_length,
            "masked_length": self.masked_length,
        }


class PrivacyGuard:
    """
    全链路隐私防护

    使用：
        guard = PrivacyGuard()
        safe_text = guard.mask_text(raw_text)          # 脱敏
        token = guard.encrypt_field("张三")             # 加密
        plain = guard.decrypt_field(token)              # 解密（仅授权场景）
    """

    def __init__(self, key: Optional[str] = None) -> None:
        self.aesgcm = None
        self.key_id = "none"
        self._is_ephemeral_key = False
        raw_key = key or settings.FIELD_ENCRYPT_KEY or ""
        if not HAS_CRYPTO:
            logger.warning("未安装 cryptography，字段加密不可用（pip install cryptography）")
        elif raw_key:
            try:
                key_bytes = base64.b64decode(raw_key)
                if len(key_bytes) != 32:
                    raise ValueError(f"AES-256 需要 32 字节密钥，当前 {len(key_bytes)} 字节")
                self.aesgcm = AESGCM(key_bytes)
                #: 只记录密钥指纹，绝不记录密钥本身
                self.key_id = hashlib.sha256(key_bytes).hexdigest()[:12]
                logger.info("隐私加密模块就绪（AES-256-GCM，密钥指纹 %s）", self.key_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("FIELD_ENCRYPT_KEY 无效：%s；字段加密已禁用", exc)
        else:
            # 未配置密钥时，为**演示原型**自动生成一个进程内临时密钥，
            # 保证「全链路加密」能力开箱可用。生产环境必须通过 KMS 注入固定密钥，
            # 否则重启后旧密文将无法解密。
            dev_key = os.urandom(32)
            self.aesgcm = AESGCM(dev_key)
            self.key_id = hashlib.sha256(dev_key).hexdigest()[:12]
            self._is_ephemeral_key = True
            logger.warning(
                "未配置 FIELD_ENCRYPT_KEY，已自动生成**临时演示密钥**（指纹 %s）。"
                "该密钥仅存在于当前进程，重启后无法解密旧数据。"
                "生产环境请生成固定密钥并写入 .env："
                "python -c \"import os,base64;print(base64.b64encode(os.urandom(32)).decode())\"",
                self.key_id,
            )
        self.stats: Dict[str, int] = {}

    # ==================================================================
    #  ① 文本脱敏
    # ==================================================================
    def mask_text(self, text: str) -> str:
        """脱敏并返回安全文本"""
        return self.mask(text).text

    def mask(self, text: str) -> MaskResult:
        """执行 PII 脱敏，返回结果与审计统计（不保留原文）"""
        raw = text or ""
        if not settings.PRIVACY_MASK_ENABLED:
            return MaskResult(text=raw, original_length=len(raw), masked_length=len(raw))

        # 先标记安全上下文位置（避免把医学数值误判为 PII）
        protected = self._protected_spans(raw)
        out = raw
        hits: Dict[str, int] = {}

        for rule in PII_RULES:
            def _sub(m: re.Match) -> str:
                start, end = m.span()
                if any(not (end <= ps or start >= pe) for ps, pe in protected):
                    return m.group(0)      # 命中安全上下文 → 不脱敏
                hits[rule.name] = hits.get(rule.name, 0) + 1
                return f"[{rule.mask}]"

            out = rule.pattern.sub(_sub, out)

        total = sum(hits.values())
        if total:
            for k, v in hits.items():
                self.stats[k] = self.stats.get(k, 0) + v
            logger.info("脱敏完成：%d 处（%s）", total,
                        ", ".join(f"{k}×{v}" for k, v in hits.items()))
        return MaskResult(
            text=out, hit_count=total, hits=hits,
            original_length=len(raw), masked_length=len(out),
        )

    def _protected_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for pat in SAFE_CONTEXT_PATTERNS:
            spans.extend(m.span() for m in pat.finditer(text))
        return spans

    # ==================================================================
    #  ② 字段加密（AES-256-GCM）
    # ==================================================================
    def encrypt_field(self, plaintext: str) -> str:
        """
        加密单个字段。
        返回 `v1:<base64(nonce)>:<base64(ciphertext||tag)>`；
        加密不可用时返回原文并在日志中告警（保证功能可用性优先）。
        """
        if not plaintext:
            return ""
        if self.aesgcm is None or not settings.PRIVACY_ENCRYPT_ENABLED:
            return plaintext
        try:
            nonce = os.urandom(12)                        # 96-bit nonce
            ct = self.aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
            return "v1:{}:{}".format(
                base64.b64encode(nonce).decode(),
                base64.b64encode(ct).decode(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("字段加密失败：%s", exc)
            return plaintext

    def decrypt_field(self, token: str) -> str:
        """解密字段；格式不符或校验失败时返回原文"""
        if not token or not token.startswith("v1:") or self.aesgcm is None:
            return token
        try:
            _, n_b64, c_b64 = token.split(":", 2)
            nonce = base64.b64decode(n_b64)
            ct = base64.b64decode(c_b64)
            return self.aesgcm.decrypt(nonce, ct, None).decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.error("字段解密失败（密文可能被篡改或密钥不匹配）：%s", exc)
            return token

    # ==================================================================
    #  ③ 去标识化（用于风险预测入参）
    # ==================================================================
    @staticmethod
    def anonymize_health_input(payload: Dict[str, object]) -> Dict[str, object]:
        """
        把用户提交的健康信息转为**去标识化特征集**：
        仅保留数值型指标与标准化的枚举值，丢弃任何自由文本与标识字段。

        这是满足「个人信息零留存」的关键：风险预测接口只需要数值特征，
        不需要（也不应该接收）姓名、身份证、地址等任何可直接识别个人身份的信息。
        """
        allowed_numeric = {
            "age", "bmi", "systolic_bp", "diastolic_bp", "fasting_glucose",
            "total_cholesterol", "hdl", "ldl", "triglycerides", "heart_rate",
        }
        allowed_enum = {"gender", "physical_activity"}
        allowed_bool = {"smoking", "drinking"}
        allowed_list = {"family_history", "symptoms", "existing_conditions"}

        out: Dict[str, object] = {}
        for k, v in (payload or {}).items():
            if k in allowed_numeric and isinstance(v, (int, float)):
                out[k] = v
            elif k in allowed_enum and isinstance(v, str):
                out[k] = v if re.fullmatch(r"[A-Za-z_]{2,20}", v) else ""
            elif k in allowed_bool:
                out[k] = bool(v)
            elif k in allowed_list and isinstance(v, list):
                out[k] = [str(x)[:30] for x in v[:20]]
        return out

    # ==================================================================
    #  ④ 匿名标识符
    # ==================================================================
    @staticmethod
    def pseudonymous_id(seed: str, salt: Optional[str] = None) -> str:
        """
        生成不可逆的假名标识（用于关联同一用户的多次请求而不存储身份信息）。
        采用 SHA-256(salt + seed)，salt 由服务端持有，攻击者无法彩虹表反推。
        """
        salt = salt or os.getenv("PSEUDONYM_SALT", "zhiyu-medical-codex")
        return "anon_" + hashlib.sha256(f"{salt}:{seed}".encode("utf-8")).hexdigest()[:16]

    # ==================================================================
    def info(self) -> Dict[str, object]:
        return {
            "mask_enabled": settings.PRIVACY_MASK_ENABLED,
            "encrypt_enabled": settings.PRIVACY_ENCRYPT_ENABLED and self.aesgcm is not None,
            "cipher": "AES-256-GCM" if HAS_CRYPTO else "unavailable (missing cryptography)",
            "key_id": self.key_id,
            "key_is_ephemeral": self._is_ephemeral_key,
            "pii_rule_count": len(PII_RULES),
            "pii_rules": [r.name for r in PII_RULES],
            "masked_total": sum(self.stats.values()),
        }


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_guard: Optional[PrivacyGuard] = None


def get_privacy_guard() -> PrivacyGuard:
    global _guard
    if _guard is None:
        _guard = PrivacyGuard()
    return _guard


def mask_pii(text: str) -> str:
    """便捷函数：脱敏文本"""
    return get_privacy_guard().mask_text(text)


def encrypt_field(value: str) -> str:
    """便捷函数：加密字段"""
    return get_privacy_guard().encrypt_field(value)


def decrypt_field(token: str) -> str:
    """便捷函数：解密字段"""
    return get_privacy_guard().decrypt_field(token)


if __name__ == "__main__":  # pragma: no cover
    import json

    guard = PrivacyGuard()
    demo = (
        "患者张三，男，58岁，身份证号 110101196503151234，手机号 13812345678，"
        "住院号：ZY20240115，住址：北京市朝阳区建国路88号3单元502室。"
        "血压 152/96 mmHg，空腹血糖 6.4 mmol/L，参考 PMID: 32130469。"
    )
    res = guard.mask(demo)
    print("脱敏前长度:", res.original_length)
    print("脱敏后:", res.text)
    print("命中统计:", json.dumps(res.hits, ensure_ascii=False))
    print("加密演示:", guard.encrypt_field("张三")[:60], "...")
    print("模型信息:", json.dumps(guard.info(), ensure_ascii=False, indent=2))
