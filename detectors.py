"""Detectors for life followups, cross-month deja vu, and long-absent friend reunions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any, Sequence
from zoneinfo import ZoneInfo

try:
    from .store import FollowupRecord, PresenceRecord, UtteranceRecord, cosine_similarity
except ImportError:
    from store import FollowupRecord, PresenceRecord, UtteranceRecord, cosine_similarity  # type: ignore[no-redef]

_MEDIA_TAIL_RE = re.compile(r"\[动图\][\s\S]*")
_SYSTEM_BRACKET_RE = re.compile(
    r"\[(?:表情包|图片|语音|视频|小程序|文件|音乐|分享|转发|合并转发|QQ红包|红包|事件-[^\]\n]*|网页链接摘要|B站视频信息)[^\]]*\]"
)
_AT_TAG_RE = re.compile(r"<at\b[^>]*>|@\S+")
_URL_RE = re.compile(r"https?://\S+", re.I)
_REPLY_PREFIX_RE = re.compile(r"^(?:回复了[^\n:：]{1,32}?的消息[:：]?\s*|\[引用[^\]]*\]\s*)+")
_SYSTEM_NOTICE_RE = re.compile(
    r"^(?:【[^】]{1,20}(?:提醒|通知|播报|报告|告警|警告|状态|结果|防撤回)】|\[系统通知\])"
)
_PURE_QUESTION_RE = re.compile(r"(?:[?？]|吗|呢)\s*$")
_THIRD_PERSON_RE = re.compile(r"(?:我朋友|我室友|我同学|我哥|我姐|我弟|别人|群友|大佬|他|她)")

# Avoid overlapping with flag-boomerang vows
_FLAG_EXCLUSION_RE = re.compile(
    r"(?:倒立|吃键盘|吃显示器|我是狗|立[帖贴]为证|话放这|毒奶一口|绝不抽|再氪|再抽|再熬夜|剁手)"
)

_FOLLOWUP_EVENT_RE = re.compile(
    r"(?:面试|笔试|答辩|路考|科[一二三四]|科目[一二三四]|考驾照|驾考|四六级|期末考|体测|"
    r"看牙|拔牙|洗牙|体检|复查|去医院|打疫苗|带猫去|带狗去|绝育|"
    r"寄去?售后|返修|寄修了|修电脑|修手机|抢票|开票|等快递|"
    r"去旅游|去漫展|去相亲|去面基|约了饭|去聚餐|去体检|"
    r"搬家|去理发|去烫头|剪头发|投了?简历)"
)
_TIME_CONTEXT_RE = re.compile(
    r"(?:今天下午|今下午|今晚|今天|明天|明早|明晚|后天|大后天|周[一二三四五六日天]|周末|"
    r"这几天|刚|刚才|刚刚|准备去|马上要去|要去|待会儿去)"
)

_SMALLTALK_RE = re.compile(
    r"^(?:早安|早|午安|晚安|睡了|醒了|哈哈+|确实|好耶|牛逼|nb|6+|谢谢|感谢|收到|好的|嗯嗯|草|笑死|"
    r"吃饭了吗|吃了吗|在干嘛|在吗|晚饭吃啥|午饭吃啥|早饭吃啥)+[。.!！~～]*$",
    re.I,
)
_OPINION_HINT_RE = re.compile(
    r"(?:再也不|这游戏|坐牢|好难|好累|好烦|绝了|想换|真难吃|再买我是|受不了|崩溃|加班|太肝|好贵|"
    r"卡死了|又崩了|又卡|真离谱|每次都|怎么老是|永远都|实在太|打算换)"
)


@dataclass(frozen=True)
class FollowupCandidate:
    event_summary: str
    ask_after_ts: float
    expire_ts: float
    confidence: float


def resolve_tz(tz_name: str) -> Any:
    name = str(tz_name or "Asia/Shanghai").strip() or "Asia/Shanghai"
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone(timedelta(hours=8))


def parse_hhmm(value: str, fallback: tuple[int, int]) -> int:
    text = str(value or "").strip()
    try:
        parts = text.split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60 + m
    except Exception:
        pass
    return fallback[0] * 60 + fallback[1]


def in_quiet_hours(now_dt: datetime, quiet_start: str = "01:00", quiet_end: str = "08:00") -> bool:
    cur_min = now_dt.hour * 60 + now_dt.minute
    s_min = parse_hhmm(quiet_start, (1, 0))
    e_min = parse_hhmm(quiet_end, (8, 0))
    if s_min == e_min:
        return False
    if s_min < e_min:
        return s_min <= cur_min < e_min
    return cur_min >= s_min or cur_min < e_min


def sanitize_text(text: str) -> str:
    cleaned = _MEDIA_TAIL_RE.sub(" ", str(text or ""))
    cleaned = _SYSTEM_BRACKET_RE.sub(" ", cleaned)
    cleaned = _AT_TAG_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _REPLY_PREFIX_RE.sub("", cleaned.strip())
    return " ".join(cleaned.split()).strip()


def char_bigrams(text: str) -> set[str]:
    clean = "".join(ch for ch in str(text or "").lower() if "\u4e00" <= ch <= "\u9fff" or ch.isalnum())
    if len(clean) < 2:
        return {clean} if clean else set()
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def bigram_jaccard(a: str, b: str) -> float:
    sa, sb = char_bigrams(a), char_bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --- 1. Followup detectors ---

def estimate_ask_after_ts(text: str, now_dt: datetime) -> float:
    clean = str(text or "")
    if any(k in clean for k in ("刚寄", "寄去售后", "寄修", "投了简历", "投简历", "返修")):
        return (now_dt + timedelta(hours=48)).timestamp()
    if "大后天" in clean:
        dt = now_dt + timedelta(days=3)
        return datetime(dt.year, dt.month, dt.day, 19, 0, tzinfo=now_dt.tzinfo).timestamp()
    if "后天" in clean:
        dt = now_dt + timedelta(days=2)
        return datetime(dt.year, dt.month, dt.day, 19, 0, tzinfo=now_dt.tzinfo).timestamp()
    if "明天" in clean or "明早" in clean or "明晚" in clean:
        dt = now_dt + timedelta(days=1)
        h = 21 if "晚" in clean else (12 if "早" in clean else 18)
        return datetime(dt.year, dt.month, dt.day, h, 0, tzinfo=now_dt.tzinfo).timestamp()
    if "周末" in clean:
        days_to_mon = (7 - now_dt.weekday()) % 7 or 7
        dt = now_dt + timedelta(days=days_to_mon)
        return datetime(dt.year, dt.month, dt.day, 10, 30, tzinfo=now_dt.tzinfo).timestamp()
    m_wk = re.search(r"(?:这|本)?周([一二三四五六日天])", clean)
    if m_wk:
        wk_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        tw = wk_map[m_wk.group(1)]
        delta = (tw - now_dt.weekday()) % 7
        dt = now_dt + timedelta(days=delta)
        if delta == 0 and now_dt.hour >= 18:
            dt = dt + timedelta(days=1)
            h = 10
        else:
            h = 19
        return datetime(dt.year, dt.month, dt.day, h, 0, tzinfo=now_dt.tzinfo).timestamp()
    if "今晚" in clean:
        dt = now_dt + timedelta(days=1)
        return datetime(dt.year, dt.month, dt.day, 10, 0, tzinfo=now_dt.tzinfo).timestamp()
    if "下午" in clean or "今天" in clean or "准备去" in clean or "要去" in clean:
        if now_dt.hour < 19:
            return datetime(now_dt.year, now_dt.month, now_dt.day, 19, 30, tzinfo=now_dt.tzinfo).timestamp()
        dt = now_dt + timedelta(days=1)
        return datetime(dt.year, dt.month, dt.day, 10, 0, tzinfo=now_dt.tzinfo).timestamp()
    return (now_dt + timedelta(hours=24)).timestamp()


def prefilter_followup_candidate(
    text: str,
    now_dt: datetime,
    *,
    expire_days: int = 5,
) -> FollowupCandidate | None:
    clean = sanitize_text(text)
    if len(clean) < 5 or len(clean) > 160:
        return None
    if _SYSTEM_NOTICE_RE.match(clean):
        return None
    if _PURE_QUESTION_RE.search(clean):
        return None
    if _FLAG_EXCLUSION_RE.search(clean):
        return None
    if _THIRD_PERSON_RE.search(clean) and "我" not in _THIRD_PERSON_RE.sub("", clean):
        return None

    m_ev = _FOLLOWUP_EVENT_RE.search(clean)
    if not m_ev:
        return None
    if not _TIME_CONTEXT_RE.search(clean):
        return None

    ask_ts = estimate_ask_after_ts(clean, now_dt)
    exp_ts = ask_ts + max(1, int(expire_days)) * 86400.0
    return FollowupCandidate(
        event_summary=clean[:50],
        ask_after_ts=ask_ts,
        expire_ts=exp_ts,
        confidence=0.82,
    )


def _extract_json_block(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fenced:
        text = fenced.group(1).strip()
    else:
        brace = re.search(r"\{[\s\S]*\}", text)
        if brace:
            text = brace.group(0)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def build_followup_extract_prompt(user_name: str, text: str, now_dt: datetime) -> str:
    now_str = now_dt.strftime("%Y-%m-%d %H:%M (%A)")
    return (
        "你是群友日常关心助手。判断下面这句话是不是群友提到了一件「自己即将经历、且事后让人关心结果的真实生活事件」"
        "（例如：面试、考试、看牙体检、寄修售后、重要相亲/面基、旅游、搬家等）。\n"
        "单纯口嗨玩梗、转述别人、或纯粹提问不算。\n"
        f"当前时间：{now_str}\n"
        f"说话人：{user_name}\n"
        f"原话：{text}\n\n"
        "仅输出一行合法 JSON：\n"
        '{"is_followup": true/false, "event_summary": "不超过20字的简短事件名", '
        '"ask_after_iso": "YYYY-MM-DD HH:MM（事件大概结束之后再问的合理时间）", "confidence": 0.0}'
    )


def parse_followup_llm_response(
    raw_text: str,
    candidate: FollowupCandidate,
    now_dt: datetime,
    *,
    expire_days: int = 5,
) -> FollowupCandidate | None:
    data = _extract_json_block(raw_text)
    if data is None:
        return None
    if not bool(data.get("is_followup", False)):
        return FollowupCandidate(
            event_summary="",
            ask_after_ts=0.0,
            expire_ts=0.0,
            confidence=0.0,
        )
    summary = str(data.get("event_summary") or candidate.event_summary).strip()[:60]
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", candidate.confidence))))
    except (TypeError, ValueError):
        conf = candidate.confidence

    ask_ts = candidate.ask_after_ts
    iso_s = str(data.get("ask_after_iso") or "").strip()
    if iso_s:
        try:
            dt = datetime.fromisoformat(iso_s.replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=now_dt.tzinfo)
            if dt.timestamp() >= now_dt.timestamp():
                ask_ts = dt.timestamp()
        except Exception:
            pass
    exp_ts = ask_ts + max(1, int(expire_days)) * 86400.0
    return FollowupCandidate(
        event_summary=summary,
        ask_after_ts=ask_ts,
        expire_ts=exp_ts,
        confidence=conf,
    )


# --- 2. DejaVu detectors ---

def is_dejavu_candidate(text: str) -> bool:
    clean = sanitize_text(text)
    if len(clean) < 10 or len(clean) > 120:
        return False
    if _SYSTEM_NOTICE_RE.match(clean):
        return False
    if _PURE_QUESTION_RE.search(clean):
        return False
    if _SMALLTALK_RE.match(clean):
        return False
    cjk_count = sum(1 for ch in clean if "\u4e00" <= ch <= "\u9fff")
    if _OPINION_HINT_RE.search(clean) and cjk_count >= 6:
        return True
    return cjk_count >= 12


def find_best_dejavu_match(
    new_text: str,
    new_emb: Sequence[float] | None,
    old_records: Sequence[UtteranceRecord],
    *,
    lexical_threshold: float = 0.42,
    embedding_threshold: float = 0.86,
) -> tuple[UtteranceRecord, float, str] | None:
    best_rec: UtteranceRecord | None = None
    best_score = 0.0
    best_mode = ""

    for rec in old_records:
        if rec.text == new_text:
            return rec, 1.0, "exact"
        if new_emb and rec.embedding:
            sim = cosine_similarity(new_emb, rec.embedding)
            if sim >= embedding_threshold and sim > best_score:
                best_rec = rec
                best_score = sim
                best_mode = "embedding"
                continue
        jac = bigram_jaccard(new_text, rec.text)
        if jac >= lexical_threshold and jac > best_score:
            best_rec = rec
            best_score = jac
            best_mode = "lexical"

    if best_rec is None:
        return None
    return best_rec, best_score, best_mode


def build_dejavu_verify_prompt(user_name: str, old_text: str, old_date_str: str, new_text: str) -> str:
    return (
        "判断同一位群友隔了一段时间后发的这两句话，是否属于“令人会心一笑的既视感 / 梅开二度”"
        "（例如又发了几乎相同的吐槽、又说要退游/换手机/戒东西、或隔了很久重复同样的感叹）。\n"
        "如果只是正常高频用语（比如都在说“好的”“哈哈”）或完全不相关，判 false。\n"
        f"群友：{user_name}\n"
        f"{old_date_str} 的历史发言：{old_text}\n"
        f"今天的最新发言：{new_text}\n\n"
        '仅输出一行 JSON：{"is_dejavu": true/false, "angle": "简短吐槽点"}'
    )


def parse_dejavu_llm_response(raw_text: str) -> tuple[bool, str]:
    data = _extract_json_block(raw_text)
    if data is None:
        return False, ""
    return bool(data.get("is_dejavu", False)), str(data.get("angle") or "").strip()[:60]


# --- 3. Hint formatters (zero extra injection unless triggered) ---

def format_followup_hint(rec: FollowupRecord, now_dt: datetime) -> str:
    when = datetime.fromtimestamp(rec.created_at, tz=now_dt.tzinfo).strftime("%m月%d日")
    return (
        f"[老友近况] {rec.user_name} 在 {when} 曾顺嘴提过「{rec.event_summary}」（原话：{rec.original_text[:50]}）。"
        "趁他现在刚好又在群里说话，像真把人家的事放心里一样，自然随口问一句后来怎么样了，一句话带过即可，不要像发问卷。"
    )


def format_reunion_hint(rec: PresenceRecord, absent_days: int) -> str:
    return (
        f"[久别重逢] 平常聊过不少的老群友 {rec.user_name} 已经大概 {absent_days} 天没露面了，刚刚突然冒泡。"
        "顺口像熟人一样自然打个轻招呼、感叹一句稀客即可，不要过度煽情或盘问隐私。"
    )


def format_dejavu_hint(old_rec: UtteranceRecord, new_text: str, days_apart: int, angle: str, now_dt: datetime) -> str:
    when = datetime.fromtimestamp(old_rec.created_at, tz=now_dt.tzinfo).strftime("%m月%d日")
    note = f"（点：{angle}）" if angle else ""
    return (
        f"[既视感] {old_rec.user_name} 早在 {days_apart} 天前（{when}）就说过类似的话：「{old_rec.text[:55]}」，"
        f"今天又来了句「{new_text[:50]}」{note}。用损友熟人口吻顺嘴调侃一句梅开二度即可。"
    )
