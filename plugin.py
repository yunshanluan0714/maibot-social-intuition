"""Main plugin wiring for github.yunshanluan0714.social-intuition v1.0.1."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
import time
from typing import Any
from uuid import uuid4

from maibot_sdk import HookHandler, MaiBotPlugin, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType

from .config import SocialIntuitionConfig
from .detectors import (
    build_dejavu_verify_prompt,
    build_followup_extract_prompt,
    find_best_dejavu_match,
    format_dejavu_hint,
    format_followup_hint,
    format_reunion_hint,
    in_quiet_hours,
    is_dejavu_candidate,
    parse_dejavu_llm_response,
    parse_followup_llm_response,
    prefilter_followup_candidate,
    resolve_tz,
    sanitize_text,
)
from .store import SocialIntuitionStore


def _extract_llm_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for key in ("response", "text", "content"):
            val = resp.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return ""


def _build_reminder_item(text: str) -> dict[str, Any]:
    return {
        "item_type": "UserMessageItem",
        "meta": {"item_id": uuid4().hex, "logical_turn_id": None, "timestamp": datetime.now().isoformat()},
        "parts": [{"type": "text", "text": text}],
    }


def _inject_reminder(items: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    updated = list(items)
    pos = len(updated)
    if updated and updated[-1].get("item_type") == "UserMessageItem":
        parts = [str(p.get("text") or "") for p in updated[-1].get("parts") or [] if isinstance(p, dict)]
        if parts and not parts[0].lstrip().startswith("<message"):
            pos -= 1
    updated.insert(pos, _build_reminder_item(text))
    return updated


class SocialIntuitionPlugin(MaiBotPlugin):
    """熟人社交直觉套件：近况追问、跨月既视感与失踪老友回归。"""

    config_model = SocialIntuitionConfig

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._store: SocialIntuitionStore | None = None
        self._bot_qq_account = ""
        self._session_to_group: dict[str, str] = {}
        self._session_to_private: dict[str, str] = {}
        self._member_api_targets: list[tuple[str, str]] | None = None
        self._pending_hints: dict[str, tuple[int, str]] = {}
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._backfill_task: asyncio.Task[None] | None = None

    async def on_load(self) -> None:
        self._store = SocialIntuitionStore(self.ctx.paths.data_dir)
        await self._refresh_bot_identity()
        await self._warm_stream_mappings()
        self._running = True
        pending = self._store.pending_manual_names()
        if pending:
            task = asyncio.create_task(self._bind_legacy_participants(pending))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        if self.config.guard.backfill_on_startup:
            self._backfill_task = asyncio.create_task(self._run_startup_backfill(), name="social-intuition-backfill")
        self.ctx.logger.info("熟人社交直觉套件 v1.0.1 已加载")

    async def on_unload(self) -> None:
        self._running = False
        if self._backfill_task is not None:
            self._backfill_task.cancel()
            try:
                await self._backfill_task
            except asyncio.CancelledError:
                pass
            self._backfill_task = None
        for task in list(self._bg_tasks):
            task.cancel()
        self._bg_tasks.clear()

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        del config_data, version
        for session_id in list(self._pending_hints):
            if not self._scope_allowed(self._resolve_scope_group_id(session_id)):
                self._pending_hints.pop(session_id, None)
        self.ctx.logger.info("熟人社交直觉套件配置已热更新: scope=%s", scope)

    async def _refresh_bot_identity(self) -> None:
        try:
            qq = await self.ctx.config.get("bot.qq_account", "")
            if isinstance(qq, str) and qq.strip():
                self._bot_qq_account = qq.strip()
        except Exception:
            pass

    async def _warm_stream_mappings(self) -> None:
        try:
            getter = getattr(self.ctx.chat, "get_all_streams", None) or getattr(self.ctx.chat, "get_group_streams", None)
            if getter is None:
                return
            raw = await getter(platform="qq")
            streams = raw.get("streams", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
            if not isinstance(streams, list):
                return
            for st in streams:
                if not isinstance(st, dict):
                    continue
                sid = str(st.get("session_id") or st.get("stream_id") or "").strip()
                if not sid:
                    continue
                gid = str(st.get("group_id") or "").strip()
                uid = str(st.get("user_id") or "").strip()
                if st.get("is_group_session") or st.get("chat_type") == "group" or gid:
                    if gid:
                        self._session_to_group[sid] = gid
                else:
                    self._session_to_group.setdefault(sid, "")
                    if uid:
                        self._session_to_private[sid] = uid
        except Exception:
            pass

    def _resolve_scope_group_id(self, session_id: str, *, explicit_group_id: str = "", user_id: str = "") -> str:
        if explicit_group_id:
            return explicit_group_id
        sid = str(session_id or "").strip()
        if sid in self._session_to_group and self._session_to_group[sid]:
            return self._session_to_group[sid]
        if ":group:" in sid:
            return sid.split(":group:")[-1]
        puid = user_id or self._session_to_private.get(sid, "")
        if not puid and ":user:" in sid:
            puid = sid.split(":user:")[-1]
        return f"private:{puid}" if puid else ""

    def _is_scope_enabled(self, *, group_id: str, is_private: bool) -> bool:
        cfg = self.config
        if not cfg.plugin.enabled:
            return False
        if is_private:
            return bool(cfg.guard.allow_private)
        blocked = {str(x).strip() for x in cfg.guard.blocked_groups if str(x).strip()}
        if group_id in blocked:
            return False
        allowed = {str(x).strip() for x in cfg.guard.allowed_groups if str(x).strip()}
        if allowed and group_id not in allowed:
            return False
        return True

    def _scope_allowed(self, group_id: str) -> bool:
        return bool(group_id) and self._is_scope_enabled(group_id=group_id, is_private=group_id.startswith("private:"))

    async def _discover_member_api_targets(self) -> list[tuple[str, str]]:
        if self._member_api_targets is not None:
            return self._member_api_targets
        defaults: list[tuple[str, str]] = [
            ("call_data", "maibot-team.snowluma-adapter.adapter.snowluma.action.call_data"),
            ("call_data", "maibot-team.snowluma-adapter.adapter.napcat.action.call_data"),
            ("call_data", "maibot-team.snowluma-adapter.adapter.onebot.action.call_data"),
        ]
        list_fn = getattr(getattr(self.ctx, "api", None), "list", None)
        if not callable(list_fn):
            self._member_api_targets = defaults
            return self._member_api_targets
        try:
            payload = await list_fn()
        except Exception:
            self._member_api_targets = defaults
            return self._member_api_targets
        items = payload.get("apis", []) if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
        discovered: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for kind, suffix in (
            ("call_data", "action.call_data"),
            ("call", "action.call"),
            ("direct", "get_group_member_list"),
        ):
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                pid = str(entry.get("plugin_id") or "").strip()
                name = str(entry.get("name") or "").strip()
                if not pid or not name:
                    continue
                if name == suffix or name.endswith("." + suffix):
                    full = name if name.startswith(pid + ".") else f"{pid}.{name}"
                    pair = (kind, full)
                    if pair not in seen:
                        seen.add(pair)
                        discovered.append(pair)
        self._member_api_targets = discovered or defaults
        return self._member_api_targets

    @staticmethod
    def _unwrap_member_list(resp: Any) -> list[dict[str, Any]]:
        if isinstance(resp, dict):
            if resp.get("success") is False:
                return []
            if "result" in resp:
                resp = resp["result"]
            if isinstance(resp, dict) and "data" in resp:
                resp = resp["data"]
        return [m for m in resp if isinstance(m, dict)] if isinstance(resp, list) else []

    async def _group_members(self, group_id: str) -> list:
        api_cap = getattr(self.ctx, "api", None)
        call_fn = getattr(api_cap, "call", None)
        if callable(call_fn):
            targets = await self._discover_member_api_targets()
            for kind, api_name in targets:
                try:
                    if kind == "direct":
                        resp = await call_fn(api_name, group_id=group_id)
                    else:
                        resp = await call_fn(
                            api_name,
                            action_name="get_group_member_list",
                            params={"group_id": group_id},
                        )
                except Exception:
                    continue
                members = self._unwrap_member_list(resp)
                if members:
                    return members
        return self._store.list_group_members(group_id) if self._store is not None else []

    async def _manual_participant(
        self, group_id: str, target: str, sender_id: str, members: list | None = None
    ) -> tuple[str, str] | None:
        target = target.strip()
        if group_id.startswith("private:"):
            uid = group_id.removeprefix("private:")
            if target.isdigit() and target != uid:
                return None
            return uid, target or uid
        if sender_id and (not target or target == sender_id):
            return sender_id, sender_id
        if not target:
            return None
        if members is None:
            members = await self._group_members(group_id)
        matches = {}
        for member in members:
            if not isinstance(member, dict):
                continue
            uid = str(member.get("user_id") or "").strip()
            card = str(member.get("card") or "").strip()
            nickname = str(member.get("nickname") or "").strip()
            if uid and target == uid:
                return uid, card or nickname or uid
            if uid and target in (card, nickname):
                matches[uid] = card or nickname or uid
        return next(iter(matches.items())) if len(matches) == 1 else None

    async def _bind_legacy_participants(self, pending: list[tuple[str, str]]) -> None:
        members_by_group = {}
        for group_id, name in pending:
            if not self._scope_allowed(group_id):
                continue
            if not group_id.startswith("private:") and group_id not in members_by_group:
                members_by_group[group_id] = await self._group_members(group_id)
            person = await self._manual_participant(group_id, name, "", members_by_group.get(group_id))
            if person and self._store and self._scope_allowed(group_id):
                self._store.bind_manual_name(group_id, name, person[0])

    def _now_dt(self, now_ts: float | None = None) -> datetime:
        tz = resolve_tz(self.config.guard.timezone)
        if now_ts is None:
            return datetime.now(tz)
        return datetime.fromtimestamp(float(now_ts), tz=tz)

    def _set_pending_hint(self, session_id: str, priority: int, text: str) -> None:
        if not session_id or not text:
            return
        cur = self._pending_hints.get(session_id)
        if cur is None or priority >= cur[0]:
            self._pending_hints[session_id] = (priority, text)

    async def _run_startup_backfill(self) -> None:
        if self._store is None:
            return
        try:
            await asyncio.sleep(3.0)
            if not self._running:
                return
            raw = await self.ctx.chat.get_all_streams(platform="qq")
            streams = raw.get("streams", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
            if not isinstance(streams, list):
                return
            now_ts = time.time()
            start_ts = now_ts - float(self.config.guard.backfill_days) * 86400.0
            total_groups = 0
            total_msgs = 0

            for st in streams:
                if not self._running:
                    break
                if not isinstance(st, dict):
                    continue
                sid = str(st.get("session_id") or st.get("stream_id") or "").strip()
                gid = str(st.get("group_id") or "").strip()
                uid_priv = str(st.get("user_id") or "").strip()
                is_private = not bool(gid)
                scope_gid = gid or (f"private:{uid_priv}" if uid_priv else "")
                if not sid or not scope_gid:
                    continue
                if not self._is_scope_enabled(group_id=scope_gid, is_private=is_private):
                    continue
                if self._store.is_group_backfilled(scope_gid):
                    continue

                resp = await self.ctx.message.get_by_time_in_chat(
                    chat_id=sid,
                    start_time=str(start_ts),
                    end_time=str(now_ts),
                    limit=800,
                    limit_mode="latest",
                    filter_mai=True,
                    filter_command=True,
                )
                msgs = resp.get("messages", []) if isinstance(resp, dict) else (resp if isinstance(resp, list) else [])
                if not isinstance(msgs, list):
                    msgs = []

                pres_agg: dict[str, list[Any]] = defaultdict(lambda: ["", float("inf"), 0.0, 0])
                utt_rows: list[tuple[str, str, str, float]] = []
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    mi = m.get("message_info") if isinstance(m.get("message_info"), dict) else {}
                    ui = mi.get("user_info") if isinstance(mi.get("user_info"), dict) else {}
                    uid = str(ui.get("user_id") or m.get("user_id") or "").strip()
                    if not uid or (self._bot_qq_account and uid == self._bot_qq_account):
                        continue
                    uname = str(ui.get("user_cardname") or ui.get("user_nickname") or uid).strip()
                    ts_val = m.get("timestamp") if m.get("timestamp") is not None else (m.get("time") or mi.get("time"))
                    try:
                        mts = float(ts_val)
                    except (TypeError, ValueError):
                        continue
                    rec = pres_agg[uid]
                    if uname:
                        rec[0] = uname
                    rec[1] = min(rec[1], mts)
                    rec[2] = max(rec[2], mts)
                    rec[3] += 1

                    txt = sanitize_text(str(m.get("processed_plain_text") or ""))
                    if is_dejavu_candidate(txt):
                        utt_rows.append((uid, uname, txt, mts))

                pres_rows = [
                    (uid, rec[0], rec[1], rec[2], rec[3])
                    for uid, rec in pres_agg.items()
                    if rec[3] > 0 and rec[2] > 0
                ]
                if not self._scope_allowed(scope_gid):
                    continue
                self._store.bulk_import_history(
                    scope_gid,
                    presence_rows=pres_rows,
                    utterance_rows=utt_rows,
                    total_scanned=len(msgs),
                    completed_at=now_ts,
                    max_utterances_per_group=self.config.dejavu.max_utterances_per_group,
                )
                total_groups += 1
                total_msgs += len(msgs)
                await asyncio.sleep(0.2)

            if total_groups > 0:
                self.ctx.logger.info("熟人社交直觉历史回填完成: 会话数=%d 扫描消息数=%d", total_groups, total_msgs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.ctx.logger.debug("历史回填跳过: %s", type(exc).__name__)

    @HookHandler(
        "chat.receive.after_process",
        name="social_intuition_observe_incoming",
        description="静默观察生活近况、既视感观点与老友活跃状态（零主动刷屏）",
        mode=HookMode.OBSERVE,
        order=HookOrder.NORMAL,
        timeout_ms=4000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def on_receive_after_process(self, **kwargs: Any) -> None:
        if not self._running or not self.config.plugin.enabled or self._store is None:
            return
        msg = kwargs.get("message")
        if not isinstance(msg, dict):
            return

        msg_info = msg.get("message_info") if isinstance(msg.get("message_info"), dict) else {}
        user_info = msg_info.get("user_info") if isinstance(msg_info.get("user_info"), dict) else {}
        group_info = msg_info.get("group_info") if isinstance(msg_info.get("group_info"), dict) else None
        user_id = str(user_info.get("user_id") or "").strip()
        if not user_id or (self._bot_qq_account and user_id == self._bot_qq_account):
            return

        sender_name = str(user_info.get("user_cardname") or user_info.get("user_nickname") or user_id).strip()
        raw_group_id = str(group_info.get("group_id") or "").strip() if group_info else ""
        is_private = not bool(raw_group_id)
        session_id = str(msg.get("session_id") or "").strip()
        platform = str(msg_info.get("platform") or "qq").strip()
        if not session_id and platform:
            session_id = f"{platform}:{'group:' + raw_group_id if raw_group_id else 'user:' + user_id}"
        scope_group_id = raw_group_id or f"private:{user_id}"

        if session_id:
            self._session_to_group[session_id] = raw_group_id
            if is_private:
                self._session_to_private[session_id] = user_id

        if not self._is_scope_enabled(group_id=scope_group_id, is_private=is_private):
            return

        plain_text = str(msg.get("processed_plain_text") or "").strip()
        if not plain_text:
            return

        task = asyncio.create_task(
            self._process_incoming(
                scope_group_id=scope_group_id,
                session_id=session_id,
                user_id=user_id,
                sender_name=sender_name,
                plain_text=plain_text,
                now_ts=time.time(),
            )
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _process_incoming(
        self,
        *,
        scope_group_id: str,
        session_id: str,
        user_id: str,
        sender_name: str,
        plain_text: str,
        now_ts: float,
    ) -> None:
        if self._store is None or not self._scope_allowed(scope_group_id):
            return
        cfg = self.config
        now_dt = self._now_dt(now_ts)
        quiet = in_quiet_hours(now_dt, cfg.guard.quiet_start, cfg.guard.quiet_end)
        clean_text = sanitize_text(plain_text)

        try:
            triggered_this_turn = False

            # 1. 到点近况追问（优先级最高 = 3）
            if not quiet and cfg.followup.enabled:
                due_fu = self._store.get_due_followup(
                    scope_group_id,
                    user_id,
                    now_ts,
                    user_cooldown_days=cfg.followup.user_cooldown_days,
                )
                if due_fu is not None and self._store.mark_followup_asked(due_fu.id, scope_group_id, user_id, now_ts):
                    self._set_pending_hint(session_id, 3, format_followup_hint(due_fu, now_dt))
                    triggered_this_turn = True

            # 2. 失踪老友回归感应（优先级 = 2，若本轮已有追问线索则仅更新发言不重复触发）
            pres_rec, absent_days = self._store.touch_member_and_check_reunion(
                group_id=scope_group_id,
                user_id=user_id,
                user_name=sender_name,
                now_ts=now_ts,
                min_absent_days=cfg.reunion.min_absent_days,
                min_history_messages=cfg.reunion.min_history_messages,
                cooldown_days=cfg.reunion.cooldown_days,
                allow_trigger=(not quiet and cfg.reunion.enabled and not triggered_this_turn),
            )
            if absent_days is not None:
                self._set_pending_hint(session_id, 2, format_reunion_hint(pres_rec, absent_days))
                triggered_this_turn = True

            # 3. 跨月既视感检索与观点语料入库（优先级 = 1）
            if is_dejavu_candidate(clean_text):
                new_emb: list[float] | None = None
                if cfg.dejavu.enabled and not quiet and not triggered_this_turn:
                    before_ts = now_ts - float(cfg.dejavu.min_days_apart) * 86400.0
                    old_recs = self._store.get_old_utterances(
                        scope_group_id,
                        user_id,
                        before_ts=before_ts,
                        now_ts=now_ts,
                        user_cooldown_days=cfg.dejavu.user_cooldown_days,
                        limit=250,
                    )
                    if old_recs:
                        has_old_embs = any(r.embedding is not None for r in old_recs)
                        if cfg.dejavu.use_embedding and has_old_embs:
                            new_emb = await self._embed_text(clean_text)
                            if not self._scope_allowed(scope_group_id):
                                return
                        hit = find_best_dejavu_match(
                            clean_text,
                            new_emb,
                            old_recs,
                            lexical_threshold=cfg.dejavu.lexical_threshold,
                            embedding_threshold=cfg.dejavu.embedding_threshold,
                        )
                        if hit is not None:
                            old_rec, _, _ = hit
                            days_apart = max(1, int((now_ts - old_rec.created_at) // 86400.0))
                            old_date_str = datetime.fromtimestamp(old_rec.created_at, tz=now_dt.tzinfo).strftime("%m月%d日")
                            ok_dj, angle = await self._verify_dejavu_with_llm(sender_name, old_rec.text, old_date_str, clean_text)
                            if not self._scope_allowed(scope_group_id):
                                return
                            if ok_dj:
                                self._store.mark_utterance_triggered(old_rec.id, scope_group_id, user_id, now_ts)
                                self._set_pending_hint(
                                    session_id,
                                    1,
                                    format_dejavu_hint(old_rec, clean_text, days_apart, angle, now_dt),
                                )
                if new_emb is None and cfg.dejavu.enabled and cfg.dejavu.use_embedding:
                    new_emb = await self._embed_text(clean_text)
                if not self._scope_allowed(scope_group_id):
                    return
                self._store.add_utterance(
                    group_id=scope_group_id,
                    user_id=user_id,
                    user_name=sender_name,
                    text=clean_text,
                    created_at=now_ts,
                    embedding=new_emb,
                    max_per_group=cfg.dejavu.max_utterances_per_group,
                )

            # 4. 静默捕获新的生活近况
            if cfg.followup.enabled:
                cand = prefilter_followup_candidate(plain_text, now_dt, expire_days=cfg.followup.expire_days)
                if cand is not None:
                    verified = None
                    if cfg.followup.use_llm_verification:
                        verified = await self._verify_followup_with_llm(sender_name, clean_text, cand, now_dt)
                    if not self._scope_allowed(scope_group_id):
                        return
                    final_cand = verified if verified is not None else cand
                    if final_cand.event_summary and final_cand.confidence >= float(cfg.followup.min_confidence):
                        rec, inserted = self._store.add_followup(
                            group_id=scope_group_id,
                            session_id=session_id,
                            user_id=user_id,
                            user_name=sender_name,
                            original_text=clean_text,
                            event_summary=final_cand.event_summary,
                            created_at=now_ts,
                            ask_after_ts=final_cand.ask_after_ts,
                            expire_ts=final_cand.expire_ts,
                            confidence=final_cand.confidence,
                            max_pending_per_user=cfg.followup.max_pending_per_user,
                        )
                        if inserted:
                            self.ctx.logger.info("静默记下生活近况: id=%d summary=%s", rec.id, rec.event_summary)
        except Exception as exc:
            self.ctx.logger.debug("社交直觉观察异常: %s", type(exc).__name__)

    async def _embed_text(self, text: str) -> list[float] | None:
        try:
            resp = await self.ctx.llm.embed(text=text, task_name=self.config.dejavu.embedding_task_name)
            emb = resp.get("embedding") if isinstance(resp, dict) else None
            if isinstance(emb, list) and emb:
                return [float(x) for x in emb]
        except Exception:
            pass
        return None

    async def _verify_followup_with_llm(self, user_name: str, text: str, cand: Any, now_dt: datetime) -> Any:
        try:
            prompt = build_followup_extract_prompt(user_name, text, now_dt)
            resp = await self.ctx.llm.generate(
                prompt,
                task_name=self.config.followup.llm_task_name,
                temperature=0.1,
                max_tokens=180,
            )
            raw = _extract_llm_text(resp)
            if not raw:
                return None
            return parse_followup_llm_response(raw, cand, now_dt, expire_days=self.config.followup.expire_days)
        except Exception:
            return None

    async def _verify_dejavu_with_llm(
        self,
        user_name: str,
        old_text: str,
        old_date_str: str,
        new_text: str,
    ) -> tuple[bool, str]:
        try:
            prompt = build_dejavu_verify_prompt(user_name, old_text, old_date_str, new_text)
            resp = await self.ctx.llm.generate(
                prompt,
                task_name=self.config.followup.llm_task_name,
                temperature=0.1,
                max_tokens=140,
            )
            raw = _extract_llm_text(resp)
            if not raw:
                return False, ""
            return parse_dejavu_llm_response(raw)
        except Exception:
            return False, ""

    @HookHandler(
        "maisaka.planner.before_request",
        name="social_intuition_planner_hint",
        description="仅在触发近况追问/久别重逢/既视感时注入最多一条一次性线索（常态零注入）",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=1500,
        error_policy=ErrorPolicy.SKIP,
    )
    async def on_planner_before_request(self, **kwargs: Any) -> dict[str, Any]:
        unchanged = {"action": "continue", "modified_kwargs": kwargs}
        if not self._running or not self.config.plugin.enabled:
            return unchanged
        items = kwargs.get("items")
        if not isinstance(items, list):
            return unchanged
        session_id = str(kwargs.get("session_id") or "").strip()
        picked = self._pending_hints.pop(session_id, None)
        if not picked or not self._scope_allowed(self._resolve_scope_group_id(session_id)):
            return unchanged
        _, hint = picked
        updated = dict(kwargs)
        updated["items"] = _inject_reminder(items, hint)
        return {"action": "continue", "modified_kwargs": updated}

    @Tool(
        "social_intuition_query",
        description="查询当前聊天内静默记住的生活近况待追问事项或统计信息（仅限当前聊天，绝不跨群泄露）。",
        parameters=[
            ToolParameterInfo(
                name="view",
                param_type=ToolParamType.STRING,
                description="查看视图：followups(近况档案，默认) 或 stats(统计概览)",
                required=False,
                enum_values=["followups", "stats"],
                default="followups",
            ),
            ToolParameterInfo(
                name="user_name",
                param_type=ToolParamType.STRING,
                description="可选：按群友昵称筛选",
                required=False,
            ),
            ToolParameterInfo(
                name="limit",
                param_type=ToolParamType.INTEGER,
                description="返回条数上限（默认 8）",
                required=False,
                default=8,
            ),
        ],
    )
    async def tool_query(
        self,
        view: str = "followups",
        user_name: str = "",
        limit: int = 8,
        stream_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self._store is None:
            return {"success": False, "content": "直觉仓库尚未初始化。"}
        group_id = self._resolve_scope_group_id(
            stream_id, explicit_group_id=str(kwargs.get("group_id") or ""), user_id=str(kwargs.get("user_id") or "")
        )
        if not stream_id or not self._scope_allowed(group_id):
            return {"success": False, "content": "无法确认当前所属聊天，出于隔离限制拒绝查询。"}

        if str(view).strip().lower() == "stats":
            st = self._store.get_group_stats(group_id)
            return {
                "success": True,
                "content": (
                    f"本群社交直觉统计：待追问近况 {st['pending_followups']} 条、已追问 {st['asked_followups']} 条、"
                    f"观点句语料 {st['utterances']} 条、已记录群友 {st['members']} 人。"
                ),
            }

        rows = self._store.list_followups(group_id, status_filter="all", user_name=str(user_name or ""), limit=limit)
        if not rows:
            return {"success": True, "content": "本群当前还没有记录任何生活近况。"}
        lines = [f"本群生活近况档案（共 {len(rows)} 条）："]
        for r in rows:
            ask_dt = datetime.fromtimestamp(r.ask_after_ts).strftime("%m-%d %H:%M")
            lines.append(f"#{r.id} [{r.status}] {r.user_name}: 「{r.event_summary}」（预计 {ask_dt} 后可问）")
        return {"success": True, "content": "\n".join(lines)}

    @Tool(
        "social_intuition_manage",
        description="在本群手动新增一条以后要关心的近况，或作废取消一条不需要再问的近况。",
        parameters=[
            ToolParameterInfo(
                name="action",
                param_type=ToolParamType.STRING,
                description="操作类型：add_followup 或 cancel_followup",
                required=True,
                enum_values=["add_followup", "cancel_followup"],
            ),
            ToolParameterInfo(
                name="followup_id",
                param_type=ToolParamType.INTEGER,
                description="cancel_followup 时必填的近况编号",
                required=False,
            ),
            ToolParameterInfo(
                name="user_name",
                param_type=ToolParamType.STRING,
                description="add_followup 时的群友 QQ 号或唯一昵称；私聊可留空",
                required=False,
            ),
            ToolParameterInfo(
                name="event_summary",
                param_type=ToolParamType.STRING,
                description="add_followup 时的事件简述",
                required=False,
            ),
            ToolParameterInfo(
                name="delay_hours",
                param_type=ToolParamType.INTEGER,
                description="多少小时后可以追问后续（默认 24）",
                required=False,
                default=24,
            ),
        ],
    )
    async def tool_manage(
        self,
        action: str = "",
        followup_id: int | None = None,
        user_name: str = "",
        event_summary: str = "",
        delay_hours: int = 24,
        stream_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self._store is None:
            return {"success": False, "content": "直觉仓库尚未初始化。"}
        group_id = self._resolve_scope_group_id(
            stream_id, explicit_group_id=str(kwargs.get("group_id") or ""), user_id=str(kwargs.get("user_id") or "")
        )
        if not stream_id or not self._scope_allowed(group_id):
            return {"success": False, "content": "无法确认当前所属聊天，出于隔离限制拒绝修改。"}

        act = str(action or "").strip().lower()
        if act == "cancel_followup":
            if followup_id is None:
                return {"success": False, "content": "请提供要取消的近况编号（followup_id）。"}
            cancelled = self._store.cancel_followup(group_id, int(followup_id))
            if cancelled is None:
                return {"success": False, "content": f"本群未找到编号 #{followup_id} 的近况记录。"}
            return {"success": True, "content": f"已取消近况 #{cancelled.id}（{cancelled.user_name}：{cancelled.event_summary}）。"}

        if act == "add_followup":
            summary = sanitize_text(event_summary)
            person = await self._manual_participant(group_id, str(user_name or ""), str(kwargs.get("user_id") or ""))
            if not person or not self._scope_allowed(group_id):
                return {"success": False, "content": "无法唯一确认当前会话的成员，请提供当事人的 QQ 号。"}
            participant_id, who = person
            if not summary:
                return {"success": False, "content": "请提供事件简述。"}
            now_ts = time.time()
            hrs = max(1, min(24 * 30, int(delay_hours or 24)))
            ask_ts = now_ts + hrs * 3600.0
            exp_ts = ask_ts + float(self.config.followup.expire_days) * 86400.0
            rec, _ = self._store.add_followup(
                group_id=group_id,
                session_id=stream_id,
                user_id=participant_id,
                user_name=who,
                original_text=summary,
                event_summary=summary[:50],
                created_at=now_ts,
                ask_after_ts=ask_ts,
                expire_ts=exp_ts,
                confidence=0.95,
                max_pending_per_user=self.config.followup.max_pending_per_user,
            )
            return {"success": True, "content": f"已记下 #{rec.id}（{who}：{rec.event_summary}），约 {hrs} 小时后再聊到时提醒问候。"}

        return {"success": False, "content": f"不支持的操作：{action}"}


def create_plugin() -> SocialIntuitionPlugin:
    return SocialIntuitionPlugin()
