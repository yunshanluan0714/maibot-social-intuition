"""SDK integration regressions for scope and participant identity."""

import asyncio
import importlib
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec("maibot_sdk"), "requires the real MaiBot SDK")
class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module(f"{Path(__file__).resolve().parent.name}.plugin")
        self.plugin = module.SocialIntuitionPlugin()
        self.plugin.set_plugin_config(self.plugin.build_default_config())
        self.trigger = AsyncMock()
        self.plugin._set_context(SimpleNamespace(
            logger=logging.getLogger(__name__),
            api=SimpleNamespace(call=AsyncMock(return_value=[{"user_id": "31", "card": "甲"}])),
            maisaka=SimpleNamespace(proactive=SimpleNamespace(trigger=self.trigger)),
        ))
        data = tempfile.TemporaryDirectory()
        self.addCleanup(data.cleanup)
        self.plugin._store = module.SocialIntuitionStore(data.name)
        self.plugin._running = True

    async def test_manual_identity_and_scope(self):
        context = {"stream_id": "qq:group:21", "group_id": "21", "user_id": "32"}
        result = await self.plugin.tool_manage(action="add_followup", user_name="甲", event_summary="面试", **context)
        self.assertTrue(result["success"])
        self.assertEqual(self.plugin._store.get_due_followup("21", "31", time.time() + 25 * 3600).user_id, "31")
        self.plugin.ctx.api.call.return_value.append({"user_id": "33", "card": "甲"})
        result = await self.plugin.tool_manage(action="add_followup", user_name="甲", event_summary="面试", **context)
        self.assertFalse(result["success"])
        self.plugin.config.guard.blocked_groups = ["21"]
        result = await self.plugin.tool_query(**context)
        self.assertFalse(result["success"])
        result = await self.plugin.tool_manage(action="add_followup", user_name="甲", event_summary="面试", **context)
        self.assertFalse(result["success"])
        self.plugin._pending_hints["qq:group:21"] = (2, "hint")
        planner = await self.plugin.on_planner_before_request(session_id="qq:group:21", items=[])
        self.assertEqual(planner["modified_kwargs"]["items"], [])

    async def test_legacy_manual_record_waits_for_unique_identity(self):
        record, _ = self.plugin._store.add_followup(group_id="21", session_id="qq:group:21", user_id="manual:甲",
                                                   user_name="甲", original_text="旧约定", event_summary="旧约定",
                                                   created_at=1, ask_after_ts=2, expire_ts=100)
        pending = self.plugin._store.pending_manual_names()
        self.assertEqual(pending, [("21", "甲")])
        self.plugin.ctx.api.call.return_value.append({"user_id": "33", "card": "甲"})
        await self.plugin._bind_legacy_participants(pending)
        self.assertEqual(self.plugin._store.pending_manual_names(), pending)
        self.plugin.ctx.api.call.return_value.pop()
        await self.plugin._bind_legacy_participants(pending)
        self.assertEqual(self.plugin._store.pending_manual_names(), [])
        self.assertEqual(self.plugin._store.get_due_followup("21", "31", 3).user_id, "31")

    async def test_disable_scope_after_receive_before_background_processing(self):
        await self.plugin.on_receive_after_process(message={
            "message_id": "msg", "session_id": "qq:group:21", "processed_plain_text": "今晚我一定十二点前睡觉",
            "message_info": {"group_info": {"group_id": "21"}, "user_info": {"user_id": "31"}},
        })
        self.assertEqual(len(self.plugin._bg_tasks), 1)
        self.plugin.config.guard.blocked_groups = ["21"]
        await self.plugin.on_config_update("self", {}, "test")
        await asyncio.gather(*self.plugin._bg_tasks)
        self.assertEqual(self.plugin._store.get_group_stats("21")["members"], 0)
        self.assertFalse(self.plugin._pending_hints)

    async def test_dynamic_adapter_discovery_and_local_presence_fallback(self):
        # 1) Dynamic discovery via ctx.api.list() works with generic OneBot/NapCat adapters.
        async def custom_call(api_name, **kwargs):
            if api_name == "third-party.napcat.action.call_data":
                return [{"user_id": "77", "nickname": "乙"}]
            return {"success": False, "error": "not found"}

        self.plugin._member_api_targets = None
        self.plugin.ctx.api = SimpleNamespace(
            list=AsyncMock(return_value={"apis": [{"plugin_id": "third-party.napcat", "name": "action.call_data"}]}),
            call=AsyncMock(side_effect=custom_call),
        )
        res = await self.plugin.tool_manage(
            action="add_followup", user_name="乙", event_summary="考驾照", stream_id="qq:group:21", group_id="21", user_id="32"
        )
        self.assertTrue(res["success"])
        self.assertEqual(self.plugin._store.get_due_followup("21", "77", time.time() + 25 * 3600).user_id, "77")

        # 2) Falls back to member_presence when adapter API is unavailable.
        self.plugin._member_api_targets = None
        self.plugin.ctx.api = SimpleNamespace(call=AsyncMock(return_value={"success": False, "error": "offline"}))
        self.plugin._store.touch_member_and_check_reunion(
            group_id="21", user_id="99", user_name="丙", now_ts=time.time()
        )
        res2 = await self.plugin.tool_manage(
            action="add_followup", user_name="丙", event_summary="搬家", stream_id="qq:group:21", group_id="21", user_id="32"
        )
        self.assertTrue(res2["success"])
        self.assertEqual(self.plugin._store.get_due_followup("21", "99", time.time() + 25 * 3600).user_id, "99")
