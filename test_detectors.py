"""Unit tests for social-intuition detectors."""

from datetime import datetime, timedelta, timezone
import unittest

from detectors import (
    bigram_jaccard,
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
)
from store import FollowupRecord, PresenceRecord, UtteranceRecord


class DetectorsTests(unittest.TestCase):
    def _dt(self, hour: int = 14, minute: int = 0, day: int = 25) -> datetime:
        return datetime(2026, 11, day, hour, minute, 0, tzinfo=timezone(timedelta(hours=8)))

    def test_followup_prefilter_and_exclusions(self) -> None:
        now_dt = self._dt(hour=14)
        c1 = prefilter_followup_candidate("明天下午去考科二，有点小慌", now_dt)
        self.assertIsNotNone(c1)
        assert c1 is not None
        self.assertEqual(datetime.fromtimestamp(c1.ask_after_ts, tz=now_dt.tzinfo).day, 26)

        c2 = prefilter_followup_candidate("刚把显卡寄去售后返修", now_dt)
        self.assertIsNotNone(c2)
        assert c2 is not None
        self.assertAlmostEqual(c2.ask_after_ts - now_dt.timestamp(), 48 * 3600, delta=60)

        # Must exclude flag vows, pure questions, third-person chatter, system notices
        self.assertIsNone(prefilter_followup_candidate("明天再抽卡我倒立吃键盘", now_dt))
        self.assertIsNone(prefilter_followup_candidate("你们明天去面试吗？", now_dt))
        self.assertIsNone(prefilter_followup_candidate("我朋友明天去拔牙", now_dt))
        self.assertIsNone(prefilter_followup_candidate("【防撤回提醒】明天面试", now_dt))

        prompt = build_followup_extract_prompt("阿强", "明天下午去考科二", now_dt)
        self.assertIn("科二", prompt)
        parsed = parse_followup_llm_response(
            '```json\n{"is_followup": true, "event_summary": "去考科二", "ask_after_iso": "2026-11-26 19:00", "confidence": 0.91}\n```',
            c1,
            now_dt,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.event_summary, "去考科二")
        self.assertGreater(parsed.confidence, 0.9)

    def test_dejavu_candidate_and_matching(self) -> None:
        self.assertFalse(is_dejavu_candidate("哈哈哈哈"))
        self.assertFalse(is_dejavu_candidate("早安"))
        self.assertFalse(is_dejavu_candidate("今天晚饭吃什么好呢？"))
        self.assertTrue(is_dejavu_candidate("这游戏破日常也太坐牢了，明天再玩我是真的服"))

        old = UtteranceRecord(
            id=1,
            group_id="10001",
            user_id="u1",
            user_name="阿明",
            text="这破游戏日常太坐牢了再也不想打开了",
            created_at=1000.0,
            embedding=(0.9, 0.4, 0.1),
            triggered=False,
        )
        hit = find_best_dejavu_match(
            "今天打这破游戏感觉像上班坐牢一样完全玩不动",
            (0.89, 0.42, 0.11),
            [old],
            lexical_threshold=0.45,
            embedding_threshold=0.86,
        )
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit[2], "embedding")
        self.assertGreater(bigram_jaccard("这破游戏太坐牢了", "这破游戏真的太坐牢了"), 0.5)

        p = build_dejavu_verify_prompt("阿明", old.text, "11月01日", "今天又坐牢了")
        self.assertIn("既视感", p)
        ok_dj, angle = parse_dejavu_llm_response('{"is_dejavu": true, "angle": "又说这游戏坐牢"}')
        self.assertTrue(ok_dj)
        self.assertIn("坐牢", angle)

    def test_hint_formatters_and_quiet_hours(self) -> None:
        now_dt = self._dt(hour=15)
        self.assertTrue(in_quiet_hours(self._dt(hour=2), "01:00", "08:00"))
        self.assertFalse(in_quiet_hours(now_dt, "01:00", "08:00"))

        fu = FollowupRecord(1, "g1", "s1", "u1", "阿强", "明天考科二", "考科二", 1000.0, 2000.0, 5000.0, "pending", None, 0.9)
        pr = PresenceRecord("g1", "u1", "阿强", 100.0, 200.0, 25, 0.0, 0.0, 0.0)
        ut = UtteranceRecord(1, "g1", "u1", "阿强", "老早说过要退游", 1000.0, None, False)

        self.assertIn("[老友近况]", format_followup_hint(fu, now_dt))
        self.assertIn("[久别重逢]", format_reunion_hint(pr, 18))
        self.assertIn("[既视感]", format_dejavu_hint(ut, "又想退游了", 20, "退游", now_dt))


if __name__ == "__main__":
    unittest.main()
