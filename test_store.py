"""Unit tests for social-intuition SQLite store."""

from pathlib import Path
import tempfile
import unittest

from store import SocialIntuitionStore, cosine_similarity, pack_embedding, unpack_embedding


class StoreTests(unittest.TestCase):
    def test_embedding_codec_and_cosine(self) -> None:
        blob = pack_embedding([0.6, 0.8, 0.0])
        self.assertIsNotNone(blob)
        vec = unpack_embedding(blob)
        self.assertIsNotNone(vec)
        assert vec is not None
        self.assertAlmostEqual(cosine_similarity(vec, (0.6, 0.8, 0.0)), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity(vec, (0.0, 0.0, 1.0)), 0.0, places=5)

    def test_followup_reunion_backfill_and_group_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            st = SocialIntuitionStore(Path(td))

            # Followup dedup, due query, cooldown, and cross-group isolation
            f1, ins1 = st.add_followup(
                group_id="10001",
                session_id="s1",
                user_id="u1",
                user_name="阿强",
                original_text="明天去考科二",
                event_summary="考科二",
                created_at=1000.0,
                ask_after_ts=2000.0,
                expire_ts=100000.0,
                max_pending_per_user=2,
            )
            self.assertTrue(ins1)
            _, ins_dup = st.add_followup(
                group_id="10001",
                session_id="s1",
                user_id="u1",
                user_name="阿强",
                original_text="明天去考科二",
                event_summary="考科二",
                created_at=1100.0,
                ask_after_ts=2000.0,
                expire_ts=100000.0,
                max_pending_per_user=2,
            )
            self.assertFalse(ins_dup)
            st.touch_member_and_check_reunion(
                group_id="10001",
                user_id="u1",
                user_name="阿强",
                now_ts=1000.0,
                min_absent_days=15,
                min_history_messages=10,
            )
            due = st.get_due_followup("10001", "u1", 2500.0, user_cooldown_days=7)
            self.assertIsNotNone(due)
            assert due is not None
            self.assertEqual(due.id, f1.id)
            self.assertTrue(st.mark_followup_asked(f1.id, "10001", "u1", 2500.0))

            # Another followup right away is blocked by user_cooldown_days
            st.add_followup(
                group_id="10001",
                session_id="s1",
                user_id="u1",
                user_name="阿强",
                original_text="后天寄修键盘",
                event_summary="寄修键盘",
                created_at=2600.0,
                ask_after_ts=2700.0,
                expire_ts=200000.0,
            )
            self.assertIsNone(st.get_due_followup("10001", "u1", 3000.0, user_cooldown_days=7))
            self.assertEqual(st.list_followups("20002", status_filter="all"), [])
            self.assertIsNone(st.cancel_followup("20002", f1.id))

            # Backfill + Reunion test: u2 has 12 messages ending at t=1000
            st.bulk_import_history(
                "10001",
                presence_rows=[("u2", "小林", 100.0, 1000.0, 12)],
                utterance_rows=[("u2", "小林", "这游戏破日常太坐牢了", 500.0)],
                total_scanned=12,
                completed_at=1200.0,
            )
            self.assertTrue(st.is_group_backfilled("10001"))

            # Absent 18 days -> reunion triggers once
            t_reunion = 1000.0 + 18 * 86400.0
            _, absent_days = st.touch_member_and_check_reunion(
                group_id="10001",
                user_id="u2",
                user_name="小林",
                now_ts=t_reunion,
                min_absent_days=15,
                min_history_messages=10,
                cooldown_days=30,
            )
            self.assertEqual(absent_days, 18)

            # Next message shortly after must NOT trigger reunion
            _, absent_again = st.touch_member_and_check_reunion(
                group_id="10001",
                user_id="u2",
                user_name="小林",
                now_ts=t_reunion + 60.0,
                min_absent_days=15,
                min_history_messages=10,
                cooldown_days=30,
            )
            self.assertIsNone(absent_again)

            # Old utterance query works for min_days_apart=14
            olds = st.get_old_utterances(
                "10001",
                "u2",
                before_ts=t_reunion - 14 * 86400.0,
                now_ts=t_reunion,
                user_cooldown_days=14,
            )
            self.assertEqual(len(olds), 1)


if __name__ == "__main__":
    unittest.main()
