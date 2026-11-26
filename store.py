"""SQLite storage and compact float32 embedding codec for github.yunshanluan0714.social-intuition."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import struct
import threading
import time
from typing import Iterable, Sequence


def pack_embedding(vec: Sequence[float] | None) -> bytes | None:
    if not vec:
        return None
    floats = [float(x) for x in vec]
    return struct.pack(f"<{len(floats)}f", *floats)


def unpack_embedding(blob: bytes | None) -> tuple[float, ...] | None:
    if not blob or len(blob) % 4 != 0:
        return None
    count = len(blob) // 4
    return struct.unpack(f"<{count}f", blob)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return (dot / denom) if denom > 1e-12 else 0.0


@dataclass(frozen=True)
class FollowupRecord:
    id: int
    group_id: str
    session_id: str
    user_id: str
    user_name: str
    original_text: str
    event_summary: str
    created_at: float
    ask_after_ts: float
    expire_ts: float
    status: str
    asked_at: float | None
    confidence: float


@dataclass(frozen=True)
class UtteranceRecord:
    id: int
    group_id: str
    user_id: str
    user_name: str
    text: str
    created_at: float
    embedding: tuple[float, ...] | None
    triggered: bool


@dataclass(frozen=True)
class PresenceRecord:
    group_id: str
    user_id: str
    user_name: str
    first_seen: float
    last_seen: float
    msg_count: int
    last_reunion_at: float
    last_followup_at: float
    last_dejavu_at: float


class SocialIntuitionStore:
    """Thread-safe SQLite store scoped strictly by group_id (or private:<uid>)."""

    def __init__(self, data_dir: Path | str) -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "social_intuition.db"
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def pending_manual_names(self) -> list[tuple[str, str]]:
        with self._lock, closing(self._connect()) as conn:
            return [
                (row["group_id"], row["user_name"]) for row in conn.execute(
                    "SELECT DISTINCT group_id, user_name FROM followups "
                    "WHERE user_id = 'manual:' || user_name AND status IN ('pending')"
                )
            ]

    def bind_manual_name(self, group_id: str, name: str, user_id: str) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE followups SET user_id = ? WHERE group_id = ? AND user_id = ? AND status IN ('pending')",
                (user_id, group_id, f"manual:{name}"),
            )

    def list_group_members(self, group_id: str) -> list[dict[str, str]]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT user_id, user_name FROM member_presence WHERE group_id = ? ORDER BY last_seen DESC",
                (group_id,),
            ).fetchall()
            return [
                {"user_id": str(r["user_id"]), "card": str(r["user_name"]), "nickname": str(r["user_name"])}
                for r in rows
                if str(r["user_id"] or "").strip()
            ]

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS followups (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            group_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            user_name TEXT NOT NULL,
                            original_text TEXT NOT NULL,
                            event_summary TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            ask_after_ts REAL NOT NULL,
                            expire_ts REAL NOT NULL,
                            status TEXT NOT NULL DEFAULT 'pending',
                            asked_at REAL,
                            confidence REAL NOT NULL DEFAULT 0.8
                        )
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_fu_group_user_status ON followups (group_id, user_id, status)"
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS utterances (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            group_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            user_name TEXT NOT NULL,
                            text TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            embedding BLOB,
                            triggered INTEGER NOT NULL DEFAULT 0
                        )
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_utt_group_user_time ON utterances (group_id, user_id, created_at)"
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS member_presence (
                            group_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            user_name TEXT NOT NULL,
                            first_seen REAL NOT NULL,
                            last_seen REAL NOT NULL,
                            msg_count INTEGER NOT NULL DEFAULT 0,
                            last_reunion_at REAL NOT NULL DEFAULT 0,
                            last_followup_at REAL NOT NULL DEFAULT 0,
                            last_dejavu_at REAL NOT NULL DEFAULT 0,
                            PRIMARY KEY (group_id, user_id)
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS backfill_state (
                            group_id TEXT PRIMARY KEY,
                            completed_at REAL NOT NULL,
                            imported_messages INTEGER NOT NULL DEFAULT 0
                        )
                        """
                    )
            finally:
                conn.close()

    @staticmethod
    def _row_to_followup(row: sqlite3.Row) -> FollowupRecord:
        return FollowupRecord(
            id=int(row["id"]),
            group_id=str(row["group_id"]),
            session_id=str(row["session_id"]),
            user_id=str(row["user_id"]),
            user_name=str(row["user_name"]),
            original_text=str(row["original_text"]),
            event_summary=str(row["event_summary"]),
            created_at=float(row["created_at"]),
            ask_after_ts=float(row["ask_after_ts"]),
            expire_ts=float(row["expire_ts"]),
            status=str(row["status"]),
            asked_at=float(row["asked_at"]) if row["asked_at"] is not None else None,
            confidence=float(row["confidence"]),
        )

    @staticmethod
    def _row_to_utterance(row: sqlite3.Row) -> UtteranceRecord:
        return UtteranceRecord(
            id=int(row["id"]),
            group_id=str(row["group_id"]),
            user_id=str(row["user_id"]),
            user_name=str(row["user_name"]),
            text=str(row["text"]),
            created_at=float(row["created_at"]),
            embedding=unpack_embedding(row["embedding"]),
            triggered=bool(row["triggered"]),
        )

    @staticmethod
    def _row_to_presence(row: sqlite3.Row) -> PresenceRecord:
        return PresenceRecord(
            group_id=str(row["group_id"]),
            user_id=str(row["user_id"]),
            user_name=str(row["user_name"]),
            first_seen=float(row["first_seen"]),
            last_seen=float(row["last_seen"]),
            msg_count=int(row["msg_count"]),
            last_reunion_at=float(row["last_reunion_at"]),
            last_followup_at=float(row["last_followup_at"]),
            last_dejavu_at=float(row["last_dejavu_at"]),
        )

    # --- Followup CRUD ---

    def add_followup(
        self,
        *,
        group_id: str,
        session_id: str,
        user_id: str,
        user_name: str,
        original_text: str,
        event_summary: str,
        created_at: float,
        ask_after_ts: float,
        expire_ts: float,
        confidence: float = 0.85,
        max_pending_per_user: int = 3,
    ) -> tuple[FollowupRecord, bool]:
        clean_orig = str(original_text or "").strip()[:300]
        clean_sum = str(event_summary or clean_orig).strip()[:100]
        cap = max(1, int(max_pending_per_user))

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    dup = conn.execute(
                        """
                        SELECT * FROM followups
                        WHERE group_id = ? AND user_id = ? AND status = 'pending'
                          AND created_at >= ?
                          AND (original_text = ? OR event_summary = ?)
                        ORDER BY id DESC LIMIT 1
                        """,
                        (group_id, user_id, float(created_at) - 600.0, clean_orig, clean_sum),
                    ).fetchone()
                    if dup is not None:
                        return self._row_to_followup(dup), False

                    active_rows = conn.execute(
                        """
                        SELECT id FROM followups
                        WHERE group_id = ? AND user_id = ? AND status = 'pending'
                        ORDER BY created_at ASC, id ASC
                        """,
                        (group_id, user_id),
                    ).fetchall()
                    overflow = len(active_rows) - (cap - 1)
                    if overflow > 0:
                        expire_ids = [(int(r["id"]),) for r in active_rows[:overflow]]
                        conn.executemany("UPDATE followups SET status = 'expired' WHERE id = ?", expire_ids)

                    cur = conn.execute(
                        """
                        INSERT INTO followups (
                            group_id, session_id, user_id, user_name,
                            original_text, event_summary, created_at,
                            ask_after_ts, expire_ts, status, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                        """,
                        (
                            group_id,
                            session_id,
                            user_id,
                            user_name or user_id,
                            clean_orig,
                            clean_sum,
                            float(created_at),
                            float(ask_after_ts),
                            float(expire_ts),
                            float(confidence),
                        ),
                    )
                    row = conn.execute("SELECT * FROM followups WHERE id = ?", (cur.lastrowid,)).fetchone()
                    assert row is not None
                    return self._row_to_followup(row), True
            finally:
                conn.close()

    def get_due_followup(
        self,
        group_id: str,
        user_id: str,
        now_ts: float,
        *,
        user_cooldown_days: int = 7,
    ) -> FollowupRecord | None:
        cutoff_cooldown = float(now_ts) - max(1, int(user_cooldown_days)) * 86400.0
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        "UPDATE followups SET status = 'expired' WHERE group_id = ? AND status = 'pending' AND expire_ts < ?",
                        (group_id, float(now_ts)),
                    )
                    pres = conn.execute(
                        "SELECT last_followup_at FROM member_presence WHERE group_id = ? AND user_id = ?",
                        (group_id, user_id),
                    ).fetchone()
                    last_fup = float(pres["last_followup_at"] or 0.0) if pres is not None else 0.0
                    if last_fup > 0.0 and last_fup > cutoff_cooldown:
                        return None

                    row = conn.execute(
                        """
                        SELECT * FROM followups
                        WHERE group_id = ? AND user_id = ? AND status = 'pending'
                          AND ask_after_ts <= ? AND expire_ts >= ?
                        ORDER BY ask_after_ts ASC, id ASC LIMIT 1
                        """,
                        (group_id, user_id, float(now_ts), float(now_ts)),
                    ).fetchone()
                    return self._row_to_followup(row) if row is not None else None
            finally:
                conn.close()

    def mark_followup_asked(self, followup_id: int, group_id: str, user_id: str, now_ts: float) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    cur = conn.execute(
                        "UPDATE followups SET status = 'asked', asked_at = ? WHERE id = ? AND group_id = ? AND status = 'pending'",
                        (float(now_ts), int(followup_id), group_id),
                    )
                    if cur.rowcount > 0:
                        conn.execute(
                            "UPDATE member_presence SET last_followup_at = ? WHERE group_id = ? AND user_id = ?",
                            (float(now_ts), group_id, user_id),
                        )
                        return True
                    return False
            finally:
                conn.close()

    def list_followups(
        self,
        group_id: str,
        *,
        status_filter: str = "pending",
        user_name: str = "",
        limit: int = 8,
    ) -> list[FollowupRecord]:
        lim = max(1, min(15, int(limit)))
        clauses = ["group_id = ?"]
        params: list[object] = [group_id]
        sf = str(status_filter or "pending").strip().lower()
        if sf in ("pending", "asked", "cancelled", "expired"):
            clauses.append("status = ?")
            params.append(sf)
        elif sf == "all":
            clauses.append("status != 'cancelled'")
        if user_name.strip():
            clauses.append("(user_name LIKE ? OR user_id = ?)")
            like_u = f"%{user_name.strip()}%"
            params.extend([like_u, user_name.strip()])
        sql = f"SELECT * FROM followups WHERE {' AND '.join(clauses)} ORDER BY ask_after_ts ASC, id DESC LIMIT ?"
        params.append(lim)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, tuple(params)).fetchall()
                return [self._row_to_followup(r) for r in rows]
            finally:
                conn.close()

    def cancel_followup(self, group_id: str, followup_id: int) -> FollowupRecord | None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    cur = conn.execute(
                        "UPDATE followups SET status = 'cancelled' WHERE id = ? AND group_id = ?",
                        (int(followup_id), group_id),
                    )
                    if cur.rowcount <= 0:
                        return None
                    row = conn.execute("SELECT * FROM followups WHERE id = ? AND group_id = ?", (int(followup_id), group_id)).fetchone()
                    return self._row_to_followup(row) if row is not None else None
            finally:
                conn.close()

    # --- Utterances & DejaVu ---

    def add_utterance(
        self,
        *,
        group_id: str,
        user_id: str,
        user_name: str,
        text: str,
        created_at: float,
        embedding: Sequence[float] | None = None,
        max_per_group: int = 4000,
    ) -> int:
        clean = str(text or "").strip()[:200]
        if not clean:
            return 0
        blob = pack_embedding(embedding)
        cap = max(100, int(max_per_group))
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    dup = conn.execute(
                        """
                        SELECT id FROM utterances
                        WHERE group_id = ? AND user_id = ? AND text = ? AND created_at >= ?
                        LIMIT 1
                        """,
                        (group_id, user_id, clean, float(created_at) - 3600.0),
                    ).fetchone()
                    if dup is not None:
                        return int(dup["id"])

                    cur = conn.execute(
                        """
                        INSERT INTO utterances (group_id, user_id, user_name, text, created_at, embedding, triggered)
                        VALUES (?, ?, ?, ?, ?, ?, 0)
                        """,
                        (group_id, user_id, user_name or user_id, clean, float(created_at), blob),
                    )
                    cnt_row = conn.execute("SELECT COUNT(*) AS c FROM utterances WHERE group_id = ?", (group_id,)).fetchone()
                    cnt = int(cnt_row["c"]) if cnt_row else 0
                    if cnt > cap:
                        conn.execute(
                            """
                            DELETE FROM utterances WHERE id IN (
                                SELECT id FROM utterances WHERE group_id = ?
                                ORDER BY created_at ASC, id ASC LIMIT ?
                            )
                            """,
                            (group_id, cnt - cap),
                        )
                    return int(cur.lastrowid or 0)
            finally:
                conn.close()

    def get_old_utterances(
        self,
        group_id: str,
        user_id: str,
        *,
        before_ts: float,
        now_ts: float,
        user_cooldown_days: int = 14,
        limit: int = 300,
    ) -> list[UtteranceRecord]:
        cutoff_cd = float(now_ts) - max(1, int(user_cooldown_days)) * 86400.0
        with self._lock:
            conn = self._connect()
            try:
                pres = conn.execute(
                    "SELECT last_dejavu_at FROM member_presence WHERE group_id = ? AND user_id = ?",
                    (group_id, user_id),
                ).fetchone()
                last_dj = float(pres["last_dejavu_at"] or 0.0) if pres is not None else 0.0
                if last_dj > 0.0 and last_dj > cutoff_cd:
                    return []
                rows = conn.execute(
                    """
                    SELECT * FROM utterances
                    WHERE group_id = ? AND user_id = ? AND triggered = 0 AND created_at <= ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (group_id, user_id, float(before_ts), max(1, int(limit))),
                ).fetchall()
                return [self._row_to_utterance(r) for r in rows]
            finally:
                conn.close()

    def mark_utterance_triggered(self, utterance_id: int, group_id: str, user_id: str, now_ts: float) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("UPDATE utterances SET triggered = 1 WHERE id = ? AND group_id = ?", (int(utterance_id), group_id))
                    conn.execute(
                        "UPDATE member_presence SET last_dejavu_at = ? WHERE group_id = ? AND user_id = ?",
                        (float(now_ts), group_id, user_id),
                    )
            finally:
                conn.close()

    # --- Member Presence & Reunion ---

    def touch_member_and_check_reunion(
        self,
        *,
        group_id: str,
        user_id: str,
        user_name: str,
        now_ts: float,
        min_absent_days: int = 15,
        min_history_messages: int = 10,
        cooldown_days: int = 30,
        allow_trigger: bool = True,
    ) -> tuple[PresenceRecord, int | None]:
        now = float(now_ts)
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    row = conn.execute(
                        "SELECT * FROM member_presence WHERE group_id = ? AND user_id = ?",
                        (group_id, user_id),
                    ).fetchone()
                    if row is None:
                        conn.execute(
                            """
                            INSERT INTO member_presence (group_id, user_id, user_name, first_seen, last_seen, msg_count)
                            VALUES (?, ?, ?, ?, ?, 1)
                            """,
                            (group_id, user_id, user_name or user_id, now, now),
                        )
                        new_row = conn.execute(
                            "SELECT * FROM member_presence WHERE group_id = ? AND user_id = ?",
                            (group_id, user_id),
                        ).fetchone()
                        assert new_row is not None
                        return self._row_to_presence(new_row), None

                    old_last = float(row["last_seen"])
                    old_cnt = int(row["msg_count"])
                    old_reunion = float(row["last_reunion_at"])
                    absent_sec = now - old_last
                    absent_days: int | None = None
                    new_reunion = old_reunion

                    if (
                        allow_trigger
                        and old_cnt >= max(1, int(min_history_messages))
                        and absent_sec >= max(1, int(min_absent_days)) * 86400.0
                        and (old_reunion <= 0.0 or (now - old_reunion) >= max(1, int(cooldown_days)) * 86400.0)
                    ):
                        absent_days = int(absent_sec // 86400.0)
                        new_reunion = now

                    conn.execute(
                        """
                        UPDATE member_presence
                        SET user_name = ?, last_seen = MAX(last_seen, ?), msg_count = msg_count + 1, last_reunion_at = ?
                        WHERE group_id = ? AND user_id = ?
                        """,
                        (user_name or str(row["user_name"]), now, new_reunion, group_id, user_id),
                    )
                    upd = conn.execute(
                        "SELECT * FROM member_presence WHERE group_id = ? AND user_id = ?",
                        (group_id, user_id),
                    ).fetchone()
                    assert upd is not None
                    return self._row_to_presence(upd), absent_days
            finally:
                conn.close()

    # --- Backfill & Stats ---

    def is_group_backfilled(self, group_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT 1 FROM backfill_state WHERE group_id = ?", (group_id,)).fetchone()
                return row is not None
            finally:
                conn.close()

    def bulk_import_history(
        self,
        group_id: str,
        *,
        presence_rows: Iterable[tuple[str, str, float, float, int]],
        utterance_rows: Iterable[tuple[str, str, str, float]],
        total_scanned: int,
        completed_at: float | None = None,
        max_utterances_per_group: int = 4000,
    ) -> tuple[int, int]:
        """Import historical member presence and candidate utterances idempotently."""
        now = float(completed_at if completed_at is not None else time.time())
        p_count = 0
        u_count = 0
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    for uid, uname, fseen, lseen, mcnt in presence_rows:
                        conn.execute(
                            """
                            INSERT INTO member_presence (group_id, user_id, user_name, first_seen, last_seen, msg_count)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(group_id, user_id) DO UPDATE SET
                                user_name = CASE WHEN excluded.user_name != '' THEN excluded.user_name ELSE member_presence.user_name END,
                                first_seen = MIN(member_presence.first_seen, excluded.first_seen),
                                last_seen = MAX(member_presence.last_seen, excluded.last_seen),
                                msg_count = member_presence.msg_count + excluded.msg_count
                            """,
                            (group_id, uid, uname or uid, float(fseen), float(lseen), int(mcnt)),
                        )
                        p_count += 1

                    for uid, uname, text, ts in utterance_rows:
                        clean = str(text or "").strip()[:200]
                        if not clean:
                            continue
                        dup = conn.execute(
                            "SELECT 1 FROM utterances WHERE group_id = ? AND user_id = ? AND text = ? LIMIT 1",
                            (group_id, uid, clean),
                        ).fetchone()
                        if dup is not None:
                            continue
                        conn.execute(
                            """
                            INSERT INTO utterances (group_id, user_id, user_name, text, created_at, embedding, triggered)
                            VALUES (?, ?, ?, ?, ?, NULL, 0)
                            """,
                            (group_id, uid, uname or uid, clean, float(ts)),
                        )
                        u_count += 1

                    cap = max(100, int(max_utterances_per_group))
                    cnt_row = conn.execute("SELECT COUNT(*) AS c FROM utterances WHERE group_id = ?", (group_id,)).fetchone()
                    cnt = int(cnt_row["c"]) if cnt_row else 0
                    if cnt > cap:
                        conn.execute(
                            """
                            DELETE FROM utterances WHERE id IN (
                                SELECT id FROM utterances WHERE group_id = ?
                                ORDER BY created_at ASC, id ASC LIMIT ?
                            )
                            """,
                            (group_id, cnt - cap),
                        )

                    conn.execute(
                        """
                        INSERT INTO backfill_state (group_id, completed_at, imported_messages)
                        VALUES (?, ?, ?)
                        ON CONFLICT(group_id) DO UPDATE SET completed_at = excluded.completed_at, imported_messages = excluded.imported_messages
                        """,
                        (group_id, now, int(total_scanned)),
                    )
                    return p_count, u_count
            finally:
                conn.close()

    def get_group_stats(self, group_id: str) -> dict[str, int]:
        with self._lock:
            conn = self._connect()
            try:
                fu_pending = conn.execute(
                    "SELECT COUNT(*) AS c FROM followups WHERE group_id = ? AND status = 'pending'", (group_id,)
                ).fetchone()["c"]
                fu_asked = conn.execute(
                    "SELECT COUNT(*) AS c FROM followups WHERE group_id = ? AND status = 'asked'", (group_id,)
                ).fetchone()["c"]
                utt_cnt = conn.execute(
                    "SELECT COUNT(*) AS c FROM utterances WHERE group_id = ?", (group_id,)
                ).fetchone()["c"]
                mem_cnt = conn.execute(
                    "SELECT COUNT(*) AS c FROM member_presence WHERE group_id = ?", (group_id,)
                ).fetchone()["c"]
                return {
                    "pending_followups": int(fu_pending),
                    "asked_followups": int(fu_asked),
                    "utterances": int(utt_cnt),
                    "members": int(mem_cnt),
                }
            finally:
                conn.close()
