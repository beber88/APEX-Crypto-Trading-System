"""Strategy performance statistics repository.

Queries the trades database (SQLite or TimescaleDB) to compute
per-strategy stats used by PositionSizingEngine for Kelly sizing
and anti-martingale adjustments.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("risk.stats_repository")


class StatsRepository:
    """Reads strategy performance from the trades database.

    Supports SQLite (local) — can be extended to TimescaleDB by passing
    a psycopg2-compatible connection.

    Attributes:
        sqlite_path: Path to the SQLite trade log database.
    """

    def __init__(self, config: dict) -> None:
        """Initialise from the data config section.

        Args:
            config: The ``data`` section of config.yaml. Expects
                ``sqlite_path`` key.
        """
        self.sqlite_path: str = config.get("sqlite_path", "./data/trades.db")
        self._conn: Optional[sqlite3.Connection] = None

        log_with_data(logger, "info", "StatsRepository initialised", {
            "sqlite_path": self.sqlite_path,
        })

    def _get_conn(self) -> sqlite3.Connection:
        """Lazy-open the SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.sqlite_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ------------------------------------------------------------------
    # Per-strategy stats
    # ------------------------------------------------------------------

    def get_strategy_stats(self, strategy_name: str) -> Optional[dict[str, Any]]:
        """Query rolling performance stats for a single strategy.

        Returns:
            Dictionary with win_rate, avg_win_r, avg_loss_r,
            recent_win_streak, recent_loss_streak.
            None if no closed trades exist for this strategy.
        """
        try:
            conn = self._get_conn()
            cur = conn.cursor()

            # Win rate and average R-multiples from last 1000 closed trades
            cur.execute(
                """
                SELECT
                    AVG(CASE WHEN r_multiple > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                    AVG(CASE WHEN r_multiple > 0 THEN r_multiple END) AS avg_win_r,
                    ABS(AVG(CASE WHEN r_multiple <= 0 THEN r_multiple END)) AS avg_loss_r,
                    COUNT(*) AS total_trades
                FROM trades
                WHERE strategy = ?
                  AND status = 'closed'
                  AND r_multiple IS NOT NULL
                ORDER BY exit_time DESC
                LIMIT 1000
                """,
                (strategy_name,),
            )
            row = cur.fetchone()

            if not row or row["total_trades"] == 0 or row["win_rate"] is None:
                log_with_data(logger, "debug",
                              "No closed trades for strategy", {
                                  "strategy": strategy_name,
                              })
                return None

            win_rate = float(row["win_rate"])
            avg_win_r = float(row["avg_win_r"] or 1.0)
            avg_loss_r = float(row["avg_loss_r"] or 1.0)

            # Calculate recent streaks from last 20 trades
            win_streak, loss_streak = self._compute_streaks(cur, strategy_name)

            result = {
                "win_rate": win_rate,
                "avg_win_r": avg_win_r,
                "avg_loss_r": avg_loss_r,
                "recent_win_streak": win_streak,
                "recent_loss_streak": loss_streak,
                "total_trades": int(row["total_trades"]),
            }

            log_with_data(logger, "debug", "Strategy stats loaded", {
                "strategy": strategy_name,
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in result.items()},
            })

            return result

        except Exception as exc:
            log_with_data(logger, "warning", "Failed to load strategy stats", {
                "strategy": strategy_name,
                "error": str(exc),
            })
            return None

    def _compute_streaks(
        self, cur: sqlite3.Cursor, strategy_name: str
    ) -> tuple[int, int]:
        """Compute current win/loss streak from recent trades."""
        cur.execute(
            """
            SELECT r_multiple
            FROM trades
            WHERE strategy = ?
              AND status = 'closed'
              AND r_multiple IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT 20
            """,
            (strategy_name,),
        )
        rows = cur.fetchall()

        if not rows:
            return 0, 0

        # Count streak from most recent trade
        first_is_win = rows[0]["r_multiple"] > 0
        streak = 0
        for row in rows:
            is_win = row["r_multiple"] > 0
            if is_win == first_is_win:
                streak += 1
            else:
                break

        if first_is_win:
            return streak, 0
        else:
            return 0, streak

    # ------------------------------------------------------------------
    # All strategies summary
    # ------------------------------------------------------------------

    def get_all_strategy_stats(self) -> dict[str, dict[str, Any]]:
        """Load stats for all strategies that have closed trades.

        Returns:
            Dictionary mapping strategy_name → stats dict.
        """
        try:
            conn = self._get_conn()
            cur = conn.cursor()

            cur.execute(
                """
                SELECT DISTINCT strategy
                FROM trades
                WHERE status = 'closed'
                  AND r_multiple IS NOT NULL
                """,
            )
            strategies = [row["strategy"] for row in cur.fetchall()]

            result = {}
            for name in strategies:
                stats = self.get_strategy_stats(name)
                if stats:
                    result[name] = stats

            log_with_data(logger, "info", "All strategy stats loaded", {
                "strategies_with_data": len(result),
            })

            return result

        except Exception as exc:
            log_with_data(logger, "warning", "Failed to load all strategy stats", {
                "error": str(exc),
            })
            return {}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
