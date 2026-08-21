# -*- coding: utf-8 -*-
"""Corrupt paper ledger must not Bitget top-up an already-open live seat."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import hl_bitget_executor as ex
from utils import hl_paper_copy as pc


class PaperAtomicIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self._path_patch = mock.patch.object(pc, "_data_dir", return_value=self.root)
        self._path_patch.start()
        self.addCleanup(self._path_patch.stop)
        self.addCleanup(self._td.cleanup)

    def test_atomic_save_survives_and_makes_bak(self):
        data = {"bots": {"bot_c": {"id": "bot_c", "paper_cleared_for_live": True}}}
        with mock.patch.object(pc, "_ensure_bots", side_effect=lambda d: d), mock.patch.object(
            pc, "_aggregate", side_effect=lambda d: d
        ):
            pc.save_paper(dict(data))
            pc.save_paper({"bots": {"bot_c": {"id": "bot_c", "paper_cleared_for_live": True, "x": 1}}})
        primary = pc._path()
        bak = pc._bak_path()
        self.assertTrue(primary.is_file())
        self.assertTrue(bak.is_file())
        self.assertIsNotNone(pc._try_read_paper_dict(primary))
        self.assertIsNotNone(pc._try_read_paper_dict(bak))

    def _good_book(self) -> dict:
        return {
            "bots": {
                "bot_c": {
                    "id": "bot_c",
                    "live_only": True,
                    "copy_current": False,
                    "paper_cleared_for_live": True,
                    "target_av": 250000,
                    "target_positions": {"BTC": {"sz": -15.0}},
                }
            },
            "updated_at": "t0",
        }

    def test_corrupt_primary_recovers_from_bak(self):
        good = self._good_book()
        # Two writes: first creates primary, second refreshes .bak from good primary.
        pc._atomic_write_json(pc._path(), good)
        pc._atomic_write_json(pc._path(), good)
        # Destroy primary the way the production bug looked.
        pc._path().write_text("", encoding="utf-8")

        align: list[str] = []
        with mock.patch.object(pc, "load_watchlist", return_value=[]), mock.patch.object(
            pc, "paper_config", return_value={"bot_balance": 1000.0}
        ), mock.patch.object(
            pc, "_queue_live_align", side_effect=lambda ids: align.extend(ids)
        ), mock.patch.object(pc, "_aggregate", side_effect=lambda d: d):
            book = pc.load_paper()
        bot = (book.get("bots") or {}).get("bot_c") or {}
        self.assertTrue(bot.get("paper_cleared_for_live"))
        self.assertEqual(align, [])
        # Primary restored; bak still readable.
        self.assertIsNotNone(pc._try_read_paper_dict(pc._path()))
        self.assertIsNotNone(pc._try_read_paper_dict(pc._bak_path()))

    def test_missing_primary_recovers_from_bak(self):
        good = self._good_book()
        pc._bak_path().write_text(json.dumps(good), encoding="utf-8")
        self.assertFalse(pc._path().exists())
        with mock.patch.object(pc, "load_watchlist", return_value=[]), mock.patch.object(
            pc, "paper_config", return_value={"bot_balance": 1000.0}
        ), mock.patch.object(pc, "_queue_live_align"), mock.patch.object(
            pc, "_aggregate", side_effect=lambda d: d
        ):
            book = pc.load_paper()
        self.assertTrue(
            ((book.get("bots") or {}).get("bot_c") or {}).get("paper_cleared_for_live")
        )
        self.assertIsNotNone(pc._try_read_paper_dict(pc._path()))

    def test_unrecovered_save_does_not_clobber_bak(self):
        good = self._good_book()
        pc._bak_path().write_text(json.dumps(good), encoding="utf-8")
        shell = {
            "bots": {},
            "updated_at": "x",
            "error": "paper_corrupt_unrecovered",
        }
        with mock.patch.object(pc, "_ensure_bots", side_effect=lambda d: d), mock.patch.object(
            pc, "_aggregate", side_effect=lambda d: d
        ):
            pc.save_paper(shell)
        # Bak untouched; primary not overwritten with empty shell.
        self.assertFalse(pc._path().exists())
        self.assertEqual(
            json.loads(pc._bak_path().read_text(encoding="utf-8"))["bots"]["bot_c"][
                "paper_cleared_for_live"
            ],
            True,
        )


class EnterLiveNoAlignTests(unittest.TestCase):
    def test_copy_current_off_skips_align_queue(self):
        data = {"bots": {}}
        w = {
            "id": "bot_c",
            "address": "0xd05d2e015b9ed6f17f2111cf1ac7ae229155816e",
            "copy_current": False,
        }
        queued: list[str] = []
        with mock.patch.object(pc, "load_watchlist", return_value=[w]), mock.patch.object(
            pc, "paper_config", return_value={"bot_balance": 1000.0}
        ), mock.patch.object(
            pc,
            "_queue_live_align",
            side_effect=lambda ids: queued.extend(ids),
        ), mock.patch(
            "utils.hl_bitget_subaccounts.seat_enabled_by_env",
            return_value=True,
        ), mock.patch(
            "utils.hl_bitget_subaccounts.route_id_for_bot",
            return_value="C",
        ):
            out = pc._ensure_bots(data)
        self.assertTrue((out.get("bots") or {}).get("bot_c", {}).get("paper_cleared_for_live"))
        self.assertEqual(queued, [])

    def test_copy_current_off_to_on_queues_sync_align(self):
        data = {
            "bots": {
                "bot_c": {
                    "id": "bot_c",
                    "address": "0xd05d2e015b9ed6f17f2111cf1ac7ae229155816e",
                    "live_only": True,
                    "paper_cleared_for_live": True,
                    "copy_current": False,
                    "positions": {},
                    "fills": [],
                    "balance": 0.0,
                    "equity": 0.0,
                    "realized_pnl": 0.0,
                    "paper_balance": 0.0,
                }
            }
        }
        w = {
            "id": "bot_c",
            "address": "0xd05d2e015b9ed6f17f2111cf1ac7ae229155816e",
            "copy_current": True,
        }
        queued: list[str] = []
        with mock.patch.object(pc, "load_watchlist", return_value=[w]), mock.patch.object(
            pc, "paper_config", return_value={"bot_balance": 1000.0}
        ), mock.patch.object(
            pc,
            "_queue_live_align",
            side_effect=lambda ids: queued.extend(ids),
        ), mock.patch(
            "utils.hl_bitget_subaccounts.seat_enabled_by_env",
            return_value=True,
        ), mock.patch(
            "utils.hl_bitget_subaccounts.route_id_for_bot",
            return_value="C",
        ):
            out = pc._ensure_bots(data)
        self.assertTrue((out.get("bots") or {}).get("bot_c", {}).get("copy_current"))
        self.assertEqual(queued, ["bot_c"])


class AlignNoIncreaseGateTests(unittest.TestCase):
    def setUp(self) -> None:
        with ex._pending_fresh_lock:
            ex._pending_fresh_opens.clear()
        ex._pending_fresh_loaded = True
        self._persist_patch = mock.patch.object(ex, "_persist_pending_fresh_opens")
        self._persist_patch.start()
        self.addCleanup(self._persist_patch.stop)

    def test_align_clamps_top_up(self):
        """The C incident: align want=0.037 have=0.0204 → keep have."""
        bot = {"id": "bot_c", "live_only": True, "copy_current": False}
        out = ex._gate_desired_no_copy_current(
            bot,
            {"BTCUSDT": -0.0372},
            {"BTCUSDT": -0.0204},
            [{"id": "align-bot_c", "action": "live_align", "source": "bot_c"}],
            account_id="C",
        )
        self.assertAlmostEqual(out.get("BTCUSDT") or 0.0, -0.0204)

    def test_fill_sync_without_delta_does_not_top_up(self):
        """live_sync row missing target_delta is not a size-up signal."""
        bot = {"id": "bot_c", "live_only": True, "copy_current": False}
        with mock.patch.object(ex, "hl_coin_to_bitget", return_value="BTCUSDT"):
            out = ex._gate_desired_no_copy_current(
                bot,
                {"BTCUSDT": -0.0372},
                {"BTCUSDT": -0.0204},
                [{"action": "live_sync", "source": "bot_c", "coin": "BTC"}],
                account_id="C",
            )
        self.assertAlmostEqual(out.get("BTCUSDT") or 0.0, -0.0204)

    def test_fill_sync_with_delta_allows_top_up(self):
        bot = {"id": "bot_c", "live_only": True, "copy_current": False}
        with mock.patch.object(ex, "hl_coin_to_bitget", return_value="BTCUSDT"):
            out = ex._gate_desired_no_copy_current(
                bot,
                {"BTCUSDT": -0.0372},
                {"BTCUSDT": -0.0204},
                [
                    {
                        "action": "live_sync",
                        "source": "bot_c",
                        "coin": "BTC",
                        "target_delta": -6.0,
                        "dir": "Open Short",
                        "start_position": -9.0,
                    }
                ],
                account_id="C",
            )
        self.assertAlmostEqual(out.get("BTCUSDT") or 0.0, -0.0372)

    def test_c_incident_rebuild_align_holds(self):
        """Exact 2026-08-05 failure mode: corrupt paper → align must not sell again."""
        bot = {"id": "bot_c", "live_only": True, "copy_current": False}
        out = ex._gate_desired_no_copy_current(
            bot,
            {"BTCUSDT": -0.0372},
            {"BTCUSDT": -0.02047678},
            [
                {
                    "id": "align-bot_c",
                    "action": "live_align",
                    "source": "bot_c",
                    "bot_id": "bot_c",
                    "live_only": True,
                }
            ],
            account_id="C",
        )
        self.assertAlmostEqual(out.get("BTCUSDT") or 0.0, -0.02047678)

    def test_align_still_allows_reduce(self):
        bot = {"id": "bot_c", "live_only": True, "copy_current": False}
        out = ex._gate_desired_no_copy_current(
            bot,
            {"BTCUSDT": -0.01},
            {"BTCUSDT": -0.0204},
            [{"id": "align-bot_c", "action": "live_align"}],
            account_id="C",
        )
        self.assertAlmostEqual(out.get("BTCUSDT") or 0.0, -0.01)

    def test_prune_paper_only_does_not_queue_live_flatten(self):
        """Shrinking watchlist to bot_c must not flatten Bitget for any retiree."""
        data = {
            "bots": {
                "bot_c": {
                    "id": "bot_c",
                    "address": "0xd05d2e015b9ed6f17f2111cf1ac7ae229155816e",
                    "live_only": True,
                    "paper": False,
                    "paper_cleared_for_live": True,
                    "copy_current": False,
                    "positions": {},
                    "fills": [],
                    "balance": 0.0,
                    "equity": 0.0,
                    "realized_pnl": 0.0,
                },
                "bot_a": {
                    "id": "bot_a",
                    "address": "0x20bc9cd229dfd681740834d9b4f55641ce435da3",
                    "paper": True,
                    "live_only": False,
                    "positions": {"BTC": {"szi": 0.01}},
                    "fills": [],
                    "balance": 1000.0,
                    "equity": 1000.0,
                    "realized_pnl": 0.0,
                },
                "bot_b": {
                    "id": "bot_b",
                    "address": "0xb315067ae6b8ae6dfafd052b630d95b72a91cc25",
                    "live_only": True,
                    "paper": False,
                    "paper_cleared_for_live": True,
                    "positions": {},
                    "fills": [],
                    "balance": 0.0,
                    "equity": 0.0,
                },
            }
        }
        w = {
            "id": "bot_c",
            "address": "0xd05d2e015b9ed6f17f2111cf1ac7ae229155816e",
            "copy_current": False,
        }
        flat_q: list = []
        align_q: list = []
        with mock.patch.object(pc, "load_watchlist", return_value=[w]), mock.patch.object(
            pc, "paper_config", return_value={"bot_balance": 1000.0}
        ), mock.patch.object(
            pc,
            "_queue_live_flatten",
            side_effect=lambda ids, reason="leave_live": flat_q.append((list(ids), reason)),
        ), mock.patch.object(
            pc,
            "_queue_live_align",
            side_effect=lambda ids: align_q.extend(ids),
        ), mock.patch.object(
            pc, "_rebase_desk_peak_anchor", return_value=None
        ), mock.patch(
            "utils.hl_bitget_subaccounts.seat_enabled_by_env",
            return_value=True,
        ), mock.patch(
            "utils.hl_bitget_subaccounts.route_id_for_bot",
            return_value="C",
        ):
            out = pc._ensure_bots(data)
        self.assertEqual(set((out.get("bots") or {}).keys()), {"bot_c"})
        self.assertEqual(flat_q, [])
        self.assertEqual(align_q, [])
        # C stays live_only + copy_current off — no re-enter align
        c = out["bots"]["bot_c"]
        self.assertTrue(c.get("live_only"))
        self.assertTrue(c.get("paper_cleared_for_live"))
        self.assertFalse(pc._bot_copy_current(c))

    def test_prune_reason_is_noop(self):
        before = list(pc._pending_live_flatten)
        pc._queue_live_flatten(["bot_a", "bot_b"], reason="prune")
        self.assertEqual(pc._pending_live_flatten, before)

    def test_hard_portfolio_rebase_skips_live_only(self):
        book = {
            "bots": {
                "bot_c": {
                    "id": "bot_c",
                    "live_only": True,
                    "paper": False,
                    "positions": {},
                    "fills": [],
                    "balance": 0.0,
                    "equity": 0.0,
                }
            },
            "portfolio_anchor_equity": 5000.0,
            "portfolio_peak_equity": 5000.0,
        }
        rows = pc._hard_portfolio_rebase(
            book,
            {},
            pc.paper_config(),
            reason="portfolio_peak_dd",
            ret=-0.2,
            anchor=5000.0,
            equity=0.0,
        )
        self.assertEqual(rows, [])
        self.assertEqual(book["bots"]["bot_c"].get("live_only"), True)


if __name__ == "__main__":
    unittest.main()
