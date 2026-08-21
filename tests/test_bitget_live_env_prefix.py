# -*- coding: utf-8 -*-
"""Enabled seats without SUB_* keys must fall back to main BITGET_*."""

from __future__ import annotations

import unittest
from unittest import mock

from utils import hl_bitget_subaccounts as sub


class _Creds:
    def __init__(self, ok: bool):
        self._ok = ok

    def ok(self) -> bool:
        return self._ok


class LiveEnvPrefixTests(unittest.TestCase):
    def setUp(self) -> None:
        sub._fallback_logged.clear()

    def test_enabled_missing_sub_falls_back_to_main(self):
        def load(prefix: str = ""):
            p = (prefix or "").strip()
            if p in ("", "BITGET"):
                return _Creds(True)
            return _Creds(False)

        with mock.patch(
            "quant.engine.exchanges.bitget.account.load_creds_from_env",
            side_effect=load,
        ):
            got = sub.resolve_live_env_prefix(
                "BITGET_SUB_C", route_id="C", enabled=True
            )
        self.assertEqual(got, "BITGET")

    def test_enabled_uses_sub_when_present(self):
        def load(prefix: str = ""):
            p = (prefix or "").strip()
            return _Creds(p == "BITGET_SUB_C")

        with mock.patch(
            "quant.engine.exchanges.bitget.account.load_creds_from_env",
            side_effect=load,
        ):
            got = sub.resolve_live_env_prefix(
                "BITGET_SUB_C", route_id="C", enabled=True
            )
        self.assertEqual(got, "BITGET_SUB_C")

    def test_disabled_keeps_configured(self):
        with mock.patch(
            "quant.engine.exchanges.bitget.account.load_creds_from_env",
            return_value=_Creds(True),
        ):
            got = sub.resolve_live_env_prefix(
                "BITGET_SUB_C", route_id="C", enabled=False
            )
        self.assertEqual(got, "BITGET_SUB_C")

    def test_enabled_routes_refuse_shared_main(self):
        routes = [
            sub.SubAccountRoute(
                id="C",
                label="C",
                bot_id="bot_c",
                coins=None,
                enabled=True,
                env_prefix="BITGET",
                scale=1.0,
            ),
            sub.SubAccountRoute(
                id="J",
                label="J",
                bot_id="bot_j",
                coins=None,
                enabled=True,
                env_prefix="BITGET",
                scale=1.0,
            ),
        ]
        with mock.patch.object(sub, "parse_routes", return_value=routes):
            with mock.patch.object(sub, "max_subaccounts", return_value=10):
                self.assertEqual(sub.enabled_routes(), [])

    def test_validate_flags_shared_prefix(self):
        routes = [
            sub.SubAccountRoute(
                id="C",
                label="C",
                bot_id="bot_c",
                coins=None,
                enabled=True,
                env_prefix="BITGET",
                scale=1.0,
            ),
            sub.SubAccountRoute(
                id="J",
                label="J",
                bot_id="bot_j",
                coins=None,
                enabled=True,
                env_prefix="",
                scale=1.0,
            ),
        ]
        problems = sub.validate_routes(routes)
        self.assertTrue(any("share API prefix" in p for p in problems))

    def test_flatten_refuses_shared_main_with_live_survivor(self):
        """Pruning bot_a must not resolve to BITGET_* while bot_c is live on it."""
        live_c = sub.SubAccountRoute(
            id="C",
            label="C",
            bot_id="bot_c",
            coins=None,
            enabled=True,
            env_prefix="BITGET",
            scale=1.0,
        )
        raw_a = sub.SubAccountRoute(
            id="A",
            label="A",
            bot_id="bot_a",
            coins=None,
            enabled=False,
            env_prefix="BITGET_SUB_A",
            scale=1.0,
        )

        def load(prefix: str = ""):
            p = (prefix or "").strip()
            if p in ("", "BITGET"):
                return _Creds(True)
            return _Creds(False)

        with mock.patch.object(sub, "route_for_bot_any", return_value=raw_a), mock.patch.object(
            sub, "enabled_routes", return_value=[live_c]
        ), mock.patch(
            "quant.engine.exchanges.bitget.account.load_creds_from_env",
            side_effect=load,
        ):
            self.assertIsNone(sub.route_for_flatten("bot_a"))

    def test_flatten_allows_own_disabled_seat_when_no_clash(self):
        raw_c = sub.SubAccountRoute(
            id="C",
            label="C",
            bot_id="bot_c",
            coins=None,
            enabled=False,
            env_prefix="BITGET",
            scale=1.0,
        )

        def load(prefix: str = ""):
            return _Creds((prefix or "").strip() in ("", "BITGET"))

        with mock.patch.object(sub, "route_for_bot_any", return_value=raw_c), mock.patch.object(
            sub, "enabled_routes", return_value=[]
        ), mock.patch(
            "quant.engine.exchanges.bitget.account.load_creds_from_env",
            side_effect=load,
        ):
            got = sub.route_for_flatten("bot_c")
        self.assertIsNotNone(got)
        self.assertEqual(got.env_prefix, "BITGET")
        self.assertTrue(got.enabled)


if __name__ == "__main__":
    unittest.main()
