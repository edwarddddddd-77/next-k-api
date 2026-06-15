"""Pickle 兼容：旧 .pkl 内模块名 orb.breakout_gbm。"""
import orb.ml.gbm as _impl
import sys

sys.modules[__name__] = _impl
