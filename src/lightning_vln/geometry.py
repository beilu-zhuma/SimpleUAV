"""兼容入口；实现只有一份。"""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module('lightning_vln.core.geometry')
