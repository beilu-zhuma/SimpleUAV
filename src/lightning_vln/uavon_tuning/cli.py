"""旧模块名兼容入口；正式实现位于适配器。"""
import importlib as _importlib
import sys as _sys
_target = _importlib.import_module('lightning_vln.adapters.uavon.tuning.cli')
if __name__ == '__main__':
    _target.main()
else:
    _sys.modules[__name__] = _target
