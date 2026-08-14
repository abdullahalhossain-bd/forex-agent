import sys, importlib, os
sys.path.insert(0, '.')
mod = importlib.import_module('core.constants')
print('core.constants file ->', getattr(mod, '__file__', None))
print('MIN_RR_PROD ->', getattr(mod, 'MIN_RR_PROD', None))
print('env MIN_RR_PROD ->', repr(os.getenv('MIN_RR_PROD')))
print('\nmodules containing "constants":')
for k, v in sys.modules.items():
    if 'constants' in k:
        print(k, getattr(v, '__file__', None))
