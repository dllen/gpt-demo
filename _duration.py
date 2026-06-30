#!/usr/bin/env python3
"""辅助脚本：传入 epochs [device]，输出预估耗时"""
import sys

epochs = int(sys.argv[1])
device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
steps_per_epoch = 76

# CPU 实测 ~1.2s/step; GPU (CuPy) 实测 ~0.05s/step (V100 级别)
sec_per_step = 0.05 if device == 'gpu' else 1.2

total_sec = epochs * steps_per_epoch * sec_per_step
h = int(total_sec // 3600)
m = int((total_sec % 3600) // 60)

if h > 0:
    print(f"{h}h{m:02d}m")
else:
    print(f"{m}m")
