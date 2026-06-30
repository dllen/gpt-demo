#!/usr/bin/env python3
"""辅助脚本：传入 epochs，输出预估耗时"""
import sys

epochs = int(sys.argv[1])
steps_per_epoch = 76
sec_per_step = 1.2  # 实测

total_sec = epochs * steps_per_epoch * sec_per_step
h = int(total_sec // 3600)
m = int((total_sec % 3600) // 60)

if h > 0:
    print(f"{h}h{m:02d}m")
else:
    print(f"{m}m")
