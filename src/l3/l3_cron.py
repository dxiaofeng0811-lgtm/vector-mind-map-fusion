#!/usr/bin/env python3
"""
L3 Cron Entry Point
被 OpenClaw cron 触发，执行完整的 L3 pipeline。
触发时间：每两天 03:00（Asia/Shanghai）
"""

import sys
import os

sys.path.insert(0, "/workspace/fusion/l3")
os.chdir("/workspace/fusion/l3")

from l3_biweekly_consolidate import run
from datetime import datetime

def main():
    print(f"[L3 Cron] 开始执行: {datetime.now().isoformat()}")
    run()
    print(f"[L3 Cron] 结束: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()