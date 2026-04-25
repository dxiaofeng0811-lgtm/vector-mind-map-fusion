#!/usr/bin/env python3
"""
L2 Cron Entry Point
被 OpenClaw cron 触发，执行完整的 L2 pipeline。
触发时间：每天 00:30（Asia/Shanghai）
"""

import sys
import os

sys.path.insert(0, "/workspace/fusion/l2")
os.chdir("/workspace/fusion/l2")

from l2_daily import run
from datetime import datetime

def main():
    print(f"[L2 Cron] 开始执行: {datetime.now().isoformat()}")
    run()
    print(f"[L2 Cron] 结束: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()