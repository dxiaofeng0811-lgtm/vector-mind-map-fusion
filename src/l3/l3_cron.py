#!/usr/bin/env python3
"""
L3 Cron Entry Point
被 OpenClaw cron 触发，执行完整的 L3 pipeline。
触发时间：每两天 03:00（Asia/Shanghai）

路径基于项目根目录（不依赖 /workspace/fusion）
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src" / "l3"))
os.chdir(str(PROJECT_ROOT / "src" / "l3"))

from l3_biweekly_consolidate import run


def main():
    print(f"[L3 Cron] 开始执行: {datetime.now().isoformat()}")
    run()
    print(f"[L3 Cron] 结束: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()