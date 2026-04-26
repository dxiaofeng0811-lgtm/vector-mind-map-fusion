#!/usr/bin/env python3
"""
L3 Cron Entry Point
被 OpenClaw cron 触发，执行完整的 L3 pipeline。
触发时间：每两天 03:00（Asia/Shanghai）

路径基于项目根目录（不依赖 /workspace/fusion）
"""

import sys
import os
import time as time_module
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src" / "l3"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT / "src" / "l3"))

from l3_biweekly_consolidate import run
from utils.cost_tracker import track_layer


def main():
    t0 = time_module.time()
    print(f"[L3 Cron] 开始执行: {datetime.now().isoformat()}")

    stats = run()
    duration_ms = int((time_module.time() - t0) * 1000)

    if stats and stats.get("neurons_written", 0) > 0:
        track_layer(
            layer="l3",
            ollama_calls=stats.get("ollama_calls", 0),
            tokens_approx=stats.get("tokens_approx", 0),
            chunks_in=stats.get("chunks_in", 0),
            chunks_out=stats.get("neurons_written", 0),
            duration_ms=duration_ms,
            extra={
                "schemas_written": stats.get("schemas_written", 0),
                "relations_written": stats.get("relations_written", 0),
            },
        )
    else:
        # 无 chunks 也记录一条
        track_layer(
            layer="l3",
            chunks_in=0,
            chunks_out=0,
            duration_ms=duration_ms,
        )

    print(f"[L3 Cron] 结束: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
