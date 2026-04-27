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

from l3_biweekly_consolidate import run, L3Processor, INFINITYDB_DIR, InfinityDBLite
from utils.cost_tracker import track_layer
from l3_manifest import write_manifest


def main():
    t0 = time_module.time()
    print(f"[L3 Cron] 开始执行: {datetime.now().isoformat()}")

    # 记录运行前的 InfinityDB 节点数
    infinitydb_before = InfinityDBLite(str(INFINITYDB_DIR))
    nodes_before = len(infinitydb_before.data.get("neurons", {}))
    del infinitydb_before

    stats = run()
    duration_ms = int((time_module.time() - t0) * 1000)

    # 记录运行后的 InfinityDB 节点数
    infinitydb_after = InfinityDBLite(str(INFINITYDB_DIR))
    nodes_after = len(infinitydb_after.data.get("neurons", {}))
    del infinitydb_after

    # 写 cost tracker
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
        track_layer(
            layer="l3",
            chunks_in=0,
            chunks_out=0,
            duration_ms=duration_ms,
        )

    # 写 manifest
    l2_files_cleared = stats.get("l2_files_cleared", []) if stats else []
    stats["duration_ms"] = duration_ms
    write_manifest(
        l3_stats=stats,
        infinitydb_nodes_before=nodes_before,
        infinitydb_nodes_after=nodes_after,
        l2_files_cleared=l2_files_cleared,
    )
    print(f"[L3 Cron] manifest 已写入")

    print(f"[L3 Cron] 结束: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
