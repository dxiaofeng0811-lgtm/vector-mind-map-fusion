#!/usr/bin/env python3
"""
L1 Cron Entry Point
被 OpenClaw cron 触发，执行完整的 L1 pipeline。
触发时间：每天 00:30（Asia/Shanghai）

路径基于项目根目录（不依赖 /workspace/fusion）
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src" / "l1"))
os.chdir(str(PROJECT_ROOT / "src" / "l1"))

from scan_sessions_incremental import ByteOffsetScanner
from l1_classifier import L1Classifier, save_to_l2a

RAW_CHUNKS_TMP_FILE = str(PROJECT_ROOT / "memory" / "_state" / "l1_raw_chunks_tmp.jsonl")


def main():
    print(f"[L1 Cron] 开始执行: {datetime.now().isoformat()}")

    scanner = ByteOffsetScanner()
    raw_chunks = scanner.scan()

    if raw_chunks:
        classifier = L1Classifier()
        processed = classifier.process(raw_chunks)

        if processed:
            drop_count = sum(1 for c in processed if c.get("_dropped"))
            new_count = sum(1 for c in processed if c.get("dedup_level") == 0)
            print(f"[L1 Cron] 处理完成: {new_count} 新增, {drop_count} 过滤")
            date_str = datetime.now().strftime("%Y-%m-%d")
            save_to_l2a(processed, date_str)
        else:
            print("[L1 Cron] 无有效 chunks")
    else:
        if os.path.exists(RAW_CHUNKS_TMP_FILE):
            print(f"[L1 Cron] 发现 stale tmp，尝试恢复...")
            classifier = L1Classifier()
            processed = classifier.process_from_tmp()

            if processed:
                drop_count = sum(1 for c in processed if c.get("_dropped"))
                new_count = sum(1 for c in processed if c.get("dedup_level") == 0)
                print(f"[L1 Cron] 恢复完成: {new_count} 新增, {drop_count} 过滤")
                date_str = datetime.now().strftime("%Y-%m-%d")
                save_to_l2a(processed, date_str)
            else:
                print(f"[L1 Cron] 恢复完成: 0 新增")

            os.remove(RAW_CHUNKS_TMP_FILE)
            print(f"[L1 Cron] 清理 stale tmp")
        else:
            print("[L1 Cron] 无新 chunks")

    print(f"[L1 Cron] 结束: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()