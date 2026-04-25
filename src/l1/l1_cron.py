#!/usr/bin/env python3
"""
L1 Cron Entry Point
被 OpenClaw cron 触发，执行完整的 L1 pipeline。

流程（5 Stage）：
  Stage 0: 保护性写入（scan 后立即落盘，不在内存堆积）
  Stage 1: Scanner 过滤（只丢弃100%确定是噪音的内容）
  Stage 2: Classifier 处理（去噪 → Chunk → Type → Priority → Hash → Cosine）
  Stage 3: 质量过滤（语义密度，长度预检）
  Stage 4: 写入 L2A（atomic write）

防断裂原则：
  - Scanner 只丢弃100%噪音，边界情况保留
  - Classifier 做二次语义密度检查
  - crash 后可从 tmp 恢复
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, "/workspace/fusion/l1")
os.chdir("/workspace/fusion/l1")

from scan_sessions_incremental import ByteOffsetScanner
from l1_classifier import L1Classifier, save_to_l2a

RAW_CHUNKS_TMP_FILE = "/workspace/fusion/memory/_state/l1_raw_chunks_tmp.jsonl"


def main():
    print(f"[L1 Cron] 开始执行: {datetime.now().isoformat()}")

    # ============================================================
    # Stage 0+1: Scanner 扫描 + 保护性落盘 + Stage 1 过滤
    # ============================================================
    # scanner.scan() 会：
    #   1. 读取 byte offset 状态
    #   2. 扫描新增内容
    #   3. 应用 Stage 1 过滤（只丢弃100%确定是噪音的内容）
    #   4. 保护性写入 tmp 文件
    #   5. 更新 offset 状态
    # scanner.scan() → raw_chunks（已过滤）
    scanner = ByteOffsetScanner()
    raw_chunks = scanner.scan()

    # ============================================================
    # Stage 2+3+4: Classifier 处理 + 质量过滤 + 写入 L2A
    # ============================================================
    if raw_chunks:
        # 正常流程：直接从 scanner 返回的 raw_chunks 处理
        # 不读 tmp（tmp 是 crash 恢复用的）
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
        # raw_chunks 为空，检查 tmp 是否有残留（crash 恢复）
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

            # 不管 processed 是否为空，都清理 tmp
            # （processed 为空说明没有有效内容）
            os.remove(RAW_CHUNKS_TMP_FILE)
            print(f"[L1 Cron] 清理 stale tmp")
        else:
            print("[L1 Cron] 无新 chunks")

    print(f"[L1 Cron] 结束: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()