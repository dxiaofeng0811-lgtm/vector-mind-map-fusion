#!/usr/bin/env python3
"""
Fusion Recall Layer
实现 spreading activation recall：
  1. query encoding → hnsw 种子候选
  2. spreading activation through synapse graph（recall_config 参数）
  3. Dynamic priority 加权
  4. Tier 过滤 + top-k 返回

触发方式：Agent 通过 tool_call 直接调用
"""

import json
import math
import sqlite3
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional

# 配置
BRAIN_DB_PATH = os.environ.get("NEURALMEMORY_DIR", os.path.expanduser("~/.local/share/neural-memory/brains.db"))
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
INFINITYDB_DIR = PROJECT_ROOT / "memory" / "layers" / "infinitydb"
HNSW_INDEX_PATH = PROJECT_ROOT / "memory" / "layers" / "hnsw" / "index.jsonl"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "bge-m3")
VECTOR_DIM = 1024

# 确保 fusion 模块可导入（standalone 运行时需要）
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# InfinityDB-lite（邻接表 + HNSW，向量检索 + <1ms/hop BFS）
from infinitydb_lite import InfinityDBLite

# recall_config 默认值
DEFAULT_RECALL_CONFIG = {
    "max_spread_hops": 3,
    "activation_threshold": 0.3,
    "diminishing_returns_enabled": True,
    "diminishing_returns_threshold": 0.15,
    "diminishing_returns_min_neurons": 2,
    "diminishing_returns_grace_hops": 1,
}


def compute_cosine(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class OllamaEncoder:
    """Ollama 向量编码器"""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model

    def encode(self, text: str) -> list[float]:
        try:
            import urllib.request
            url = f"{self.base_url}/api/embeddings"
            payload = json.dumps({"model": self.model, "prompt": text}).encode('utf-8')
            req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                embedding = result.get("embedding")
                if embedding and len(embedding) == VECTOR_DIM:
                    return embedding
                return [0.0] * VECTOR_DIM
        except Exception as e:
            print(f"[OllamaEncoder] 失败: {e}")
            return [0.0] * VECTOR_DIM


class HnswSearch:
    """HNSW 向量检索（从 index.jsonl 加载）"""

    def __init__(self, index_path: Path = HNSW_INDEX_PATH):
        self.index_path = index_path
        self.vectors: dict[str, list[float]] = {}
        self._load()

    def _load(self):
        if not self.index_path.exists():
            print(f"[HnswSearch] 索引文件不存在: {self.index_path}")
            return
        with open(self.index_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    self.vectors[obj["id"]] = obj["vector"]
        print(f"[HnswSearch] 加载 {len(self.vectors)} 条向量")

    def search(self, query_vector: list[float], k: int = 20) -> list[tuple[str, float]]:
        """暴力搜索 top-k（hnswlib 不可用时的 fallback）"""
        if not self.vectors:
            return []
        scores = []
        for nid, vec in self.vectors.items():
            cos = compute_cosine(query_vector, vec)
            scores.append((cos, nid))
        scores.sort(reverse=True)
        return [(nid, cos) for cos, nid in scores[:k]]


class SpreadingActivationRecall:
    """Spread Activation Recall（使用 InfinityDB-lite 作为主索引）"""

    def __init__(self, db_path: str = BRAIN_DB_PATH, recall_config: dict = None):
        self.db_path = db_path
        self.recall_config = recall_config or DEFAULT_RECALL_CONFIG
        self.conn: Optional[sqlite3.Connection] = None
        # InfinityDB-lite：HNSW 向量 + 邻接表（<1ms/hop BFS）
        self.infinitydb = InfinityDBLite(str(INFINITYDB_DIR))
        # 回退：旧版 HNSW search（index.jsonl）
        self.hnsw = HnswSearch()

    def connect(self):
        if not os.path.exists(self.db_path):
            print(f"[Recall] Brain.db 不存在: {self.db_path}")
            return
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        if self.conn:
            self.conn.close()

    def load_recall_config(self) -> dict:
        """从 Brain.db 加载 recall_config"""
        if not self.conn:
            return DEFAULT_RECALL_CONFIG
        try:
            cursor = self.conn.execute(
                'SELECT config FROM brains WHERE id = "default"'
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except Exception as e:
            print(f"[Recall] 加载 recall_config 失败: {e}")
        return DEFAULT_RECALL_CONFIG

    def get_seeds_by_hnsw(self, query_vector: list[float], k: int = 10) -> dict[str, float]:
        """通过 HNSW 找到种子 neurons（优先用 InfinityDB-lite，回退到 index.jsonl）"""
        # 优先用 InfinityDB（内存 HNSW，更快）
        if self.infinitydb.hnsw.nodes:
            results = self.infinitydb.vector_search(query_vector, k=k)
            # 取 top-k 不过滤阈值（InfinityDB HNSW 已经 beam search 过）
            return {nid: float(score) for nid, score in results}
        # 回退：旧版 index.jsonl
        results = self.hnsw.search(query_vector, k=k)
        return {nid: float(score) for nid, score in results if score > 0.0}

    def get_seeds_by_keyword(self, query: str, k: int = 10) -> dict[str, float]:
        """通过 Brain.db 关键词匹配找到种子 neurons"""
        if not self.conn:
            return {}
        try:
            cursor = self.conn.execute("""
                SELECT id, priority FROM neurons
                WHERE content LIKE ? OR memory_type LIKE ?
                ORDER BY priority DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", k))
            return {row[0]: float(row[1]) / 10.0 for row in cursor.fetchall()}
        except Exception as e:
            print(f"[Recall] 关键词搜索失败: {e}")
            return {}

    def spreading_activation(self, seed_activations: dict[str, float]) -> dict[str, float]:
        """
        Spreading activation through synapse graph.
        模拟神经网络中的信号传播：
        - 从种子节点开始，激活沿边传播
        - 每跳应用 weight 衰减
        - 累积激活值超过阈值则加入结果
        """
        cfg = self.recall_config
        max_hops = cfg["max_spread_hops"]
        threshold = cfg["activation_threshold"]
        dr_enabled = cfg["diminishing_returns_enabled"]
        dr_threshold = cfg["diminishing_returns_threshold"]
        dr_min = cfg["diminishing_returns_min_neurons"]
        dr_grace = cfg["diminishing_returns_grace_hops"]

        # 初始化激活值
        activations: dict[str, float] = dict(seed_activations)
        frontier = dict(seed_activations)  # 当前层的节点
        visited = set(seed_activations.keys())

        for hop in range(1, max_hops + 1):
            next_frontier = {}
            dr_factor = 1.0

            # Diminishing returns: 当已激活节点超过阈值时降低衰减
            if dr_enabled and len(activations) > dr_min:
                depth_penalty = dr_threshold ** (hop - dr_grace)
                dr_factor = max(0.5, depth_penalty)

            for node_id, activation in frontier.items():
                # 优先用 InfinityDB adjacency BFS（<1ms/hop）
                if self.infinitydb.adj:
                    bfs_result = self.infinitydb.adjacency_bfs(node_id, max_hops=1)
                    neighbors = bfs_result.get(1, [])
                    for neighbor_id in neighbors:
                        if neighbor_id in visited:
                            continue
                        # 从邻接表获取权重
                        weight = self.infinitydb.adj.get(node_id, {}).get(neighbor_id, 0.5)
                        new_activation = activation * weight * dr_factor
                        if new_activation < threshold:
                            continue
                        next_frontier[neighbor_id] = next_frontier.get(neighbor_id, 0.0) + new_activation
                        visited.add(neighbor_id)
                elif self.conn:
                    # 回退：SQL 查询 synapses
                    cursor = self.conn.execute("""
                        SELECT target_id, weight, rel_type FROM synapses
                        WHERE source_id = ?
                    """, (node_id,))
                    rows = cursor.fetchall()

                    if not rows:
                        continue

                    for target_id, weight, rel_type in rows:
                        if target_id in visited:
                            continue
                        new_activation = activation * weight * dr_factor
                        if new_activation < threshold:
                            continue
                        next_frontier[target_id] = next_frontier.get(target_id, 0.0) + new_activation
                        visited.add(target_id)

            # 更新全局激活
            for nid, act in next_frontier.items():
                activations[nid] = activations.get(nid, 0.0) + act

            frontier = next_frontier
            if not frontier:
                break

        return activations

    def apply_dynamic_priority(self, activations: dict[str, float]) -> dict[str, float]:
        """应用 Dynamic priority 加权"""
        if not self.conn:
            return activations
        try:
            cursor = self.conn.execute(
                "SELECT id, priority FROM neurons WHERE id IN ({})".format(
                    ",".join("?" * len(activations))
                ),
                list(activations.keys())
            )
            priority_map = {row[0]: row[1] for row in cursor.fetchall()}
            # priority 越高激活值加权越多
            for nid, act in activations.items():
                priority = priority_map.get(nid, 3)
                activations[nid] = act * (1.0 + (priority - 3) * 0.1)
        except Exception as e:
            print(f"[Recall] Dynamic priority 加权失败: {e}")
        return activations

    def fetch_neurons(self, neuron_ids: list[str]) -> list[dict]:
        """获取 neurons 详情"""
        if not self.conn or not neuron_ids:
            return []
        try:
            placeholders = ",".join("?" * len(neuron_ids))
            cursor = self.conn.execute(f"""
                SELECT id, content, memory_type, priority, tier, abstraction_level
                FROM neurons WHERE id IN ({placeholders})
            """, neuron_ids)
            return [
                {
                    "id": row[0],
                    "content": row[1],
                    "memory_type": row[2],
                    "priority": row[3],
                    "tier": row[4],
                    "abstraction_level": row[5],
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            print(f"[Recall] fetch_neurons 失败: {e}")
            return []

    def recall(
        self,
        query: str,
        query_vector: list[float] = None,
        top_k: int = 10,
        tier_filter: str = None,
        memory_type_filter: str = None,
        min_score: float = 0.3,
    ) -> list[dict]:
        """
        主 recall 函数。
        1. query encoding → hnsw 种子
        2. spreading activation
        3. dynamic priority 加权
        4. tier/type 过滤
        5. top-k 返回
        """
        # Step 1: 获取种子激活
        if query_vector is None:
            encoder = OllamaEncoder()
            query_vector = encoder.encode(query)

        seed_activations = self.get_seeds_by_hnsw(query_vector, k=10)
        keyword_seeds = self.get_seeds_by_keyword(query, k=10)
        for nid, score in keyword_seeds.items():
            seed_activations[nid] = max(seed_activations.get(nid, 0.0), score)

        if not seed_activations:
            print("[Recall] 无种子节点")
            return []

        # Step 2: Spreading activation
        activations = self.spreading_activation(seed_activations)

        # Step 3: Dynamic priority 加权
        activations = self.apply_dynamic_priority(activations)

        # Step 4: 过滤
        neuron_ids = list(activations.keys())
        neurons = self.fetch_neurons(neuron_ids)
        neuron_map = {n["id"]: n for n in neurons}

        results = []
        for nid, score in activations.items():
            if score < min_score:
                continue
            neuron = neuron_map.get(nid)
            if not neuron:
                continue
            # tier 过滤
            if tier_filter and neuron.get("tier") != tier_filter:
                continue
            # memory_type 过滤
            if memory_type_filter and neuron.get("memory_type") != memory_type_filter:
                continue
            results.append({
                **neuron,
                "activation_score": round(score, 4),
            })

        # Step 5: 排序 + top-k
        results.sort(key=lambda x: x["activation_score"], reverse=True)
        return results[:top_k]


def fusion_recall(
    query: str,
    top_k: int = 10,
    tier: str = None,
    memory_type: str = None,
    min_score: float = 0.3,
) -> list[dict]:
    """
    对外暴露的 recall 接口。
    用法：
        results = fusion_recall("dxiaofeng 的项目", top_k=10)
    """
    recall = SpreadingActivationRecall()
    recall.connect()
    recall.recall_config = recall.load_recall_config()

    try:
        results = recall.recall(
            query=query,
            top_k=top_k,
            tier_filter=tier,
            memory_type_filter=memory_type,
            min_score=min_score,
        )
        return results
    finally:
        recall.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python3 recall.py <query> [--top-k N] [--tier TIER] [--type TYPE]")
        sys.exit(1)

    query = sys.argv[1]
    top_k = 10
    tier = None
    memory_type = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--tier" and i + 1 < len(sys.argv):
            tier = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--type" and i + 1 < len(sys.argv):
            memory_type = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    print(f"[Fusion Recall] Query: {query}")
    results = fusion_recall(query, top_k=top_k, tier=tier, memory_type=memory_type)
    print(f"\n返回 {len(results)} 条结果：\n")
    for r in results:
        print(f"  [{r['activation_score']:.4f}] {r['memory_type']} | tier={r['tier']} | {r['content'][:80]}...")