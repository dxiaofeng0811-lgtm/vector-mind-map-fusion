#!/usr/bin/env python3
"""
Fusion Recall Layer（双写模式：Brain.db + InfinityDB）

架构：
  Brain.db（SQLite，主存储）
    └─ get_seeds_by_keyword() → priority 排序精确召回
  InfinityDB（向量 + 图结构）
    ├─ vector_search()     → HNSW 向量搜索（语义）
    ├─ adjacency_bfs()     → 图扩散
    └─ get_neurons()       → 元数据读取

并行搜索路径：
  Path1: HNSW 向量搜索 → top-N 候选（语义查全）
  Path2: Brain.db keyword → priority 排序（字面查准）
  合并 → spreading activation → 元数据过滤 → top-k 返回

触发方式：Agent 通过 tool_call 直接调用
"""

import json
import math
import os
import sqlite3
from pathlib import Path
from collections import defaultdict
from typing import Optional

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
INFINITYDB_DIR = PROJECT_ROOT / "memory" / "layers" / "infinitydb"
BRAIN_DB_PATH = os.environ.get("NEURALMEMORY_DIR", os.path.expanduser("~/.local/share/neural-memory/brains.db"))
HNSW_INDEX_PATH = PROJECT_ROOT / "memory" / "layers" / "hnsw" / "index.jsonl"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "bge-m3")
VECTOR_DIM = 1024

import sys
sys.path.insert(0, str(Path(__file__).parent))

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
        self._client = None

    @property
    def client(self):
        import httpx
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=30)
        return self._client

    def encode(self, text: str) -> Optional[list[float]]:
        try:
            resp = self.client.post(
                "/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            if resp.status_code == 200:
                data = resp.json()
                emb = data.get("embedding", [])
                if isinstance(emb, list) and len(emb) == VECTOR_DIM:
                    return emb
        except Exception as e:
            print(f"[OllamaEncoder] encode failed: {e}")
        return None


class HnswSearch:
    """旧版 HNSW search（fallback，当 InfinityDB HNSW 不可用时）"""
    def __init__(self):
        self.vectors: dict[str, list[float]] = {}
        index_file = Path(HNSW_INDEX_PATH)
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        self.vectors[obj["id"]] = obj["vector"]
            print(f"[HnswSearch] 加载 {len(self.vectors)} 条向量")

    def search(self, query_vector: list[float], k: int = 20) -> list[tuple[str, float]]:
        if not self.vectors:
            return []
        scores = []
        for nid, vec in self.vectors.items():
            cos = compute_cosine(query_vector, vec)
            scores.append((cos, nid))
        scores.sort(reverse=True)
        return [(nid, cos) for cos, nid in scores[:k]]


class SpreadingActivationRecall:
    """
    Spread Activation Recall（单一数据源：InfinityDB + 并行搜索）

    并行搜索：
      Path1（HNSW）: vector_search() → 语义 top-50
      Path2（关键词）: keyword_search() → 字面 top-50
      合并 → spreading activation → 元数据过滤 → top-k
    """

    def __init__(self, recall_config: dict = None):
        self.recall_config = recall_config or DEFAULT_RECALL_CONFIG
        self.infinitydb = InfinityDBLite(str(INFINITYDB_DIR))
        self.hnsw = HnswSearch()
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """连接 Brain.db（lazy）"""
        if self._conn is None:
            os.makedirs(os.path.dirname(BRAIN_DB_PATH), exist_ok=True)
            self._conn = sqlite3.connect(BRAIN_DB_PATH)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_seeds_by_hnsw(self, query_vector: list[float], k: int = 50) -> dict[str, float]:
        """
        Path1：HNSW 向量搜索（语义查全）
        优先 warm，不够再扩 cold。
        """
        if not self.infinitydb.hnsw.nodes:
            # fallback：旧版 index.jsonl
            results = self.hnsw.search(query_vector, k=k)
            return {nid: float(score) for nid, score in results if score > 0.0}

        # 先搜 warm（优先扩散热数据）
        warm_results = self.infinitydb.vector_search(query_vector, k=k * 2, tier_filter="warm")
        warm_ids = set(nid for nid, _ in warm_results)

        # warm 不够再扩 cold
        if len(warm_ids) < k:
            cold_results = self.infinitydb.vector_search(query_vector, k=k * 2, tier_filter="cold")
            cold_ids = [nid for nid, score in cold_results if nid not in warm_ids]
            all_ids = list(warm_ids) + cold_ids
        else:
            all_ids = list(warm_ids)[:k * 2]

        return {nid: 1.0 for nid in all_ids}


    def get_seeds_by_keyword(self, query: str, k: int = 50) -> dict[str, float]:
        """
        Path2：Brain.db 关键词搜索（字面查准）
        使用 SQL LIKE 匹配 content 和 memory_type，
        按 priority 字段排序（priority 高的 neuron 天然质量高，规避 hub neuron 问题）
        """
        self.connect()
        if not self._conn:
            return {}
        try:
            cursor = self._conn.execute(
                """SELECT id, priority FROM neurons
                   WHERE content LIKE ? OR memory_type LIKE ?
                   ORDER BY priority DESC
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", k)
            )
            return {row[0]: float(row[1]) / 10.0 for row in cursor.fetchall()}
        except Exception as e:
            print(f"[Recall] 关键词搜索失败: {e}")
            return {}

    def merge_seeds(self, vector_seeds: dict[str, float], keyword_seeds: dict[str, float]) -> dict[str, float]:
        """
        合并两个搜索路径的种子
        两者都命中的神经元得双倍激活分
        """
        merged: dict[str, float] = {}
        all_ids = set(vector_seeds.keys()) | set(keyword_seeds.keys())

        for nid in all_ids:
            vec_score = vector_seeds.get(nid, 0.0)
            kw_score = keyword_seeds.get(nid, 0.0)

            if vec_score > 0 and kw_score > 0:
                # 两者都命中 → 加权叠加（boost）
                merged[nid] = vec_score + kw_score * 1.5
            elif vec_score > 0:
                merged[nid] = vec_score
            else:
                merged[nid] = kw_score * 0.7  # 纯关键词命中降权

        return merged

    def spreading_activation(self, seed_activations: dict[str, float]) -> dict[str, float]:
        """Spreading activation through adjacency graph"""
        cfg = self.recall_config
        max_hops = cfg["max_spread_hops"]
        threshold = cfg["activation_threshold"]
        dr_enabled = cfg["diminishing_returns_enabled"]
        dr_threshold = cfg["diminishing_returns_threshold"]
        dr_min = cfg["diminishing_returns_min_neurons"]
        dr_grace = cfg["diminishing_returns_grace_hops"]

        activations: dict[str, float] = dict(seed_activations)
        frontier = dict(seed_activations)
        visited = set(seed_activations.keys())

        for hop in range(1, max_hops + 1):
            next_frontier = {}
            dr_factor = 1.0

            if dr_enabled and len(activations) > dr_min:
                depth_penalty = dr_threshold ** (hop - dr_grace)
                dr_factor = max(0.5, depth_penalty)

            for node_id, activation in frontier.items():
                bfs_result = self.infinitydb.adjacency_bfs(node_id, max_hops=1)
                neighbors = bfs_result.get(1, [])
                for neighbor_id in neighbors:
                    if neighbor_id in visited:
                        continue
                    weight = self.infinitydb.adj.get(node_id, {}).get(neighbor_id, 0.5)
                    new_activation = activation * weight * dr_factor
                    if new_activation < threshold:
                        continue
                    next_frontier[neighbor_id] = next_frontier.get(neighbor_id, 0.0) + new_activation
                    visited.add(neighbor_id)

            for nid, act in next_frontier.items():
                activations[nid] = activations.get(nid, 0.0) + act

            frontier = next_frontier
            if not frontier:
                break

        return activations

    def apply_dynamic_priority(self, activations: dict[str, float]) -> dict[str, float]:
        """Dynamic priority 加权（从 Brain.db 读取 priority）"""
        self.connect()
        if not self._conn:
            return activations
        try:
            ids = list(activations.keys())
            placeholders = ",".join("?" * len(ids))
            cursor = self._conn.execute(
                f"SELECT id, priority FROM neurons WHERE id IN ({placeholders})",
                ids
            )
            priority_map = {row[0]: row[1] for row in cursor.fetchall()}
            for nid, act in activations.items():
                priority = priority_map.get(nid, 3)
                activations[nid] = act * (1.0 + (priority - 3) * 0.1)
        except Exception as e:
            print(f"[Recall] Dynamic priority 加权失败: {e}")
        return activations

    def fetch_neurons(self, neuron_ids: list[str]) -> list[dict]:
        """获取 neurons 详情（从 Brain.db 读取）"""
        self.connect()
        if not self._conn or not neuron_ids:
            return []
        try:
            placeholders = ",".join("?" * len(neuron_ids))
            cursor = self._conn.execute(f"""
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

    def load_recall_config(self) -> dict:
        """从 InfinityDB 加载 recall_config（fallback 到默认）"""
        config = self.infinitydb.get_config()
        return config if config else DEFAULT_RECALL_CONFIG

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
        主 recall 函数（旧双写模式）：
        1. query encoding → hnsw 种子（k=10）
        2. keyword seeds 合并（max，不叠加）
        3. spreading activation
        4. dynamic priority 加权（Brain.db）
        5. tier/type 过滤
        6. top-k 返回（Brain.db fetch）
        """
        # Step 0: 向量编码
        if query_vector is None:
            encoder = OllamaEncoder()
            query_vector = encoder.encode(query)

        # Step 1: HNSW 种子 + keyword 合并（max，不叠加）
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
            if tier_filter and neuron.get("tier") != tier_filter:
                continue
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
    对外暴露的 recall 接口（旧双写模式）
    """
    recall = SpreadingActivationRecall()
    recall.connect()
    recall.recall_config = recall.load_recall_config()
    try:
        return recall.recall(
            query=query,
            top_k=top_k,
            tier_filter=tier,
            memory_type_filter=memory_type,
            min_score=min_score,
        )
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
        print(f"  [{r['activation_score']:.4f}] {r.get('memory_type', 'unknown')} | tier={r.get('tier', '?')} | {r.get('content', '')[:80]}...")
