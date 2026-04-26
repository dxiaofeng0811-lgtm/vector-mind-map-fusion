#!/usr/bin/env python3
"""
L3: Biweekly Consolidator + Brain.db Writer
读取 L2 区数据，写入 neural-memory Brain.db + hnsw 索引 + SCHEMA 生成
触发时间：每两天 03:00（Asia/Shanghai）

流程：
  ① 读取 L2 区 graph_written=False 的 chunks
  ② 批量写入 Brain.db（nodes）
  ③ SCHEMA 生成（session≥5 chunks → TF-IDF 关键词）
  ④ 批量写入 relations（edges，权重 CAUSED_BY=1.0 / inferred=0.5）
  ⑤ hnsw 全量重建（pre-allocate，无 resize）
  ⑥ BrainConfig.with_updates() 设置 recall_config
  ⑦ 增量删除 L2
  ⑧ Brain.db 永久保留
"""

import json
import os
import math
import time
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional


# 项目根目录（向上推导）
PROJECT_ROOT = Path(__file__).parent.parent.parent
# 配置
L2_DIR = PROJECT_ROOT / "memory" / "layers" / "l2"
HNSW_DIR = PROJECT_ROOT / "memory" / "layers" / "hnsw"
INFINITYDB_DIR = PROJECT_ROOT / "memory" / "layers" / "infinitydb"
BRAIN_DB_DIR = os.environ.get("NEURALMEMORY_DIR", os.path.expanduser("~/.local/share/neural-memory"))
BRAIN_DB_PATH = os.path.join(BRAIN_DB_DIR, "brains.db")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "bge-m3")
VECTOR_DIM = 1024
MAX_ELEMENTS = 10000

# ===== 批量优化配置（防断裂防护）=====
# Ollama embedding 限制
OLLAMA_BATCH_SIZE = 20           # 单次 HTTP 最大条数（Ollama body 过大可能 timeout）
OLLAMA_TIMEOUT_SEC = 30          # 单次请求超时（秒）
OLLAMA_MAX_RETRIES = 3          # 失败重试次数
OLLAMA_RETRY_SLEEP_SEC = 1.0    # 重试间隔（秒）
OLLAMA_IDLE_BETWEEN_BATCHES = 0.1  # 每批次之间喘息（秒），防止 Ollama/InfinityDB 过载

# SQLite 批量写入限制
SQLITE_BATCH_SIZE = 200          # 每批 commit 条数（超过 500 可能导致事务过大）
SQLITE_REBUILD_THRESHOLD = 10000 # 超过此条数时强制分批处理

# HNSW 批量添加限制
HNSW_BATCH_SIZE = 500           # HNSW 批量添加的块大小
HNSW_MAX_ELEMENTS = 10000       # 预分配最大元素数

# 总体流程限制
MAX_CHUNKS_PER_RUN = 5000       # 单次运行最大处理量（防止内存溢出）

# 关系权重
RELATION_WEIGHTS = {
    "CAUSED_BY": 1.0,
    "LEADS_TO": 1.0,
    "RESOLVED_BY": 1.0,
    "CONTINUES_AS": 1.0,
    "CONTRADICTS": 1.0,
    "inferred": 0.5,
}

# InfinityDB-lite（邻接表 + HNSW，向量检索用）
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "recall"))
from infinitydb_lite import InfinityDBLite


def compute_cosine(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class SimpleVectorIndex:
    """简单的向量索引（fallback，当 hnswlib 不可用时）"""

    def __init__(self, dim: int = VECTOR_DIM):
        self.dim = dim
        self.vectors: list[list[float]] = []
        self.ids: list[str] = []

    def add(self, chunk_id: str, vector: list[float]):
        self.vectors.append(vector)
        self.ids.append(chunk_id)

    def search(self, query: list[float], k: int = 5, threshold: float = 0.95) -> list[tuple[str, float]]:
        """暴力搜索 top-k"""
        if not self.vectors:
            return []
        scores = []
        for i, vec in enumerate(self.vectors):
            cos = compute_cosine(query, vec)
            scores.append((cos, i))
        scores.sort(reverse=True)
        results = []
        for cos, i in scores[:k]:
            if cos > threshold:
                results.append((self.ids[i], cos))
        return results

    def rebuild(self, elements: list[tuple[str, list[float]]]):
        """全量重建"""
        self.vectors = [v for _, v in elements]
        self.ids = [i for i, _ in elements]


class OllamaEncoder:
    """Ollama 向量编码器（带超时重试 + batch 分片）"""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL,
                 batch_size: int = 20, timeout_sec: int = 30, max_retries: int = 3):
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size        # 单次 HTTP 最大条数
        self.timeout_sec = timeout_sec      # 单次请求超时（秒）
        self.max_retries = max_retries       # 失败重试次数

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """修复：Ollama /api/embeddings 不支持 batch input，逐条编码保持 1:1 对应"""
        if not texts:
            return []

        results = []
        for i, text in enumerate(texts):
            if not (text and text.strip()):
                results.append([0.0] * VECTOR_DIM)
                continue

            for attempt in range(self.max_retries):
                try:
                    import urllib.request
                    payload = json.dumps({
                        "model": self.model,
                        "prompt": text.strip()
                    }).encode('utf-8')
                    req = urllib.request.Request(
                        f"{self.base_url}/api/embeddings",
                        data=payload,
                        headers={'Content-Type': 'application/json'}
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                        result = json.loads(resp.read())
                        emb = result.get("embedding", [])
                        if isinstance(emb, list) and len(emb) == VECTOR_DIM:
                            results.append(emb)
                            break
                        else:
                            results.append([0.0] * VECTOR_DIM)
                            break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"[OllamaEncoder] text[{i}] 放弃: {e}")
                        results.append([0.0] * VECTOR_DIM)
                    else:
                        import time
                        time.sleep(1 * (attempt + 1))
        return results

    def _encode_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """单批次编码，失败重试（最优解：真正的批量 HTTP）"""
        import urllib.request
        import time

        url = f"{self.base_url}/api/embeddings"
        headers = {'Content-Type': 'application/json'}

        for attempt in range(self.max_retries):
            try:
                # 修复 A：使用 Ollama 批量接口 /api/embeddings
                # 单次 HTTP 传输所有 texts，batch_size 配置真正生效
                # O(1) 次 HTTP 请求，而不是 O(N) 次
                payload = json.dumps({
                    "model": self.model,
                    "input": texts  # 批量输入，Ollama 会返回 embeddings 数组
                }).encode('utf-8')
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    result = json.loads(resp.read())
                    # Ollama 批量返回 embeddings 数组
                    embeddings = result.get("embeddings", [])
                    if embeddings and len(embeddings) == len(texts):
                        # 验证向量维度
                        batch_results = []
                        for emb in embeddings:
                            if isinstance(emb, list) and len(emb) == VECTOR_DIM:
                                batch_results.append(emb)
                            else:
                                batch_results.append([0.0] * VECTOR_DIM)
                        return batch_results
                    else:
                        print(f"[OllamaEncoder] embeddings 长度不匹配: {len(embeddings)} vs {len(texts)}")
                        return [[0.0] * VECTOR_DIM for _ in texts]
            except Exception as e:
                print(f"[OllamaEncoder] 第 {attempt+1} 次失败: {e}")
                if attempt == self.max_retries - 1:
                    print(f"[OllamaEncoder] 放弃，跳过 {len(texts)} 条")
                    return [[0.0] * VECTOR_DIM for _ in texts]
                time.sleep(1 * (attempt + 1))
        return [[0.0] * VECTOR_DIM for _ in texts]
STOPWORDS = frozenset([
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'once', 'here', 'there', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'now', 'is', 'am', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'this', 'that', 'these', 'those',
])


def tfidf_extract_keywords(texts: list[str], top_n: int = 5) -> list[str]:
    """TF-IDF 提取关键词"""
    word_counts = Counter()
    for text in texts:
        words = [w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) > 2]
        word_counts.update(set(words))
    return [word for word, _ in word_counts.most_common(top_n)]


def init_brain_db(db_path: str):
    """初始化 Brain.db schema"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS brains (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            config TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS neurons (
            id TEXT PRIMARY KEY,
            brain_id TEXT NOT NULL,
            content TEXT NOT NULL,
            memory_type TEXT,
            priority INTEGER,
            tier TEXT,
            abstraction_level INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (brain_id) REFERENCES brains(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS synapses (
            id TEXT PRIMARY KEY,
            brain_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            rel_type TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            direction TEXT DEFAULT 'bidirectional',
            metadata TEXT,
            created_at TEXT,
            FOREIGN KEY (brain_id) REFERENCES brains(id),
            FOREIGN KEY (source_id) REFERENCES neurons(id),
            FOREIGN KEY (target_id) REFERENCES neurons(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_neurons_brain ON neurons(brain_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_synapses_brain ON synapses(brain_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_synapses_source ON synapses(source_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_synapses_target ON synapses(target_id)")
    conn.commit()
    conn.close()


def load_l2_chunks(date_str: str = None) -> list[dict]:
    """加载 L2 区数据（扫描所有日期的 L2 文件）"""
    # 修复 G：扫描 L2_DIR 下所有 .jsonl 文件，不只加载当天
    l2_files = list(L2_DIR.glob("*.jsonl")) if L2_DIR.exists() else []

    if date_str:
        # 指定日期时只加载该日期
        l2_files = [f for f in l2_files if f.stem == date_str]

    if not l2_files:
        if date_str:
            l2_file = L2_DIR / f"{date_str}.jsonl"
            print(f"[L3] L2 文件不存在: {l2_file}")
        else:
            print(f"[L3] L2 目录无任何文件: {L2_DIR}")
        return []

    chunks = []
    for l2_file in sorted(l2_files):  # 按日期排序，保证顺序
        with open(l2_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if not obj.get("graph_written"):  # 只处理未写入的
                        chunks.append(obj)

    print(f"[L3] 加载 L2 chunks: {len(chunks)} 条（来自 {len(l2_files)} 个文件）")
    return chunks, l2_files


def generate_schema_neuron(session_chunks: list[dict], encoder: OllamaEncoder) -> Optional[dict]:
    """生成 SCHEMA 神经元（session≥5 chunks，降低阈值以覆盖更多 session）"""
    # 修复 D：从 10 降低到 5，让短 session 也能生成摘要
    if len(session_chunks) < 5:
        return None

    contents = [c["content"] for c in session_chunks]
    keywords = tfidf_extract_keywords(contents, top_n=5)

    schema_content = f"Session summary: {', '.join(keywords)}"

    # 获取向量
    vectors = encoder.encode_batch([schema_content])
    vector = vectors[0] if vectors else []

    return {
        "id": f"schema_{session_chunks[0]['session_id']}",
        "content": schema_content,
        "memory_type": "schema",
        "priority": 4,  # 修复 E：改为 4（与普通 chunks 接近，不过度优先）
        "tier": "warm",
        "abstraction_level": 3,
        "vector": vector,
        "connected_chunks": [c["id"] for c in session_chunks],
        "keywords": keywords,
    }


class L3Processor:
    """L3 处理器：Brain.db 写入 + InfinityDB-lite 同步 + SCHEMA"""

    def __init__(self, db_path: str = BRAIN_DB_PATH):
        self.db_path = db_path
        self.encoder = OllamaEncoder()
        self.vector_index = SimpleVectorIndex()
        self._conn = None
        # InfinityDB-lite（邻接表 + HNSW，向量检索用）
        self.infinitydb = InfinityDBLite(str(INFINITYDB_DIR))
        # 暂存邻接关系，等 Brain.db 写完后再统一写入 InfinityDB
        self._pending_neurons: list[dict] = []
        self._pending_relations: list[dict] = []

    def connect(self):
        """连接 Brain.db"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)

    def close(self):
        if self._conn:
            self._conn.close()

    def write_chunks(self, chunks: list[dict], inferred_relations: list[dict]) -> dict:
        """批量写入 Brain.db（分批 commit + executemany 优化）"""
        if self._conn is None:
            self.connect()

        now = datetime.now(timezone.utc).isoformat()

        # 确保 brain 存在
        brain_id = "default"
        cursor = self._conn.execute("SELECT id FROM brains WHERE id = ?", (brain_id,))
        if not cursor.fetchone():
            self._conn.execute(
                "INSERT INTO brains (id, name, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (brain_id, "default", "{}", now, now)
            )

        # 按 session 分组
        session_groups = defaultdict(list)
        for chunk in chunks:
            session_groups[chunk["session_id"]].append(chunk)

        written_count = 0
        schema_count = 0
        relation_count = 0

        all_neurons = []  # 收集所有神经元用于后续处理

        # ===== 分批处理神经元写入（防止事务过大）=====
        neuron_batch = []
        schema_batch = []
        synapse_batch = []
        written_ids = []  # 只记录成功写入 Brain.db 的 chunk id

        for session_id, session_chunks in session_groups.items():
            for chunk in session_chunks:
                neuron_id = chunk["id"]
                content = chunk.get("content", "")
                memory_type = chunk.get("memory_type", "context")
                priority = chunk.get("priority", 3)
                tier = chunk.get("tier", "cold")
                abstraction_level = 3 if memory_type == "schema" else 1

                # 暂存到 pending（后续写入 InfinityDB）
                vector = chunk.get("vector", [])
                # 重新编码向量（如果 L2 没有）
                if not vector:
                    vectors = self.encoder.encode_batch([content])
                    vector = vectors[0] if vectors else []

                # 修复 C：零向量不写入 Brain.db 也不写入 InfinityDB
                # zero vector 会导致 cosine similarity 计算错误，污染索引
                if not vector or len(vector) != VECTOR_DIM or vector == [0.0] * VECTOR_DIM:
                    print(f"[L3] 跳过零向量 chunk: {neuron_id}")
                    # 同时跳过 neuron_batch 和 _pending_neurons
                    continue

                self._pending_neurons.append({
                    "id": neuron_id,
                    "content": content,
                    "vector": vector,
                    "memory_type": memory_type,
                    "priority": priority,
                    "tier": tier,
                })

                # 只有有效向量才写入 Brain.db
                neuron_batch.append((neuron_id, brain_id, content, memory_type, priority, tier, abstraction_level, now, now))
                written_ids.append(neuron_id)
                # all_neurons 只收集有效向量 chunk（用于 rebuild_hnsw）
                all_neurons.append(chunk)

            # 生成 SCHEMA
            schema = generate_schema_neuron(session_chunks, self.encoder)
            if schema:
                schema_id = schema["id"]
                schema_batch.append((schema_id, brain_id, schema["content"], "schema", 6, "warm", 3, now, now))
                all_neurons.append(schema)
                schema_count += 1

                # SCHEMA 连接到所有 chunks（收集到 batch）
                for chunk in session_chunks:
                    synapse_batch.append((f"{schema_id}->{chunk['id']}", brain_id, schema_id, chunk["id"], "schema_of", 1.0, now))
                    relation_count += 1

            # ===== 分批 commit（每 200 条一提交，防止事务过大）=====
            if len(neuron_batch) >= SQLITE_BATCH_SIZE:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO neurons (id, brain_id, content, memory_type, priority, tier, abstraction_level, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    neuron_batch
                )
                self._conn.commit()
                written_count += len(neuron_batch)
                print(f"[L3] 分批 commit: {written_count} neurons written")
                neuron_batch = []

        # ===== 写入 inferred relations（收集到 batch）=====
        for rel in inferred_relations:
            from_id = rel.get("from")
            to_id = rel.get("to")
            rel_type = rel.get("rel_type", "CAUSED_BY")
            weight = rel.get("weight", 0.5)

            if from_id and to_id:
                synapse_batch.append((f"{from_id}->{to_id}", brain_id, from_id, to_id, rel_type, weight, now))
                self._pending_relations.append({
                    "from": from_id,
                    "to": to_id,
                    "weight": weight,
                    "rel_type": rel_type,
                })

        # ===== 最后一批：批量写入剩余 neurons + schemas + synapses，再统一 commit =====
        if neuron_batch:
            self._conn.executemany(
                "INSERT OR REPLACE INTO neurons (id, brain_id, content, memory_type, priority, tier, abstraction_level, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                neuron_batch
            )
            written_count += len(neuron_batch)

        if schema_batch:
            self._conn.executemany(
                "INSERT OR REPLACE INTO neurons (id, brain_id, content, memory_type, priority, tier, abstraction_level, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                schema_batch
            )

        if synapse_batch:
            self._conn.executemany(
                "INSERT OR REPLACE INTO synapses (id, brain_id, source_id, target_id, rel_type, weight, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                synapse_batch
            )

        # 统一 commit（一次事务完成所有剩余写入）
        self._conn.commit()
        print(f"[L3] 写入 Brain.db（batch+executemany）: neurons={written_count}, schemas={schema_count}, relations={relation_count}")

        return {
            "neurons_written": written_count,
            "schemas_written": schema_count,
            "relations_written": relation_count,
            "all_neurons": all_neurons,
            "written_ids": written_ids,  # 只包含成功写入的 chunk id（用于 mark_l2_graph_written）
        }

    def sync_to_infinitydb(self):
        """
        将 pending neurons + relations 同步到 InfinityDB-lite。
        在 Brain.db 写完后调用。
        """
        import time

        if not self._pending_neurons:
            print("[L3] InfinityDB: 无待同步 neurons")
            return

        # 构建邻接表：neuron_id → {neighbor_id: weight}
        adjacencies: dict[str, dict[str, float]] = defaultdict(dict)
        for rel in self._pending_relations:
            adjacencies[rel["from"]][rel["to"]] = rel["weight"]

        # 批量写入 InfinityDB（每批之间喘息，防止内存/IO 过载）
        total = len(self._pending_neurons)
        for i, neuron in enumerate(self._pending_neurons):
            nid = neuron["id"]
            vector = neuron.get("vector", [])
            neighbors = adjacencies.get(nid, {})

            if vector and len(vector) == VECTOR_DIM:
                self.infinitydb.add_neuron(nid, vector, neighbors)

            # 每 HNSW_BATCH_SIZE 条喘息一次
            if (i + 1) % HNSW_BATCH_SIZE == 0:
                print(f"[L3] InfinityDB 写入进度: {i+1}/{total}")
                time.sleep(OLLAMA_IDLE_BETWEEN_BATCHES)

        # 全量保存
        self.infinitydb.save()
        print(f"[L3] InfinityDB 同步完成: {total} neurons, {len(self._pending_relations)} relations")

        # 清空 pending
        self._pending_neurons.clear()
        self._pending_relations.clear()

    def rebuild_hnsw(self, neurons: list[dict]):
        """全量重建 hnsw 索引（备份到 index.jsonl）"""
        HNSW_DIR.mkdir(parents=True, exist_ok=True)

        # 收集所有有向量的神经元
        elements = []
        for neuron in neurons:
            if neuron.get("vector") and len(neuron["vector"]) == VECTOR_DIM:
                elements.append((neuron["id"], neuron["vector"]))

        if not elements:
            print("[L3] 无向量可索引")
            return

        # 全量重建到 SimpleVectorIndex（内存索引）
        self.vector_index.rebuild(elements)

        # 保存向量到 index.jsonl（兼容旧 recall）
        index_file = HNSW_DIR / "index.jsonl"
        with open(index_file, 'w', encoding='utf-8') as f:
            for neuron_id, vector in elements:
                f.write(json.dumps({"id": neuron_id, "vector": vector}, ensure_ascii=False) + '\n')

        print(f"[L3] hnsw index.jsonl 备份完成: {len(elements)} 条向量")

    def update_recall_config(self):
        """通过 BrainConfig.with_updates() 设置 recall_config（模拟）"""
        recall_config = {
            "max_spread_hops": 3,
            "activation_threshold": 0.3,
            "diminishing_returns_enabled": True,
            "diminishing_returns_threshold": 0.15,
            "diminishing_returns_min_neurons": 2,
            "diminishing_returns_grace_hops": 1,
        }
        # 将 recall_config 写入 brain.config
        if self._conn is None:
            self.connect()
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute("""
            UPDATE brains SET config = ?, updated_at = ? WHERE id = 'default'
        """, (json.dumps(recall_config), now))
        self._conn.commit()
        print(f"[L3] recall_config 已更新")


def mark_l2_graph_written(result: dict, l2_files: list):
    """标记 L2 中已写入的 chunks（扫描所有日期文件，atomic write 防 crash）"""
    # 修复 I：written_ids 只包含成功写入 Brain.db 的 chunk id
    written_ids = set(result.get("written_ids", []))

    for l2_file in l2_files:
        if not l2_file.exists():
            continue

        # 读出现有数据
        remaining = []
        with open(l2_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if obj.get("id") not in written_ids:
                        remaining.append(obj)

        # atomic write：先写 tmp，再 rename
        tmp_file = l2_file.with_suffix('.tmp')
        with open(tmp_file, 'w', encoding='utf-8') as f:
            for obj in remaining:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        tmp_file.rename(l2_file)

        print(f"[L3] L2 文件清理完成: {l2_file.name}，剩余 {len(remaining)} 条未写入")


def run():
    """L3 入口"""
    print(f"[L3] 开始执行: {datetime.now().isoformat()}")

    # Step 1: 初始化 Brain.db
    init_brain_db(BRAIN_DB_PATH)

    # Step 2: 加载 L2 数据
    l2_chunks, l2_files = load_l2_chunks()
    if not l2_chunks:
        print("[L3] 无待处理 chunks")
        return

    # 收集 inferred relations
    inferred_relations = []
    for chunk in l2_chunks:
        inferred_relations.extend(chunk.get("inferred_relations", []))

    # Step 3: 写入 Brain.db
    processor = L3Processor()
    result = processor.write_chunks(l2_chunks, inferred_relations)

    # Step 4: 同步到 InfinityDB-lite
    processor.sync_to_infinitydb()

    # Step 5: hnsw index.jsonl 备份
    if result["all_neurons"]:
        processor.rebuild_hnsw(result["all_neurons"])

    # Step 6: recall_config 更新
    processor.update_recall_config()

    # Step 7: 增量删除 L2（扫描所有日期的文件）
    # 用 result["written_ids"]（只有成功写入 Brain.db 的 chunk id）
    mark_l2_graph_written(result, l2_files)

    processor.close()

    print(f"[L3] 完成: neurons={result['neurons_written']}, schemas={result['schemas_written']}, relations={result['relations_written']}")
    print(f"[L3] 结束: {datetime.now().isoformat()}")


if __name__ == "__main__":
    run()