# Vector-Mind Map-Fusion

三层向量记忆融合系统：L1 提取 → L2 整理 → L3 检索

## 核心架构

```
用户输入
    ↓
L1 (Session Scanner + Classifier)
    ├─ ByteOffsetScanner: 扫描 session JSONL，断点续扫
    ├─ Stage1 过滤: 噪音/UUID/cron/metadata
    ├─ Classifier: 去噪→质量检查→分类→分块→向量
    └─ 输出: L2A/ (每日增量)
    ↓
L2 (Daily Consolidator)
    ├─ 加载 L2A: 扫描所有日期文件
    ├─ session 分组 + 滑动窗口
    ├─ 四级去重: content_hash → cosine → simhash → hnsw
    ├─ session graph: N-gram 中文分词
    ├─ transitive closure: 关系补全
    └─ 输出: L2/ (每日增量)
    ↓
L3 (Biweekly Consolidator)
    ├─ 加载 L2: Brain.db 写入
    ├─ SCHEMA 生成: session≥5 → TF-IDF 摘要
    ├─ InfinityDB 同步: brain.graph + brain.vec + hnsw
    └─ 增量删除 L2 (written_ids 追踪)
```

## 环境准备

### 1. 安装 Ollama

```bash
# 安装 Ollama（macOS/Linux）
curl -fsSL https://ollama.com/install.sh | sh

# Windows: 下载 https://ollama.com/download
```

### 2. 拉取向量模型

```bash
# 安装 bge-m3 模型（用于语义向量编码）
ollama pull bge-m3

# 验证模型已安装
ollama list
# 应看到: bge-m3  ...  ...
```

### 3. 启动 Ollama 服务

```bash
# 启动 Ollama（后台运行）
ollama serve

# 验证服务正常运行
curl http://localhost:11434/api/tags
```

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 运行整个系统
python main.py run --all

# 只运行 L1 提取
python main.py run --layer l1

# 只运行 L2 整理
python main.py run --layer l2

# 只运行 L3 检索
python main.py run --layer l3

# 搜索记忆
python main.py search "查询内容"

# 查看状态
python main.py stats
```

## 项目结构

```
vector-mind-map-fusion/
├── README.md              # 本文件
├── requirements.txt        # Python 依赖
├── .gitignore
├── main.py                # 项目入口
├── src/
│   ├── l1/                # L1 提取层
│   │   ├── __init__.py
│   │   ├── l1_cron.py    # L1 入口
│   │   ├── scan_sessions_incremental.py
│   │   └── l1_classifier.py
│   ├── l2/                # L2 整理层
│   │   ├── __init__.py
│   │   ├── l2_cron.py
│   │   └── l2_daily.py
│   ├── l3/                # L3 检索层
│   │   ├── __init__.py
│   │   ├── l3_cron.py
│   │   └── l3_biweekly_consolidate.py
│   └── recall/            # 召回工具
│       ├── __init__.py
│       ├── recall.py      # VectorMindRecall
│       └── infinitydb_lite.py
├── tests/                 # 测试
└── docs/                  # 文档
```

## 触发条件

| 用户意图 | 对应层 | 说明 |
|---------|-------|------|
| "记住 XXX"、"存入记忆" | L1 | 立即提取 |
| "之前有没有"、"我的记忆里" | L2+L3 | 语义查询 |
| "整理一下"、"归类" | L2 | 结构化整理 |
| "搜索记忆"、"语义搜索" | L3 | 向量召回 |
| 每日定时 | L1+L2 (00:30) | 增量扫描 |
| 每两天定时 | L3 (03:00) | 归档整理 |

## 质量保证

| 保证 | 实现 |
|------|------|
| 防断裂 | 50字 overlap、atomic write、byte offset |
| 防丢失 | tmp 保护、written_ids 追踪、graph_written 标记 |
| 防质量下降 | denoise→quality→classify 顺序、零向量过滤 |
| 防关系错乱 | session graph + transitive closure |
| 防索引混乱 | 四级去重、content_hash_index 隔离 |

## 性能指标

| 操作 | 速度 |
|------|------|
| L1 Scanner | ~87,000 条/秒 |
| L1 Classifier | ~32,000 条/秒 |
| L2 处理 | ~14,500 条/秒 |
| L3 Brain.db 写入 | ~100 条/秒 |
| Vector Search | ~60 QPS |
| Adjacency BFS | ~437,000 QPS |
| Combined Recall | ~59 QPS |
| Brain.db SQL | ~60,000 QPS |

## 数据路径

| 层 | 代码路径 | 数据路径 |
|---|---------|---------|
| L1 | src/l1/ | memory/layers/l2a/ |
| L2 | src/l2/ | memory/layers/l2/ |
| L3 | src/l3/ | memory/layers/infinitydb/ + Brain.db |
| Recall | src/recall/ | Brain.db + infinitydb/ |

## 详细文档

- [L1 提取流程](docs/l1_flow.md)
- [L2 整理流程](docs/l2_flow.md)
- [L3 检索流程](docs/l3_flow.md)
- [Recall API](docs/recall_api.md)
- [修复清单](docs/fixes.md)