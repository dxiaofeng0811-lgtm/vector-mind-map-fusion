---
name: vector-mind-map-fusion
description: L1→L2→L3 向量记忆融合系统。用于构建、查询和管理语义记忆图谱。当用户需要提取、加工、记忆、或检索结构化知识时触发。具体场景：(1) 用户说"记住"、"存入记忆"、"这个很重要" → L1 提取；(2) 用户问"之前有没有"、"有没有记录过"、"我的记忆里" → L2+L3 查询；(3) 用户要求"整理一下"、"归类"、"形成知识体系" → L2 整理；(4) 用户说"搜索记忆"、"查找相关内容"、"语义搜索" → L3 召回；(5) 主动记忆扫描、增量更新、跨 session 知识关联时也触发。项目位于 /workspace/vector-mind-map-fusion。
---

# Vector-Mind Map-Fusion

三层记忆融合系统：L1 提取 → L2 整理 → L3 检索

## 快速使用

```bash
# 安装依赖
pip install -r /workspace/vector-mind-map-fusion/requirements.txt

# 运行全部
python /workspace/vector-mind-map-fusion/main.py run --all

# 只运行 L1
python /workspace/vector-mind-map-fusion/main.py run --layer l1

# 搜索记忆
python /workspace/vector-mind-map-fusion/main.py search "查询内容"

# 查看状态
python /workspace/vector-mind-map-fusion/main.py stats
```

## 项目结构

```
/workspace/vector-mind-map-fusion/
├── README.md           # 核心说明书
├── requirements.txt    # Python 依赖
├── main.py             # 项目入口
├── src/
│   ├── l1/             # L1 提取层
│   ├── l2/             # L2 整理层
│   ├── l3/             # L3 检索层
│   └── recall/         # 召回工具
├── docs/               # 详细文档
├── memory/layers/      # 数据目录
│   ├── l2a/            # L1 输出
│   ├── l2/             # L2 输出
│   ├── hnsw/           # HNSW 索引
│   └── infinitydb/     # InfinityDB 数据
└── tests/
```

## 触发条件

| 用户意图 | 对应层 | 说明 |
|---------|-------|------|
| "记住 XXX"、"存入记忆" | L1 | 立即提取 |
| "之前有没有"、"我的记忆里" | L2+L3 | 语义查询 |
| "搜索记忆"、"语义搜索" | L3 | 向量召回 |
| 每日定时 | L1+L2 (00:30) | 增量扫描 |
| 每两天定时 | L3 (03:00) | 归档整理 |

## 详细文档

- [L1 提取流程](references/l1_flow.md)
- [L2 整理流程](references/l2_flow.md)
- [L3 检索流程](references/l3_flow.md)
- [Recall API](references/recall_api.md)
- [修复清单](references/fixes.md)

## 质量保证

| 保证 | 实现 |
|------|------|
| 防断裂 | 50字 overlap、atomic write、byte offset |
| 防丢失 | tmp 保护、written_ids 追踪、graph_written 标记 |
| 防质量下降 | denoise→quality→classify 顺序、零向量过滤 |
| 防关系错乱 | session graph + transitive closure |
| 防索引混乱 | 四级去重、content_hash_index 隔离 |