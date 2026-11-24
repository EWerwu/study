# 《平凡的世界》三部梗概的中文信息抽取与知识图谱

从三张课程图片出发，抽取人物等实体、定位证据句、归并简称，再构建可查询的图谱。交付物包含实验输入、参考标注、各模型原始输出、可复现实验代码和实际测量结果。

## 直接查看作业

| 内容 | 文件 |
| --- | --- |
| 三张原图的逐句转录 | [原文与句号](reports/00_原文与句号.md) |
| 实体词典、简称和出现位置 | [实体词典](reports/01_实体词典.md) |
| BERT、Qwen、BiLSTM-CRF 逐句识别结果 | [实体识别对照](reports/02_实体识别逐句对照.md) |
| spaCy、BERT、Qwen 三种方法的三元组和原句 | [关系三元组](reports/03_关系三元组.md) |
| 精确率、召回率、F1、耗时及错例 | [实验比较](reports/04_实验比较.md) |
| 双击即可打开的交互图谱 | [知识图谱](reports/05_知识图谱.html) |
| 在自己的 Neo4j 中重建三张图 | [图谱导入脚本](reports/06_图谱导入.cypher) |
| Qwen 原始提示和回复、模型预测、Neo4j 回读、计时 | [实验记录目录](reports/runs/) |

解压后打开 HTML，可切换参考图、spaCy 图和 Qwen 图，按部别筛选，点击节点查看证据，也可展开整张关系表。查看和重算已有结果均可离线完成。

## 一条命令重算报告

在项目目录打开终端。Python 3.10 或更新版本可以直接使用以下命令，根据随包的真实测量记录重新生成 Markdown 和 HTML。

```powershell
python pipeline.py report
```

## 从语料重新运行所有实验

准备 Python 3.10、Ollama 和一个运行中的 Neo4j 5.x 数据库。本项目这一轮使用 Neo4j Community 5.26.0 和本地 Qwen2.5 7B，BERT 与 BiLSTM-CRF 默认在 CPU 上训练。首次下载模型与依赖需要联网和磁盘空间。请在已解压的项目目录执行以下命令。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:7b
```

确保 Ollama 已运行；如果使用独立命令行服务，可在另一窗口执行 `ollama serve`。在 Neo4j Desktop 中启动一个本地数据库实例，记下连接地址、用户名和自己的密码。PowerShell 可通过交互提示设置当前窗口的密码，再运行整条流水线。

```powershell
$env:NEO4J_URI = 'bolt://127.0.0.1:7687'
$env:NEO4J_USER = 'neo4j'
$env:NEO4J_PASSWORD = [System.Net.NetworkCredential]::new('', (Read-Host 'Neo4j 密码' -AsSecureString)).Password
.\.venv\Scripts\python.exe pipeline.py all --fresh
```

BERT 默认从 `hfl/rbt3` 加载。已有本地模型时，可把 `KG_BERT` 设为包含 config.json、vocab.txt 和权重的目录。Ollama 默认地址为 `http://127.0.0.1:11434`，使用其他地址时设置 `OLLAMA_HOST`。其他可选设置包括 `KG_DEVICE`、`KG_THREADS`、`NEO4J_DATABASE` 和 `KG_LLM`。更换大模型后应重新生成全部结果，并同步调整实验报告中的模型名称。

| 阶段参数 | 操作 |
| --- | --- |
| `ocr` | 用 RapidOCR 读取三张原图，与已校对转录对齐；保留原始行、改字位置和分句结果 |
| `neural` | 生成独立合成短句，训练并运行 BiLSTM-CRF、中文 BERT 的实体和关系任务 |
| `rules` | 运行 spaCy 中文分词、词面匹配与关系触发规则 |
| `llm --fresh` | 逐部运行实体抽取、三元组抽取、直接建图，总计九次本地请求 |
| `graph` | 分别导入参考、spaCy 和 Qwen 直接生成的图，计时、查询并回读核验 |
| `report` | 只读取已保存的结果，重建全部交付文档 |
| `all --fresh` | 按上述顺序完整复现；会替换本目录中的实验记录和报告 |

省略 `--fresh` 时，LLM 阶段复用模型名和提示均匹配的原始回复，报告中的耗时来自历史真实请求。训练在固定种子下执行；硬件、驱动、模型版本和系统负载会影响复现时间与数值。

## 数据和解释

`data/passages.tsv` 是逐句转录；`dictionary.tsv` 定义实体规范名、类型与简称；`reference.tsv` 提供逐句复核的参考词面和关系。神经模型的训练脚本不读取参考文件。spaCy 与 BERT 的关系抽取共用词典候选，便于观察局部规则与学习分类器的差异。Qwen 独立处理每一部，直接建图另行调用，保留原始输出。

例如，2.03 的“获得了当煤矿工人的好机遇”对应（孙少平，就业机会，煤矿工人），3.01 的“到了煤矿，尽心尽力干活”另记（孙少平，任职于，煤矿）。查询人物经历时可据此区分机会与已经发生的事实。图中每条边保留句号；同一语义三元组在不同句得到印证时会有多条证据边。

## 在 Neo4j 中查看

可在已配置的数据库上执行 `python pipeline.py graph`，或通过 Neo4j 自带的 cypher-shell 导入随包脚本，在交互终端输入 `:source reports/06_图谱导入.cypher`。导入按项目来源和数据指纹区分图谱，可重复执行。下面的查询列出当前库中的项目图及其节点数。

```cypher
MATCH (n:NovelCourseEntity)
RETURN n.scope AS 图谱, count(*) AS 节点数
ORDER BY 图谱
```

实验比较文档末段的 Cypher 示例可在 Neo4j Browser 中打开参考图的“孙少平”一跳邻域。HTML 已包含相同的数据库回读结果，可直接用于课堂展示。
