# extract_keywords
## 主要功能

实现的主要功能是处理文本数据，从中提取关键词，计算关键词之间的相似度，并基于相似度进行聚类，最终将结果保存为网络图。其工作流程细分：

1. **关键词提取**：使用 KeyBERT 模型从文本中提取关键词，并结合 TF-IDF 算法计算每个关键词的重要性。
2. **词向量生成**：使用 SciBERT 模型将关键词转换成词向量，以便进行数值化的相似度比较。
3. **相似度计算与聚类**：计算关键词之间的相似度，并基于设定的阈值将相似的关键词聚类在一起，形成关键词网络。
4. **网络图构建和保存**：构建一个关键词网络图，其中节点代表关键词，边代表关键词之间的相似性，并将网络图保存为 GEXF 格式文件。

需要准备以下输入：

1. **文本数据**：以 Excel 文件形式提供，其中包含需要分析的文本数据。
2. **处理参数**：
   - `sheet_name`：指定 Excel 文件中要处理的工作表名称。
   - `column_list`：指定要合并并分析的列，用于从 Excel 文件中提取并合并文本数据。
3. **关键词提取参数**：
   - `num_keywords`：指定每段文本提取的关键词数量。
   - `similarity_threshold`：相似度聚类时使用的阈值，用于决定哪些关键词应该被聚类在一起。
4. **模型和工具**：
   - **KeyBERT**：用于关键词提取的模型。
   - **SciBERT**：用于生成词向量的预训练语言模型。
   - **TF-IDF**：用于计算文本中单词的重要性。
   - **UMAP**：用于降维，以便于进行聚类。
   - **NetworkX**：用于创建和操作网络图。
5. **自定义文本预处理功能**：在 `processdata` 模块中定义，用于在关键词提取之前对文本进行清洗和规范化。



## 代码中类的详细介绍：

### `KeywordExtractor` 类

- `__init__`: 初始化，接收一个 KeyBERT 模型并创建一个 TF-IDF 向量化器。
- `fit_vectorizer`: 使用输入文本训练 TF-IDF 向量化器。
- `extract_keywords`: 提取文本的关键词，然后使用 TF-IDF 计算每个关键词的权重，返回权重最高的关键词。

### `ScibertEmbedding` 类

- `__init__`: 初始化，接收一个分词器和一个模型。
- `__call__`: 将关键词转换为词向量，通过分词器处理后，使用模型生成每个关键词的词向量。

### `SimilarCombine` 类

- `__init__`: 初始化，接收一个分词器和一个模型。
- `remove_self_loops`: 移除网络图中的自环边。
- `merge_networks_with_threshold`: 合并相似的关键词子网络，基于词向量的相似度和给定的阈值，聚类关键词，并构建一个网络图，其中节点表示关键词，边表示关键词之间的相似性。

### `FileHandler` 类

- `get_xlsx_files`: 获取指定文件夹中所有 `.xlsx` 文件的路径。
- `remove_file`: 删除指定路径的文件。
- `save_network_as_gexf`: 将网络图保存为 GEXF 格式的文件。

### `process_file` 函数

- 这个函数处理单个 Excel 文件，读取文件内容，合并指定列的文本，并使用预处理函数处理文本。
- 提取文本的关键词，使用 `ScibertEmbedding` 生成关键词的词向量，并通过 `SimilarCombine` 类合并相似的关键词网络。
- 最终将生成的网络图保存为 GEXF 格式的文件。

### `process_files` 函数

- 用于批量处理文件夹中的 Excel 文件。
- 读取文件夹中的所有 Excel 文件，合并指定列的文本，并进行预处理和关键词提取。
- 使用 `KeywordExtractor` 类训练关键词提取模型并提取关键词。
- 使用 `ThreadPoolExecutor` 并发处理每个文件，生成网络图并保存。



## 调用示例

以下是如何使用这段代码来处理特定文件夹中的 Excel 文件的示例。假设我们要分析的 Excel 文件包含了需要提取关键词的文本数据，并且这些文本数据分布在名为 `"AB"` 和 `"TI"` 的列中。我们想要提取每段文本的前 10 个关键词，并设定相似度阈值为 0.2 用于聚类：

```python
# 导入必要的模块
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# 设置文件夹路径
folder_path = 'path/to/your/folder'

# 设置处理参数
sheet_name = 'Sheet1'  # Excel 文件中的工作表名称
column_list = ['AB', 'TI']  # 需要分析的列
similarity_threshold = 0.2  # 相似度聚类阈值
num_keywords = 10  # 每段文本提取的关键词数量

# 调用 process_files 函数来处理文件夹中的所有 Excel 文件
process_files(
    folder_path=folder_path,
    sheet_name=sheet_name,
    column_list=column_list,
    similarity_threshold=similarity_threshold,
    num_keywords=num_keywords
)
```

`process_files` 函数会处理指定文件夹中的每个 Excel 文件，首先读取每个文件，合并指定的列（在这个例子中是 `"AB"` 和 `"TI"`），然后对合并后的文本进行预处理和关键词提取。之后，使用 SciBERT 模型生成词向量，并基于这些词向量以及设定的相似度阈值进行聚类。最终，将构建的关键词网络图保存为 GEXF 格式的文件，可以使用图形可视化工具（如 Gephi）查看和分析这些网络图。
