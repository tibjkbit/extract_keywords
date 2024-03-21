import os
from concurrent.futures import ThreadPoolExecutor  # 线程池执行器
from functools import reduce
import networkx as nx  # 网络分析库
import numpy as np
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer  # 句子嵌入模型
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF算法
from transformers import AutoTokenizer, AutoModel  # BERT预训练语言模型
from umap import UMAP  # 降维算法（降维词向量以进行聚类）

# 自定义的文本预处理函数
from processdata import NLP_Prpcessing


# 关键词提取类
class KeywordExtractor:
    def __init__(self, model):
        self.model = model  # KeyBERT 模型
        self.vectorizer = TfidfVectorizer()  # TF-IDF 向量化器

    def fit_vectorizer(self, texts):
        self.vectorizer.fit(texts)  # 模型训练

    def extract_keywords(self, text, num_keywords=10):
        # 提取关键词
        keywords = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=50)

        keyword_strings = [kw for kw, _ in keywords]

        # 计算词的TF-IDF权重
        tfidf_matrix = self.vectorizer.transform([text] + keyword_strings)

        # 获得关键词对应的TF-IDF值
        keyword_indices = list(range(1, tfidf_matrix.shape[0]))
        tfidf_scores = tfidf_matrix.todense()[keyword_indices].tolist()[0]

        # 按TF-IDF值降序排序
        keyword_scores = list(zip(keywords, tfidf_scores))
        sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)

        # 返回前num_keywords = 10个关键词
        final_keywords = [kw for kw, _ in sorted_keywords[:num_keywords]]

        return final_keywords


# 嵌入模型
class ScibertEmbedding:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, keywords):
        embeddings = []
        for keyword in keywords:
            inputs = self.tokenizer(keyword, return_tensors="pt")
            outputs = self.model(**inputs)

            # 模型获取关键词的词向量
            embeddings.append(outputs.last_hidden_state[:, 0, :].detach().numpy().flatten())

        return np.array(embeddings)
        # 返回关键词对应的词向量数组


# 相似度计算与聚类
class SimilarCombine:

    # 初始化设置
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    # 定义移除图中的自环边方法
    @staticmethod
    def remove_self_loops(graph):
        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)

    # 定义合并相似的关键词子网络的方法
    def merge_networks_with_threshold(self, networks, threshold, focal_keywords):

        # 构建包含所有关键词的完整网络图
        merged_network = nx.Graph()
        all_keywords = reduce(lambda x, y: x + y, networks)  # 将networks中的所有元素（每个元素是一个列表）合并成一个单一的列表 -> all_keywords
        unique_keywords = list(set([kw for kw, _ in all_keywords]))  # 利用set性质去重 -> unique_keywords

        # 遍历每个关键词子网络
        for network in networks:

            # 遍历当前子网络的每个关键词
            for i, (term, weight) in enumerate(network):

                # 如果该关键词已在merged_network中,更新其次数和标签
                if merged_network.has_node(term):
                    merged_network.nodes[term]["size"] += 1
                    if term in focal_keywords:
                        merged_network.nodes[term]["label"] = "merge"  # merger标签设置

                # 否则,将其添加到merged_network中
                else:
                    if term in focal_keywords:
                        merged_network.add_node(term, size=1, label="output")  # output标签设置
                    else:
                        merged_network.add_node(term, size=1, label="input")  # input标签设置

                # 在子网络中添加该关键词与网络中其他关键词的边（全联通、无向图）
                for j in range(i):
                    merged_network.add_edge(network[j][0], network[i][0])
                    merged_network.add_edge(network[i][0], network[j][0])

        # 获取所有唯一关键词的词向量
        keyword_embeddings = ScibertEmbedding(self.tokenizer, self.model)(unique_keywords)

        # 对词向量进行降维
        umap = UMAP(n_neighbors=5, n_components=2)
        reduced_embeddings = umap.fit_transform(keyword_embeddings)

        # 按词向量聚类
        clusters = []
        # 对每一个唯一关键词（unique_keywords）及其对应的降维后的词向量（reduced_embeddings）进行遍历。
        for i, (node, embedding) in enumerate(zip(unique_keywords, reduced_embeddings)):

            # 获取当前关键词的信息
            label = merged_network.nodes[node]["label"]
            size = merged_network.nodes[node]["size"]

            min_distance = float('inf')  # 初始最小距离为无穷大
            closest_cluster = None  # 最近的聚类初始为None

            for cluster in clusters:
                # 计算与该聚类的中心的距离
                centroid = np.mean(cluster['embeddings'], axis=0)
                distance = np.linalg.norm(embedding - centroid)

                # 如果距离小于阈值且小于当前的最小距离，更新最小距离和最近的聚类
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster

            if closest_cluster is not None:
                # 如果找到了最近的聚类，将当前节点添加到该聚类
                closest_cluster['nodes'].append(node)
                closest_cluster['embeddings'].append(embedding)
                closest_cluster['label'].append(label)
                closest_cluster['size'].append(size)
            else:
                # 如果没有找到合适的聚类,则新建一个聚类
                clusters.append({'nodes': [node],
                                 'embeddings': [embedding],
                                 'label': [label],
                                 'size': [size]})

        # 合并属于同一聚类的关键词
        for cluster in clusters:

            merged_node = cluster['nodes'][0]

            # 找到除聚类中心外的其他关键词
            nodes_to_remove = [n for n in cluster['nodes'] if n != merged_node]

            # 将这些关键词的边转移到聚类中心（删除被合并的节点的边）
            for node in nodes_to_remove:

                neighbors = list(merged_network.neighbors(node))

                for neighbor in neighbors:

                    if not merged_network.has_edge(merged_node, neighbor):
                        merged_network.add_edge(merged_node, neighbor)

            # 删除这些关键词（删除被合并的节点）
            for node in nodes_to_remove:
                merged_network.remove_node(node)

            # 更新聚类中心的标签和大小
            if len(set(cluster['label'])) != 1:
                merged_network.nodes[merged_node]["label"] = "merge"

            merged_network.nodes[merged_node]["size"] = sum(cluster['size'])

        # 删除自环
        self.remove_self_loops(merged_network)
        return merged_network


# 文件处理类
class FileHandler:

    # 定义文件路径获取方法
    @staticmethod
    def get_xlsx_files(folder_path):
        xlsx_files = []
        for file in os.listdir(folder_path):
            if file.endswith('.xlsx'):
                xlsx_files.append(os.path.join(folder_path, file))
        return xlsx_files

    # 定义文件路径删除方法
    @staticmethod
    def remove_file(file_path):
        os.remove(file_path)

    # 定义保存网络图为gexf格式文件方法
    def save_network_as_gexf(self, graph, filename):
        nx.write_gexf(graph, filename)


# 定义处理单个Excel文件函数
def process_file(file, sheet_name, column_list, similarity_threshold, num_keywords, keyword_extractor):
    df = pd.read_excel(file, sheet_name=sheet_name)
    print(file + " is processing")

    # 连接指定列中的文本
    df['combined'] = df[column_list].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    combined_data = df['combined']

    # 使用NLP_Prpcessing库（自定义库）预处理文本
    combined_data = combined_data.apply(NLP_Prpcessing)

    keywords_list = []

    # 获取focal_keywords，以便添加标签
    focal_keywords = set(keyword_extractor.extract_keywords(combined_data.iloc[0], num_keywords=num_keywords))
    focal_keywords = [tup[0] for tup in focal_keywords]

    for text in combined_data:
        if not isinstance(text, str):
            text = str(text)

        keywords = keyword_extractor.extract_keywords(text, num_keywords=num_keywords)
        keywords_list.append(keywords)

    # 提取每段文本的关键词
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    similar_combine = SimilarCombine(tokenizer, model)

    merged_network = similar_combine.merge_networks_with_threshold(keywords_list, similarity_threshold, focal_keywords)

    # 构建关键词网络图
    file_handler = FileHandler()
    file_handler.save_network_as_gexf(merged_network, f"{file.replace('.xlsx', '.gexf')}")
    print(file + " is done")


# 定义批量处理文件夹中的Excel文件函数
def process_files(folder_path, sheet_name='Sheet1', column_list=None, similarity_threshold=0.2, num_keywords=10):
    # 默认"AB",实际调用时调整为了"AB"+"TI"
    if column_list is None:
        column_list = ['AB']

    xlsx_files = FileHandler.get_xlsx_files(folder_path)

    all_texts = []
    for file in xlsx_files:
        df = pd.read_excel(file, sheet_name=sheet_name)
        df['combined'] = df[column_list].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        combined_data = df['combined']
        all_texts.extend(combined_data.tolist())

    # 训练关键词提取模型
    keyword_extractor = KeywordExtractor(KeyBERT(model=SentenceTransformer('allenai/scibert_scivocab_uncased')))
    keyword_extractor.fit_vectorizer(all_texts)

    # 批量处理
    with ThreadPoolExecutor(max_workers=5) as executor:
        for file in xlsx_files:
            executor.submit(process_file, file, sheet_name, column_list, similarity_threshold, num_keywords,
                            keyword_extractor)
