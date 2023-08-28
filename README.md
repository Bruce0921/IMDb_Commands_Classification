# IMDb_Commands_Classification
Trying NLP project

# 情感分析项目的基本步骤：
```
数据收集:
你可以使用现有的数据集，例如IMDb电影评论数据集，这是一个经常被用于情感分析的数据集。
也可以考虑使用API（如Twitter API）来收集自己的数据。

数据预处理:
分词：将句子分解成单词或标记。
去除停用词：例如“的”、“和”、“是”等常见词。
词形还原：将单词还原到其基本形式。
向量化：将文本转换为数字形式，常用的方法有TF-IDF、Word2Vec或BERT embeddings。

模型选择:
从逻辑回归、朴素贝叶斯或支持向量机等开始。
如果你熟悉深度学习，可以尝试使用RNN、LSTM、GRU或Transformer。

模型训练:
使用训练数据集训练模型。
使用验证数据集调整超参数。

评估:
使用测试数据集评估模型的性能。
常用的评估指标有准确率、F1分数、精确度和召回率。

模型优化:
根据评估结果，可能需要返回并调整模型结构、超参数或预处理步骤。

部署:
一旦模型训练完成并满意其性能，你可以考虑将其部署为Web应用或API。

工具和库:
数据处理和机器学习：Scikit-learn, Pandas, NumPy
深度学习：TensorFlow, Keras, PyTorch
文本处理：NLTK, SpaCy
```

# First Step: Preprocessing Data

### 数据清洗:
移除HTML标签：评论可能包含HTML标签，如<br>。
移除非字母字符：例如数字、标点符号等。
将所有文本转换为小写：这样可以确保词汇的统一性。

### 分词:
将评论分解成单词或标记。

### 去除停用词:
停用词是那些在文本中频繁出现但对分析没有太大意义的词，如“the”、“and”、“is”等。

### 词形还原:
将单词还原到其基本形式，例如将“running”还原为“run”。

### 向量化:
将文本转换为数字形式，常用的方法有TF-IDF、Word2Vec等。

# Learning about the Data

### Distribution of Sentiments
![image](https://github.com/Bruce0921/IMDb_Commands_Classification/blob/main/graphs/Distribution_of_Sentiments.jpg)

### Positive WordCloud
![image](https://github.com/Bruce0921/IMDb_Commands_Classification/blob/main/graphs/MostFrequentPositive.jpg)

### Negative WordCloud
![image](https://github.com/Bruce0921/IMDb_Commands_Classification/blob/main/graphs/MostFrequentNegative.jpg)

![image](https://github.com/Bruce0921/IMDb_Commands_Classification/blob/main/graphs/t-SNE_visutalization_word2vecEmbeddings.jpg)

![image](https://github.com/Bruce0921/IMDb_Commands_Classification/blob/main/graphs/regression_model_trained.png)


# Results
seems both regression model or BERT transformer model are having a somewhat promising results with accuracy around 88%
