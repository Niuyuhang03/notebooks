### EMBEDDING ENTITIES AND RELATIONS FOR LEARNING AND INFERENCE IN KNOWLEDGE BASES（DistMult）

https://blog.csdn.net/damuge2/article/details/87990277

### Complex Embeddings for Simple Link Prediction（complEx）

### Convolutional 2D Knowledge Graph Embeddings（ConvE，图结构的多层卷积网络）

方向：实体关系连接预测

![ConvE](https://i.loli.net/2019/05/08/5cd279b39126c.jpg)

知识图谱中实体间关系缺失是普遍问题。针对KG连接预测问题，提出了一种多层卷积神经网络模型 ConvE，主要优点就是参数利用率高（相同表现下参数是DistMult的8分之一，R-GCN的17分之一），擅长学习有复杂结构的KG，并利用1-N scoring来加速训练和极大加速测试过程，即以实体关系对(s,r)为输入，并同时与所有entity进行打分。

NLP中的CNN常用的是Conv1D，即把embedding拼接起来进行卷积，而本文用的模型用的是Conv2D，把输入的实体关系二元组的embedding reshape成一个矩阵，并将其看成是一个图形用二维卷积核提取embedding之间的联系，这个模型最耗时的部分就是卷积计算部分。为了加快feed-forward速度，作者在最后把二元组的特征与KG中所有实体的embedding进行点积，同时计算N个三元组的score（即1-N scoring），这样可以极大地减少计算时间。
