# 机器学习基础算法

+ 数据集来源为[100-Days-Of-ML-Code](https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/datasets/Social_Network_Ads.csv)、[digit-recognizer](https://www.kaggle.com/c/digit-recognizer/data)。

## 模型代码

+ 数据处理：读取数据，将非数值数据编码，处理丢失数据，拆分数据集为训练集、验证集和测试集，归一化。
  
+ Pytorch模型
  
  ```python
  import torch
  import torch.nn as nn
  import torch.utils.data
  import torch.nn.functional as F
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  import seaborn as sns
  import matplotlib.pyplot as plt
  from keras.utils.np_utils import to_categorical
  
  
  # 读取数据
  test_x = pd.read_csv("../input/digit-recognizer/test.csv")
  train = pd.read_csv("../input/digit-recognizer/train.csv")
  train_y = train['label']
  train_x = train.drop('label', axis=1)
  
  # 查看数据前几行预览
  print(train_x.head())
  # 查看数据集详情
  print(train_x.describe())
  # 查看某一列的数据集分布
  print(train_x['pixel0'].value_counts())
  # 查看某一列的数据集分布图
  sns.countplot(train_y)
  plt.show()
  # 对特定某行转化为1和0
  train_x['Sex'] = pd.factorize(train_x.Sex)[0]
  # 查看nan
  print("train_x describe: \n", train_x.isnull().any().describe())
  print("\ntest_x describe: \n", test_x.isnull().any().describe())
  print("\ntrain_y describe: \n", train_y.value_counts())
  # 将缺失数据替换为平均值
  if np.isnan(train_x.astype(float)).sum() > 0:
      print("NaN exists in train_X.")
      train_x = train_x.fillna(train_x.mean())
  
  # 归一化
  train_x /= 255
  test_x /= 255
  
  # 改变维度
  train_x = train_x.values.reshape(-1, 1, 28, 28)
  test_x = test_x.values.reshape(-1, 1, 28, 28)
  
  # 划分训练集和验证集。为防止数据不均衡，设置stratify可以保证训练集和验证集里各类比例相同
  train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.1, stratify=train_y)
  
  # 对y从0-9转化为独热码，结果为np.array。具体是否需要独热吗，根据loss函数确定，二分类BCELoss需要，多分类CELoss不需要。
  train_y = to_categorical(train_y, num_classes=10)
  validation_y = to_categorical(validation_y, num_classes=10)
  
  # cpu or gpu
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  
  # 数据变为tensor
  train_x = torch.FloatTensor(np.array(train_x)).to(device)
  validation_x = torch.FloatTensor(np.array(validation_x)).to(device)
  test_x = torch.FloatTensor(np.array(test_x)).to(device)
  train_y = torch.LongTensor(np.array(train_y)).to(device)
  validation_y = torch.LongTensor(np.array(validation_y)).to(device)
  
  # 定义参数
  num_epochs = 10
  num_classes = 10
  learning_rate = 0.01
  batch_size = 100
  
  # 分batch
  train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
  validation_dataset = torch.utils.data.TensorDataset(validation_x, validation_y)
  test_dataset = torch.utils.data.TensorDataset(test_x)
  
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
  
  # 定义模型（CNN）
  class ConvNet(nn.Module):
      def __init__(self, num_classes):
          super(ConvNet, self).__init__()
          # 28*28*1
          self.layer1 = nn.Sequential(
              nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=3),  # 28*28*8
              nn.BatchNorm2d(8),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2),  # 14*14*8
              nn.Dropout(0.5))
  
          self.layer2 = nn.Sequential(
              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3),  # 14*14*16
              nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2),  # 7*7*16
              nn.Dropout(0.5))
  
          self.fc = nn.Linear(7*7*16, num_classes)
  
      def forward(self, x):
          out = self.layer1(x)
          out = self.layer2(out)
          out = out.reshape(out.size(0), -1)  # 即将batch*7*7*32的数据集改为batch*(7*7*32)的大小，进入全连接层
          out = self.fc(out)
          return out
  
  # 实例化模型
  model = ConvNet(num_classes).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  train_losses = []
  total_train_step = len(train_loader)
  total_validation_step = len(validation_loader)
  
  # 训练和验证
  for epoch in range(num_epochs):
      correct = 0
      for step, (train_x, train_y) in enumerate(train_loader):
          train_x, train_y = train_x.to(device), train_y.to(device)
          model.train()
          outputs = model(train_x)
          train_loss = criterion(outputs, train_y)
          correct += (torch.max(outputs, 1)[1] == train_y).sum().item()
  
          optimizer.zero_grad()
          train_loss.backward()
          optimizer.step()
          
          if (step + 1) % 100 == 0:
              print("step[{}/{}], loss:{:.4f}".format(step + 1, total_train_step, train_loss))
  
      train_losses.append(train_loss.item())
      train_acc = correct / (total_train_step * batch_size)
              
      model.eval()
      correct = 0
      for step, (validation_x, validation_y) in enumerate(validation_loader):
          validation_x, validation_y = validation_x.to(device), validation_y.to(device)
          outputs = model(validation_x)
          validation_loss = criterion(outputs, validation_y)
          correct += (torch.max(outputs, 1)[1] == validation_y).sum().item()
      validation_acc = correct / (total_validation_step * batch_size)
      print('epoch[{}/{}], train loss:{:.4f}, train acc:{:.4f}, validation loss:{:.4f}, validation acc:{:.4f}'.format(epoch + 1, num_epochs, train_loss.item(), train_acc, validation_loss.item(), validation_acc))
  
  # 测试
  model.eval()
  with torch.no_grad():
      for step, test_x in enumerate(test_loader):
          test_x = test_x[0].to(device)
          output = model(test_x)
          if not step:
              test_y = outputs
          else:
              test_y = torch.cat((test_y, output), 0)
  # 写入数据到文件
  if torch.cuda.is_available():
      test_y = test_y.cpu()
  test_y = pd.DataFrame(torch.argmax(test_y, 1).numpy())
  test_id = pd.DataFrame([i for i in range(1, test_y.shape[0] + 1)])
  test_y = pd.concat([test_id, test_y], axis=1)
  test_y.columns = ['ImageId', 'Label']
  test_y.to_csv('submission.csv', index=False, encoding='utf8')
  
  # loss可视化
  plt.plot([i + 1 for i in range(num_epochs)], train_losses, 'r-', lw=1)
  plt.yticks([x * 0.1 for x in range(15)])
  plt.show()
  ```

## 贝叶斯公式

+ 后验概率$P(w_i|x)=\frac{P(x|w_i)P(w_i)}{P(x)}$，$P(w_i)$为先验概率。已知是黑人x，求是非洲人wi概率：$P(非洲人|黑人)=P(非洲人)*\frac{P(黑人|非洲人)}{P(黑人)}$，$P(非洲人)$是先验概率。对于朴素贝叶斯，假设各个特征之间独立且相同重要，在现实世界不可能，故朴素。
+ 概率模型不需要归一化。

## 分类

+ 有监督，离散。

### kNN（k-邻近算法）

+ kNN是监督学习的一种**分类**算法。其核心思想为根据特征，**找到距离测试集距离最近的k个训练集，统计训练集的label，将出现次数最多的label作为预测的label**，不具有显示学习过程。kNN计算量很大，通常k不大于20，k过小则噪声影响很大，k过大则计算速度较慢，推荐遍历k选取最优值。**kNN通常需要归一化**，使得样本的权重相同。

+ kNN和k-means中使用的是欧氏距离，即两点空间的距离。而不是曼哈顿距离，即每个坐标轴下的距离之和。相比之下，欧氏距离是更可行的，没有维度限制。

+ ```python
def classify(normal_train_X, train_Y, normal_test_X, k):
    '''
    预测
    :param normal_train_X: 归一化后的训练集特征
    :param train_Y: 训练集标签
    :param normal_test_X: 归一化后的测试集特征
    :param k: 最近k个点
    :return: 测试集标签
    '''
    num_normal_train_X = normal_train_X.shape[0]
    num_normal_test_X = normal_test_X.shape[0]
    predict_test_Y = []
    for i in range(num_normal_test_X):
        # 赋值测试集每一行，与训练集由欧式距离求出距离后排序
        sq_diff = (np.tile(normal_test_X[i, :], (num_normal_train_X, 1)) - normal_train_X) ** 2
        diff = (sq_diff.sum(axis=1)) ** 0.5
        # 使用排序后的索引
        sorted_diff_index = diff.argsort()
        predict_label = {}
        for j in range(k):
            label = train_Y[sorted_diff_index[j]]
            predict_label[label] = predict_label.get(label, 0) + 1
        sorted_predict_label = sorted(predict_label.items(), key=operator.itemgetter(1), reverse=True)
        predict_test_Y.append(sorted_predict_label[0][0])
    return np.array(predict_test_Y)
  ```

### 逻辑回归

+ 逻辑回归是有监督的**分类**算法（而不是回归算法），一般的逻辑回归只能用于二分类，多分类需要多个逻辑回归模型。**其原理和线性回归类似，只是给回归方程加上了`sigmoid`函数，使得结果映射为0和1**。相应的，修改损失函数，梯度下降求出最优$\theta$。理解逻辑回归需要先理解线性回归。逻辑回归实质上是找到一个决策边界。逻辑回归的回归函数$h_\theta(x)=sigmoid(\theta^Tx)$，$sigmoid(t)=\frac{1}{1+e^{-t}}$。

  ![sigmoid](http://ww1.sinaimg.cn/large/96803f81ly1fzfkiptauxj20dv0dtq3f.jpg)

  可以看到，S函数将在t>0时，y>0.5，在t<0时，y<0.5，因此可以将测试集标签分为1和0。对于已知m*n维的X，逻辑回归的损失函数为$J(\theta)=\frac{1}{m}\Sigma^m_{i=1}[-y^{(i)}\log(h_\theta(x^{(i)}))-(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$。逻辑回归需要用***正则化**，正则化后的梯度为$\frac{\partial J(\theta)}{\partial\theta_j}=\frac{1}{m}\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j+\frac{\lambda}{m}\theta_j(j>0)$。


+ ```python
  def gradient_descent(normal_train_X, train_Y, alpha, num_iters, lamb):
      '''
      梯度下降求theta
      :param lamb：正则化系数
      :param normal_train_X:归一化后的训练集特征
      :param train_Y:训练集标签
      :param alpha:学习速率
      :param num_iters:迭代次数
      :return:回归系数theta
      '''
      num_train_Y = train_Y.shape[0]
      m_normal_train_X = normal_train_X.shape[1]
      add_mar = np.tile([1],(num_train_Y, 1))
      normal_train_X_add = np.hstack((add_mar, normal_train_X))
      theta = np.zeros((m_normal_train_X + 1,1))
      for iter in range(num_iters):
          error = np.zeros((m_normal_train_X + 1,1))
          for j in range(m_normal_train_X + 1):
              error[j] =  np.dot(np.transpose(normal_train_X_add[:, j]), (sigmoid(np.dot(normal_train_X_add, theta)) - np.transpose([train_Y]))) + lamb / num_train_Y * theta[j]
          error[0] -= lamb / num_train_Y * theta[0]
          theta -= error * alpha / num_train_Y
      return theta
  
  def sigmoid(z):
      '''
      sigmoid函数
      :param z:
      :return: sigmoid(z)
      '''
      return 1.0 / (1 + np.exp(-z))
  
  def classify(normal_train_X, train_Y, normal_test_X, alpha, num_iters, lamb):
      '''
      预测
      :param lamb：正则化系数
      :param normal_train_X: 归一化后的训练集特征
      :param train_Y: 训练集标签
      :param normal_test_X: 归一化后的测试集特征
      :param alpha：学习速率
      :param num_iters：迭代次数
      :return: 测试集标签，theta
      '''
      theta = gradient_descent(normal_train_X, train_Y, alpha, num_iters, lamb)
      add_mar = np.tile([1], (normal_test_X.shape[0], 1))
      normal_train_X_add = np.hstack((add_mar, normal_test_X))
      predict_test_Y = sigmoid(np.dot(normal_train_X_add, theta).flatten())
      for i in range(predict_test_Y.shape[0]):
          if predict_test_Y[i] > 0.5:
              predict_test_Y[i] = 1
          else:
              predict_test_Y[i] = 0
      return predict_test_Y.astype(int), theta
  ```

### 决策树

+ 决策树是有监督**分类**算法，过程包括决策树生成和剪枝两个过程。决策树如果不经过剪枝过程，很容易过拟合。决策树生成以ID3算法为例，即从根节点开始，计算每个特征的信息增益，取信息增益最大的特征作为该节点上的特征。

  + 信息增益：对于每个特征，计算其信息熵$H(D)=-\Sigma_{c=1}^Cp_c\log_2(p_c)$，其中c为最终的label，$p_c$为label为c的比例。再计算该特征中每个具体小特征i的信息熵$H(D_i)=-\Sigma_{c=1}^Cp_{c,i}\log_2(p_{c,i})$。计算该特征的信息增益$G(D,feature1)=H(D)-\Sigma_{i=1}^lp_iH(D_i)$，$p_i$为小特征i的比例。取信息增益最大的特征作为下一个节点。
  + 剪枝：分为预剪枝和后剪枝。预剪枝每生成一步都要判断是否剪掉后泛化性能更好，不好则不生成。后剪枝是在都生成完后，自下而上判断。

+ ```python
  def calcShannonEnt(dataSet):
      '''
      计算香农熵
      :param dataSet:数据集
      :return: 香农熵
      '''
      numEntires = len(dataSet)
      labelCounts = {}
      for featVec in dataSet:
          currentLabel = featVec[-1]
          if currentLabel not in labelCounts.keys():
              labelCounts[currentLabel] = 0
          labelCounts[currentLabel] += 1
      shannonEnt = 0.0
      for key in labelCounts:
          prob = float(labelCounts[key]) / numEntires
          shannonEnt -= prob * log(prob, 2)
      return shannonEnt
  
  def splitDataSet(dataSet, axis, value):
      '''
      划分数据集
      :param dataSet: 待划分的数据集（特征+标签）
      :param axis: 划分数据集的特征
      :param value: 需要返回的特征的值
      :return：划分后的数据集
      '''
      retDataSet = []
      dataSet = np.array(dataSet).tolist()
      for featVec in dataSet:
          if featVec[axis] == value:
              reducedFeatVec = featVec[:axis]
              reducedFeatVec.extend(featVec[axis+1:])
              retDataSet.append(reducedFeatVec)
      return np.array(retDataSet)
  
  def chooseBestFeatureToSplit(dataSet):
      '''
      选择最优特征
      :param dataSet: 训练集（特征+标签）
      :return: 信息增益最大的特征的索引值
      '''
      numFeatures = len(dataSet[0]) - 1
      baseEntropy = calcShannonEnt(dataSet)
      bestInfoGain = 0.0
      bestFeature = -1
      for i in range(numFeatures):
          featList = [example[i] for example in dataSet]
          uniqueVals = set(featList)
          newEntropy = 0.0
          for value in uniqueVals:
              subDataSet = splitDataSet(dataSet, i, value)
              prob = subDataSet.shape[0] / float(len(dataSet))
              newEntropy += prob * calcShannonEnt(subDataSet)
          infoGain = baseEntropy - newEntropy
          if (infoGain > bestInfoGain):
              bestInfoGain = infoGain
              bestFeature = i
      return bestFeature
  
  def majorityCnt(classList):
      '''
      统计classList中出现此处最多的元素
      :param classList: 类标签列表
      :return: 现此处最多的元素
      '''
      classCount = {}
      for vote in classList:
          classCount[vote] = classCount.get(vote, 0) + 1
      sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
      return sortedClassCount[0][0]
  
  def createTree(dataSet, labels, featLabels):
      '''
      创建决策树
      :param dataSet: 训练数据集（特征+标签）
      :param labels: 特征名称
      :param featLabels: 存储选择的最优特征标签
      :return:
      '''
      classList = [example[-1] for example in dataSet]
      if classList.count(classList[0]) == len(classList):
          return classList[0]
      if len(dataSet[0]) == 1 or len(labels) == 0:
          return majorityCnt(classList)
      bestFeat = chooseBestFeatureToSplit(dataSet)
      bestFeatLabel = labels[bestFeat]
      featLabels.append(bestFeatLabel)
      myTree = {bestFeatLabel:{}}
      del(labels[bestFeat])
      dataSet = np.delete(dataSet, bestFeat, 1)
      featValues = [example[bestFeat] for example in dataSet]
      uniqueVals = set(featValues)
      for value in uniqueVals:
          myTree[bestFeatLabel][value] = createTree(dataSet, labels, featLabels)
      return myTree
  
  def getNumLeafs(myTree):
      '''
      获取决策树叶子结点的数目
      :param myTree: 决策树
      :return: 决策树的叶子结点的数目
      '''
      numLeafs = 0
      firstStr = next(iter(myTree))
      secondDict = myTree[firstStr]
      for key in secondDict.keys():
          if type(secondDict[key]).__name__=='dict':
              numLeafs += getNumLeafs(secondDict[key])
          else:   numLeafs +=1
      return numLeafs
  
  def getTreeDepth(myTree):
      '''
      获取决策树的层数
      :param myTree: 决策树
      :return: 层数
      '''
      maxDepth = 0
      firstStr = next(iter(myTree))
      secondDict = myTree[firstStr]
      for key in secondDict.keys():
          if type(secondDict[key]).__name__=='dict':
              thisDepth = 1 + getTreeDepth(secondDict[key])
          else:   thisDepth = 1
          if thisDepth > maxDepth: maxDepth = thisDepth
      return maxDepth
  
  def plotNode(nodeTxt, centerPt, parentPt, nodeType):
      '''
      绘制结点
      :param nodeTxt: 结点名
      :param centerPt: 文本位置
      :param parentPt: 标注的箭头位置
      :param nodeType: 结点格式
      :return:
      '''
      arrow_args = dict(arrowstyle="<-")
      font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
      createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)
  
  def plotMidText(cntrPt, parentPt, txtString):
      '''
      标注有向边属性值
      :param cntrPt: 用于计算标注位置
      :param parentPt: 用于计算标注位置
      :param txtString: 标注的内容
      '''
      xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
      yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
      createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
  
  def plotTree(myTree, parentPt, nodeTxt):
      '''
      绘制决策树
      :param myTree: 决策树
      :param parentPt: 标注的内容
      :param nodeTxt: 结点名
      '''
      decisionNode = dict(boxstyle="sawtooth", fc="0.8")
      leafNode = dict(boxstyle="round4", fc="0.8")
      numLeafs = getNumLeafs(myTree)
      depth = getTreeDepth(myTree)
      firstStr = next(iter(myTree))
      cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
      plotMidText(cntrPt, parentPt, nodeTxt)
      plotNode(firstStr, cntrPt, parentPt, decisionNode)
      secondDict = myTree[firstStr]
      plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
      for key in secondDict.keys():
          if type(secondDict[key]).__name__=='dict':
              plotTree(secondDict[key],cntrPt,str(key))
          else:
              plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
              plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
              plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
      plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
  
  def createPlot(inTree):
      '''
      创建绘制面板
      :param inTree: 决策树
      '''
      fig = plt.figure(1, facecolor='white')
      fig.clf()
      axprops = dict(xticks=[], yticks=[])
      createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
      plotTree.totalW = float(getNumLeafs(inTree))
      plotTree.totalD = float(getTreeDepth(inTree))
      plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
      plotTree(inTree, (0.5,1.0), '')
      plt.show()
  
  def classify(inputTree, featLabels, testVec):
      '''
      预测
      :param inputTree: 已经生成的决策树
      :param featLabels: 存储选择的最优特征标签
      :param testVec: 测试数据列表，顺序对应最优特征标签
      :return: 分类结果
      '''
      firstStr = next(iter(inputTree))
      secondDict = inputTree[firstStr]
      featIndex = featLabels.index(firstStr)
      for key in secondDict.keys():
          if testVec[featIndex] == key:
              if type(secondDict[key]).__name__ == 'dict':
                  classLabel = classify(secondDict[key], featLabels, testVec)
              else:
                  classLabel = secondDict[key]
      return classLabel
  
  def storeTree(inputTree, filename):
      '''
      存储决策树
      :param inputTree: 决策树
      :param filename: 决策树的存储文件名
      '''
      with open(filename, 'wb') as fw:
          pickle.dump(inputTree, fw)
  
  def grabTree(filename):
      """
      函数说明:读取决策树
  
      Parameters:
          filename - 决策树的存储文件名
      Returns:
          pickle.load(fr) - 决策树字典
      """
      fr = open(filename, 'rb')
      return pickle.load(fr)
  
  def Classify_tree(normal_train_X, train_Y, normal_test_X):
      '''
      预测
      :param normal_train_X: 归一化后的训练集特征
      :param train_Y: 训练集标签
      :param normal_test_X: 归一化后的测试集特征
      :return: 测试集标签
      '''
      num_normal_train_X = normal_train_X.shape[0]
      num_normal_test_X = normal_test_X.shape[0]
      predict_test_Y = []
      featLabels = []
      dataset = np.append(normal_train_X, train_Y.reshape((num_normal_train_X,1)), axis=1)
      labels = ['Gender','Age','EstimatedSalary']
      myTree = createTree(dataset, labels, featLabels)
      storeTree(myTree, 'classifierStorage.txt')
      createPlot(myTree)
      for i in range(num_normal_test_X):
          result = classify(myTree, featLabels, normal_test_X[i, :])
          predict_test_Y.append(result)
      return np.array(predict_test_Y)
  ```

### SVM

+ SVM是有监督的**分类**算法。将所有样本投射到一个n维空间中，在空间中找到一条超平面，且SVM要求这条超平面距离两侧样本都最远。

  ![超平面](http://ww1.sinaimg.cn/large/96803f81ly1fzgefaiockj209d0bbjsf.jpg)

  同逻辑回归一样，SVM直接使用时只能用来二分类。但SVM泛化错误率低，计算开销小。

+ SVM用序列最小优化算法SMO可以较快求解，SMO主要有如下步骤：求解误差，计算上下界，计算学习速率，更新乘子$\alpha_i$，修剪$\alpha_i$，更新$\alpha_i$，更新$b_1$和$b_2$，更新b。

+ ```python
  def selectJrand(i, m):
      '''
      随机选择alpha
      :param i:alpha
      :param m:alpha参数个数
      :return:
      '''
      j = i
      while (j == i):
          j = int(random.uniform(0, m))
      return j
  
  def clipAlpha(aj,H,L):
      '''
      修剪alpha
      :param aj:alpha值
      :param H:alpha上限
      :param L:alpha下限
      :return:
      '''
      if aj > H:
          aj = H
      if L > aj:
          aj = L
      return aj
  
  def smoSimple(normal_train_X, train_Y, C, toler, maxIter):
      '''
      简化版SMO算法
      :param normal_train_X:归一化后的训练集样本
      :param train_Y:训练集标签
      :param C:惩罚系数
      :param toler:松弛变量
      :param maxIter:最大迭代次数
      :return:
      '''
      normal_train_X = np.mat(normal_train_X)
      train_Y = np.mat(train_Y).transpose()
      b = 0
      m_normal_train_X, n_normal_train_X = np.shape(normal_train_X)
      alphas = np.mat(np.zeros((m_normal_train_X, 1)))
      iter_num = 0
      while (iter_num < maxIter):
          alphaPairsChanged = 0
          for i in range(m_normal_train_X):
              fXi = float(np.multiply(alphas, train_Y).T * (normal_train_X * normal_train_X[i, :].T)) + b
              Ei = fXi - float(train_Y[i])
              if ((train_Y[i] * Ei < -toler) and (alphas[i] < C)) or ((train_Y[i] * Ei > toler) and (alphas[i] > 0)):
                  j = selectJrand(i, m_normal_train_X)
                  # 计算误差Ej
                  fXj = float(np.multiply(alphas, train_Y).T * (normal_train_X * normal_train_X[j, :].T)) + b
                  Ej = fXj - float(train_Y[j])
                  alphaIold = alphas[i].copy()
                  alphaJold = alphas[j].copy()
                  # 计算上下界L和H
                  if (train_Y[i] != train_Y[j]):
                      L = max(0, alphas[j] - alphas[i])
                      H = min(C, C + alphas[j] - alphas[i])
                  else:
                      L = max(0, alphas[j] + alphas[i] - C)
                      H = min(C, alphas[j] + alphas[i])
                  if L == H:
                      continue
                  # 计算eta
                  eta = 2.0 * normal_train_X[i, :] * normal_train_X[j, :].T - normal_train_X[i, :] * normal_train_X[i, :].T - normal_train_X[j,:] * normal_train_X[j, :].T
                  if eta >= 0:
                      continue
                  # 更新alpha_j
                  alphas[j] -= train_Y[j] * (Ei - Ej) / eta
                  # 修剪alpha_j
                  alphas[j] = clipAlpha(alphas[j], H, L)
                  if (abs(alphas[j] - alphaJold) < 0.00001):
                      continue
                  # 更新alpha_i
                  alphas[i] += train_Y[j] * train_Y[i] * (alphaJold - alphas[j])
                  # 更新b_1和b_2
                  b1 = b - Ei - train_Y[i] * (alphas[i] - alphaIold) * normal_train_X[i, :] * normal_train_X[i, :].T - train_Y[j] * (alphas[j] - alphaJold) * normal_train_X[i, :] * normal_train_X[j, :].T
                  b2 = b - Ej - train_Y[i] * (alphas[i] - alphaIold) * normal_train_X[i, :] * normal_train_X[j, :].T - train_Y[j] * (alphas[j] - alphaJold) * normal_train_X[j, :] * normal_train_X[j, :].T
                  # 根据b_1和b_2更新b
                  if (0 < alphas[i]) and (C > alphas[i]):
                      b = b1
                  elif (0 < alphas[j]) and (C > alphas[j]):
                      b = b2
                  else:
                      b = (b1 + b2) / 2.0
                  alphaPairsChanged += 1
                  print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alphaPairsChanged))
          if (alphaPairsChanged == 0):
              iter_num += 1
          else:
              iter_num = 0
      return b, alphas
  
  def get_w(normal_train_X, train_Y, alphas):
      '''
      计算w
      :param normal_train_X:
      :param train_Y:
      :param alphas:
      :return:
      '''
      alphas, dataMat, labelMat = np.array(alphas), np.array(normal_train_X), np.array(train_Y)
      w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, np.shape(dataMat)[1])) * dataMat).T, alphas)
      return w.tolist()
  
  def classify(normal_train_X, train_Y, normal_test_X):
      b, alphas = smoSimple(normal_train_X, train_Y, 0.6, 0.001, 4000)
      w = get_w(normal_train_X, train_Y, alphas)
      predict_test_Y = np.dot(np.mat(normal_test_X), np.mat(w)) + b
      return predict_test_Y.flatten().tolist()[0]
  ```

## 回归

+ 有监督，连续。

### 线性回归

+ 线性回归是一种有监督的**回归**（预测）算法，目的是对于已知m*n维的X，**通过拟合出回归系数$\theta$，得到回归方程$h_\theta(x)=\theta_0+\Sigma_{i=1}^n\theta_ix_i=\theta^Tx$进行预测**。拟合回归系数$\theta$的过程，实际上就是求使得预测的平方误差和（SSE）最小时的$\theta$，即对平方误差和求导，导数为0时的$\theta$。求解$\theta$时，有正规方程和梯度下降两种求解方法。

  + 正规方程即直接求解出导数为0时的$\theta$值$\theta=(X^TX)^{-1}X^Ty$，然而矩阵有逆要求为非奇异矩阵，且这个方法在矩阵X较大时速度慢于梯度下降，因此常用梯度下降法求解线性回归。
  + 梯度下降即通过迭代求损失函数收敛时的$\theta$，迭代有多种方法，如批量梯度下降法（BGD），随机梯度下降（SGD）等。
    + 批量梯度下降法，定义损失函数$J(\theta)=\frac{1}{2m}\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$，类似于均方误差MSE，损失函数的最小值应该在导数（梯度）为0时得到。求出第j个特征的梯度$\frac{\partial J(\theta)}{\partial\theta_j}=\frac{1}{m}\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$，注意**计算每个$\theta_j$时都用到了完整的$\theta$**。更新$\theta$在每个特征处的值$\theta_j=\theta_j-\alpha\frac{\partial J(\theta)}{\partial\theta_j}$，θ向着损失函数梯度变化最大的方向移动，直至函数$J$损失收敛，即可得到回归系数$\theta$。注意BGD要求在一轮内，上一个$\theta$在所有特征的值都计算完后，再更新所有的$\theta_j$。最后通过$\theta$和测试集X相乘，得到测试集标签。梯度下降法要求对X进行**归一化**，来加快迭代速度。
    + 随机梯度下降在求梯度时，每次更新$\theta$只随机用了一个样本，而不是计算m个样本的平均值，收敛速度加快。

+ ```python
  def gradient_descent(normal_train_X, train_Y, alpha, num_iters):
      '''
      梯度下降求theta
      :param normal_train_X:归一化后的训练集特征
      :param train_Y:训练集标签
      :param alpha:学习速率
      :param num_iters:迭代次数
      :return:回归系数theta
      '''
      num_train_Y = train_Y.shape[0]
      m_normal_train_X = normal_train_X.shape[1]
      theta = np.zeros((m_normal_train_X,1))
      for iter in range(num_iters):
          error = np.zeros((m_normal_train_X,1))
          for j in range(m_normal_train_X):
              error[j] =  np.dot(np.transpose(normal_train_X[:, j]), (np.dot(normal_train_X, theta) - np.transpose([train_Y])))
          theta -= error * alpha / num_train_Y
      return theta
  
  def classify(normal_train_X, train_Y, normal_test_X, alpha, num_iters):
      '''
      预测
      :param normal_train_X: 归一化后的训练集特征
      :param train_Y: 训练集标签
      :param normal_test_X: 归一化后的测试集特征
      :param alpha：学习速率
      :param num_iters：迭代次数
      :return: 测试集标签，theta
      '''
      theta = gradient_descent(normal_train_X, train_Y, alpha, num_iters)
      predict_test_Y = np.dot(normal_test_X, theta).flatten()
      return predict_test_Y, theta
  ```

## 聚类

+ 无监督，离散。

### k-means

+ k-means是非监督学习的一种分类算法。通过选取k个样本作为初始聚类中心，计算其他样本到各个聚类中心的距离，将所有样本划分给最近的一个聚类中心，再在每个聚类中求出中心值作为新的聚类中心，重新分配，直至样本不再移动。k-means收敛较慢，且可能会收敛到局部最小值。可以通过误差平方和（SSE）来求出最优k值。

+ ```python
  def get_init_center(normal_train_X, k):
      '''
      得到k个初始聚类中心
      :param normal_train_X: 归一化后的训练集样本
      :param k: 参数k
      :return: 包含k个聚类中心的数组
      '''
      cluster_center = []
      m_normal_train_X = normal_train_X.shape[0]
      for i in range(k):
          cluster_center.append(normal_train_X[int(random.uniform(0, m_normal_train_X)), :].tolist())
      return cluster_center
  
  def classify(normal_train_X, k):
      '''
      预测
      :param normal_train_X: 归一化后的训练集特征
      :param normal_test_X: 归一化后的测试集特征
      :return:聚类结果
      '''
      cluster_center = get_init_center(normal_train_X, k)
      m_normal_train_X = normal_train_X.shape[0]
      all_index = {}
      changed = True
      cnt = 0
      while changed:
          changed = False
          for i in range(m_normal_train_X):
              cur_index = all_index.get(i, -1)
              min_distance = -1
              min_index = -1
              for j in range(k):
                  sq_diff = (normal_train_X[i, :] - np.array(cluster_center[j])) ** 2
                  diff = sum(sq_diff.tolist()) ** 0.5
                  if min_distance == -1 or diff < min_distance:
                      min_distance = diff
                      min_index = j
              if min_index != cur_index:
                  changed = True
                  all_index[i] = min_index
          if changed is True:
              for i in range(k):
                  sum_dis = []
                  for j in range(m_normal_train_X):
                      if all_index[j] == i:
                          sum_dis.append(normal_train_X[i].tolist())
                  cluster_center[i] = np.mean(sum_dis, axis=0)
      predict_test_Y = []
      for i in range(m_normal_train_X):
          predict_test_Y.append(all_index[i])
      return np.array(predict_test_Y)
  ```

### k-means++

+ 在选择初始聚类中心时，尽可能选择距离已选中心最远的点作为下一个中心。

## 数据降维

+ 无监督，连续。

### PCA

+ 主成分分析法，使用较少的数据维度保留住较多的原数据特性，主要有最大方差和最小均方误差两种思想。

## 深度学习

### ANN

+ 人工神经网络ANN是最简单的深度学习算法之一。模型包含一个输入层，若干个隐藏层和一个输出层。隐藏层越多效果越好，训练难度也越大，全连接神经网络隐藏层一般不超过三层。每层有若干个神经元，隐藏层神经元个数可以用$\sqrt{n*l}$来作为初始值，其中n和l为输入层和输出层的神经元个数。

  ![ANN](http://ww1.sinaimg.cn/large/96803f81ly1g06zla4re8j20if09cq73.jpg)

  神经元之间采用全连接，连接上有权重w。当要计算下一层某个神经元的值时，所用公式为$y_1=sigmoid(w^Ta)$。即每个要计算的神经元的值等于上一层与它连接的所有神经元与连接上的权重的乘积和，输入到sigmoid的结果，只需要计算出权重即可。

+ 权重由反向传播更新。

  + 对于输出层，求得误差$\delta_i=y_{i,true}-y_i$}。对于隐藏层，求得误差$\delta_i=\Sigma w\delta$。
  + 更新权重。从输入层开始，$w=w+\eta\delta\frac{\partial y}{\partial\Sigma w} x$，其中$\eta$是学习速率。
  
+ ```python
  num_epochs = 500
  learning_rate = 0.02
  
  class AnnNet(nn.Module):
      def __init__(self):
          super(AnnNet, self).__init__()
          self.fc1 = nn.Linear(7, 5)
          self.tanh = nn.Tanh()
          self.fc2 = nn.Linear(5, 5)
          self.relu = nn.ReLU()
          self.fc3 = nn.Linear(5, 2)
  
      def forward(self, x):
          out = self.fc1(x)
          out = self.tanh(out)
          out = self.fc2(out)
          out = self.relu(out)
          out = self.fc3(out)
          return out
  
  
  model = AnnNet()
  print(model)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  losses = []
  
  for epoch in range(num_epochs):
      model.train()
      optimizer.zero_grad()
      pred = model(train_x)
      loss = criterion(pred, train_y)
      losses.append(loss)
      loss.backward()
      optimizer.step()
      correct = (torch.max(pred.data, 1)[1] == train_y).sum()
      acc = correct.item() / len(pred)
      print("Epoch:[{}/{}], acc: {:.4f}, Loss:{:.4f}".format(epoch + 1, num_epochs, acc, loss.item()))
  
  model.eval()
  with torch.no_grad():
      test_y = model(test_x)
  test_y = pd.DataFrame(torch.argmax(test_y, 1).numpy())
  test_y = pd.concat([test_id, test_y], axis=1)
  test_y.columns = ['PassengerId', 'Survived']
  test_y.to_csv('submission.csv', index=False, encoding='utf8')
  ```

### CNN

+ 一个CNN往往由**N个卷积层和ReLU层叠加，再加一个可选的池化层，将该结构重复M次，最后叠加K个全连接层**，`INPUT -> [[CONV -> ReLU]*N -> POOL?]*M -> [FC]*K`。

+ 卷积层（Convolutional Layer）：卷积层通过滤波器（filter）将图片卷积为Feature Map，即从原始图像提取出的特征，这个过程成为卷积，是一种局部连接的网络。Feature Map计算公式如下：

  ![Feature Map](https://i.loli.net/2019/06/18/5d08e73aecf1920876.jpg)

  其中w为滤波器中第m行n列的权重，w为偏置，x为图片的每个像素，D为图片深度，F为Filter的边长。对于一个5\*5的图片，设置滤波器大小为3\*3，步幅stride为1，偏置为0，则得到的Feature Map大小为3\*3，计算过程如下。

  ![滤波器](https://i.loli.net/2019/06/18/5d08e57a404ea88450.gif)

  Feature Map大小W_2、Filter大小W_1、步幅S、零填充P满足如下公式：

  ![Feature Map大小](https://i.loli.net/2019/06/18/5d08e6bd6f02080619.jpg)

  其中零填充是在图片外侧填充P层0，有利于图像边缘的特征提取。

+ 池化层（Pooling Layer）:池化pooling即汇总，作用是下采样，去掉Feature Map中不必要的参数。常用Max Pooling，Mean Pooling等。

  + Max Pooling是对每个部分取最大值。一个2\*2的Max Pooling如图：

    ![Max Pooling](https://i.loli.net/2019/06/19/5d091615af29320884.jpg)

  对于深度为D的Feature Map，各层独立做Pooling，因此Pooling后的深度仍然为D。

+ 全连接层：计算梯度，反向传播更新权重。

    ```python
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms
    
    x_train = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    x_test = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=x_test, batch_size=batch_size, shuffle=False)
    
    
    class ConvNet(nn.Module):
        def __init__(self, num_classes):
            super(ConvNet, self).__init__()
            # 28*28*1
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),  # 28*28*16
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 14*14*16
                nn.Dropout(0.5))
    
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),  # 14*14*32
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 7*7*32
                nn.Dropout(0.5))
    
            self.fc = nn.Linear(7*7*32, num_classes)
    
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out
    ```

+ CNN为什么可用于CV，NLP，speech等领域：以上领域都存在局部和整体的关系，低层次特征组合为高层次特征的共性。CNN通过卷积、池化等实现此共性。

## 集成学习

+ 以Boosting方法为例。所谓Boosting方法，即迭代提升，故为串行算法。先训练弱学习器，然后根据前一个弱学习器分错的样本，改变样本的概率分布构成新的训练集，从而训练出一个更强的学习器。这样反复迭代提升，就能得到一系列分类器。最后，将这些分类器组合起来，就能构成一个很强的学习器。

### AdaBoost

+ 在不改变训练数据的情况下，通过在迭代训练弱学习器中，不断提升被错分类样本的权重（也就是使被错分的样本在下一轮训练时得到更多的重视），不断减少正确分类样本的权重。最后通过加权线性组合M个弱分类器得到最终的分类器，正确率越高的弱分类器的投票权数越高，正确率低的弱分类器自然投票权数就低。

### GBDT

+ 迭代的决策树算法，由多棵决策树组成，所有树的结论累加起来做最终答案。GBDT中的树是回归树（不是分类树），GBDT用来做回归预测。

### xgboost

+ 相比GBDT，使用了正则化防止过拟合，loss采用了泰勒展开实现二阶导数而不是一阶导数，寻找最佳分割点标准是最大化Lsplit。xgboost在精度和效率上都有了提升。

## Q&A

+ CNN的Pytorch和TF版，对于500张28\*28\*1的图片，为什么train_x.shape=(500, 1, 28, 28)？
  
  + 输入格式为` (batch, channel, H, W)`，即样本数、通道数、高度、宽度
+ model在train之后，需要验证cv集，为什么反传的loss是train的而不是cv的？
  
+ cv集的作用就是查看表现，由于test没有label，只能通过找一个不会用来训练、模型从没见过的数据集（即cv集）来进行验证效果，但模型还是应该根据train集来训练。
  
+ 归一化

  + **先划分后归一化，但只是用训练集进行归一化的fit**，保证不从测试集得到任何数据。在有交叉验证集的数据，也应该**只根据训练集**归一化，在交叉验证集调到最佳参数，在测试集上测试。且每一个特征独自进行归一化，保证各个特征最后都在相同量级内。
  + 具体方法
    + 均值-方差归一化：$x=\frac{x-\mu}{\sigma}$，结果符合标准正态分布，均值为0方差为1，最大最小值没有范围，常用于距离相关的算法如K-means。
    + 最大-最小归一化：$x=\frac{x-x_{min}}{x_{max}-x_{min}}$，结果都在$[0,1]$范围内，常用于图像处理（都在0-255之内）。由于测试数据可能在训练集的最值之外，使用最大-最小归一化可能导致测试数据归一化后不在$[0,1]$范围内，出现问题。

+ 混淆矩阵：二分类问题时可用

  ![混淆矩阵](http://ww1.sinaimg.cn/large/96803f81ly1fzf7rkjiqaj20d406oglx.jpg)

  ```python
  # confusion_matrix混淆矩阵
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(cv_Y, predict_cv_Y)
  print(cm)
  ```

+ 准确率：

  ![准确率](http://ww1.sinaimg.cn/large/96803f81ly1fzf7ucwtzaj207w02rt8m.jpg)

  ```python
  # 准确率
  from sklearn.metrics import accuracy_score
  score = accuracy_score(cv_Y, predict_cv_Y)
  ```

+ F1值：

  ![F1值](http://ww1.sinaimg.cn/large/96803f81ly1fzf7y11jopj2095034t8q.jpg)

  ```python
  # f1_score， 用于二分类
  from sklearn.metrics import f1_score
  score = f1_score(cv_Y, predict_cv_Y)
  print(cm)
  ```

+ 均方误差MSE：

  ![均方误差](http://ww1.sinaimg.cn/large/96803f81ly1fzgd80u212j207a034wed.jpg)

  ```python
  # 均方误差MSE
  from sklearn.metrics import mean_squared_error
  score = mean_squared_error(cv_Y, predict_cv_Y)
  print(score)
  ```

+ 决定系数r2：回归问题常用

  ![决定系数](http://ww1.sinaimg.cn/large/96803f81ly1fzgd9y3y8vj206i01ja9y.jpg)

  ```python
  # 决定系数，可用于回归
  from sklearn.metrics import r2_score
  score = r2_score(cv_Y, predict_cv_Y)
  ```

+ 过拟合解决方法

  + dropout
  + 正则化：利用正则化系数$\lambda$作为惩罚，以放大逻辑回归的梯度中每个$\theta$的影响，从而减小$\theta$的值，防止过拟合。$\theta_0$恒为1，不需惩罚。==L1正则化是绝对值之和，L2正则化是平方和的开方==，常用L2来防止过拟合。
  + batch normalizatin：经过每一层计算，数据分布会发生变化，学习越来越困难。BN即在一个mini-batch内对数据的每个feature进行均值-方差归一化，归一化后再做一个线性变换$y=\gamma x+\beta$，通过两个可学习的参数，一定程度上保留数据原本特征。极端的情况下，$\gamma$和$\beta$就是方差和均值，数据完全还原。batch_size一般塞满卡即可。

+ 梯度爆炸和消失：BP反传中，多个导数连乘可能导致梯度非常小，无法更新参数，导致梯度消失。同理，可能梯度非常大，导致梯度爆炸。将Sigmoid换用ReLU或LeakyReLU激活函数可解决，其正数的导数恒为1，不会带来梯度爆炸和消失。

+ loss

    + `nn.CrossEntropyLoss(outputs, labels)`：n分类问题pred为m\*n维，y为m\*==1==维（`LongTensor`）。过程为先进行`nn.LogSoftmax()`，再`nn.NLLLoss()`，其中`nn.NLLLoss()`即为对于每个样本，`loss += -input[class]`。
    + `nn.BCEWithLogitsLoss(outputs, labels)`：二分类问题pred为m\*n维，y为m\*==n==维独热码（`FloatTensor`）。过程为先进行`nn.Sigmoid()`，再`nn.BCELoss()`，其中`nn.BCELoss()`即为对于每个样本，求$loss=-\frac{1}{n}\Sigma_{i=0}^n(y_i\ln pred_i+(1-y_i)\ln(1-pred_i))$，总loss为$\frac{1}{m}\Sigma loss$。

+ 激活函数：加入非线性运算，否则所有全连接层合并后等价于一个线性变换
  + ReLU速度快。
  + ReLU效果不好时，尝试LeakyReLU或Maxout。
  + 模型不深，激活函数作用不大。

+ 优化器：
  
    + Adam：为每一个参数适应性地保留1个学习率，不向SGD一样学习速率不变