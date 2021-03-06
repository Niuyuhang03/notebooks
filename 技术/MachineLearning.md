# 机器学习基础算法

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
  import numpy as np
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.utils.data as Data
  import torch.optim as optim
  
  
  class LRNet(nn.Module):
      def __init__(self, nfeature):
          super(LRNet, self).__init__()
          self.LR = nn.Linear(nfeature, 1)
      
      def forward(self, x):
          y = self.LR(x)
          return y
  
  
  num_inputs = 2
  num_examples = 1000
  true_w = [2, -3.4]
  true_b = 4.2
  features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
  labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
  labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
  
  batch_size = 10
  dataset = Data.TensorDataset(features, labels)
  data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
  
  model = LRNet(num_inputs)
  loss = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=0.03)
  
  epochs = 100
  for epoch in range(epochs):
      for X, y in data_iter:
          output = model(X)
          train_loss = loss(output, y.view(output.size()))
  
          optimizer.zero_grad()
          train_loss.backward()
          optimizer.step()
      print("epoch: {:3d}, train loss: {:.7f}".format(epoch, train_loss))
  
  print(model.LR.weight, model.LR.bias)
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
