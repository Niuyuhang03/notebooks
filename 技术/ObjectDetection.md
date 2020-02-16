# Object Detection

## CV分类

* 图像分类：判断图中有哪些东西，给出图片的描述。以ImageNet为权威评测集
* 物体检测：检测每个物体类别并用矩形检测框定位
* 语义分割：对每个类别和其背景分隔开
* 实体分割：按单个目标分割

## 物体检测

![Object Detection](https://img-blog.csdn.net/20180712105327622?)

### 2-stage模型：局部裁剪region proposal+分类

* [R-CNN](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1311.2524)：Selective Search得到ROI，AlexNet做分类。模型本身存在的问题也很多，如需要训练三个不同的模型（proposal, classification, regression）、重复计算过多导致的性能问题
* [Fast R-CNN](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1504.08083)：将图片进行特征提取后得到Feature Map，Selective Search得到ROI映射到Feature Map上，Pooling后传入R-CNN子网络，共享了大部分计算
* [Faster R-CNN](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1506.01497)：提出（RPN）Regional Proposal Networks代替Selective Search，RPN

### 1-stage模型

* [YOLO](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1506.02640)：速度快，但网格粗糙
* [SSD](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1512.02325)

## 参考文章

* [object_detection](https://github.com/hoya012/deep_learning_object_detection)
* [目标检测入门](https://zhuanlan.zhihu.com/p/34142321)
* [计算机视觉知识点总结](http://bbs.cvmart.net/articles/380)
* 吴恩达深度学习课程