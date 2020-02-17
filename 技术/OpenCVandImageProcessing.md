# OpenCV

## 安装

+ windows+python：[教程](https://blog.csdn.net/Toby_Cho/article/details/81001382)

+ OpenCV教程：[PyImageSearch blog](https://www.pyimagesearch.com/start-here/)

## 基本操作

### 引用

+ 引用opencv，引用辅助库imutils，引用缩放函数rescale_intensity

  ```python
  import cv2 as cv
  import imutils
  from skimage.exposure import rescale_intensity
  ```

### 图像读取

+ 读取图像

    ```python
    img = cv.imread('img path')
    ```

    `cv.imread`在路径错误时不会返回error，而是得到None，在之后的操作中报NoneType Error

+ 批量读取图片

    ```python
    from imutils import paths
    import random
    
    
    imageDir = list(paths.list_images("dir_name"))
    random.shuffle(imageDir)
    for imagePath in imageDir:
        image = cv.imread(imagePath)
        imageslist.append(image)
    ```

+ 显示图像，按任意键关闭，关闭所有窗口

    ```python
    cv.imshow('window name', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    ```

    显示图像一定要对np.uint8格式图片，如果是np.float32格式会导致大片白色

+ 复制图像

    ```python
    output = image.copy()
    ```

+ 获取图像**行数、列数、通道数（BGR）**

    ```python
    h, w, d = img.shape
    ```

+ 取图像某一通道

    ```python
    img[..., 0]
    ```

+ 获取roi（regions of interest）

    ```python
    roi = image[60:160, 320:420]
    ```

+ 得到小于阈值的像素，设置映射

    ```python
    img[img < t]
    img[img < t] = 0
    ```

+ 得到小于阈值的像素坐标

    ```python
    np.where(img < t)
    ```

+ 保存图像

    ```python
    cv.imwrite('file name', img)
    ```

+ 读取摄像头

  ```python
  vs = VideoStream(usePiCamera=0 > 0).start()
  time.sleep(2.0)
  
  while True:
      frame = vs.read()
      frame = imutils.resize(frame, width=400)
  
      timestamp = datetime.datetime.now()
      ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
      cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
  	cv2.imshow('frame', frame)
      key = cv2.waitKey(1) & 0xFF
  
      if key == ord("q"):
          break
  
  cv2.destroyAllWindows()
  vs.stop()
  ```

### 调整大小

+ 调整大小，不保持比例

    ```python
    resized = cv.resize(image, (300, 200))
    ```

+ 调整大小，保持比例

    ```python
    r = 300.0 / w	// 比例
    resized = cv.resize(image, (300, int(h * r)))	//注意int
    ```

    或用imutils库函数：

    ```
    resized = imutils.resize(image, width=300)
    ```

### 旋转

+ 围绕中心旋转，图片不完整（角被切割）

    ```
    center  = (w // 2, h // 2)	// 整除
    M = cv.getRotationMatrix2D(center, -45, 1.0)	// M旋转矩阵，-45为顺时针
    rotated = cv.warpAffine(image, M, (w, h))
    ```

    或用库函数

    ```python
    rotated = imutils.rotate(image, -45)
    ```

+ 围绕中心旋转，图片完整

    ```python
    rotated = imutils.rotate_bound(image, 45)	// +45为顺时针
    ```

### 模糊

减少高频噪声，易于检测。

+ 高斯模糊，原理为卷积

  ```python
  blurred = cv.GaussianBlur(image, (11, 11), 0)	// 11*11卷积核，卷积核越大越模糊
  ```

  + 卷积需要先padding一圈或几圈，使得原图像边缘部分能够放在卷积核中心。通常复制边缘作为padding，使得padding和边缘最接近；在CNN中常用0 padding，即填充0

  + 卷积核边长必须为奇数，这样卷积核才有中心点

  + 卷积核在图像上移动，将对应矩阵按元素相乘（而不是矩阵乘法）后求和，作为该点输出。对于图像，最后还要缩放到0-255范围

    ```python
    from skimage.exposure import rescale_intensity
    gray = rescale_intensity(gray, in_range=(0, 255))
    gray = (gray * 255).astype("uint8")
    ```

  + 算法不同，通常为卷积核的值不同，高斯模糊卷积核为3*3的全1核

### 绘图

直接在图上修改，无需接收返回值。

+ 画长方形

  ```python
  cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
  ```

+ 画实心圆

  ```python
  cv.circle(output, (300, 150), 20, (255, 0, 0), -1)	// 厚度为-1代表实心
  ```

+ 画线

  ```python
  cv.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
  ```

+ 写字

  ```python
  cv.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
  ```

### 转换

+ 转灰度图，Y = 0.2126*R + 0.7152*G + 0.0722*B：

    ```math
    gray = image[..., 0] * 0.0722 + image[..., 1] * 0.7152* + image[..., 2] * 0.2126
    ```

    或库函数

    ```python
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ```

+ 转二值图：对灰度图根据阈值映射为0**和**255

    ```python
thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)[1]
    ```
    
    + 大津二值化算法：对**灰度**图遍历所有阈值，分为两类。将最大类间方差时的阈值作为最佳阈值。最大类间方差如下：

        ```math
        S_b^2=w_0*w_1*(M_0-M_1)^2
        ```
    
        其中w为比例，M为平均值

### 侵蚀和膨胀

减少二值图噪声。

+ 侵蚀：黑色增大，输入为二值图

    ```python
    mask = cv.erode(thresh.copy(), None, iterations=5)
    ```

+ 膨胀：白色增大，输入为二值图

    ```python
    mask = cv.dilate(thresh.copy(), None, iterations=5)
    ```

#### mask和位运算

遮盖不关注的部分。

+ mask的实现通过按位与运算实现。mask为二值图，黑色背景为0，按位与时黑色部分变为黑色，其他保持原图颜色

  ```python
  output = cv.bitwise_and(image, image, mask=thresh)
  ```

### 拼接图片

```python
montages = build_montages(imageslist, (128, 196), (7, 3))

for montage in montages:
    cv.imshow("montage", montage)
    cv.waitKey(0)
```

## 颜色

### 颜色识别

+ 使用`cv.inRange`找到颜色，用mask画出

  ```python
  boundaries = [
  	([17, 15, 100], [50, 56, 200]),
  	([86, 31, 4], [220, 88, 50]),
  	([25, 146, 190], [62, 174, 250]),
  	([103, 86, 65], [145, 133, 128])
  ]
  
  for (lower, upper) in boundaries:
      lower = np.array(lower, dtype = "uint8")	// opencv接受的应该为np格式数组，且为uint8
      upper = np.array(upper, dtype="uint8")
  
      mask = cv.inRange(image, lower, upper)
      output = cv.bitwise_and(image, image, mask=mask)
      
      cv.imshow("output", output)
      cv.waitKey(0)
  ```

+ 提取BGR三种颜色通道

  ```python
  for chan in cv.split(image):
      xxx
  ```

### 颜色转换

+ 将source的颜色风格加在target上，使用L\*a\*b色彩空间

  ```python
  output = color_transfer.color_transfer(source, target)
  ```

## 轮廓

### 边缘检测

+ 检测像素亮度变化明显的点，输入为**灰度图**。Canny算法：

    ```python
    edged = cv.Canny(gray, 30, 150)
    ```

### 轮廓提取

findContour会修改输入图像，要先copy。

+ 提取物体轮廓，输入为**二值图**，识别二值图白色部分边界，因此二值化时阈值选择很重要

    ```python
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)	// 找到轮廓
    cnts = imutils.grab_contours(cnts)	// 对opencv不同版本的兼容性操作
    output = image.copy()
    for c in cnts:
        M = cv.moments(c)		// 图像矩，用于找轮廓中心
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        
        cv.drawContours(output, [c], -1, (240, 0, 159), 3)	// 画出每个轮廓
        cv.circle(image, (cX, cY), 7, (255, 255, 255), -1)	// 轮廓中心画圈
        cv.imshow("output", output)
        cv.waitKey(0)
    
    c = max(cnts, key=cv.contourArea)	// 最大轮廓
    cv.drawContours(output, [c], -1, (240, 0, 159), 3)
    
    extLeft = tuple(c[c[..., 0].argmin()][0])	// 最大轮廓的最左点
    extRight = tuple(c[c[..., 0].argmax()][0])
    extUp = tuple(c[c[..., 1].argmin()][0])
    extDown = tuple(c[c[..., 1].argmax()][0])
    
    cv.imshow("output", output)
    cv.waitKey(0)
    ```

### 形状检测

+ 根据边个数判断形状

  ```python
  def detect(self, c):
  	shape = "unidentified"
  	peri = cv2.arcLength(c, True)	// 求出轮廓周长
  	approx = cv2.approxPolyDP(c, 0.04 * peri, True)		// 提取轮廓近似值
  	if len(approx) == 3:
  		shape = "triangle"
  	elif len(approx) == 4:
  		(x, y, w, h) = cv2.boundingRect(approx)
  		ar = w / float(h)
  		shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
  	elif len(approx) == 5:
  		shape = "pentagon"
  	else:
  		shape = "circle"
  	return shape
  ```

## NoneType errors

+ `cv.imread`的图片路径错误。涉及路径错误、图片格式不支持。路径错误时返回None但不报错，之后的操作得到NoneType Error。图片格式不支持（如jpg）需要重新编译OpenCV
+ `cv.ViodeoCapture`截取视频错误。涉及路径错误、驱动错误、解码器错误

# Image Processing

## 尺度不变特征转换SIFT

用于特征点检测，效果极好，对目标的缩放、平移、旋转、光线容忍大。专利不可商用

+ 输入两张灰度图，

  ```python
  for gray in grays:
  	sift_initialize = cv.xfeatures2d.SIFT_create()
      key_points, descriptors = 	sift_initialize.detectAndCompute(gray, None)
  cv.imshow('sift_features', cv.drawKeypoints(gray1, key_points1, image1.copy()))	// 查看key points
  
  bruteForce = cv.BFMatcher(cv.NORM_L2)	// 用曼哈顿距离
  matches = bruteForce.match(descriptors1, descriptors2)	// 找到key points之间的连接
  matches = sorted(matches, key=lambda match:match.distance)
  
  matched_img = cv.drawMatches(image1, key_points1, image2, key_points2, matches[:100], image2.copy())	// 画出对应特征连线
  ```

## 随机抽样一致算法RANSAC

最小二乘法是兼顾所有点找到最佳曲线，RANSAC就是不考虑错误点，只找到正确点的最佳曲线。具体思想是迭代n次，每次随机取m个点，找到m个点连线后包含其他点的数量（误差），直至最优。

## 人脸检测

+ 用haar cascade方法。所需文件从[opencv](https://github.com/opencv/opencv/tree/master/data/haarcascades)下载

```python
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('face', frame)
```

## 人脸识别

+ 采用基于LBPH特征方法

```python
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

label= face_recognizer.predict(face)[0]
```

