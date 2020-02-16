# [Spark](https://github.com/Y1ran/Spark-The-Definitive-Guide-Chinese-Traslation-2019)

## 1. 安装（ubuntu-python环境）

+ spark需要java1.8环境：

    ```
    sudo apt update
    sudo apt install openjdk-8-jdk
    ```

    配置环境变量~/.baserc：

    ```
    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
    export JRE_HOME=${JAVA_HOME}/jre
    export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib 
    export PATH=${JAVA_HOME}/bin:$PATH
    java -version
    ```

+ 在[官方地址](http://spark.apache.org/downloads.html)选择spark 2.2.3和Pre-built for Hadoop 2.7 and later进行下载。

    在.tgz路径下执行：

    ```
    tar -xf spark-2.2.0-bin-hadoop2.7.tgz
    cd spark-2.2.0-bin-hadoop2.7.tgz
    ```

    配置环境变量~/.bashrc：

    ```
    export SPARK_HOME=/opt/spark-2.2.3-bin-hadoop2.7
    export PATH=${SPARK_HOME}/bin:$PATH
    ```

+ 启动python-spark控制台：

    ```
    pyspark
    ```

+ 如果在/opt/spark/bin/路径下可以执行pyspark，但任意路径下执行pyspark报错，可能为出现权限问题，尝试执行：

  ```
  sudo chown -R username:username /opt/spark-2.2.3-bin-hadoop2.7/
  ```

  其中username是用户名。

+ pycharm运行spark：

    选择“Run” ->“Edit Configurations” ->“Environment variables”，增加SPARK_HOME目录与PYTHONPATH目录，以及HADOOP_HOME、JAVA_HOME。
    
    ```
    SPARK_HOME=/opt/spark-2.2.3-bin-hadoop2.7	# SPARK_HOME:Spark安装目录
    PYTHONPATH=/opt/spark-2.2.3-bin-hadoop2.7/python	# PYTHONPATH:Spark安装目录下的Python目录
    ```

+ 链接postgres

    [postgres driver](https://jdbc.postgresql.org/download.html)下载后，放入SPARK_HOME的Jars中。

## 2. 名词

+ 集群：多台计算机的资源集合。

+ 弹性分布数据集RDD：

+ 分区partition：spark将数据分解，这些数据块称为分区。

+ 转换transformation：spark对数据的读取、计算等操作，这些操作会延迟到action出现再执行。

+ action：如take、show等，只有要执行action操作前才会执行transformation。

## 3. Spark概述

spark是一个分布式编程模型，通过管理和协调跨集群计算机上的数据，来执行任务。通过多个transformation构成一个有向无环图（DAG）。通过一个action操作开始执行DAG的过程，作为一个job作业，它被分解为多个stages阶段和task任务，以便在整个集群中执行。

### Spark应用程序

一个spark应用程序包括一个driver进程和多个executor进程。

+ dirver：驱动，负责维护spark应用，响应输入，给executor分配工作。

+ executor：执行器，负责执行driver分配的工作，将状态报告给驱动。

spark应用程序通过SparkSession对象进行管理。当通过shell启动spark的交互平台时，隐式创建一个SparkSession。而当在独立代码启动spark时，要手动创建一个SparkSession对象。

### Spark的两种模式

+ 集群模式cluster

  spark向集群管理器提交应用程序。集群管理器（包括Spark Standalone、YARN和Mesos三种集群管理器）用来跟踪可用的资源。

+ 单机模式lacal

  在一台机器上运行驱动和执行器。

### Spark的三种核心数据类型

+ DataFrame：与pandas里的DataFrame不同，有相应的转换接口。spark中的dataframe是经过优化后的内部格式。

+ Dataset：在python中不适用。

+ SQL：通过SQL语言进行transformation操作。

### Spark UI

spark UI可用来监视作业进度。

启动spark UI：

```
http://localhost:4040
```

## 4. 注意

+ spark不会立即执行对数据的transformation，而是在出现action操作时再一起执行。

+ 使用spark SQL和spark DateFrame处理的效率相同。

+ transformation不会改变原有的DataFrame，而是创建新的DF保存结果。

| transform | action  |
| :-------: | :-----: |
|    map    | collect |
|  filter   |  count  |
|  select   |  take   |
|           |  first  |

+ 数据持久化：对于以后还会用到的数据，通过.cache()来存储数据，再次使用时不用重复计算。单单独一行的xxx.cache()不起作用，应跟在transformation操作后。

+ spark.mllib中的算法接口是基于RDDs的，spark.ml中的算法接口是基于DataFrames / Dataset 的。尽量使用spark.ml。

## 5. 常用操作

+ 不要去获取df每一行或每一元素，分布式数据库是不可分的，获取违背了用spark的目的

+ spark.dataframe读取特定一行：
  
    ```py
    def getrows(df, rownum):
        return df.rdd.zipWithIndex().filter(lambda x: x[1] = rownum).map(lambda x: x[0])
    print(getrows(df, 0).collect())
    ```
    
    或
    
    ```py
    album_id_with_index = album_id.rdd.zipWithIndex().map(lambda x: (x[1], x[0])).cache()
    print(album_id_with_index.lookup(0)[0]['album_id'])
    ```
    
    或
    
    ```py
    album_id.createTempView("album_id_temp_view")
    album_id_with_index = spark.sql("select row_number() over (order by album_id) as rowNum, * from album_id_temp_view").cache()
    recommend_album_id = album_id_with_index.filter("rowNum=" + str(matrix_rec_sorted_index[i] + 1)).first()['album_id']
    ```

+ spark.df和pandas.df转换

    ```py
    pandas_df = spark_df.toPandas()
    spark_df = spark.createDataFrame(pandas_df)
    ```

+ spark.df和rdd转换

    ```py
    rdd_df = df.rdd
    df = rdd_df.toDF()
    ```

+ 为spark.df增加行号

    ```py
    df.rdd.zipWithIndex()
    ```

+ spark.df填充空为‘’

    ```py
    df.fillna()
    ```

+ 按行获取spark.df（会导入本地）

    ```py
    list = df.collect()
    ```

+ spark.df查询行数

    ```py
    int_num = df.count()
    ```

+ spark.df选择某列

    ```py
    df.select(“name”)
    ```

+ spark.df过滤

    ```py
    df = df.filter(df['age']>21)
    df = df.where(df['age']>21)
    ```

+ spark.df显示

    ```py
    df.show()
    df.show(50)
    df.show(50, False)
    ```
    
+ spark.df转稀疏矩阵

    ```py
    raw_tfidf_matrix_rdd = raw_tfidf_matrix.rdd.map(attrgetter("features"))
    def as_matrix(vec):
        data, indices = vec.values, vec.indices
        shape = 1, vec.size
        return csr_matrix((data, indices, np.array([0, vec.values.size])), shape)

    mats = raw_tfidf_matrix_rdd.map(as_matrix)
    tfidf_matrix = vstack(mats.collect())
    ```
    
+ 矩阵乘法最高效方法

[矩阵乘法](http://hejunhao.me/archives/1503)