# 基于内容推荐的spark程序中稀疏矩阵的探究

基于内容推荐算法中，要求只使用该用户自己的数据进行推荐。在我设计的通过矩阵相乘得到推荐结果的算法中，涉及几次较稀疏的矩阵相乘。故在此探究一下稀疏矩阵的选择。

## 为什么使用稀疏矩阵？

由于需要相乘的矩阵包括用户评分矩阵、tf-idf矩阵等，为了使得最后的推荐结果矩阵和所有user及album对应，会将所有从数据库读出的dataframe扩展为类似“行对应**所有**user，列对应**所有**album”的矩阵形式，此时矩阵中就会出现大量值为0的部分，使用稠密矩阵直接相乘会出现OOM等问题。此外对于第一版中使用的sklearn库的tf-idf算法，得到的结果正是scipy稀疏矩阵。因此考虑使用稀疏矩阵来加速矩阵乘法。

## 关于scipy稀疏矩阵

在第一版推荐算法中，我们使用了scipy稀疏矩阵（由数据库读出的pandas.df手动转换而来）进行计算，效率比较高。

第二版推荐算法重构为spark代码后，依然使用scipy稀疏矩阵。在了解了分布式数据集会通过spark sql的Catalyst优化器生成优化后的执行代码，其执行速度会更快，因此尝试在第三版更换为分布式数据集，以加速性能。

## 关于spark稀疏矩阵

在第三版中尝试将scipy稀疏矩阵更换为spark稀疏矩阵时，遇到了一些困难。

在[spark的矩阵乘法文档](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html?highlight=multiply#pyspark.mllib.linalg.distributed.BlockMatrix.multiply)中，我们可以看到：

>Left multiplies this BlockMatrix by other, another BlockMatrix. The colsPerBlock of this matrix must equal therowsPerBlock of other. If other contains any SparseMatrix blocks, they will have to be converted to DenseMatrix blocks. The output BlockMatrix will only consist of DenseMatrix blocks. This may cause some performance issues until support for multiplying two sparse matrices is added.

即spark中矩阵乘法函数暂不支持稀疏矩阵相乘，而是把稀疏矩阵转换为稠密矩阵后相乘，得到的结果为稠密矩阵。这样以来就违背了使用稀疏矩阵的目的。

此外，了解到分布式系统并不会加速矩阵操作，且目前scipy的coo格式稀疏矩阵耗时很低，更换为spark的crs格式稀疏矩阵由于暂不支持稀疏矩阵相乘，有可能影响性能，故第三版没有更改，依然使用scipy稀疏矩阵进行计算。

## scipy稀疏矩阵相乘原理

### 稀疏矩阵Sparse Matrix

正常的矩阵为稠密矩阵。当矩阵中0元素较多，且非零元素呈不规律分布时，可以使用稀疏矩阵，只保存矩阵中的非0项，来节省空间，加速计算。

### 常用稀疏矩阵类型

+ coo稀疏矩阵

    ![coo](https://i.loli.net/2019/05/08/5cd254b7a0be8.jpg)
    
    ```py
    sparse.coo_matrix((values, (rows, cols)), shape=(m,n))
    ```
    
    coo稀疏矩阵的格式最简单，rows代表数据所处的行，columns代表数据所处的列。注意coo稀疏矩阵无法修改，一般创建好后更改为其他矩阵类型。

+ **csr稀疏矩阵**、csc稀疏矩阵

    ![csr](https://i.loli.net/2019/05/08/5cd254d5a522a.jpg)
    
    ```py
    sparse.csr_matrix((values, cols, rowsptr), shape=(m, n))
    ```
    
    csr稀疏矩阵中，values和cols和coo相同，但rowsptr是长度为矩阵行数+1的列表，[rowsptr[i], rowsptr[i+1])的左闭右开区间为矩阵第i行元素在values里的索引。
    
    csc稀疏矩阵则是以列方式压缩的。

### 稀疏矩阵乘法原理

稠密矩阵的乘法相对规律，需要遍历矩阵每一行和每一列，对应元素相乘后累加。而在稀疏矩阵中，存在大量0元素，其乘积结果为0，而在三元组中，所有数据都是非零的，因此稀疏矩阵可以加速乘积。

普通的矩阵相乘，可以直接通过两个for循环遍历，逐个相乘累加。对于三元组格式的稀疏矩阵乘法，可以直接计算：

```C
#define MAX_SIZE 1500 //稀疏矩阵的非零元素的最大个数
#define MAX_ROW 1500 //稀疏矩阵的行数的最大个数
class Triple
{
    public:
    int i,j; //表示非零元素的行下标和列下标
    int val; //非零元素的值
};
class RLSMatrix
{
    public:
    Triple data[MAX_SIZE]; //非零元三元组表
    int rpos[MAX_ROW]; //每行第一个非零元素的位置
    int row_num, col_num, cnt; //稀疏矩阵的行数、列数以及非零元素的个数
};
 
void MultRLSMatrix(RLSMatrix M,RLSMatrix N,RLSMatrix &result){
    int arow, brow, p, q, ccol, ctemp[MAX_ROW + 1], t, tp;
    
    //不能相乘
    if(M.col_num != N.row_num)
        return;
    //有一个是零矩阵
    if(0 == M.cnt * N.cnt )
        return;
    
    //result初始化
    result.row_num = M.row_num;
    result.col_num = N.col_num;
    result.cnt = 0;
    
    //从M的第一行开始到最后一行，arow是M的当前行
    for(arow = 1; arow <= M.row_num; arow++){
        for(ccol=1;ccol <= result.col_num;ccol++)
            ctemp[ccol] = 0; //result的当前行的各列元素清零
            
        result.rpos[arow] = result.cnt + 1; //开始时从第一个存储位置开始存，后面是基于前面的
        if(arow < M.row_num)
            tp = M.rpos[arow+1]; //下一行的起始位置
        else
            tp = M.cnt + 1; //最后一行的边界

        for(p = M.rpos[arow]; p < tp; p++){
            //对M当前行的每一个非零元
            //找到对应元素在N中的行号，即M中当前元的列号
            brow = M.data[p].j;
            if(brow < N.row_num)
                t = N.rpos[brow + 1];
            else
                t = N.cnt + 1;

            for(q = N.rpos[brow]; q < t; q++){
                ccol = N.data[q].j; //乘积元素在result中列的位置
                ctemp[ccol] += M.data[p].val * N.data[q].val;
            }
        }
        
        //压缩存储该行非零元
        for(ccol = 1; ccol <= result.col_num; ccol++){
            if(0 != ctemp[ccol]){
                if(++result.cnt > MAX_SIZE)
                    return;
                result.data[result.cnt].i = arow;
                result.data[result.cnt].j = ccol;
                result.data[result.cnt].val = ctemp[ccol];
            }
        }
    }
}
```

rpos[arow]表示矩阵N中第arow行，第一个非零元在N.data中的序号。即直接计算非零元素所在行列的乘积和来求出稀疏矩阵乘积。

此外，稀疏矩阵的存储存在访存不规则的问题，会影响乘积效率，故在底层使用[稀疏矩阵向量乘法](https://jackgittes.github.io/2017/08/23/matrix-multip-optimization/) (Sparse matrix-vector multiplication，SpMV)进行加速。可以加速的主要方面如下：

+ 分割：将稀疏矩阵按照特定的方法进行分割，得到更小的矩阵，目的是使每个矩阵分块的存储尽量和芯片上的存储空间和计算资源相匹配，避免一个分块的处理过程中频繁访问片外存储，提高存储复用率。

+ 编码和解码：编码中，分为格式压缩、索引压缩、值压缩等。格式压缩就是采用COO、CSR的格式。索引压缩是在压缩非0元素索引的位宽。而值压缩则是在矩阵值为特定值时，给可能取值建立索引。

+ 此外还有负载平衡等方法，加速稀疏矩阵乘法。

注意矩阵乘法使用.dot()或*，而不能用.multiply()，该函数为按元素相乘。