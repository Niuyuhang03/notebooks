# 常用指令

- ls

- ls -a

- ll：ls -l

- tree

- clear

- ctrl+r：查找命令记录，输入关键字后，查找后一个结果再次输入ctrl+r，编辑命令用左右键

- which：查看java路径

- pwd

# 安装

## java8

    ```
    sudo apt update
    sudo apt install openjdk-8-jdk
    java -version
    ```
    
    修改.bashrc：
    
    ```
    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
    export JRE_HOME=${JAVA_HOME}/jre
    export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib 
    export PATH=${JAVA_HOME}/bin:$PATH
    ```
    
    ```
    source ~/.bashrc
    ```

## spark

    [spark.md](http://note.youdao.com/noteshare?id=2e0c4e5e3756c78baeb82b7fe8cca920)

## pipenv

    ```
    pipenv install
    ```
    
    如果lock很慢考虑换源。在项目路径下：
    ```
    vim Pifile
    url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
    ```
    
    ```
    pipenv shell
    pipenv install requests
    ctrl+D
    ```
    
    进入pycharm项目设置，搜索Project Interpreter。在Project Interpreter的右上角配置按钮上选择Add Local，选择VirtualEnv Environment，复制刚才的环境路径"/Users/zyt/.local/share/virtualenvs/new-cRH-55u9/bin/python"到粘贴板，粘贴到existing environment的interpreter下面，点击确定。

## pytorch(非服务器，有root权限)

+ 下载CUDA

    [CUDA](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=runfilelocal)地址，对于ubuntu18.04可以选17.04/16.04，CUDA版本一定选9.0
    
    ![CUDA](http://ww1.sinaimg.cn/large/96803f81ly1g2czl9b5c5j20s20bhjru.jpg)

+ 下载cudnn

    [cudnn](https://developer.nvidia.com/cudnn)，需要注册，选择CUDA9.0对应版本。


+ GCC降级（6.0以下），否则无法安装cuda

    ```
    查看版本
    gcc --version 
    g++ --version
    
    sudo apt-get install gcc-4.8 
    sudo apt-get install g++-4.8
    
    d /usr/bin 
    sudo rm gcc
    sudo ln -s gcc-4.8 gcc
    sudo rm g++
    sudo ln -s g++-4.8 g++
    
    gcc --version
    g++ --version
    ```

+ 安装CUDA及其补丁，注意cuda和cudnn应安装同一文件夹下（cuda文件夹）

    安装CUDA
    
    ```
    sudo sh cuda_9.0.176_384.81_linux.run
    ```
    
    注意询问是否安装驱动选no，刚才已经安装过
    
    安装补丁
    
    ```
    sudo sh xxxx
    ```

    配置环境变量
    
    ```
    vi ~/.bashrc
    export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}　　
    
    source ~/.bashrc
    ```
    
+ 安装cudnn

    ```
    tar -xzvf cudnn-9.0-linux-x64-v7.tgz
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    ```

+ 查看cuda版本

    ```
    nvcc -V
    cat /usr/local/cuda/version.txt
    ```

+ 安装显卡驱动

    ```
    查看本机支持的驱动（一般为390）
    ubuntu-drivers devices
    
    卸载驱动
    sudo apt-get purge nvidia*
    
    添加显卡驱动PPA
    sudo add-apt-repository ppa:graphics-drivers
    sudo apt-get update
    
    装对应版本的显卡驱动
    sudo apt-get install nvidia-390
    ```
    
    重启，可能要求configuring secure boot，使用右方向键确定，设置密码。开机后输入密码。
    
    ```
    检查是否安装好
    lsmod | grep nvidia
    
    固定版本
    sudo apt-mark hold nvidia-390
    ```

+ 安装pytorch，必须在官网找指令

    ```
    conda安装
    9.0
    conda install pytorch torchvision cudatoolkit=9.0
    9.1
    conda install pytorch cuda91
    ```

## pip临时换源

    ```
    pip install xxxx -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

## conda换源

    ```
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    安装指令去掉-c conda
    ```

## act GPU作业

见act.md。