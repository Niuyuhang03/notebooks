# ACT GPU集群

## 注意

+ 无root权限，sudo不可用，apt指令不可用
+ 预装cuda，虽然检查不到，但直接创建虚拟环境、安装torch即可
+ 纯cpu程序，使用`#SBATCH -p cpu`

## 连接

+ 使用xshell访问`192.168.5.201`管理节点，管理节点下不可直接运行程序，需要通过sbatch或srun提交
+ linux使用ssh user@192.168.5.201访问

## 配置环境

+ 首次配置环境：在**管理**节点下

    ```bash
    # 创建conda指令所需环境
    source /home/LAB/anaconda3/etc/profile.d/conda.sh
    
    # 创建虚拟环境my_env_name：
    conda create -n my_env_name python=3.6
    
    # 进入conda虚拟环境
    source /home/LAB/anaconda3/etc/profile.d/conda.sh
    conda activate cuda9.1
    
    # conda换源
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    conda config --set show_channel_urls yes
    
    # 安装pytorch
    通过conda安装pytorch，指令见pytorch官网。但指令结尾去掉-c torch
    
    # 安装所需包
    在虚拟环境内conda install xxx
    或在虚拟环境内python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --user
    
    # 退出虚拟环境
    conda deactivate
    ```

+ 提交作业：在**管理**节点下，编辑脚本`vi xxx.slurm`：

    ```
    #!/bin/bash
    #SBATCH -o ConvE_result.log         # 输出日志，可以查看ckpt文件夹
    #SBATCH -J ConvE             # 任务名称，只能显示前6字符
    #SBATCH --gres=gpu:V100:1   # 申请1个v100的gpu
    #SBATCH -c 5        # 申请5个cpu核心
    #SBATCH -p sugon	# 指定使用曙光gpu，参数为cpu则在cpu上跑
    #SBATCH --mail-user=user@mail.com	# 作业结束后发送邮件

    source /home/LAB/anaconda3/etc/profile.d/conda.sh
    conda activate cuda9.1
    指令xxx
    ```

+ 提交脚本

    ```
    sbatch xxx.slurm
    ```

+ 查看结果

    ```
    tail -f test.log		// 动态输出log结果
    squeue -u your_username		// 查看该用户所有job状态
    ```

+ 查看`sbatch -p`参数对应cpu和gpu

  ```
  sinfo
  ```

+ 查看自己内存、cpu、gpu占用情况

    ```
    top
    http://192.168.5.201/gpustat/current.log
    ```

+ 粘贴时不缩进

  ```
  :set paste
  ```

  粘贴后关闭

  ```
  :set nopaste
  ```

## 文件传输

xftp
