# ACT GPU集群

## 注意

+ ==无root权限，sudo不可用，apt指令不可用==
+ ==管理节点下不可直接运行程序，需要通过sbatch或srun提交==
+ 预装cuda，虽然管理节点下没有gpu检查不到。需要检查cuda时先`srun --pty bash`即可，同理检查gpu占用时`srun --gres=gpu:V100:1 nvidia-smi`
+ 安装torch：创建虚拟环境、安装torch即可
+ 纯cpu程序，使用`#SBATCH -p cpu`

## 连接

+ ==校内act内网==
    + 管理节点
        + windows下使用xshell访问`192.168.5.201`管理节点
        + linux下使用ssh user@192.168.5.201访问
    + Gitlab
        + `http://gitlab.act.buaa.edu.cn/ACT/gpu-cluster/wikis/home`
    + smb文件系统
        + windows下直接在此电脑上`\\192.168.0.1`，账号actuser，密码act123
        + mac下在finder的前往-连接服务器下，连接`smb://192.168.0.1`
+ ==校外==
    + 管理节点
        + d.buaa.edu.cn下，ssh连接219.224.171.201访问
        + EsayConnect下访问vpn.buaa.edu.cn，ssh usr@219.224.171.201（==推荐==）
    + Gitlab
        + d.buaa.edu.cn下，http使用`http://gitlab.act.buaa.edu.cn/ACT/gpu-cluster/wikis/home`访问gitlab
    + smb文件系统
        + d.buaa.edu.cn下，ssh连接ubuntu@188.131.175.193:7000，密码1234567，路径/home/yujinze/share下访问smb文件系统

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

+ 修改`.bashrc`，每次进入虚拟环境配置包时无需source conda

    ```bash
    source /home/LAB/anaconda3/etc/profile.d/conda.sh
    ```
    
+ 提交作业：在**管理**节点下，编辑脚本`vi xxx.slurm`：

    ```bash
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

    ```bash
    sbatch xxx.slurm
    ```

+ 查看结果

    ```bash
    tail -f test.log		// 动态输出log结果
    squeue -u your_username		// 查看该用户所有job状态
    ```

+ 查看`sbatch -p`参数对应cpu和gpu

  ```bash
  sinfo
  ```

+ ==查看自己内存、cpu、gpu占用情况==

    ```bash
    cpu、memory：
    top
    htop
    
    gpu：
    srun提交后ctrl+z，nvidia-smi
    http://192.168.5.201/gpu-util-stat/current.log（校外为d.buaa.edu.cn，http访问219.224.171.201/gpu-util-stat/current.log）
    ```

+ Vim粘贴时不缩进

  ```
  :set paste
  ```

  粘贴后关闭

  ```
  :set nopaste
  ```

## 文件传输

+ xftp