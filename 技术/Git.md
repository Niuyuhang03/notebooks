# git常用指令

```bash
从github克隆项目到本地
在正确的本地路径下
git clone git@github.com:rRetr0Git/rateMyCourse.git

查看本地分支
git branch

查看本地和远程分支
git branch -a

切换本地分支xxx
git checkout xxx

创建新本地分支xxx
git branch xxx

创建新本地分支xxx并切换过去
git checkout -b xxx

拉取远程分支xxx并切换过去
git checkout -b xxx origin/xxx

查看本地文件状态
git status

提交xxx文件
git add xxx
git commit -m "xxx"

推送到远程xxx分支
git push origin xxx

拉取远程xxx分支的更新
git pull origin xxx
```

# git指令撤销

+ 撤销修改

```bash
git checkout filename
```

+ 撤销add，保留修改

```bash
git reset HEAD filename
```

+ 撤销commit

```bash
git log # 查上一commit版本号commitid
git reset --mixed commitid # 撤销commit，不撤销add，--mixed可省略
git reset --soft commitid # 撤销commit和add，不撤销修改
git reset --hard commitid # 撤销commmit、add和修改，重置到之前某一commit状态
```

+ 删除远程文件，但不删除本地文件

```bash
git rm -n --cached xxx  # -n参数显示预览，不执行删除
git rm --cached xxx
vi .gitignore
commit push
```

# git加速

## 下载releases的文件太慢

+ 直接复制链接到迅雷

## 下载整个仓库的zip太慢

+ 在码云gitee导入github仓库链接，在码云下载
+ 或用下述方法clone

##  clone仓库太慢

### 为 github使用代理

如果你已经拥有代理软件，直接为 `git` 设置代理是最好的提速方法。假设本地代理地址为 `127.0.0.1：1080`（端口号详见ss或v2ray的设置中socks的端口号），那么你可以使用以下命令为 `git` 设置代理：

```shell
# 首先取消已有代理
git config --global --unset http.proxy
git config --global --unset https.proxy

# 代理github
git config --global http.https://github.com.proxy socks5://127.0.0.1:1080  # 端口号需要根据v2ray的sock的端口设置
git config --global https.https://github.com.proxy socks5://127.0.0.1:1080  # 端口号需要根据v2ray的sock的端口设置
```

```python
# 不推荐全局代理git，会将国内git变慢
git config --global http.proxy http://127.0.0.1:1080  # 不要做
git config --global https.proxy http://127.0.0.1:1080  # 不要做
```

==**必须使用https**方式clone，且全程开启代理软件==

+ 出现`LibreSSL SSL_connect: SSL_ERROR_SYSCALL in connection to github.com:443`问题，大概率为代理软件挂了，尝试更新订阅或切换节点

### 修改 host 文件（不推荐）

在 `git clone` 或 `git push` 时，实际上并不是直接向 `github.com` 发送请求，而是对 `github.global.ssl.fastly.net` 发送请求与通信，Fastly 公司在中国有着众多的 CDN 节点，GitHub 可能因为成本或者其他原因，并没有在中国搭设自己专属的 CDN 节点，我们可以通过修改 `host` 文件来加速对这个域名的访问。

```
# windows下修改C:\Windows\System32\drivers\etc\hosts# Linux/Mac下修改/etc/hosts# 在最后加上151.101.77.194  github.global.ssl.fastly.net13.229.188.59   github.com185.199.109.153 assets-cdn.github.com151.101.76.249  global-ssl.fastly.net
```

然后刷新 DNS 缓存。

```
# windowsipconfig /flushdns# linux/macsudo /etc/init.d/network-manager restart
```

如果网络没问题的话，修改后的速度一般都能达到 `MB/s` 的级别。

# git多平台问题

## mac上换行符引起`^M`问题

```bash
git config --global core.autocrlf input
git config --global core.safecrlf true
```

其中git config可以通过`git config —list`参看，手动删除可以通过编辑`~/.gitconfig`进行修改。

# git分支管理


## 一些约定

+ 任何一次add前，请用git branch查看所在分支是否正确，用git status查看修改的文件。

+ 任何一次commit前，请用git status查看add的文件有没有多余的。

+ 任何一次出现错误的add、commit、push行为，请通知群里不要改动分支，然后处理。

## 主分支

### master分支

+ master的每次commit都必须是一个正式的大版本更新，必须打tag。
+ 只能通过merge release分支来修改，其他时候不要修改master。

### develop分支

+ 用于开发工作，每个人都从develop上分出新分支开发，开发完成后合入develop分支。

## 其他分支

### feature分支

+ 用于每个人分配到的新功能相关代码。

+ 必须从develop分出来，必须合入develop，合入后删除该分支。

+ feature分支不能出现在origin，即只能commit不能push。

+ feature分支的命名基本无要求，如“wwj”

```bash
# 创建feature分支
任意分支下git checkout -b wwj develop

# 每次有代码更新，在feature分支提交，只commit不push
git add xxx
git commit -m "xxx"

# 完成该阶段工作，合并到develop分支，并删除wwj分支
git checkout develop
git pull origin develop
git merge --no-ff wwj
git branch -d wwj
git push origin develop
```

### release分支

+ 用于发布正式版本。当develop上所有功能都完成后，创建release分支，此后不允许添加新功能，但可以在release分支上debug。而后会合并如master（为了发布）和develop分支（为了将fixbug部分移入develop）。

+ 必须从develop分出来，必须合入develop和master。

+ release创建出来后到删除前，所有的bugfix请直接在release分支上修改。

+ 命名必须为“release-\*”，如“release-0.2”，数字为版本号。版本号具体怎么来一般以公司dalao心情决定，一般是A.B.C的格式，A是大版本号，B是增加新功能，C是修复bug。我们发布的次数有限，我推荐直接用A.B格式，把新功能和修复bug都在B递增，预计会在网站可以完全运行时更新大版本到1.0。即从0.1,0.2...0.10,0.11,0.12....1.0,1.1,1.2这样来。

```bash
# 创建release分支
任意分支下git checkout -b release-0.1 develop

# 在release分支下fix bug
git branch -a
git checkout release
git add xxx
git commit -m "xxx"
git push origin release

# 把release分支合并到master分支和develop分支
git checkout master
git merge --no-ff release-0.1
git tag 0.1
git checkout develop
git merge --no-ff release-0.1
git branch -d release-0.1
git push --tags
git push
```