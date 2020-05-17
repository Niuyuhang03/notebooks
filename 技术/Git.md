# git常用指令

## git安装和初始化

```bash
查看是否安装
git

安装git，推荐从homebrew安装。安装homebrew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

安装git
brew install git

配置你的用户名和邮箱，否则每次提交都要输入
git config --global user.name "xxxxxx"
git config --global user.email "xxxxx@xxx.com"

配置ssh key，这个key在一台电脑上每个git账户有一个key，因此命名推荐可以识别出哪台设备
ssh-keygen -t rsa -C "xxxxxx@xxxxx.com"

复制.ssh/id_rsa.pub的内容，打开github网页的settings里的ssh key部分，新建，粘贴

验证连接，出现successfully即可
ssh -T git@github.com
```

## 创建项目

+ github网页创建项目，clone

```bash
git clone git@github.com:rRetr0Git/rateMyCourse.git
```

```bash
或在github网页新建项目，看指示
```

## 分支管理

```bash
查看本地分支
git branch

查看本地和远程分支
git branch -a

切换到本地分支xxx
git checkout xxx

新建本地分支xxx并关联远程分支xxx，并切换过去
git checkout -b xxx origin/xxx

创建新本地分支xxx
git branch xxx

创建新本地分支xxx并切换过去
git checkout -b xxx

查看所有本地分支和远程分支的关联情况
git branch -vv

将本地分支关联到远程xxx分支
git branch -u origin/xxx
```

## 项目提交

```bash
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

## 查看项目的远程地址

```bash
git remote -v
```

# ==git指令撤销==

+ 撤销修改

```bash
git checkout filename
```

+ 撤销add，保留修改

```bash
git reset HEAD filename
```

+ 撤销commit或pull

```bash
git log # 查上一commit版本号commitid
git reset --mixed commitid # 撤销commitid之后的commit，不撤销add，--mixed可省略
git reset --soft commitid # 撤销commitid之后的commit和add，不撤销修改
git reset --hard commitid # 撤销commmitid之后的commit、add和修改，重置到之前某一commit状态
```

+ 删除远程文件，但不删除本地文件

```bash
git rm -n --cached xxx  # -n参数显示预览，不执行删除
git rm --cached xxx
vi .gitignore
commit push
```

+ 版本回退

```bash
# 回退
git log
git reset --hard commitid

# 撤销版本回退
git reflog
git reset --hard commitid
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
git config --global http.https://github.com.proxy socks5://127.0.0.1:1080  # 端口号需要根据v2ray的sock的端口设置，mac默认为1080
git config --global https.https://github.com.proxy socks5://127.0.0.1:1080  # 端口号需要根据v2ray的sock的端口设置，mac默认为1080
```

```python
# 不推荐不推荐不推荐不推荐不推荐全局代理git，会将国内git变慢
git config --global http.proxy http://127.0.0.1:1080  # 不要做
git config --global https.proxy http://127.0.0.1:1080  # 不要做
```

==**必须使用https**方式clone，且全程开启代理软件==

+ 出现`LibreSSL SSL_connect: SSL_ERROR_SYSCALL in connection to github.com:443`问题，大概率为代理软件挂了，尝试更新订阅或切换节点
+ Unpacking objects卡住，文件可能过大，也可能该仓库是用ssh而不是http建立的连接，使用`git remote set-url origin https://xxx`

# git多平台问题（win不要用）

## mac上换行符引起`^M`问题

linux换行使用LF，Windows换行使用CRLF，即\r\n，在mac使用cat -e filename中可以看到^M\$即为crlf，\$即为lf。mac配置如下

```bash
git config --global core.eol lf  # 统一换行符为 lf
git config --global core.autocrlf input  # 打开push时自动转换关闭，保证push时一定是lf
git config --global core.safecrlf true  # 禁止混用 lf 和 crlf 两种换行符
```

其中git config可以通过`git config —list`参看，手动删除可以通过编辑`~/.gitconfig`进行修改。

虽然通过设置了 git 全局参数解决了问题，但是作为团队协作的话，并不能保证所有人都正确配好了。git 提供了.gitattributes文件解决了这个问题。在项目根目录新建.gitattributes文件，添加一下内容：

```bash
# Set the default behavior, in case people don't have core.autocrlf set.
* text eol=lf
```

通过这种方式避免有人没有设置 core.autocrlf 参数，并且将该文件加入版本控制中。

如果已经出现crlf，批量转换为lf，需要在brew安装dos2unix，然后`find . -name "*" | xargs dos2unix`

# Mac git 自动补全

+ 安装homebrew：`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
+ 安装bash-completion：`brew install bash-completion`
+ 访问`https://github.com/git/git.git`，找到`contrib/completion/git-completion.bash`，复制到`~/.git-completion.bash`
+ 将如下代码添加到`~/.bash_profile`（不存在则创建）：

```bash
if [ -f ~/.git-completion.bash ]; then
   . ~/.git-completion.bash
fi
```

+ 由于新的mac已经将zsh作为默认shell，如果打开terminal后最上面不是bash而是zsh，则不会自动在启动terminal时执行`source ~/.bash_profile`，而是`source ~/.zshrc`，故修改`~/.zshrc`，加入`source ~/.bash_profile`

# git分支管理

+ 任何一次add前，请用git branch查看所在分支是否正确，用git status查看修改的文件。

+ 任何一次commit前，请用git status查看add的文件有没有多余的。

+ 任何一次出现错误的add、commit、push行为，请及时通知其他人员，然后处理。

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
# 创建名叫wwj的feature分支，该分支是从develop分出来的
任意分支下git checkout -b wwj develop

# 每次有代码更新，在wwj的feature分支提交，只commit不push
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

+ 命名必须为“release-\*”，如“release-0.2”，数字为版本号。版本号具体怎么来一般以公司dalao心情决定，一般是A.B.C的格式，A是大版本号，B是增加新功能，C是修复bug。发布的次数有限时，推荐直接用A.B格式，把新功能和修复bug都在B递增，预计会在网站可以完全运行时更新大版本到1.0。即从0.1,0.2...0.10,0.11,0.12....1.0,1.1,1.2这样来。

```bash
# 创建release分支
任意分支下git checkout -b release-0.1 develop

# 在release分支下fix bug，没有则跳过
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