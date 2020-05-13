# Mac相关

## 技巧

+ terminal中的ctrl和command
    + command：command+C复制，command+V复制
    + ctrl：ctrl+C结束进程，ctrl+R搜素
    
+ finder复制路径：option+command+c

+ spotlight搜索：command+space

+ 关闭软件：command+W

+ 输入法切换：caps lock

+ 大小写：shift+字母，或长按caps lock

+ 修改默认打开方式：option+右键-默认打开方式

+ 修改dock速度：`defaults write com.apple.Dock autohide-delay -float 0 && killall Dock`。重置dock速度：`defaults delete com.apple.Dock autohide-delay && killall Dock`

+ bash & zsh

    + bash
        + `.bash_profile`系统启动时执行，因此主要为全局设置
        + `.bashrc`打开shell时执行，因此主要为用户设置
    + zsh
        + 新版本mac默认使用zsh，且不存在`.zsh_profile`，而是只有`.zshrc`。在使用zsh时，bash相关的环境不会执行。因此有必要时，要在`.zshrc`里调用`source .bash_profile`。

+ Markdown：

    + 页内跳转：`[](# xxx)`，注意必须只用一个#

    + 操作

        + ==高亮==：`==高亮==`，快捷键command+Shift+H
        + **加粗**：`**加粗**`
        + *斜体*：`*斜体*`

    + 公式

        | 公式               | 效果                                       | markdown原文                                 |
        | ------------------ | ------------------------------------------ | -------------------------------------------- |
        | 行内公式           | `行内公式`                                 | \`行内公式\`                                 |
        | 行间公式           | ```行间公式```                             | \`\`\`行间公式\`\`\`                         |
        | 引用               | > 引用                                     | \> 引用                                      |
        | 行内数学公式       | $公式$                                     | \$公式\$                                     |
        | 上下角标           | $a^2,a^{22},a_2,a_{22}$                    | \$a^2,a^{22},a_2,a_{22}\$                    |
        | 分数线             | $\frac{分子}{分母}$                        | \$\frac{分子}{分母}​\$                        |
        | 开方               | $\sqrt{x}$                                 | \$\sqrt{x}​\$                                 |
        | 积分               | $\int^{+\infty}_{-\infty}$                 | \$\int^{+\infty}_{-\infty}\$                 |
        | 闭曲线积分         | $\oint$                                    | \$\oint​\$                                    |
        | 求和               | $\sum^n_{i=1}x$                            | \$\sum^n\_{i=1}x​\$                           |
        | 连乘               | $\prod$                                    | \$\prod​\$                                    |
        | 大于等于、小于等于 | $\ge\le$                                   | \$\ge\le​\$                                   |
        | 空格               | $\ $                                       | \$\ $                                        |
        | 分数线             | $\begin{cases}1&x>1\\0&x\le 0\end{cases}$  | \$\begin{cases}1&x>1\\\0&x\le 0\end{cases}​\$ |
        | 箭头               | $\rightarrow\Rightarrow\Leftrightarrow$    | \$\rightarrow\Rightarrow\Leftrightarrow​\$    |
        | 箭头打撇           | $\nrightarrow\nRightarrow\nLeftrightarrow$ | \$\nrightarrow\nRightarrow\nLeftrightarrow​\$ |
        | 正负               | $\pm$                                      | \$\pm​\$                                      |
        | 交                 | $\cap$                                     | \$\cap​\$                                     |
        | 并                 | $\cup$                                     | \$\cup​\$                                     |
        | 空集               | $\emptyset$                                | \$\emptyset\$                                |

    + 希腊字母

        | 字母名称 | 大写 | markdown原文 | 小写 | markdown原文 | 专用形式 | markdown原  |
        | -------- | ---- | ------------ | ---- | ------------ | -------- | ----------- |
        | alpha    | A    | A            | α    | \alpha       |          |             |
        | beta     | B    | B            | β    | \beta        |          |             |
        | gamma    | Γ    | \Gamma       | γ    | \gamma       |          |             |
        | delta    | Δ    | \Delta       | δ    | \delta       |          |             |
        | epsilon  | E    | E            | ϵ    | \epsilon     | ε        | \varepsilon |
        | zeta     | Z    | Z            | ζ    | \zeta        |          |             |
        | eta      | E    | E            | η    | \eta         |          |             |
        | theta    | Θ    | \Theta       | θ    | \theta       |          |             |
        | iota     | I    | I            | ι    | \iota        |          |             |
        | kappa    | K    | K            | κ    | \kappa       |          |             |
        | lambda   | Λ    | \Lambda      | λ    | \lambda      |          |             |
        | Mu       | M    | M            | μ    | \mu          |          |             |
        | nu       | N    | N            | ν    | \nu          |          |             |
        | xi       | Ξ    | \Xi          | ξ    | \xi          |          |             |
        | omicron  | O    | O            | ο    | \omicron     |          |             |
        | pi       | Π    | \Pi          | π    | \pi          |          |             |
        | rho      | P    | P            | ρ    | \rho         |          |             |
        | sigma    | Σ    | \Sigma       | σ    | \sigma       |          |             |
        | tau      | T    | T            | τ    | \tau         |          |             |
        | upsilon  | Υ    | \Upsilon     | υ    | \upsilon     |          |             |
        | phi      | Φ    | \Phi         | ϕ    | \phi         | φ        | \varphi     |
        | chi      | X    | X            | χ    | \chi         |          |             |
        | omega    | Ω    | \Omega       | ω    | \omega       |          |             |
        | psi      | Ψ    | \Psi         | ψ    | \psi         |          |             |

+ VS：mac版vs无法写c

+ 美区Apple ID

    + 不可购买付费app

    + 准备工作
        + 手机上语言设置为English
        + 挂代理，美国节点
        + [https://whatismyipaddress.com/](https://link.zhihu.com/?target=https%3A//whatismyipaddress.com/) 验证ip地址为美国否united states

    + 开始注册
        + safari打开 [appleid.apple.com](https://link.zhihu.com/?target=http%3A//appleid.apple.com/) ，看网站右下角地区是否美国，不是则切换为美国
        + 选择create a new apple id
        + 按步骤填写信息，记住密保问题
        + 可能需要verify email：切记收到email以后，也要在当前一样的代理下打开
        + 登陆id

    + 完善账户信息
        + safari打开 [appleid.apple.com](https://link.zhihu.com/?target=http%3A//appleid.apple.com/)，登陆上面创建成功的apple id。可能回答密保问题
        + 登陆后点击 payment & shipping，Payment选择none
        + 填写美国地址：到google map上搜一个真实地址
        + 填写手机号码：到 [Temporary SMS and Disposable Numbers](https://link.zhihu.com/?target=https%3A//smsreceivefree.com/) （或Google “disposable number”）可以提供一个美国的虚拟手机号码，可以收到短信验证码。不需要写美国的国家代码 +1。有些号码会提示invalid phone number，换一个再试

    + 验证账号

        + 打开手机上的app store，应该会自动变成美国区商店（英文）
        + 在美国区的app store里面，随便选择一个免费app下载。然后会提示让你登陆，选择use an existing apple id

        + 输入注册好的账号，会提示this apple id has not yet been used in the itunes store。点击 Review
        + 点击review以后，进去确认terms and conditions，然后一路确认

        + 然后切换到app store随便下载一个免费app

## 安装

### Homebrew

+ 最好用的软件安装和管理程序
+ 安装：`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
+ 安装xxx包
    + `brew install xxx`
    + `brew cask install xxx`：不编译，直接下载二进制包
+ 查找xxx包：`brew search xxx`
+ 卸载
    + `brew uninstall xxx`
    + `brew cask uninstall xxx`

### Python3

+ `brew search python3`

### [Git](https://github.com/Niuyuhang03/notebooks/blob/master/技术/Git.md)

### 第三方软件

+ [马可波罗](https://www.macbl.com/app/system)

+ [macwk](https://www.macwk.com/)

+ 报错

    > + 打不开 XXX.app，因为它来自身份不明的开发者
    > + XXX.app 已损坏，打不开。您应该将它移到废纸篓

    + 打开*系统偏好设置*界面，进入安全性与隐私。点按左下角的锁头图标，解锁更改权限。将允许从以下位置下载的应用，更改为任何来源。
    + 若没有任何来源，打开终端，输入`sudo spctl --master-disable`，输入账户密码。
    + 如已经开启任何来源，但依旧打不开，输入`sudo xattr -d com.apple.quarantine /Applications/xxxx.app`。

### [代理](https://github.com/Niuyuhang03/notebooks/blob/master/记录/VPN.md)

### Latex（VS Code+MacTex+Adobe Acrobat DC）

+ [安装homebrew](# Homebrew)
+ 安装vs code：官网
+ 下面两个二选一，空间足够推荐第一个：
    + 安装MacTex：总大小约4G，`brew cask install mactex`
    + 安装BasicTeX：总大小80MB，是MacTex的无GUI和无包版，`brew cask install basictex`。没有字体没有包，很难用

+ VS Code安装LaTeX Workshop插件

+ 重启vscode，然后，打开vscode的配置界面(快捷键`cmd+,`)，需要修改下面两项配置。

    + 在`latex-workshop.latex.tools`配置中，增加xelatex项，具体如下。这一项的作用是在工具集中定义xelatex项，以便下一项配置能找到。增加的过程，即在设置中搜索配置，点击“将设置复制到json文本”，进入settings.json，粘贴，==增加==xelatex项，格式同后面几项。

        ```json
        {
            "latex-workshop.latex.tools": [
                {
                    "name": "xelatex",
                    "command": "xelatex",
                    "args": [
                        "-synctex=1",
                    	"-interaction=nonstopmode",
                    	"-file-line-error",
                    	"%DOC%"
                    ],
                    "env": {}
                },
                {
                    ...
                },
                {
                    ...
                },
            ],
        }
        ```

    + 在`latex-workshop.latex.recipes`配置中，将第一项的latexmk改为xelatex。这里定义的是编译时调用的工具顺序，默认第一个为latexmk，因为我们要支持中文，所以在latxmk==前==增加如下项目。

        ```json
        {
        	"latex-workshop.latex.recipes": [
                {
                  "name": "xelatex",
                  "tools": [
                    "xelatex"
                  ]
                },
                {
                  "name": "xelatex -> bibtex -> xelatex*2",
                  "tools": [
                    "xelatex",
                    "bibtex",
                    "xelatex",
                    "xelatex"
                  ]
                },
                {
                    ...
                },
                {
                    ...
                }
            ]
        }
        ```

+ 设置pdf阅读器

    ```json
    {
    	"latex-workshop.view.pdf.viewer": "external",
        "latex-workshop.view.pdf.external.viewer.command": "/Applications/Adobe Acrobat DC",
        "latex-workshop.view.pdf.external.viewer.args": [
          "%PDF%"
        ]
    }
    ```

+ 编译：command+s或command+option+b

+ 报错
    + `recipe terminated with fatal error: spawn xelatex enoent`：重启vs code，关闭时叉掉进程。
    + `Recipe terminated with error.`：查看vs code的compile log文件，可以发现如`! LaTeX Error: 'File multirow.sty' not found.`的错误，即缺少multirow宏，使用`sudo tlmgr install xxx`安装

### Endnote x9

+ 下载Endnote后，word会多一栏endnote
+ 在endnote中file-new新建文献仓库，推荐一个论文的所有文献放在一个仓库
+ 编辑-输出样式-打开样式管理器，查看是否有GBT7714，没有需要在[Chinese Standard GBT7714 (numeric)](https://endnote.com/style_download/chinese-standard-gb-t7714-numeric/)下载，双击打开，file-save as，起名Chinese Standard GBT7714 (numeric)，再去样式管理器，选中这个格式
+ 在word的endnote插件界面，样式改成GBT7714
+ 打开跳转：word的endnote插件界面，configure bibliography，选中link in-text项目。这个动作只能让本.doc有效，下一篇.doc还需操作
+ bibtex导入endnote：安装bibutils，bib2xml input.bib | xml2end > output.end

### Typora

+ 下载：官网
+ 图床：uPic作为Typora自动上传到图床服务商的软件，`brew cask install upic`，在Typora的设置-图像-自动上传服务，选择uPic，配置uPic的系统偏好设置-扩展-访达扩展权限。可实现在Typora加入本地图片时自动上传，并将图片链接更换为url。

### VS Code

+ 主题选择one dark pro

### 其他常用软件清单

+ One Switch
+ 自动切换输入法
+ Megnet
+ Tencent Lemon
+ Bartender
+ Sublime
+ Office
+ Endnote
+ 坚果云
+ Chrome
+ eZip
+ Pycharm
+ 微信
+ QQ
+ 迅雷
+ 百度网盘
+ HSTracker
+ 战网
+ Steam
+ 网易云音乐