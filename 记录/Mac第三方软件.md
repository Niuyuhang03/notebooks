# Mac第三方软件

+ [马可波罗](https://www.macbl.com/app/system)
+ [macwk](https://www.macwk.com/)

# Mac软件安装问题

> + 打不开 XXX.app，因为它来自身份不明的开发者
> + XXX.app 已损坏，打不开。您应该将它移到废纸篓

+ 打开*系统偏好设置*界面，进入安全性与隐私。点按左下角的锁头图标，解锁更改权限。将允许从以下位置下载的应用，更改为任何来源。
+ 若没有任何来源，打开终端，输入`sudo spctl --master-disable`，输入账户密码。
+ 如已经开启任何来源，但依旧打不开，输入`sudo xattr -d com.apple.quarantine /Applications/xxxx.app`。