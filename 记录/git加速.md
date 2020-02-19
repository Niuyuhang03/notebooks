# 下载releases的文件太慢

+ 直接复制链接到迅雷

# 下载整个仓库的zip太慢

+ 在码云gitee导入github仓库链接，在码云下载

#  clone仓库太慢

## 为 git 使用代理

如果你已经拥有了一些代理软件，那么直接为 `git` 设置代理是最好的提速方法，这里以 `ss` 为例，假设本地代理地址为 `127.0.0.1：1080`，那么你可以使用以下命令为 `git` 设置代理：

```
# http optionsgit config --global http.proxy "socks5h://127.0.0.1:1080"# unknown section namegit config --global https.proxy "socks5h://127.0.0.1:1080"# 查看更改是否成功git config --global --get http.proxygit config --global --get https.proxys# 取消代理git config --global --unset http.proxygit config --global --unset https.proxy
```

这一方法可以加速克隆使用 http/https 协议进行传输的仓库，使用 ssh 协议的需要其他设置，这里不加以阐述。

```
# http/https协议git clone https://github.com/leslievan/verilog.git# ssh协议git clone git@github.com:leslievan/verilog,git
```

## 修改 host 文件

在 `git clone` 或 `git push` 时，实际上并不是直接向 `github.com` 发送请求，而是对 `github.global.ssl.fastly.net` 发送请求与通信，Fastly 公司在中国有着众多的 CDN 节点，GitHub 可能因为成本或者其他原因，并没有在中国搭设自己专属的 CDN 节点，我们可以通过修改 `host` 文件来加速对这个域名的访问。

```
# windows下修改C:\Windows\System32\drivers\etc\hosts# Linux/Mac下修改/etc/hosts# 在最后加上151.101.77.194  github.global.ssl.fastly.net13.229.188.59   github.com185.199.109.153 assets-cdn.github.com151.101.76.249  global-ssl.fastly.net
```

然后刷新 DNS 缓存。

```
# windowsipconfig /flushdns# linux/macsudo /etc/init.d/network-manager restart
```

如果网络没问题的话，修改后的速度一般都能达到 `MB/s` 的级别。