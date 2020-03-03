# 下载releases的文件太慢

+ 直接复制链接到迅雷

# 下载整个仓库的zip太慢

+ 在码云gitee导入github仓库链接，在码云下载

#  clone仓库太慢

## 为 github使用代理

如果你已经拥有代理软件，直接为 `git` 设置代理是最好的提速方法。假设本地代理地址为 `127.0.0.1：1080`（端口号详见ss或v2ray的设置中socks的端口号），那么你可以使用以下命令为 `git` 设置代理：

```shell
# 首先取消已有代理
git config --global --unset http.proxy
git config --global --unset https.proxy

# 代理github
git config --global http.https://github.com.proxy socks5://127.0.0.1:1080  # 端口号需要根据v2ray的sock的端口设置
git config --global https.https://github.com.proxy socks5://127.0.0.1:1080  # 端口号需要根据v2ray的sock的端口设置

# 不推荐全局代理git，会将国内git变慢
git config --global http.proxy http://127.0.0.1:1080  # 不要做
git config --global https.proxy http://127.0.0.1:1080  # 不要做
```

**必须使用https**方式clone

## 修改 host 文件（不推荐）

在 `git clone` 或 `git push` 时，实际上并不是直接向 `github.com` 发送请求，而是对 `github.global.ssl.fastly.net` 发送请求与通信，Fastly 公司在中国有着众多的 CDN 节点，GitHub 可能因为成本或者其他原因，并没有在中国搭设自己专属的 CDN 节点，我们可以通过修改 `host` 文件来加速对这个域名的访问。

```
# windows下修改C:\Windows\System32\drivers\etc\hosts# Linux/Mac下修改/etc/hosts# 在最后加上151.101.77.194  github.global.ssl.fastly.net13.229.188.59   github.com185.199.109.153 assets-cdn.github.com151.101.76.249  global-ssl.fastly.net
```

然后刷新 DNS 缓存。

```
# windowsipconfig /flushdns# linux/macsudo /etc/init.d/network-manager restart
```

如果网络没问题的话，修改后的速度一般都能达到 `MB/s` 的级别。