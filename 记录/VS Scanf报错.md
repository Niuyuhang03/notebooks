如何解决：error C4996: 'scanf': This function or variable may be unsafe. Consider using scanf_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.

1. 2

很多人在使用scanf的时候一来就会遇到如下的问题：

error C4996: 'scanf': This function or variable may be unsafe. Consider using scanf_s instead. To disable deprecation, use **_CRT_SECURE_NO_WARNINGS**. See online help for details.

1. 3

解决方法

1. 4

修改属性管理器：是将错误中提到的宏 **_CRT_SECURE_NO_WARNINGS** 添加到属性管理器中，具体设置如下，这种设置方法对所有的项目都有效，如果其他的项目使用 scanf 不再需要设置；

![img](F:\笔记\1411490323@qq.com\54dc65f802984aeebcbaa41d4b1ff617\clipboard.png)

![img](F:\笔记\1411490323@qq.com\c8a1fd4806d94c5ba0d511ab18910602\clipboard.png)