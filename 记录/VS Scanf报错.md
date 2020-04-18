>  error C4996: 'scanf': This function or variable may be unsafe. Consider using scanf_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.

+ 修改属性管理器：是将错误中提到的宏 **`_CRT_SECURE_NO_WARNINGS`** 添加到视图-属性管理器-Microsoft.Cpp.Win32.user属性页-通用属性-C/C++-预处理器-预处理器定义中，具体如图，这种设置方法对所有的项目都有效，如果其他的项目使用 scanf 不再需要设置

    ![clipboard.png](https://i.loli.net/2020/03/07/U7bLHdQjcZ4IJ3P.png)

    ![clipboard _1_.png](https://i.loli.net/2020/03/07/DmqoT1aRXp93EHi.png)