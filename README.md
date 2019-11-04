# 机器学习快速学习要点

## 基本信息

作者：杨岱川

时间：2019年9月

github：https://github.com/DrDavidS/basic_Machine_Learning

开源协议：[MIT](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/LICENSE)

> 转载请注明出处，请遵循开源协议！

## 编写说明

本教程基于 [Jupyter Notebook](http://www.jupyter.org/) 编写，如果你不知道怎么使用 Jupyter Notebook ，赶紧去学一学（真的不难）。

本仓库的最初建立目的是给实验室的师弟师妹们讲课用的，后来发现也许可以帮助更多的人。

本教程的教学顺序和理论知识基本参照李航博士的《统计学习方法》（第二版）编写，其中为了更好地帮助初学者入门，快速理解枯燥的理论和数学知识，增加了一部分实际生活中例子或者是简单数据集的例子，以及机器学习中常用的、但是书中没有记录的方法（例如探索性数据分析方法，Exploratory Data Analysis，简称 EDA），同时删除了一些不常用的细节理论部分。所以最好是配合李航博士的书一起看比较好。

本教程还有一个目的，就是帮助快速掌握机器学习实践的基本要领，注重**从理论到实战**的转换。因此本教程很少有纯Python手写的算法，尽量用现成的轮子（比如SKlearn、Pytorch等），同时对轮子的使用方法有详细的说明。

>为啥不造轮子？
>
>一. 没时间写，如果读者有兴趣手写，可以参考黄海广博士的轮子[《统计学习方法》的代码实现](https://github.com/fengdu78/lihang-code)。
>
>二. 本教程的编写本意就是快速实现，所以更着重于调用现成的轮子快速完成任务，而不是培养大家的编程水平，提高编程水平可以去[LeetcodeCN](https://leetcode-cn.com/)做题。
>
>对于初学者，不管是为了工作还是写论文，快速实现才是王道。

## 学前须知

### 本教程合适

- 熟悉线性代数的人
- 熟悉概率统计的人
- 有一定英语基础的人（因为可能会阅读英文文档）
- 有一定的编程基础的人，Python3.x最好
- 刚刚入门机器学习/模式识别/统计学习方法，想要学习理论并且快速上手做事情的人
- 想要复习一下机器学习基础理论和实践的人

### 本入门教程不合适

- 对线性代数（大学本科水平）一无所知的人
- 对概率论（大学本科水平）一无所知的人
- 不知道何谓“机器学习”、“数据挖掘”的人
- 只希望学习一些针对比赛的小技巧（tricks）的人
- 机器学习大佬（有空可以去研究怎么用Lisp写深度学习框架）

## 学习路线

### Python基础

安装[Anaconda3](https://www.anaconda.com/)。如何使用基本的、必要的Python3语法。如果想学习更多Python3的相关知识，推荐阅读《笨办法学Python3》。

### Numpy和Pandas基础

如何使用Numpy进行快速矩阵运算？本节包含基本的Numpy基础操作知识。Numpy支持各类矩阵运算，不输 Matlab。

使用Pandas的基本操作，主要针对如何将数据读取为DataFrame格式，基本的数据筛选操作等。

### Python简单绘图技巧

本教程主要使用 matplotlib 作为绘图工具，也偶尔会使用 seaborn 绘图。数据可视化是一项重要的技能。

### 机器学习理论与实践

主要是《统计学习方法》中所述理论知识的简化版，有条件有时间有耐心的人建议照着书笔算一遍。同时还有部分扩展性知识。

当前内容包括（持续更新中）：

1. [统计学习与监督学习基本概念](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/%E6%9D%AD%E7%94%B5%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AF%BE%E7%A8%8B%E5%8F%8A%E4%BB%A3%E7%A0%81/2.01%20%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E5%8F%8A%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E6%A6%82%E8%AE%BA.ipynb)
2. [感知机的基本原理和快速实现](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/%E6%9D%AD%E7%94%B5%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AF%BE%E7%A8%8B%E5%8F%8A%E4%BB%A3%E7%A0%81/2.02%20%E6%84%9F%E7%9F%A5%E6%9C%BA%E5%9F%BA%E7%A1%80.ipynb)
3. [K近邻法的基本原理和快速实现](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/%E6%9D%AD%E7%94%B5%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AF%BE%E7%A8%8B%E5%8F%8A%E4%BB%A3%E7%A0%81/2.03%20K%E8%BF%91%E9%82%BB%E6%B3%95.ipynb)
4. [朴素贝叶斯的基本原理和快速实现](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/%E6%9D%AD%E7%94%B5%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AF%BE%E7%A8%8B%E5%8F%8A%E4%BB%A3%E7%A0%81/2.05%20%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF.ipynb)
5. [特征的统计学检查](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/%E6%9D%AD%E7%94%B5%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AF%BE%E7%A8%8B%E5%8F%8A%E4%BB%A3%E7%A0%81/2.04%20%E7%89%B9%E5%BE%81%E7%9A%84%E7%BB%9F%E8%AE%A1%E5%AD%A6%E6%A3%80%E6%9F%A5.ipynb)

**注意，以上内容一般还包括当节内容的基本数据的分析处理和可视化相关知识。**

## 无法浏览怎么办

无法直接在GitHub上加载ipynb格式的话，可以下载文件在本地使用 [Jupyter Notebook](http://www.jupyter.org/) 浏览，或者点击使用[nbviewer](https://nbviewer.jupyter.org/)浏览。
