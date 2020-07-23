# 机器学习快速学习要点

## 基本信息

作者：杨岱川

时间：2019年9月

github：https://github.com/DrDavidS/basic_Machine_Learning

开源协议：[MIT](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/LICENSE)

> 转载请注明出处，请遵循开源协议！

## 编写说明

本教程编写始于2019年9月，起因是部分机器学习和深度学习教材注重理论而对代码实践较少，而部分“实战”类教程又几乎不涉及理论知识，只求会运用即可。当前，同时结合机器学习理论和代码实践的教程相对很少，因此在和老师讨论之后，本人利用业余时间编写了这份兼顾理论学习与编程的教程。

本教程基于 [Jupyter Notebook](http://www.jupyter.org/) 编写，如果你不知道怎么使用 Jupyter Notebook ，赶紧去学一学（真的不难）。

本仓库的最初建立目的是给实验室的师弟师妹们讨论学习用的，编写的同时也是激励自己继续学习钻研。如果这份教程能够帮助到你，那是最好不过的！

本教程的“机器学习基础”文件夹中，编写顺序和理论知识基本参照李航博士的《统计学习方法》（第二版）编写，其中为了更好地帮助初学者入门，快速理解枯燥的理论和数学知识，增加了一部分实际生活中例子或者是简单数据集的例子，以及机器学习中常用的、但是书中没有记录的方法（例如探索性数据分析方法，Exploratory Data Analysis，简称 EDA），同时删除了一些不常用的细节理论部分。所以最好是配合李航博士的书一起看比较好。

本教程“深度学习基础”，主要是基于Oreilly日本的《深度学习入门》一书的实现，作者是斋藤康毅。其内容包括深度学习的基本知识，比如前向传播、数值微分与参数更新、反向传播等。同时介绍了 PyTorch 和 TensorFlow 等当前流行的深度学习框架，并且利用深度学习框架对神经网络的搭建和训练等内容做了相关实现，对代码也有详细的讲解和注释。

本教程还有一个目的，就是帮助快速掌握机器学习实践的基本要领，注重**从理论到实战**的转换。因此本教程会视情况，灵活使用纯Python手写算法，或者直接调用现成的轮子（比如SKlearn、Pytorch等），同时对轮子的使用方法有较为详细的说明。

>为啥不全程手写代码？
>
>一. 没时间写。如果读者有兴趣手写，可以参考黄海广博士的轮子[《统计学习方法》的代码实现](https://github.com/fengdu78/lihang-code)。
>
>二. 本教程的编写本意就是快速实现，所以相对着重于调用现成的轮子快速完成任务，而不是培养大家的编程水平。顺便一提，想提高编程和数据结构水平可以去[LeetcodeCN](https://leetcode-cn.com/)做题。
>
>对于初学者，不管是为了工作还是写论文，快速实现才是王道。

## 学前须知

### 本教程合适

- 熟悉线性代数的人
- 熟悉概率统计的人
- 有一定英语基础的人（因为可能会阅读英文文档）
- 有一定的编程基础的人，Python3.x最好
- 刚刚入门机器学习/模式识别/统计学习方法，想要学习理论并且快速上手做事情的人
- 想要复习一下机器学习、深度学习基础理论和实践的人

### 本入门教程不合适

- 没有编程基础的人
- 对线性代数（大学本科水平）一无所知的人
- 对概率论（大学本科水平）一无所知的人
- 不知道何谓“机器学习”、“数据挖掘”的人
- 只希望学习一些针对比赛的小技巧（tricks）的人
- 机器学习大佬（有空可以去研究怎么用Lisp写深度学习框架）

## 学习路线

### 01 Python基础

安装[Anaconda3](https://www.anaconda.com/)。如何使用基本的、必要的Python3语法。如果想学习更多Python3的相关知识，推荐阅读《笨办法学Python3》。

另外包括 Numpy、Pandas，以及 Python 绘图工具也有简单涉及。

### 02 机器学习基础

主要是《统计学习方法》中所述理论知识的简化版，有条件有时间有耐心的人建议照着书笔算一遍。同时还有部分扩展性知识。

当前内容包括（持续更新中）：

- [2.01 统计学习及监督学习概论](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.01%20统计学习及监督学习概论.ipynb)
- [2.02 感知机基础](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.02%20感知机基础.ipynb)
- [2.03 K近邻法](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.03%20K近邻法.ipynb)
- [2.04 特征的统计学检查](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.04%20特征的统计学检查.ipynb)
- [2.05 朴素贝叶斯](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.05%20朴素贝叶斯.ipynb)
- [2.06 决策树](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.06%20决策树.ipynb)
- [2.07 蘑菇分类实践](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.07%20蘑菇分类实践.ipynb)
- [2.08 逻辑回归的原理与应用](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.08%20逻辑回归的原理与应用.ipynb)
- [2.09 数据的编码方法](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.09%20数据的编码方法.ipynb)
- [2.10 提升方法](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.10%20提升方法.ipynb)
- [2.11 XGBoost原理与应用](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/02机器学习基础/2.11%20XGBoost原理与应用.ipynb)

**注意，以上内容一般还包括当节内容的基本数据的分析处理和可视化相关知识。**

### 03 深度学习基础

主要是《深度学习入门》中所述理论知识的摘录和代码实现，同时扩展了这些实现在主流框架中的应用方法。

当前内容包括（持续更新中）：

- [3.01 神经网络与前向传播](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/03深度学习基础/3.01%20神经网络与前向传播.ipynb)
- [3.02 神经网络的训练](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/03深度学习基础/3.02%20神经网络的训练.ipynb)
- [3.03 误差反向传播法](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/03深度学习基础/3.03%20误差反向传播法.ipynb)
- [3.04 实际搭建并训练一个简单的神经网络](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/03深度学习基础/3.04%20实际搭建并训练一个简单的神经网络.ipynb)
- [3.05 与深度学习相关的要点（一）](https://github.com/DrDavidS/basic_Machine_Learning/blob/master/03深度学习基础/3.05%20与深度学习相关的要点.ipynb)

## 备注：无法浏览怎么办

无法直接在GitHub上加载ipynb格式的话，可以下载文件在本地使用 [Jupyter Notebook](http://www.jupyter.org/) 浏览，或者点击使用[nbviewer](https://nbviewer.jupyter.org/)浏览。
