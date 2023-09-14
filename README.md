<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# llama2
利用Transformer库从0开始搭建llama2
# 模型结构图
## 遇到的问题求解：
1. 为什么要左填充
## llama2 用到的技术梳理
（基本原则）讲清楚缘由：为什么这样子这么做，涉及到模型本身的搭建：
### 旋转位置编码 RoPE
* 绝对位置编码：$$attent(x_m, x_n, m, n) = f(x_m, x_n, m, n)$$
* 相对位置编码：
* 旋转位置编码：旋转位置编的巧妙之处在于通过绝对编码的方式实现了相对编码的效果。
个人理解：旋转矩阵的向量积
### NTK
定义：
作用：
### group attention
定义：
作用：
### RMSNorm
* ```LayerNorm```的作用：通过均值-方差归一化使得输入和权重矩阵具有重新居中和重新缩放的能力，帮助稳定训练并促进模型收敛。

* ```RMSNorm```的作用：LayerNorm的改进，他假设LayerNorm中的重新居中性质并不是必需的，于是RMSNorm根据均方根（RMS）对某一层中的神经元输入进行规范化，同时成一个可以学习的权重系数，赋予模型重新缩放的不变性属性和隐式学习率自适应能力。相比LayerNorm，RMSNorm在计算上更简单，因此更加高效。可以保证在层归一化不会改变hidden_states对应的词向量的方向，只会改变其模长。
个人理解：这样既可以将向量映射到一个梯度敏感的区间，又可以尽量的不改变向量的性质。
## 代码梳理
### decoder layer 叠加
模型的主体部分就是通过decoder layer 堆叠起来的。
### attention mask的上三角实现细节 （广播机制）
利用了torch的广播机制，就实现了上三角mask矩阵的构建。简化的示意图如下：

### 分类问题的head 映射层

### loss计算问题
