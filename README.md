# llama2
利用Transformer库从0开始搭建llama2
## 遇到的问题求解：
1. 为什么要左填充
## llama2 用到的技术梳理
（基本原则）讲清楚缘由：为什么这样子这么做，涉及到模型本身的搭建：
### 旋转位置编码 RoPE
### NTK
### group attention
### LlamaRMSNorm
## 代码梳理
### decoder layer 叠加
### attention mask的上三角实现细节 （广播机制）
### 分类问题的head 映射层
### loss计算问题
