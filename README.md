# llama2
利用Transformer库从0开始搭建llama2
## 遇到的问题求解：
1. 左填充
2. RoPE
## llama2 用到的技术梳理
讲清楚原有；为什么这样子
涉及到模型本身以及 预训练数据集的处理
## 代码梳理
1. decoder layer 叠加
2. attention mask的上三角实现细节
3. 分类问题的head 映射层
4. loss计算问题
