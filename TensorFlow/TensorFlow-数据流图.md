# TensorFlow 数据流图
- 数据流图定义：用节点和有向边描述数学运算的有向无环图
	- 节点：
		- 通常代表各类操作（operation），具体包括数学运算、数据填充、结果输出和变量读写等操作
		- 每个节点上的操作都需要分配到具体的物体设备（如CPU、GPU）上执行
	- 有向边：
		- 描述了节点间的输入、输出关系，边上流动（flow）着代表高维数据的张量（tensor）

- 基于梯度下降法优化求解的机器学习问题，包含两个计算阶段：
	- 阶段一：前向图求值：
		- 由用户编写代码完成
		- 定义模型的目标函数（object function）和损失函数（loss function）
		- 输入、输出数据的形状（shape）、类型（dtype）
    - 阶段二：后向图求梯度：
		- 由TensorFlow的优化器（optimizer）自动生成。
		- 计算模型参数的梯度值，并使用梯度值更新对应的模型参数
	
- 节点：
	- 前向图中的节点（统一称为操作）
		- 数学函数或表达式
			- 绝大多数节点都属于此类
		- 存储模型参数的变量（variable）
		- 占位符（placeholder）：
			- 通常用来描述输入、输出数据的类型和形状等，便于用户利用**数据的抽象结构**直接定义模型；
			- 在数据流图执行时，需要填充对应的数据<一般使用feed_dict = {a: 3, b: 4}字典来填充>。
	- 后向图中的节点：
		- 梯度值：
			- 经过前向图计算出的模型参数的梯度
		- 更新模型参数的操作：
			- 如何将梯度值更新到对应的模型参数
		- 更新后的模型参数：
			- 与前向图中的模型参数一一对应，但参数值得到了更新，用于模型的下一轮训练
			
- 有向边：用于定义操作之间的关系：
	- 数据边：用来传输数据，绝大部分流动着张量的边都是此类（实线）
	- 控制边：用来定义控制依赖，通过设定**节点的前置依赖**决定相关节点的执行顺序（虚线）
		- 入度为0的节点没有前置依赖，可以立即执行
		- 入度非0的节点需要等待所有依赖节点执行结束后，方可执行。
	- 所有的节点都通过数据边和控制边连接。

- 执行原理
	- 声明式编程的特点决定了在深度神经网络模型的数据流图上，各个节点的执行顺序并不完全依赖于代码中定义的顺序，而是节点之间的逻辑关系以及运行时库的实现机制相关。
	- 数据流图上节点的执行顺序的实现参考了“拓扑排序”的设计思想
		- 1、？？？？？？
		- 2、？？？？？？
		- 3、？？？？？？
		- 4、？？？？？？

		
# 数据载体（张量）：
- TensorFlow使用张量统一表示所有数据
- TensorFlow提供Tensor和SparseTensor两种张量抽象，分别表示稠密数据和稀疏数据		

## 张量：Tensor
- 在Numpy等数学计算库或TensorFlow等深度学习库中，通常使用**多维数组**的形式描述一个张量。
- 张量的阶数
	- 描述的数据所在高维空间的维数
	- 用列表表示：[2,3,3]
- 张量支持的数据类型
	- tf.float32
	- tf.int8
	- tf.uint8
	- tf.string
- 创建
	- 稠密张量抽象是Tensor类
	- Tensor构造方法的完整输入参数，同时也是张量的属性
		- dtype
		- name
		- graph
		- op
		- shape
		- value_index
	- 一般情况下，不需要使用Tensor类的构造方法直接创建张量，而是通过**操作**间接创建张量 [a,b,c]
		- a = tf.constant(1.0)
		- b = tf.constant(2.0)
		- c = tf.add(a, b)
- 求解
	- 如果想要求解特定张量的值，则需要创建会话，然后执行eval方法或**会话的run方法**
	- 推荐使用tf.Session().run(*)
- 成员方法
	- 来动态改变张量形状，以及查看张量的后置操作
		- eval
		- get_shape
		- set_shape
		- consumers
- 操作：为张量提供了大量操作，以便构建数据流图，实现算法模型；操作对象是张量
	- 一元代数操作
		- abs
	- 二元代数操作
		- add
		- sub
		- multiply
	- 形状操作
	- 归约操作
		- reduce_mean
		- reduce_sum
	- 神经网络操作
		- conv
		- pool
		- softmax
		- relu
	- 条件操作
	
## 稀疏张量：SparseTensor
- 专门用于处理高维稀疏数据的SparseTensor类
	- 该类以键值对的形式表达高维稀疏数据
		- indices
		- values
		- dense_shape
- 创建：
	- 一般可以直接使用SparseTensor类的构造方法
	- sp = tf.SparseTensor(indices=*, values=*, dense_shape=*)
		- indices:[[0,2],[1,3]]  非零元素索引值
		- values:[1,2]           指定非零元素（与上面索引值对应起来）
		- dense_shape:[3,4]      稀疏数据的维度信息
			[[0,0,1,0]
			 [0,0,0,2]
             [0,0,0,0]]
- 操作：操作对象是稀疏张量
	- 转换操作
	- 代数操作
	- 几何操作
	- 归约操作
		
# 模型载体：操作
- TensorFlow的算法模型由数据流图表示，数据流图由节点和有向边组成，每个节点均对应一个具体的操作。
- 数据流图中的节点按照功能不同可分为以下3种：
	- 计算节点：Operation
	- 存储节点：Variable
	- 数据节点：Placeholder
	
## 计算节点：Operation
- 对应的计算操作抽象是Operation类；入边（输入张量）、出边（输出张量）
- 计算操作的主要属性
	- name
	- type
	- inputs
	- control_inputs
	- outputs
	- device
	- graph
	- traceback
- 通常不需要显式构造Operation实例，只需要使用TensorFlow提供的**各类操作函数**来定义计算节点。（**操作对象**是张量）
- TensorFlow Python API提供的典型操作
	- 基础算术
	- 数组运算
	- 神经网络运算
	- 图像处理运算
- 在数据流图计算开始之前，用户通常需要执行tf.global_variables_initializer函数来进行**全局变量**的初始化
	- 本质是将initial_value传入Assign子节点，实现对变量的初次赋值
	- 什么是全局变量，这句话什么时候使用呢？？？？？？

## 存储节点：Variable
- 在多次执行相同数据流图时存储特定的参数，如深度学习或机器学习的模型参数
- 抽象是Variable类，称其为**变量**
- 变量 
	- 变量的主要属性
		- name
		- dtype
		- shape
		- inital_value 
		- initializer
		- device
		- graph
		- op
	- 一个变量通常由四种子节点构成：TensorBoard为有状态操作添加了一对括号
		- 变量的初始值：inital_value、无状态操作
		- 更新变量值的操作：Assign、无状态操作
		- 读取变量值的操作：read、无状态操作
		- 变量操作：(a)、有状态操作
- 变量操作：用于存储变量的值
	- 操作函数：variable_op_v2
		- 存储变量的形状
		- 数据类型
	- 变量两种初始化方式：
		- 初始值：用户输入初始值完成初始化 
			- _init_from_args(输入初始值完成初始化)、
		- VariableDef：用户使用Protocol Buffers
			- _init_from_proto(使用Protocol Buffers定义完成初始化)
			tf.Variable()
- read节点
	- ？？？？？？


## 数据节点：Placeholder
- 通常用户在创建模型时已经明确输入数据的类型和形状等属性
- 定义待输入数据的属性，使得用户可以描述数据特征，从而完成**模型的创建**
- 当执行数据流图时，向数据节点填充（feed）用户提供的、符合定义的数据
- tf.placeholder()、tf.sparse_placeholder()：输入参数
	- name
	- dtype
	- shape