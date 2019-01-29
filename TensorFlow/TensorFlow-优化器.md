# 训练工具：优化器
- 机器学习大致分为3种类型
	- 监督学习
		- 模型
			- 模型本身是一系列数学表达式和一组参数的组合
			- 模型训练是指在给定数据上不断拟合，以求出一组在测试数据集上使得推理值尽可能接近真实值的模型参数
		- 损失函数
			- 表达模型推理值同真实值差距的函数，用于评估模型的拟合程度
			- 即求得损失值loss
		- 优化算法
			- 使用损失值不断优化模型参数，以尽可能减小损失值的算法
			- （主流）基于梯度下降的优化算法进行模型训练
	- 无监督学习
	- 半监督学习

## 损失函数与优化算法
### 损失函数
- 损失函数：评估特定模型参数和特定输入时，表达模型输出的推理值与真实值之间不一致程度的函数。
	- 平方损失函数（Y = 推理值；Y_= 真实值）
		- loss_op = tf.reduce_sum(tf.pow(Y-Y_, 2))/(total_samples)
	- 交叉熵损失函数
		- loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_logits(labels=Y_, logits=Y))
	- 指数损失函数
		- loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
- 损失函数是一个非负实值函数，值越小说明模型对训练集拟合得越好。
	- 使用损失函数对所有训练样本求损失值，再累加求平均可得到模型的经验风险。
- 为了降低过度训练可能造成的过拟合风险，可以引入专门用来度量模型复杂度的正则化项或惩罚项
	- LO、L1、L2范数

### 优化算法
- 求解最优化问题的算法称为优化算法
	- 迭代方式实现
		- 首先设定一个初始的可行解
		- 然后**基于特定的函数**反复重新计算可行解
		- 直到找到一个最优解或达到预设的收敛条件
	- 迭代策略
		- 目标函数的一阶导数：梯度下降法 【其在优化复杂模型时几乎不会被使用】
			- 直接沿着梯度负方向，即目标函数减小最快的方向进行直线搜索。
			- 优点：计算量小，仅计算一阶导数即可
			- 缺点：收敛速度慢，只能做到线性收敛
		- 目标函数的二阶导数：牛顿法
		- 前几轮迭代的信息：Adam
	- 利用损失值和模型的拓扑结构，可以计算出模型参数的梯度值gradients；随后，优化算法以一种“高效而合理的方式”将梯度值更新到对应的模型参数，完成模型的一步迭代训练。 

## 优化器
- TensorFlow的优化器根据前向图的计算拓扑和损失值，利用**链式求导法则**依次求出每个模型参数在给定数据下的梯度值，并将其更新到对应的模型参数以完成一个完成的训练步骤
- 优化器的基类：Optimizer，用户并不会创建Optimizer类的实例，而是需要创建特定的子类实例
	- tf.train.GradientDescentOptimizer()
	- tf.train.AdadeltaOptimizer()
	- tf.train.SyncReplicasOptimizer()
- 基类方法
	- minimize
		- 内部调用compute_gradients和apply_gradeents方法
		- global_step：全局训练步数
			- 在创建global_step变量时，需要显式地将trainable参数设置为False，表示不需要在训练过程中自动计算其梯度值
			- 内部在应用梯度成功后，会将global_step变量的值加1
			- global_step = tf.Variable(0, name='global_step', trainable=False)
		- 返回操作命名为train_op或train_step
		- gate_gradients，对计算效率有一定影响
			- GATE_NONE
			- GATE_OP
			- GATE_GRAPH
	- compute_gradients
		- 计算模型所有参数的梯度值
	- apply_gradeents
		- 将梯度值更新到对应的模型参数
- 子类方法
	- _apply_dense
		- 使用稠密梯度值更新模型参数，返回数据流图上的操作
	- _apply_sparse
		- 使用稀疏梯度值更新模型参数，返回数据流图上的操作
- minimize方法训练模型：
	- 模型训练的过程需要最小化损失函数
	- TensorFlow的所有优化器均实现了用于最小化损失函数的minimize方法。

# TensorFLow训练模型典型过程
- （1）定义超参数
- （2）输入数据
- （3）构建模型
- （4）定义损失函数
- （5）创建优化器
- （6）定义单步训练操作
- （7）创建会话
- （8）迭代训练