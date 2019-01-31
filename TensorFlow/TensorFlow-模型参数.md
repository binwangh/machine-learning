# 模型参数
- 模型参数指模型的**权重值**和**偏置值**
- 模型参数的使用方法
	- 模型参数的创建、初始化和更新
		- tf.Variable类实现
	- 从模型文件中存储和恢复模型参数的方法
		- tf.train.Saver类实现
		- checkpoint文件是以<变量名，张量值>的形式来序列化存储模型参数的二进制文件，是用户持久化存储模型参数的推荐文件格式，扩展名为ckpt。
	    - 存储：是指将变量中存储的模型参数定期写入checkpoint文件
	    - 恢复：是指读取checkpoint文件中存储的模型参数，基于这些值继续训练模型
	    - TensorFlow支持：选择性存储和恢复部分任意变量的方法，使得用户可以灵活地改造模型，并基于之前的训练结果做参数微调（fine-tuning）

## 使用tf.Variable创建、初始化和更新模型参数
### （1）创建模型参数
- 确定模型的基本属性：
	- 初始值
	- 数据类型
	- 张量形状
	- 变量名称
- W = tf.Variable(initial_value=tf.random_normal(shape=(1, 4), mean=100, stddev=0.35), name='W')
	- inital_value：在会话中为变量设置的初始值
		- 1）符合统计分布的随机张量方法
			- tf.random_normal([1,4], mean=-1, stddev=4)
			- tf.random_uniform([2,3])
			- tf.multinomial(tf.log([[10.,10.]]), 5)
		- 2）符合某种生成规则的序列张量
			- tf.range(start=3, limit=18, delta=3)
		- 3）张量常量
			- tf.zeros()
			- tf.ones()
			- tf.fill
			- tf.constant()
		- 4) 已经初始化的变量值作为新创建变量的初始值
			- ***.initialized_value()
### （2）初始化模型参数
- 没有初始化的变量是无法使用的。初始化变量在**运行环境**中完成
- 两种初始化变量的选择
	- 传入初始值，然后执行初始化操作赋值
	- 从checkpoint文件中恢复变量的值
- 初始化操作
	- 最常用、最简单的初始化操作：tf.global_variables_initializers(),只要在**会话**中执行它，程序就会初始化全局的变量
	- 初始化部分变量：tf.variables_initializer([*])，并显式设置初始化变量的列表。

### （3）模型参数的更新
- 更新模型参数主要指更新变量中存储的模型参数
	- 本质上就是对变量中保存的模型参数**重新赋值**
- 更新变量的方法
	- 直接赋值：tf.assign
	- 加法赋值：tf.assign_add(w, 1.0)
	- 减法赋值：tf.assign_sub

## 使用tf.train.Saver保存和恢复模型参数
- tf.train.Saver是辅助训练的工具类，它实现了存储模型参数的变量和checkpoint文件间的读写操作。
	- SaveOp负责向checkpoint文件中写入变量
	- RestoreOp负责从checkpoint文件中读取变量
### 1）保存模型参数
- tf.train.Saver()
	- var_list
		- 设置想要存储的变量集合
		- 支持Python字典和列表两种类型
	- reshape
	- sharded
	- max_to_keep
	- restore_sequentially
- saver = tf.train.Saver({'Weight':W})
- saver.save(sess, '/test.ckpt')
	- 通过Saver.save方法：存储会话中当前时刻的变量值。
		
### 2）恢复模型参数
- 读取checkpoint文件中存储的模型参数，基于这些值继续训练模型。
- saver = tf.train.Saver()
- saver.restore(sess, 'test.ckpt')
	- 通过Saver.restore方法恢复文件中的变量值
	- 在创建变量时，指定变量的name，读取时候方便

## 变量作用域
- 如何使用更灵活和具有层次化结构的变量作用域来处理更加复杂的模型
- tf.Variable的局限
	- 模型的定义
		- 代码复杂度将随着网络层数不断增加
	- 模型的复用
		- 想要多次使用同一个模型，tf.Variable方法在每次调用模型时都会创建一份变量，但它们存储的是相同的模型参数
		- 随着模型复用次数的增加，内存开销也将不断上升，直到内存溢出
		- （不推荐）一种简单的解决办法是定义一个存储所有模型参数的Python字典variables_dict，然后每次调用时都使用variables_dict中的共享参数。
			- 缺点：破坏了模型的封装性，降低了代码的可读性。
- 变量作用域的好处
	- 编写**管理各类网络**的方法
		- 在这个方法内部定义该网络的**结构**和**参数**；
		- 同时该方法在复用模型时，允许共享该层的模型参数。
	- TensorFlow的变量作用域机制主要由tf.get_variable方法和tf.variable_scope方法实现。
		- tf.get_variable：负责“创建或获取指定名称”的变量，在运行时根据张量的形状“动态初始化变量”
			- name
			- shape
			- initializer
				- tf.constant_initializer:常量值
				- tf.random_uniform_initializer:区间值
				- tf.random_normal_initializer:符合正态分布的张量
		- tf.variable_scope：负责管理传入tf.get_variable方法的变量名称的名字空间
- 变量作用域的使用
	- with tf.variable_scope("***", resue=True)
		- 通过不同的变量作用域区分同类网络的不同层参数
		- reuse：设置为True，表示共享该作用域内的参数
		- initializer：为作用域内的所有变量设置初始化方法

### 在处理结构复杂的模型时，恰当地使用变量作用域可以简化模型的定义和初始化工作
### 有助于开发更加层次化和模块化的模型，提升代码可读性