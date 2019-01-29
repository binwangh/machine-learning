# TensorFlow 命令行参数
- 命令行参数特指启动TensorFlow程序时输入的参数
	- 模型超参数
		- 机器学习和深度学习模型中的框架参数：梯度下降的学习率和批数据大小
		- 主要用于优化模型的训练精度和速度
	- 集群参数
		- 运行TensorFlow分布式任务的集群配置参数：参数服务器主机地址和工作服务器主机地址
		- 主要用于设置TensorFlow集群

## 使用argparse解析命令行参数
- argparse模块是Python标准库提供的用于命令行参数与选项解析的模块
- 使用流程三步：[import argparse]
	- 1）创建解析器
		- parser = argparse.ArgumentParser()
			- prog
			- usage
			- epilog
			- add_help
			- argument_default
	- 2)添加待解析参数
		- parser.add_argument()
			- name or flags
			- action
			- type
			- default
			- help
	- 3）解析命令行输入的参数
		- FLAGS, unparsed = parser.parse_known_args()
			- FLAGS包含待解析的参数
			- 将解析器中未定义的参数返回给unparsed