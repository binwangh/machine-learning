# python代码
with tf.name_scope("***") as scope:
    ...
	...
	...
	...
	...
	...
	
	
sess = tf.Session()

writer = tf.summary.FileWriter("log/", sess.graph)

init = tf.global_variable_initializer()

sess.run(init)

# 运行tensorboard
- tensorboard --logdir logs
	- 生成IP链接，用浏览器打开即可


# TensorBoard遇到的坑
- 出现浏览器打不开的情况（更新浏览器）
	- sudo apt-get update
	- sudo apt-get install firefox