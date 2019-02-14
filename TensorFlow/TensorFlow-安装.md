# TensorFlow安装（联网）
- conda create -n tensorflow python=2.7
- source activate tensorflow
	- pip install tensorflow-gpu==1.2.0
	- pip install --upgrade pip
- source deactivate
- conda remove -n tensorflow --all

## libcudnn的软连接
- 解压现有的libcudnn安装文件
- 拷贝在对应的lib文件夹下（比如：envs/***/lib/）
- 删除原有的软连接文件
	- rm -rf libcudnn.so libcudnn.so.6
- 新建软连接
	- chmod u=rwx,g=rx,o=rx libcudnn.so.6.0.21
	- ln -s libcudnn.so.6.0.21 libcudnn.so.6
	- ln -s libcudnn.so.6 libcudnn.so

[TensorFlow与CUDA、CUDNN版本对应关系](https://blog.csdn.net/omodao1/article/details/83241074)

[安装对应的libcudnn，解压之后需要从新指定软连接](https://blog.csdn.net/qq_29921623/article/details/78110853)