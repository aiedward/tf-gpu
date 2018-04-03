# tf-gpu

## Tensorflow安装

tensorflow-gpu 1.0 要求 CUDA 8.0 和 cuDNN 5.1 安装

很好的一些参考文章：

ubuntu:
[CUDA 8.0 和 cuDNN 5.1 安装](https://zhuanlan.zhihu.com/p/27890924)
[Setup: Ubuntu 16.04 + NVIDIA + CUDA 8.0 + cuDNN 5.1](http://ywpkwon.github.io/post/ubuntu-setting/)

windows:
[Tensorflow+cuda+cudnn+window+Python之window下安装TensorFlow](https://blog.csdn.net/flying_sfeng/article/details/58057400)

[windows 下 TensorFlow（GPU 版）的安装](https://blog.csdn.net/lanchunhui/article/details/54964064)

第一步，是核实机器有gpu并给gpu装驱动

如果你的显卡是nvidia的而且支持CUDA Compute Capability 3.0以上（6系之后高于50甜点卡的型号），那么可以用GPU进行运算，详见[支持设备列表](https://developer.nvidia.com/cuda-gpus)

在windows中，在设备管理器中可以看到显卡配置；显卡驱动一般已经在装系统时装好，不用额外操心

在linux中，如果有图形界面，可以在系统设置 --> 软件与更新 --> 附加驱动中安装驱动或者更新驱动，比如，nvidia-384

如果没有图形界面，可以尝试

    sudo apt-get install nvidia-384 #can type nvidia then hit "tab" to view all available options

终端输入nvidia-smi，可以看到所装显卡驱动的信息

nvidia的显卡驱动是兼容之前版本的，也就是说，旧的驱动不支持新的显卡，但新的驱动是支持旧的显卡的。

[显卡支持信息](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA)

如果是旧的驱动，nvidia-smi可能不能显示足够多的信息；另外，nvidia-smi不显示足够多的信息也可能是因为显卡型号太老，所以不再支持。

在windows中，nvidia-smi并不是shell命令，所以必须用某种terminal进到nvidia-smi.exe所在文件夹中，再执行nvidia-smi.

nvidia-smi还是比较必要的，可以看到具体程序比如python对gpu的使用情况，但展示的信息还很不够，在使用tensorflow-gpu的过程中，需要探索更多的monitor gpu的方法。有了这些方法，即使没有nvidia-smi也是可以工作的。

[How to Update Nvidia Drivers](https://www.wikihow.com/Update-Nvidia-Drivers)

可以使用GeForce Experience来更新驱动，但是，系统做的太差，注册经常都无法成功。

[libcublas.so.8.0 error with tensorflow](https://stackoverflow.com/questions/44865253/libcublas-so-8-0-error-with-tensorflow)

第二步，是CUDA 8.0安装

[How can I install CUDA on Ubuntu 16.04?](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04)
[深度学习开发环境配置：Ubuntu1 6.04+Nvidia GTX 1080+CUDA 8.0](https://zhuanlan.zhihu.com/p/22635699)

在linux中，如果之前没有安装过CUDA（ls /usr/local查看），可以用apt安装。
    
    sudo apt-get install cuda-8-0
    
Note: I strongly suggest not to sue it, as it changes the paths and makes the installation of other tools more difficult.
    
如果之前装过其他版本的CUDA，可能不能成功安装，这时候需要先下载cuda-8-0的安装文件，然后手动安装。

在[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)中找到[CUDA Toolkit 8.0 - Feb 2017](https://developer.nvidia.com/cuda-80-ga2-download-archive)，选择操作系统等，然后下载。

在windows中，下载后运行exe文件，如果网络条件还可以，就选择网络版的exe，也可以选择local。双击安装时，注意关闭其他不必要的程序，安装时很耗资源，否则有可能造成死机。安装时可能需要visual studio，作为c的编译器。安装好以后，在sample文件夹有一些可以验证cuda安装成功的程序，目前invida只提供c的源文件，要生成可执行的验证文件，需要自己用visual studio编译。

先装Base Installer，然后Patch 2。CUDA自带驱动很旧，记得取消勾选，只装CUDA。(其实也可以用旧的驱动，新的驱动当然可以使用，但旧的驱动用在对应的cuda上也没问题）

There is only one requirement, that one needs to satisfy in order to install multiple CUDA on the same machine. You need to have latest Nvidia driver that is required by the highest CUDA that you’re going to install. Usually it is a good idea to install precise driver that was used during the build of CUDA.

所以，不一定最新的驱动就是最好的，有可能和相应的cuda最为匹配的驱动是最好的。

在linux中，可以用命令行下载，wget或者curl。下载一般选择runfile(local)，可能很大. 用runfile的话，可以自己控制一些安装选择。用.deb的话，不能自主控制

    sudo chmod +x cuda_8.0.61_375.26_linux.run
    sudo sh cuda_8.0.61_375.26_linux.run --tmpdir=/tmp --override
    
override是因为电脑上的gcc版本相对安装文件可能偏高，使用override可以忽略这一点。如果不加，会碰到一个错误，Installation Failed. Using unsupported Compiler. ，这是因为 Ubuntu 16.04 默认的 GCC 5.4 对于 CUDA 8.x来说过于新了，CUDA 安装脚本还不能识别新版本的 GCC。

在linux装好cuda之后，有可能需要重启电脑。

[不用runfile而用deb安装cuda 8.0](https://blog.csdn.net/xingce_cs/article/details/74079783)
    
第三步，是cuDNN 5.1安装

[How can I install CuDNN on Ubuntu 16.04?](https://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04)

cuDNN下载需要先注册，对于ubuntu等一般选择linux版本下载，比如cudnn-8.0-linux-x64-v5.1.tgz。

[cuDNN最新版本](https://developer.nvidia.com/cudnn)
[cuDNN最新版本](https://developer.nvidia.com/rdp/cudnn-download)
[cuDNN历史版本](https://developer.nvidia.com/rdp/cudnn-archive)

因为下载需要用户名和密码，所以一般的wget或curl命令没法下载，需要authenication，如果要用命令行，需要python中requests之类的工具，来模拟发出带有用户名和密码的请求信息。所以，对于远程的机器，一般是先在本地下载好，然后用scp传到远程机器上。

复制文件:

    $scp local_file remote_username@remote_ip:remote_folder
    $scp local_file remote_username@remote_ip:remote_file
    
复制目录：

    $scp -r local_folder remote_username@remote_ip:remote_folder
    
[scp 跨机远程拷贝](http://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/scp.html)

    tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
    sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64
    
windows安装    
Cudnn解压后将bin,include,lib三个文件夹里面的内容覆盖至Cuda安装目录下，默认路径为C:\Program Files\NVIDIA GPUComputing Toolkit\CUDA\v8.0（记住不是替换，是把Cudnn文件里的.dll文件添加到Cuda里面）

第四步，安装tensorflow-gpu

可以先装conda的社区包凑合着用

    conda install tensorflow-gpu=1.0
    
注：用这种方式安装会自动装上conda的cuda和cudnn，系统不用另外安装，即便装了调用的也是conda的社区版本。conda的win64通道已经删掉1.0。且win仍需在系统装CUDA和cuDNN，否则会缺DLL。

注：不推荐用conda安装，装的是社区包，官方并不支持，由此产生的问题需要自行解决。

最常用的tensorflow-gpu安装是使用pip，比如

    pip install tensorflow-gpu==1.0 -i https://pypi.mirrors.ustc.edu.cn/simple

对于windows，python3.6中的tensorflow-gpu是从1.2版本开始的，要使用1.0或者1.1，只能用python3.5

安装好tensorflow-gpu之后的验证方法：

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    
安装完成后的测试代码：

    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    
测试是否用上GPU

    import tensorflow as tf
    print(tf.test.gpu_device_name())
    
正常会输出/gpu:0

[Tensorflow: How do you monitor GPU performance during model training in real-time?](https://stackoverflow.com/questions/45544603/tensorflow-how-do-you-monitor-gpu-performance-during-model-training-in-real-tim)

[Is there a way to verify Tensorflow Serving is using GPUs on a GPU instance](https://github.com/tensorflow/serving/issues/345)

在python中，可以设置logging系统的输出强度，是info就输出，是warning就输出，还是error才输出。而且，在一个脚本中，可以有不同的logging系统，分别设置不同的输出强度，这样的结果就是，有些logger的info就输出了，有些logger的error才输出。

在tensorflow的训练中，有时候训练特别慢，就会希望输出尽可能多的信息，降低各个logger的输出设置，有时候训练特别快，就会希望少输出一些信息，提高各个logger的输出设置。

在tensorflow的logging系统打出的信息中，I代表info，W代表warning，E代表error.

tensorflow的内存（显存）溢出或者训练特别慢的话，比较明显的是三种处理办法，一是降低batch size（epochs其实没有关系，反正也是放在循环中），二是简化模型，减少模型参数，比如减少网络层数，减少网络unit个数，减少输入长度等，可能需要对模型、数据进行可视化，三是购置更加先进、更大内存与显存的机器。

在tensorflow release版本（也就是pip安装的版本） 1.0中有一个bug（在windows中可见），logging系统会打印Error：OpKernel ('op: "BestSplits" device_type: "CPU"') for unknown op: BestSplits等，不影响tensorflow的运行，但确实是bug，在tensorflow 1.1中修正了，另外，在tenforflow 1.0中，logging.info有很多，很多info信息都会输出，在tensorflow 1.1中，相比1.0输出较少的info信息。所以，能用tensorflow 1.1的话，就不要用tensorflow 1.0, 如果必须用tensorflow 1.0，用linux版本而不要用windows版本。

[TensorFlow version 1.0.0-rc2 on Windows](https://github.com/tensorflow/tensorflow/issues/7500)

tensorflow 1.1中增加了一个额外的warning, The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations等。这表明TensorFlow release版没有对特定硬件进行编译优化，如果要优化TensorFlow的运行效率，需要自己编译。

在使用tensorflow时，可以通过环境变量TF_CPP_MIN_LOG_LEVEL来设置tensorflow中logging的输出强度。TF_CPP_MIN_LOG_LEVEL is a TensorFlow environment variable responsible for the logs, to silence INFO logs set it to 1, to filter out WARNING 2 and to additionally silence ERROR logs (not recommended) set it to 3。默认值是0（?），info也会输出，如果设成1，info不再输出，如果设成2，warning不再输出，如果设成3，error也不再输出。一般情况下，是不推荐设成3的，但因为tensorflow的开发非常快，error在所难免，有些error也并不影响脚本的正常运行，所以，在某些情况下，不想被tensorflow输出的信息打扰时，可以把TF_CPP_MIN_LOG_LEVEL设成3，百度的Dureader就是这样做的。但是，这样的话，控制gpu的某些信息就没法在tensorflow运行时看到，如果想看到tensorflow-gpu的运行，建议设成1，甚至设成0.

[The TensorFlow library wasn't compiled to use SSE instructions](https://github.com/tensorflow/tensorflow/issues/7778)

[编译优化TensorFlow](http://blog.rickdyang.me/2017-05/tensorflow-build/)

在windows中，python 3.5的价值就在于可以用tensorflow 1.1（和1.0一样，还是Cuda 8.0+cuDNN 5.1），而python 3.6则是tensorflow 1.2起。

第五步，create symlink

[Symbolic link](https://en.wikipedia.org/wiki/Symbolic_link)

[How to re-install CUDA 7.5](https://devtalk.nvidia.com/default/topic/916883/cuda-setup-and-installation/how-to-re-install-cuda-7-5/)

If you have all the items in /usr/local/cuda-7.5, then you can manually create your own symlink:

ln -s -T /usr/local/cuda-7.5 /usr/local/cuda

ln is a linux command. If you need help with that, try "man ln"

补充1：在同一台电脑上安装多个cuda版本

ubuntu install different cuda versions at the same time （It can be googled）

[Multiple Version of CUDA Libraries On The Same Machine](https://blog.kovalevskyi.com/multiple-version-of-cuda-libraries-on-the-same-machine-b9502d50ae77)

[Install CUDA to different directory in 16.04](https://askubuntu.com/questions/942691/install-cuda-to-different-directory-in-16-04)

[https://stackoverflow.com/questions/41330798/install-multiple-versions-of-cuda-and-cudnn](https://stackoverflow.com/questions/41330798/install-multiple-versions-of-cuda-and-cudnn)

补充2：使用docker安装，控制gpu

使用docker控制gpu的性能究竟如何，目前还没有测试过

[NVIDIA Docker: GPU Server Application Deployment Made Easy](https://devblogs.nvidia.com/nvidia-docker-gpu-server-application-deployment-made-easy/)

[Build and run Docker containers leveraging NVIDIA GPUs](https://github.com/nvidia/nvidia-docker)

[Docker for Deep Learning](http://www.born2data.com/2017/deeplearning_install-part4.html)

[Using GPU from a docker container?](https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container)

补充3：对gpu的要求

如果需要本地运算，建议使用带nvidia独显的机器，具体支持列表在下文tensorflow部分。

若选择用GPU运算，则对CPU的要求不高（但对内存有要求），GPU显存建议至少4G。

最低配置：

奔腾G4560

8G RAM

nVIDIA GTX 1050TI （升1060要选6G版）

推荐配置：

i5 6500

16G RAM

nVIDIA GTX 1070

另：如果需要长期跑项目且对虚拟化无需求的话，性价比最高的卡是1080Ti，能耗比最高是1080。

百度的Dureader项目，推荐10GB以上显存，最好12GB以上。内存最好在50GB以上。可以适当放小batch值，段落长度适当截断，对代码进行适当修改，一次少载入一些内存。

## Tensorflow使用

tf.train.Optimizer.minimize()是简单的合并了compute_gradients()与apply_gradients()函数返回为一个优化更新后的var_list. 在实际使用过程中，似乎在compute_gradients()之后是有gradient clipping过程的，因为只用compute_gradients()与apply_gradients()是会报错的，但使用minimize()则正常运行。

不过，根据tensorflow的官方文档，minimize()是不包含gradient clipping的，[需要自己实现](https://www.tensorflow.org/versions/master/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them)。

tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。

通过gradient clipping，可以在一定程度上处理[gradient exploding](https://stats.stackexchange.com/questions/320919/why-can-rnns-with-lstm-units-also-suffer-from-exploding-gradients)的问题，因为可以把梯度的值压缩在min和max之间。

gradient vanishing的问题，首先是通过选取非None的gradient来解决一些问题，比如tensor与variable没有关系（导数为0），tensor与variable有不可微的关系等，但是，对于很小的grandient，没有好的办法，只能接受。

    gradients = self.optimizer.compute_gradients(self.loss, self.all_params)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for
    grad, var in gradients if grad is not None]
    
除了tensor与variable没有关系、tensor与variable有不可微的关系这两种情况外，如果网络不是单输入单输出（例如ac网络中有两个输出），那么compute_gradients可能会返回（None，v），即部分变量没有对应的梯度，在下一步的时候NoneType会导致错误。因此，需要将有梯度的变量提取出来，记为grads_vars。之后，对grads_vars再一次计算梯度，得到了gradient。最后, 生成一个placeholder用于存储梯度，并调用apply_gradients更新网络。

tf.train.Optimizer.compute_gradients(loss, var_list=None): 对var_list中的变量计算loss的梯度, 返回一个以元组(gradient, variable)组成的列表

[optimizer.compute_gradients](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py#L281) wraps [tf.gradients()](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/gradients_impl.py#L320). It does additional asserts

上面列表中的gradient是tensor，reuse variable，传入placeholder的值，就可以用tf.Session().run()算出当步（对应着当步的variable）的各个gradient的值，从而拿到梯度下降的方向和大小。

在一个function，或者instance，或者class中用到的variable，如果这个同名function在其他地方再次用到，需要用variable_scope和reuse, 但是在训练的过程中，在for循环中用到的时候，是不需要reuse的，一直用之前的variable，如果想要手动改变variable, 就这样来做：

[How to reset the Tensorflow Adam Optimizer internal state](https://stackoverflow.com/questions/49010772/tensorflow-reset-adam-optimizer-internal-state-every-n-minibatches):

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam').minimize(loss)
    adam_vars = [var for var in tf.all_variables() if 'Adam' in var.name]
    sess.run(tf.variables_initializer(adam_vars))
    
从tensorflow 1.1开始，os.environ["TF_CPP_MIN_LOG_LEVEL"]可以控制tensorflow内部的logging系统的输出强度，0输出info, 1输出warning, 2输出error, 3什么都不输出。tensorflow 1.0似乎并没有这个控制变量，会输出所有的info。

在tensorflow中，log_device_placement=True可以控制tensorflow内部的logging系统把每一条tf.Session().run()的所用的gpu信息输出到logging.info()中。

    sess_config = tf.ConfigProto(log_device_placement=True)
    sess_config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=sess_config)
    
多进程使用GPU会导致OUT_OF_MEMORY_ERROR，这是由于tf默认会给任一进程分配所有能分配的显存，这样除了第一个进程其他进程都无显存可用。解决办法有两个，一是在运行命令前添加 CUDA_VISIBLE_DEVICES=9999（或任一大于你的显卡数的数字）禁用显卡，推荐对ps进程使用。二是在server配置里添加gpu_options=tf.GPUOptions(allow_growth=True)（或gpu_fraction）使得tf不会一次将显存分配完而是随着使用量逐渐增加.

[分布式Tensorflow的梯度累积与异步更新](https://zhuanlan.zhihu.com/p/23060519)
    
Device placement，可以指定使用哪一个gpu

    with tf.device("/device:GPU:1"):
      v = tf.get_variable("v", [1])
    
tensorflow本质上是一种函数式的编程语言，就拿z=ax+by来说，x, y是variable, a, b是placeholder, z是狭义的tensor. z本质上是一个函数，既有未知数，也有系数。广义的tensor不仅包括狭义的tensor，还包括variable, placeholder和constant.

在tf.Session().run()的时候，可以把a, b的值传进去，比如a=1, b=2，就可以计算z这个tensor。当然也可以不用placeholder，直接用tf.constant，比如z=x+2y，这样在tf.Session().run()的时候，就不用传placeholder的值了。

tf.Session().run()只能计算狭义的tensor，传入placeholder的值，variable的值要么在初始化时生成（根据调用的函数内部生成variable的规则，在函数中会给出所用到的variable的initializer, 比如随机生成），要么reuse之前的值。

我们自己所写的函数一般是生成tensor的函数，相当于闭包函数中的外部函数，在生成tensor的过程中会执行相关语句（也就是执行闭包函数的外部函数），但是在执行tf.Session().run()计算tensor的值的时候，不执行相关语句（执行的是闭包函数的内部函数，也就是tensor如何实现的函数，这是由tensorflow来实现的）。

[Initializing variables](https://www.tensorflow.org/programmers_guide/variables)

Before you can use a variable, it must be initialized. If you are programming in the low-level TensorFlow API (that is, you are explicitly creating your own graphs and sessions), you must explicitly initialize the variables. Most high-level frameworks such as tf.contrib.slim, tf.estimator.Estimator and Keras automatically initialize variables for you before training a model.

To initialize all trainable variables in one go, before training starts, call tf.global_variables_initializer(). This function returns a single operation responsible for initializing all variables in the tf.GraphKeys.GLOBAL_VARIABLES collection. Running this operation initializes all variables. For example:

    session.run(tf.global_variables_initializer())
    # Now all variables are initialized.
    
If you do need to initialize variables yourself, you can run the variable's initializer operation. For example:

    session.run(my_variable.initializer)
    
You can also ask which variables have still not been initialized. For example, the following code prints the names of all variables which have not yet been initialized:

    print(session.run(tf.report_uninitialized_variables()))
    
Note that by default tf.global_variables_initializer does not specify the order in which variables are initialized. Therefore, if the initial value of a variable depends on another variable's value, it's likely that you'll get an error. Any time you use the value of a variable in a context in which not all variables are initialized (say, if you use a variable's value while initializing another variable), it is best to use variable.initialized_value() instead of variable:

    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    w = tf.get_variable("w", initializer=v.initialized_value() + 1)
    
Using variables

To use the value of a tf.Variable in a TensorFlow graph, simply treat it like a normal tf.Tensor:

    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
               # Any time a variable is used in an expression it gets automatically
               # converted to a tf.Tensor representing its value.
    
To assign a value to a variable, use the methods assign, assign_add, and friends in the tf.Variable class. For example, here is how you can call these methods:

    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    assignment = v.assign_add(1)
    tf.global_variables_initializer().run()
    sess.run(assignment)  # or assignment.op.run(), or assignment.eval()

Most TensorFlow optimizers have specialized ops that efficiently update the values of variables according to some gradient descent-like algorithm. 

Because variables are mutable it's sometimes useful to know what version of a variable's value is being used at any point in time. To force a re-read of the value of a variable after something has happened, you can use tf.Variable.read_value. For example:

    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    assignment = v.assign_add(1)
    with tf.control_dependencies([assignment]):
      w = v.read_value()  # w is guaranteed to reflect v's value after the
                          # assign_add operation.

[Sharing variables](https://www.tensorflow.org/programmers_guide/variables)

TensorFlow supports two ways of sharing variables:

Explicitly passing tf.Variable objects around.

Implicitly wrapping tf.Variable objects within tf.variable_scope objects

While code which explicitly passes variables around is very clear, it is sometimes convenient to write TensorFlow functions that implicitly use variables in their implementations. Most of the functional layers from tf.layer use this approach, as well as all tf.metrics, and a few other library utilities.

Variable scopes allow you to control variable reuse when calling functions which implicitly create and use variables. They also allow you to name your variables in a hierarchical and understandable way.

For example, let's say we write a function to create a convolutional / relu layer:

    def conv_relu(input, kernel_shape, bias_shape):
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape,
            initializer=tf.random_normal_initializer())
        # Create variable named "biases".
        biases = tf.get_variable("biases", bias_shape,
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, weights,
            strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(conv + biases)

调用上面这个函数会生成一个tensor。input, kernel_shape, bias_shape这些参数可以在生成tensor时就给定，相当于给定tf.constant, 也可以在执行tensor时再传入，相当于传入tf.placeholder。variable的再次使用一般情况下是没有问题的，但是，内部有variable赋值的函数的多次使用是需要小心的。

This function uses short names weights and biases, which is good for clarity. In a real model, however, we want many such convolutional layers, and calling this function repeatedly would not work:

    input1 = tf.random_normal([1,10,10,32])
    input2 = tf.random_normal([1,20,20,32])
    x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
    x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.  
    # 给生成tensor的函数传入常量

Since the desired behavior is unclear (create new variables or reuse the existing ones?) TensorFlow will fail. Calling conv_relu in different scopes, however, clarifies that we want to create new variables:

    def my_image_filter(input_images):
        with tf.variable_scope("conv1"):
            # Variables created here will be named "conv1/weights", "conv1/biases".
            relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
        with tf.variable_scope("conv2"):
            # Variables created here will be named "conv2/weights", "conv2/biases".
            return conv_relu(relu1, [5, 5, 32, 32], [32])
            
If you do want the variables to be shared, you have two options. First, you can create a scope with the same name using reuse=True:

    with tf.variable_scope("model"):
      output1 = my_image_filter(input1)
    with tf.variable_scope("model", reuse=True):
      output2 = my_image_filter(input2)

You can also call scope.reuse_variables() to trigger a reuse:

    with tf.variable_scope("model") as scope:
      output1 = my_image_filter(input1)
      scope.reuse_variables()
      output2 = my_image_filter(input2)

Since depending on exact string names of scopes can feel dangerous, it's also possible to initialize a variable scope based on another one:

    with tf.variable_scope("model") as scope:
      output1 = my_image_filter(input1)
    with tf.variable_scope(scope, reuse=True):
      output2 = my_image_filter(input2)
      
在tensorflow的debug中，可以使用tf.Session().run()来得到tensor的值，但必须是tensor才行。比如：

    loss = self.sess.run(self.loss, feed_dict)
    print('loss in _train_epoch in rc_model.py:', loss)

    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for
                                    grad, var in gradients if grad is not None]
    results_g = self.sess.run(capped_gradients[1][0], feed_dict)
    print('results_g in _train_epoch in rc_model.py:', results_g)
    # capped_gradients[1][0]是个tensor，capped_gradients[1]是tensor与variable组成的tuple
    
某些时候，在运行tf.Session().run()时非常慢，为了看的更清楚，一般采取六个办法，一是把tensor分拆，只运行最小的tensor，比如tensor的list肯定可以分拆。二是减小batch size，因为batch size表示拟合计算cost或loss的时候，一次考虑多少个样本数或者数据点，考虑的样本数越少，计算loss时越快，计算loss的梯度时也越快；当然，同一个batch内的不同样本在处理的时候是map并行处理的，只有求和算loss的时候采用reduce加总处理；减小batch size，肯定是节省显存和内存的，对于由于显存几乎占满而导致的速度问题是有效的。三是缩短样本大小（往往也要重新运行程序，得到不一样的样本分配，得到不一样的batch），如果一个batch中有多个样本，样本长度是按最长的来算的，所以batch size较大的时候，样本长度的影响还是较大的。四是重新随机初始化variable（重新运行程序），因为有可能初始化的这一批variable所在的点在计算导数时非常困难，确实算的慢。五是初始化variable的时候不再使用随机值，而是使用之前拟合后的一些值，比如启用checkpoint，这些值往往会有更好的效果，计算起来会更快。六是使用更大显存、更大内存、更强GPU的机器。

[37 Reasons why your Neural Network is not working](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607)

在tensorflow的早期版本中，tensorflow自带tensorboard，但在pip freeze中却看不到tensorboard; 后来，从pip freeze中可以看到tensorflow-tensorboard；在tensorflow 1.6以及之后的版本中，从pip freeze可以看到tensorboard.

