# tf-gpu

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

若选择用GPU运算，则对CPU的要求不高，GPU显存建议至少4G。

最低配置：

奔腾G4560

8G RAM

nVIDIA GTX 1050TI （升1060要选6G版）

推荐配置：

i5 6500

16G RAM

nVIDIA GTX 1070

另：如果需要长期跑项目且对虚拟化无需求的话，性价比最高的卡是1080Ti，能耗比最高是1080。
