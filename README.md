# tf-gpu

tensorflow-gpu 1.0 要求 CUDA 8.0 和 cuDNN 5.1 安装

很好的一些参考文章：

ubuntu:
[CUDA 8.0 和 cuDNN 5.1 安装](https://zhuanlan.zhihu.com/p/27890924)
[Setup: Ubuntu 16.04 + NVIDIA + CUDA 8.0 + cuDNN 5.1](http://ywpkwon.github.io/post/ubuntu-setting/)

windows:
[Tensorflow+cuda+cudnn+window+Python之window下安装TensorFlow](https://blog.csdn.net/flying_sfeng/article/details/58057400)

第一步，是核实机器有gpu并给gpu装驱动

在windows中，在设备管理器中可以看到显卡配置；显卡驱动一般已经在装系统时装好，不用额外操心

在linux中，如果有图形界面，可以在系统设置 --> 软件与更新 --> 附加驱动中安装驱动或者更新驱动，比如，nvidia-384

如果没有图形界面，可以尝试

    sudo apt-get install nvidia-384 #can type nvidia then hit "tab" to view all available options

终端输入nvidia-smi，可以看到所装显卡驱动的信息

nvidia的显卡驱动是兼容之前版本的，也就是说，旧的驱动不支持新的显卡，但新的驱动是支持旧的显卡的。

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

在linux中，可以用命令行下载，wget或者curl。下载一般选择runfile(local)，可能很大. 用runfile的话，可以自己控制一些安装选择。用.deb的话，不能自主控制

    sudo chmod +x cuda_8.0.61_375.26_linux.run
    sudo sh cuda_8.0.61_375.26_linux.run --tmpdir=/tmp --override
    
override是因为电脑上的gcc版本相对安装文件可能偏高，使用override可以忽略这一点。如果不加，会碰到一个错误，Installation Failed. Using unsupported Compiler. ，这是因为 Ubuntu 16.04 默认的 GCC 5.4 对于 CUDA 8.x来说过于新了，CUDA 安装脚本还不能识别新版本的 GCC。

在linux装好cuda之后，有可能需要重启电脑。
    
第三步，是cuDNN 5.1安装

[How can I install CuDNN on Ubuntu 16.04?](https://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04)

cuDNN下载需要先注册，对于ubuntu等一般选择linux版本下载，比如cudnn-8.0-linux-x64-v5.1.tgz。

[cuDNN最新版本](https://developer.nvidia.com/cudnn)
[cuDNN最新版本](https://developer.nvidia.com/rdp/cudnn-download)
[cuDNN历史版本](https://developer.nvidia.com/rdp/cudnn-archive)

因为下载需要用户名和密码，所以一般的wget或curl命令没法下载，需要authenication，如果要用命令行，需要python中requests之类的工具，来模拟发出带有用户名和密码的请求信息。

    tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
    sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64
    
windows安装    
Cudnn解压后将bin,include,lib三个文件夹里面的内容覆盖至Cuda安装目录下，默认路径为C:\Program Files\NVIDIA GPUComputing Toolkit\CUDA\v8.0（记住不是替换，是把Cudnn文件里的.dll文件添加到Cuda里面）

第四步，安装tensorflow-gpu

对于windows，python3.6中的tensorflow-gpu是从1.2版本开始的，要使用1.0或者1.1，只能用python3.5

安装好tensorflow-gpu之后的验证方法：

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

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

