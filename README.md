# tf-gpu

tensorflow-gpu 1.0 要求 CUDA 8.0 和 cuDNN 5.1 安装

很好的一个参考文章：

[CUDA 8.0 和 cuDNN 5.1 安装](https://zhuanlan.zhihu.com/p/27890924)

第一步，是核实机器有gpu并给gpu装驱动

在windows中，在设备管理器中可以看到显卡配置；显卡驱动一般已经在装系统时装好，不用额外操心

在linux中，如果有图形界面，可以在系统设置 --> 软件与更新 --> 附加驱动中安装驱动或者更新驱动，比如，nvidia-384

如果没有图形界面，可以尝试

    sudo apt-get install nvidia-384 #can type nvidia then hit "tab" to view all available options

终端输入nvidia-smi，可以看到所装显卡驱动的信息

nvidia的显卡驱动是兼容之前版本的，也就是说，旧的驱动不支持新的显卡，但新的驱动是支持旧的显卡的。

第二步，是CUDA 8.0安装

在linux中，如果之前没有安装过CUDA（ls /usr/local查看），可以用apt安装。
    
    sudo apt-get install cuda-8-0
    
如果之前装过其他版本的CUDA，可能不能成功安装，这时候需要先下载cuda-8-0的安装文件，然后手动安装。

在[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)中找到[CUDA Toolkit 8.0 - Feb 2017](https://developer.nvidia.com/cuda-80-ga2-download-archive)，选择操作系统等，然后下载。

在windows中，下载后运行exe文件，如果网络条件还可以，就选择网络版的exe。

在linux中，可以用命令行下载，wget或者curl。

    sudo chmod +x cuda_8.0.61_375.26_linux.run
    sudo sh cuda_8.0.61_375.26_linux.run --tmpdir=/tmp --override
    
第三步，是cuDNN 5.1安装

cuDNN下载需要先注册，对于ubuntu等一般选择linux版本下载，比如cudnn-8.0-linux-x64-v5.1.tgz。

因为下载需要用户名和密码，所以一般的wget或curl命令没法下载，需要authenication，如果要用命令行，需要python中requests之类的工具，来模拟发出带有用户名和密码的请求信息。

    tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
    sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64

第四步，安装tensorflow-gpu

