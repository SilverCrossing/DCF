# 人工智能开源软件开发与管理报告文档
## 3论文总结
论文的研究认为现有方法存在两个局限性。（1）损失可能和噪声并没有高度相关性，优化过程存在不稳定性，因此样本损失可能会急剧波动从而可能会导致错误丢弃，以及干净交互样本可能存在困难性，难样本通常也会表现出高损失，但是对性能提升有很大帮助，丢弃它们反而可能会降低性能。  
（2）简单丢弃样本可能会加剧数据的稀疏性，造成样本浪费，还可能导致训练空间和干净的理想空间不一致。针对问题（1），论文认为损失和噪声相关性低的原因是模型预测的观察范围有限且未考虑难样本，论文的解决方法是通过扩展观察区间、聚合多轮训练迭代的损失值来稳定预测，并识别  
和保留难样本以提升性能。针对问题（2），论文认为即使是噪声样本也存在对应的正确标签，不应简单丢弃，因此他们希望对高度确定为噪声的样本进行重标记，将其重新引入训练过程  

## 4论文公式和程序文件代码名（行数对照表）
去噪推荐模型训练目标--代码中未显式给出  
BCE损失函数--loss.py  
公式（1）损失均值--代码中未显式给出  
公式（2）带阻尼函数的平均损失计算--loss.py  
公式（3）--loss.py  
公式（4）损失下界--loss.py  
公式（5）渐进式重标记比例ri--main.py，使用的是固定值  
公式（6）基于ri的损失阈值筛选--loss.py  
公式（7）标签翻转--代码中未显式给出  
附录公式（8）~（18）--对文中的定理1和公式（4）进行证明，代码中不负责实现  


## 5安装说明
原始GitHub虽然未提供requirements.txt，但是所使用的包都在README中提及。我所使用的python版本为3.8.20，其中部分包为了适配实验室所使用的显卡，进行了升级，未完全按照原始GitHub列出的版本进行配置  
数据集作者有在GitHub中给出，所使用的数据集是Adressa、Yelp和MovieLens，可以在另一份[Github](https://github.com/WenjieWWJ/DenoisingRec)中找到，这里面的data就包含adressa和Yelp的数据，而[MovieLens](https://drive.google.com/file/d/18XDcN4Pl_NpZBp88WGhwlVQfmeKsT4WF/view)则在google盘下载  
论文所使用的依赖为numpy==1.19.5、scikit-learn==0.24.2、torch==1.8.1、CUDA==10.2，但我所使用的实验室显卡不支持太低版本的依赖，强行使用低版本反而无法训练，因此我对部分所用依赖进行了升级，numpy==1.24.4、scikit-learn==0.24.2、torch==2.4.1、CUDA==12.4  

``# 创建并激活虚拟环境``  
``conda create dcf_test``  
``conda activate dcf_test``  

``# 安装pytorch``  
``conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia``  
  
``# 安装numpy和scikit-learn``
``pip install numpy==1.24.4``
``pip install scikit-learn==0.24.2``  


``# 运行``  
``/data1/sc/.conda/envs/dcf_test/bin/python /data1/sc/DCF/DCF-main/DCF-main/main.py --epochs 40``  

## 6运行/测试结果截图



