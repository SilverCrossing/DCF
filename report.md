# 人工智能开源软件开发与管理报告文档
## 3论文总结
### 遇到的问题
论文的研究认为现有方法存在两个局限性：  
（1）损失可能和噪声并没有高度相关性，优化过程存在不稳定性，因此样本损失可能会急剧波动从而可能会导致错误丢弃，以及干净交互样本可能存在困难性，难样本通常也会表现出高损失，但是对性能提升有很大帮助，丢弃它们反而可能会降低性能。  
（2）简单丢弃样本可能会加剧数据的稀疏性，造成样本浪费，还可能导致训练空间和干净的理想空间不一致。  
### 创新点  
针对问题（1），论文认为损失和噪声相关性低的原因是模型预测的观察范围有限且未考虑难样本，论文的解决方法是通过扩展观察区间、聚合多轮训练迭代的损失值来稳定预测，并识别和保留难样本以提升性能。  
针对问题（2），论文认为即使是噪声样本也存在对应的正确标签，不应简单丢弃，因此他们希望对高度确定为噪声的样本进行重标记，将其重新引入训练过程。  
### 方法流程图

## 4论文公式和程序文件代码名（行数对照表）
| 论文公式 | 代码位置 |
|----------|---------|
| 第二章第一公式 | 代码中未提及 |
| 第二章第二公式 | loss.py的第19行代码 |
| 公式（1）损失均值 | 代码中未提及 |
| 公式（2）带阻尼函数的损失均值 | loss.py的第19到24行代码 |
| 公式（3）置信区间界定公式 | loss.py的第28到30行代码 |
| 公式（4）损失下界 | loss.py的32到33行代码 |
| 公式（5）渐进式重标记比例ri | main.py的第98行代码 |
| 公式（6）基于ri的损失阈值筛选 | loss.py的第44行代码 |
| 公式（7）标签翻转 | 代码中未提及 |
| 附录中公式（8）到（18） | 代码中未提及 |
  
第二章第一公式，去噪推荐模型训练目标--作者仅仅进行论述，将去噪模型的训练抽象为一个公式，代码中未显式给出  
<img width="125" height="35" alt="image" src="https://github.com/user-attachments/assets/01e31ca7-d40c-4c73-9665-90bc2a721761" />  
  
第二章第二公式，BCE损失函数--loss.py的第19行代码  
<img width="311" height="41" alt="image" src="https://github.com/user-attachments/assets/c3aab758-b385-4f9b-97a1-cd6bdac094e3" />  
```
# 计算每个样本的二元交叉熵（带logits输入），不做reduce得到每个样本的损失，此处为论文所使用的损失函数，对应论文第二章BCE的公式``  
# y对应模型预测，t对应真实标签，reduce=False表示不进行求和或平均，返回每个样本的损失值``  
loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)``  
```  
公式（1）损失均值--作者发现在训练过程中模型有可能会遇到极端损失值（尽管这种可能性很小）从而对均值计算造成负面影响，因此需要采取预防措施而非直接使用公式（1），代码实际使用的为公式（2），公式（1）代码中未显式给出  
<img width="313" height="37" alt="image" src="https://github.com/user-attachments/assets/a28e79cb-fb76-4aa8-b093-4c6455b9a605" />  
  
公式（2）带阻尼函数的损失均值--loss.py的第19到24行代码  
<img width="320" height="41" alt="image" src="https://github.com/user-attachments/assets/0da276d2-e1f7-4573-8f75-b25445343f1d" />  
```
# 只对正样本（t==1）关注损失，把负样本的损失置零（loss * t）
loss_mul = loss * t
# 对正样本损失做平滑处理
loss_mul = soft_process(loss_mul)  # soft process is non-decreasing damping function in the paper
# 用之前的before_loss与当前loss_mul做平均，计算历史平均损失，对应论文中第三章的公式（2）
loss_mean = (before_loss * s + loss_mul) / (s + 1)   # computing mean loss in Eq.2
```

公式（3）置信区间界定--loss.py的第28到30行代码  
<img width="315" height="46" alt="image" src="https://github.com/user-attachments/assets/44a1ffb7-00d2-45e0-8f41-bafe65e4781f" />  
```
# 计算置信界（confidence bound），对应论文中第三章的公式（3）
confidence_bound = co_lambda * (s + (co_lambda * torch.log(2 * s)) / (s * s)) / ((sn + 1) - co_lambda)
confidence_bound = confidence_bound.squeeze()
```

公式（4）损失下界--loss.py的第32到33行代码  
<img width="302" height="43" alt="image" src="https://github.com/user-attachments/assets/37a9cfec-cef7-4ca6-a691-453a07c8f0f6" />  
```
# 只保留大于置信界的部分，作为高损失的判定依据，对应论文中第三章的公式（4）
loss_mul = F.relu(loss_mean.float() - confidence_bound.cuda().float())  # loss low bound in Eq.4
```

公式（5）渐进式重标记比例ri--main.py的第98行代码，但不知为何并未实现动态改变，使用的是固定值  
<img width="304" height="31" alt="image" src="https://github.com/user-attachments/assets/3b1527ca-02db-4ecd-a249-12c9f31d7bd2" />  
```
parser.add_argument("--relabel_ratio",   # 论文中渐进式重标记比例ri，设定为固定值
	type=float,
	default=0.05,
	help="relabel ratio")
```

公式（6）基于ri的损失阈值筛选--loss.py的第44行代码  
<img width="311" height="38" alt="image" src="https://github.com/user-attachments/assets/c2d64062-872c-4d64-b48f-93e4cf041795" />  
```
# saved_ind_sorted：保留的索引（记住的前 num_remember 个）
saved_ind_sorted = ind_sorted[:num_remember]   # 对应论文中的公式（6）
```

公式（7）标签翻转--作者在论文中提到，同时也通过公式（5）和（6）的代码执行锁定了需要翻转标签的数据，但不知出于什么原因，代码中未显式给出此公式的实现  
<img width="312" height="26" alt="image" src="https://github.com/user-attachments/assets/0a7cef44-62be-468d-a84f-14a5d6da5fc0" />  

附录中的公式（8）到（18）--对文中的定理1和公式（4）进行证明所使用的，代码中不负责实现  


## 5安装说明
原始GitHub虽然未提供requirements.txt，但是所使用的包都在README中提及。我所使用的python版本为3.8.20    
论文所使用的依赖为numpy==1.19.5、scikit-learn==0.24.2、torch==1.8.1、CUDA==10.2
```
# 创建并激活虚拟环境
conda create dcf_test
conda activate dcf_test

# 安装pytorch，这里选择适配实验室显卡的cuda和pytorch版本，而非完全按照作者的配置
conda install pytorch torchvision torchaudio pytorch-cuda=10.2 -c pytorch -c nvidia
  
# 安装numpy和scikit-learn，numpy选择适配pytorch版本的，scikit-learn版本和作者给出的一致
pip install numpy==1.19.5
pip install scikit-learn==0.24.2
  
# 运行，按照默认参数进行10epochs
/data1/sc/.conda/envs/dcf_test/bin/python /data1/sc/DCF/DCF-main/DCF-main/main.py --epochs 10
```  
数据集作者有在GitHub中给出，所使用的数据集是Adressa、Yelp和MovieLens，可以在另一份[Github](https://github.com/WenjieWWJ/DenoisingRec)中找到，这里面的data就包含adressa和Yelp的数据，而[MovieLens](https://drive.google.com/file/d/18XDcN4Pl_NpZBp88WGhwlVQfmeKsT4WF/view)则在google盘下载  
作者在论文中对数据集进行了处理，Adressa中仅保留停留时间至少为10秒的交互；MovieLens中仅保留评分为5分的交互作为测试集；Yelp中仅保留评分高于3分的交互作为干净测试集  


## 6运行/测试结果截图
运行过程根据实验室的情况，对数据输出进行了部分修改，但不影响整体运行逻辑，仅仅是为了观察结果。
<img width="1504" height="219" alt="屏幕截图 2025-11-26 172512" src="https://github.com/user-attachments/assets/82610a2a-5e00-4c07-82fe-006f5ce02b7d" />  
运行的时候加载参数  
<img width="752" height="122" alt="image" src="https://github.com/user-attachments/assets/c7866b8d-6c24-4d55-838d-4eaf0ff693ee" />  
运行结果，和作者的NeuMF模型上的结果相差不大。recall的四个数字分别表示recall@5、recall@10、recall@20、recall@50，recall@K表示前K个结果中有多少个项目是用户感兴趣的，K越小则对模型要求越高。NDCG（归一化累积折损增益）的四个数字同样是分为@5、@10、@20和@50，NDCG@K不仅考虑用户是否对前K个项目感兴趣，还要考虑感兴趣的项目是否排的靠前，对模型要求更高    


