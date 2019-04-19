#-*-coding:utf-8-*-
#首先准备数据集
# 特征向量
#我们主要使用逻辑回归解决二分类的问题，那对于多分类的问题，也可以用逻辑回归来解决
#one vs rest
#将类型class1看作正样本，其他类型全部看作负样本，然后我们就可以得到样本标记类型为该类型的概率p1；
# 然后再将另外类型class2看作正样本，其他类型全部看作负样本，同理得到p2；

#one vs one.处理样本不均匀
import numpy as np
X =np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2],
             [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]]) # 自定义的数据集
# 标记
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) #　三个类别

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
#LogisticRegression和LogisticRegressionCV的主要区别是LogisticRegressionCV使用了交叉验证来选择正则化系数C。而LogisticRegression需要自己每次指定一个正则化系数。
#通过model.C_可以得到正则化系数C

classifier = LogisticRegressionCV(random_state=37,penalty='l1') # 先用默认的参数
classifier.fit(X, y) # 对国际回归分类器进行训练
print classifier.C_

# 类型权重参数：（考虑误分类代价敏感、分类类型不平衡的问题）
#
# class_weight : dictor ‘balanced’, default: None

# 逻辑回归分类器有两个最重要的参数：solver和C，其中参数solver用于设置求解系统方程的算法类型，
# 参数C表示对分类错误的惩罚值，故而C越大，表明该模型对分类错误的惩罚越大，即越不能接受分类发生错误。
#示例：
# for c in [1,5,20,50,100,200,500]:
#     classifier = LogisticRegression(C=c,random_state=37)
#     classifier.fit(X, y)
#     plot_classifier(classifier, X, y)
