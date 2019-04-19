#-*-coding:utf-8-*-
#https://blog.csdn.net/qq_22238533/article/details/79185969
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
#GBDT中的树都是回归树（cart回归树)，不是分类树，
#用损失函数的负梯度来拟合本轮损伤的近似值，得到cart回归树
# gbdt=GradientBoostingRegressor(
#     loss='ls',
#     learning_rate=0.1,
#     n_estimators=100,
#     subsample=1,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     max_depth=3,
#     init=None,
#     random_state=None,
#     max_features=None,
#     alpha=0.9,
#     verbose=0,
#     max_leaf_nodes=None,
#     warm_start=False
#
#
# )
#实操：https://zhuanlan.zhihu.com/p/27428738
#数据形式：
#1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300,g
#1,0,1,-0.18829,0.93035,-0.36156,-0.10868,-0.93597,1,-0.04549,0.50874,-0.67743,0.34432,-0.69707,-0.51685,-0.97515,0.05499,-0.62237,0.33109,-1,-0.13151,-0.45300,-0.18056,-0.35734,-0.20332,-0.26569,-0.20468,-0.18401,-0.19040,-0.11593,-0.16626,-0.06288,-0.13738,-0.02447,b

# x_columns = [x for x in df.columns if x not in 'gbflag']
# X = df[x_columns]#二维矩阵
# y = df['gbflag']  #一维
# #将数据集分成训练集，测试集
# X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

# model = GradientBoostingClassifier()
# model.fit(X_train,y_train)
# pred = gbdt.predict(X_test)

#！！！！！！！！！！！！交叉验证法：
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import KFold
# kf = KFold(n_splits = 10)
# scores = []
# X = df[x_columns]#数据，#二维矩阵
# y = df['gbflag']   #一维
# for train,test in kf.split(X):
#     train_X,test_X,train_y,test_y = X.iloc[train],X.iloc[test],y.iloc[train],y.iloc[test]
#     gbdt =  GradientBoostingClassifier(max_depth=4,max_features=9,n_estimators=100)
#     gbdt.fit(train_X,train_y)
#     prediced = gbdt.predict(test_X)
#     print(accuracy_score(test_y,prediced))
#     scores.append(accuracy_score(test_y,prediced))
# ##交叉验证后的平均得分
# np.mean(scores)

#！！！！！！！！！！！！！！网格法特征选择  https://blog.csdn.net/u012969412/article/details/72973055
# from sklearn.model_selection import GridSearchCV
#
# model = GradientBoostingClassifier()
# parameter_grid = {'max_depth':[2,3,4,5],
#                   'max_features':[1,3,5,7,9],
#                   'n_estimators':[10,30,50,70,90,100]}
# grid_search = GridSearchCV(model,param_grid = parameter_grid,cv =10,
#                            scoring = 'accuracy')
# grid_search.fit(X,y)
# #输出最高得分
# print grid_search.best_score_
# #输出最佳参数
# print grid_search.best_params_


#sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
#计算roc曲线内的面积


#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！gbdt+LR:首先说明既然gbdt已经训练好了，为啥还需要用LR，因为我们需要的shi一个概率值，如CTR预估，但是GBDT没法得到概率结果，只能用LR
# https://www.cnblogs.com/wkang/p/9657032.html
#https://blog.csdn.net/levy_cui/article/details/77168709
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,label_binarize
from sklearn.metrics import accuracy_score   # 准确率
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score,auc
from matplotlib import pyplot as plt


digits = datasets.load_digits()  # 载入mnist数据集
X_train,x_test,Y_train,y_test = train_test_split(digits.data,digits.target,test_size = 0.2,random_state = 33)     # 测试集占30%

model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,random_state=10, subsample=0.6, max_depth=7,
                                  )
model.fit(X_train, Y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))

train_new_feature = model.apply(X_train)#这个序号是针对各个树，自己是自己的,这是个三维矩阵
print train_new_feature[0].shape,'*8'
train_new_feature = train_new_feature.reshape(-1, 50*10)#50是代表50棵树，10因为这个是10分类任务，gbdt会为每一个类别训练50课树，10个类，就是10×50棵树
#a = np.array([[[1, 2], [2, 3]], [[4, 5], [5, 6]], [[7, 8], [8, 9]]])
# a.reshape(-1,4)
#array([[1, 2, 2, 3],
       # [4, 5, 5, 6],
       # [7, 8, 8, 9]])
print train_new_feature
enc = OneHotEncoder()

enc.fit(train_new_feature)  #必须fit，OneHotEncoder在训练的时候会对数据进行统计和离散化：

#例子：
# enc = OneHotEncoder()
# enc.fit([[0, 0, 3],
#          [1, 1, 0],
#          [0, 2, 1],
#          [1, 0, 2]])
#
# ans = enc.transform([[0, 1, 3]]).toarray()    ## 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，
#[[1. 0. 0. 1. 0. 0. 0. 0. 1.]]
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(train_new_feature, Y_train, test_size=0.2,random_state=10)#因为GBDT是要对数据进行特征提取，所以逻辑回归的训练数据要是GBDT的训练数据


lm = LogisticRegression(penalty='l1')
print '7',enc.transform(X_train_lr).toarray(),'7'
lm.fit(enc.transform(X_train_lr).toarray(),y_train_lr)



#predict_proba(X) 返回一个数组，数组元素依次是 X 预测为各个类别的概率的概率值。
#predict()是出标签。
#tmp=lm.predict(enc.transform(X_test_lr).toarray())

y_pred_grd_lm = lm.predict_proba(np.array(enc.transform(X_test_lr).toarray()))
print '9',y_pred_grd_lm,'9'
#y_pred_grd_lm是样本数*标签数的矩阵。predict_proba返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率。此

                            #时每一行的和应该等于1。



# 根据预测结果输出  https://blog.csdn.net/llh_1178/article/details/81016543
#roc曲线是一个二分类器常用的工具,https://blog.csdn.net/u013385925/article/details/80385873
#roc_curve
# 参数：
# y_true:实际的样本标签值（这里只能用来处理二分类问题，即为{0，1}或者{true，false}，如果有多个标签，则需要使用pos_label 指定某个标签为正例，其他的为反例）
#
# y_score:目标分数，被分类器识别成正例的分数（常使用在method="decision_function"、method="proba_predict"）

#fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test_lr, y_pred_grd_lm,pos_label=1)
# roc_auc = auc(fpr_grd_lm, tpr_grd_lm)
# print 'predict', roc_auc
#

#https://www.cnblogs.com/caiyishuai/p/9435945.html,多分类
y_one_hot = label_binarize(y_test_lr, np.arange(10))  # 装换成类似二进制的编码
print y_one_hot,'ll'
print("AUC Score macro:", (roc_auc_score(y_one_hot, y_pred_grd_lm,average='macro')))
#方法1：每种类别下，都可以得到m个测试样本为该类别的概率（矩阵P中的列）。所以，根据概率矩阵P和标签矩阵L中对应的每一列，
# 可以计算出各个阈值下的假正例率（FPR）和真正例率（TPR），从而绘制出一条ROC曲线。这样总共可以绘制出n条ROC曲线。
# 最后对n条ROC曲线取平均，即可得到最终的ROC曲线。


print("AUC Score micro:", (roc_auc_score(y_one_hot, y_pred_grd_lm,average='micro')))
#方法2：首先，对于一个测试样本：1）标签只由0和1组成，1的位置表明了它的类别（可对应二分类问题中的‘’正’’），0就表示其他类别（‘’负‘’）；
# 2）要是分类器对该测试样本分类正确，则该样本标签中1对应的位置在概率矩阵P中的值是大于0对应的位置的概率值的。
# 基于这两点，将标签矩阵L和概率矩阵P分别按行展开，转置后形成两列，这就得到了一个二分类的结果。
# 所以，此方法经过计算后可以直接得到最终的ROC曲线。

fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_one_hot.ravel(), y_pred_grd_lm.ravel())
auc_num =auc(fpr_grd_lm, tpr_grd_lm)
print auc_num


# FPR就是横坐标,TPR就是纵坐标
plt.plot(fpr_grd_lm, tpr_grd_lm, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc_num)
plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
plt.show()