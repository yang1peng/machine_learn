#-*-coding:utf-8-*-
#https://www.cnblogs.com/zhouxiaohui888/p/6008368.html
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   # 准确率
from xgboost import XGBClassifier

digits = datasets.load_digits()  # 载入mnist数据集
#这个是mnist手写题识别
#每个样本包括8*8像素的图像和一个[0, 9]整数的标签
#image就是图像，8×8
#digits.data是把图片拉成一行的结果，digits.data是二维矩阵
#这个是其中一个
# [  0.   0.   5.  13.   9.   1.   0.   0.   0.   0.  13.  15.  10.  15.   5.
#    0.   0.   3.  15.   2.   0.  11.   8.   0.   0.   4.  12.   0.   0.   8.
#    8.   0.   0.   5.   8.   0.   0.   9.   8.   0.   0.   4.  11.   0.   1.
#   12.   7.   0.   0.   2.  14.   5.  10.  12.   0.   0.   0.   0.   6.  13.
#   10.   0.   0.   0.]

#digits.target是一维矩阵，么个元素是0～9，是标签

print digits
x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size = 0.3,random_state = 33)     # 测试集占30%

model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
#n_estimators=160树的数目
#learning_rate每颗树占的比重，越小越能防止过拟合（单棵树对数据拟合太多，步子太大，容易扯蛋）
model.fit(x_train, y_train)

#完全使用默认值
# model = XGBClassifier()          # 载入模型（模型命名为model)
# model.fit(x_train,y_train)            # 训练模型（训练集）
y_pred = model.predict(x_test)        # 模型预测（测试集），y_pred为预测结果

### 性能度量

accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))

#！！！！！！！！！！！！！生成one-hot向量
from sklearn.preprocessing import OneHotEncoder
oneHot=OneHotEncoder()#声明一个编码器
oneHot.fit([[1],[2],[3],[4]])
print(oneHot.transform([[2],[3],[1],[4]]).toarray())
