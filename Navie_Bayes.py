#-*-coding:utf-8-*-
import numpy as np
import re
import random
#！！！！！！！！！！！！！‘朴素’并不是简单的意思，而是指样本的特征之间是相互独立的！！！！！！！！！！！！！！！！！！！！！
#！！！！！！！但是朴素贝叶斯却是生成方法，
#但朴素贝叶斯的缺点是：1，朴素贝叶斯算法有一个重要的使用前提：样本的特征属性之间是相互独立的，这使得朴素贝叶斯算法在满足这一条件的数据集上效果非常好，而不满足独立性条件的数据集上，效果欠佳。
#2，需要知道先验概率，且先验概率很多时候取决于假设，假设的模型可以有很多种，因此在某些时候会由于假设的先验模型的原因导致预测效果不佳。
# 3，由于通过先验和数据来决定后验的概率从而决定分类，所以分类决策存在一定的错误率。

#"""
#函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
#Parameters:
#    dataSet - 整理的样本数据集
#Returns:
#    vocabSet - 返回不重复的词条列表，也就是词汇表
#"""



def createVocablist(dataSet):
    vocabSet=set([])    ## 创建一个空的不重复列表
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)


# 函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
# Parameters:
#     vocabList - createVocabList返回的列表
#     inputSet - 切分的词条列表
# Returns:
#     returnVec - 文档向量,词集模型
def setOfWords2Vec(vocabList,inputset):
    returnVec=[0]*len(vocabList)
    for word in inputset:
        if word in vocabList:
            returnVec[vocabList.index[word]]=1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

# 函数说明:根据vocabList词汇表，构建词袋模型
# Parameters:
#     vocabList - createVocabList返回的列表
#     inputSet - 切分的词条列表
# Returns:
#     returnVec - 文档向量,词袋模型

def bagOfwords2VecMN(vocablist,inputset):
    def bagOfWords2VecMN(vocabList, inputSet):
        returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
        for word in inputSet:  # 遍历每个词条
            if word in vocabList:  # 如果词条存在于词汇表中，则计数加一
                returnVec[vocabList.index(word)] += 1
        return returnVec  # 返回词袋模型


# 函数说明:朴素贝叶斯分类器训练函数
# Parameters:
#     trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
#     trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
# Returns:
#     p0Vect - 正常邮件类的条件概率数组
#     p1Vect - 垃圾邮件类的条件概率数组
#     pAbusive - 文档属于垃圾邮件类的概率
def trainNB(trainMatrix,trainCategory): #这个是生成模型
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs) # 文档属于垃圾邮件类的概率,因为0是非垃圾邮件，1是垃圾邮件，sum之后就是总的垃圾邮件

    p0Num = np.ones(numWords)   #p0Num代表正类
    p1Num = np.ones(numWords)  # 创建numpy.ones数组,词条出现数初始化为1,拉普拉斯平滑
    p0Denom = 2.0   #正样本的总数
    p1Denom = 2.0  # 分母初始化为2 ,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])

        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = np.log(p1Num / p1Denom)    #NP除法可以用数组除以一个数，等价于每个元素除以这个数
    p0Vect = np.log(p0Num / p0Denom)   #取对数，防止下溢出
    return p0Vect, p1Vect, pAbusive

# 函数说明:朴素贝叶斯分类器分类函数
# Parameters:
# 	vec2Classify - 待分类的词条数组
# 	p0Vec - 正常邮件类的条件概率数组
# 	p1Vec - 垃圾邮件类的条件概率数组
# 	pClass1 - 文档属于垃圾邮件的概率
# Returns:
# 	0 - 属于正常邮件类
# 	1 - 属于垃圾邮件类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1  # 对应元素相乘
    #p0 = reduce(lambda x, y: x * y, vec2Classify * p0Vec) * (1.0 - pClass1)
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#函数说明:接收一个大字符串并将其解析为字符串列表
def textParse(bigString):  # 将字符串转换为字符列表
    listOfTokens = re.split(r'\W*', bigString)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 除了单个字母，例如大写的I，其它单词变成小写



#函数说明:测试朴素贝叶斯分类器，使用朴素贝叶斯进行交叉验证
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):  # 遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)  # 标记正常邮件，0表示正常文件
    vocabList = createVocablist(docList)  # 创建词汇表，不重复

    trainingSet = list(range(50))
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    trainMat = []
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB(np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount=0
    for docIndex in testSet:  # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", docList[docIndex])
        print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__=='__main__':
    spamTest()
#https://www.cnblogs.com/pinard/p/6069267.html
#朴素贝叶斯也可以处理多分类问题,朴素贝叶斯是天然的可以处理多分类问题，就是计算概率的时候考虑多少个ck的问题

#在sklearn模块中，一共有三个朴素贝叶斯分类方法，分别是GaussianNB, MultinomialNB和BernouliNB，
# ！！！！GaussianNB是先验为高斯分布的朴素贝叶斯，适用于样本特征的分布大部分是连续值的情况；
# ！！！MultinomialNB是先验为多项式分布的朴素贝叶斯，适用于样本特征的分布大部分是多元离散值的情况；
# ！！！！BernouliNB是先验为伯努利分布的朴素贝叶斯，适用于样本特征是二元离散值或者很稀疏的多元离散值的情况。
# 下面我分别用这三个分类方法来解决本项目的分类问题。


#数据形式：
# 8.73,0.31,2
# 4.71,-0.42,3
# 4.58,6.18,1
# 9.38,2.18,2
# 4.78,5.28,1
# 1.22,2.25,0
# 9.22,1.14,2
# 5.61,-0.34,3
# 7.8,0.51,2
# 1.98,1.69,0
# 7.51,1.76,2
# 0.95,2.09,0
# 3.43,0.24,3
# 4.74,4.7,1

#前面两个是特征，后面一个是label
dataset_X,dataset_y=df.iloc[:,:-1],df.iloc[:,-1] # 拆分为X和Y
dataset_X=dataset_X.values
dataset_y=dataset_y.values
#  print(dataset_X.shape) # (400, 2)
# print(dataset_y.shape) # (400,)
#dataset_X：二维
#[[8.73,0.31]
# [4.71,-0.42]
# [4.58,6.18]
# [9.38,2.18]
# [4.78,5.28]
# [1.22,2.25]]
#
#dataset_y 一维
#[1,2,0,3,...0,2]
from sklearn.naive_bayes import GaussianNB

gaussianNB = GaussianNB()
gaussianNB.fit(dataset_X, dataset_y)



from sklearn.naive_bayes import MultinomialNB
multinomialNB=MultinomialNB()
multinomialNB.fit(dataset_X,dataset_y)


from sklearn.naive_bayes import BernoulliNB
bernoulliNB=BernoulliNB()
bernoulliNB.fit(dataset_X,dataset_y)


