#-*-coding:utf-8-*-
import jieba
import os
import re
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from feature_extract import feature_selection
import codecs

def stopwordslist(file_name):
    stopwords = [line.strip().replace('\n','') for line in codecs.open(file_name,'r','utf-8').readlines()]
    return stopwords

def obtain_data(rootdir):
    datas={}
    def list_all_files(rootdir):
        files_list = []
        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(rootdir, list[i])
            if os.path.isdir(path):
                print '+++++++++++++++++++++++'
                print  path
                files_list.extend(list_all_files(path))
            if os.path.isfile(path):
                # files_list.append(path)
                print path, '***'
                label_tmp = path.split('/')[-2]
                idxs = re.findall('\d', label_tmp)  # 找出所有的数字，得到位置列表
                index = label_tmp.index(idxs[0])
                labs = label_tmp[:index]  # 得到切好的参数
                print labs, '00777'
                # 存在就返回这个列表，不存在就生成。这个要存文档内容

                with codecs.open(path, 'r', 'utf-8') as f:
                    file_data = f.readlines()
                    print len(file_data), '88888'
                    try:
                        if '【 正  文 】' in file_data[4]:
                            data = file_data[5:]
                            b = ''
                            for node in data:
                                if node.replace(' ', '') == None:
                                    continue
                                else:
                                    b += node.replace('\n', '')



                        elif '标  题:' in file_data[3]:
                            data = file_data[5:]
                            b = ''
                            for node in data:
                                if '来源:' in node:
                                    break
                                if node.replace(' ', '') == None:
                                    continue
                                else:
                                    b += node.replace('\n', '')


                        elif '新华社' in file_data[1]:
                            tmp = f.readlines()[1].index('电')
                            b = f.readlines()[1][tmp:].replace('\n', '')
                            data = f.readlines()[2:]
                            for node in data:
                                if node.replace(' ', '') == None:
                                    continue
                                else:
                                    b += node.replace('\n', '')


                        else:
                            data = file_data
                            b = ''
                            for node in data:
                                if node.replace(' ', '') == None:
                                    continue
                                else:
                                    b += node.replace('\n', '')
                    except:
                        data = file_data
                        b = ''
                        for node in data:
                            if node.replace(' ', '') == None:
                                continue
                            else:
                                b += node.replace('\n', '')
                    data_list = jieba.cut(b.strip().replace('\n', ''))  # 分词、去停用词
                    stopwords = stopwordslist('stop_word.txt')
                    tmp = []
                    for word in data_list:
                        if word not in stopwords:
                            tmp.append(word)
                    print labs, '44'
                    print tmp,'0000999'
                    datas[labs] = datas.get(labs, [])+[tmp]
                    print datas,'wowowo'
        return files_list
    list_all_files(rootdir)
    print datas, 'ljhg'
    return datas  # 返回一个字典，字典的键是类名，键值是一个二维列表，每行代表该类下一个文档，每一行是一个分好词并去了停用词的单词列表




def SVM_text_classify(data_x,data_y,num_class,test_x,test_y):
    #分类问题使用的svm
    # sklearn.svm.NuSVC()
    # sklearn.svm.LinearSVC()
    # sklearn.svm.SVC()
    #************************************************************************************88
    #SVC()是一种基于libsvm的支持向量机，由于其时间复杂度为O(n^2)，所以当样本数量超过两万时难以实现
    # sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
    #                 probability=False, tol=0.001, cache_size=200, class_weight=None,
    #                 verbose=False, max_iter=-1, decision_function_shape='ovr',
    #                 random_state=None)
    #C 表示错误项的惩罚系数C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低；
    # 相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。

    # kernel （str参数默认为‘rbf’）该参数用于选择模型所使用的核函数，算法中常用的核函数有：
    # -- linear：线性核函数
    # --  poly：多项式核函数
    # --rbf：径像核函数 / 高斯核
    # --sigmod：sigmod核函数
    # --precomputed：核矩阵，该矩阵表示自己事先计算好的，输入后算法内部将使用你提供的矩阵进行计算

    #degree （int型参数默认为3）该参数只对'kernel=poly'(多项式核函数)有用，是指多项式核函数的阶数n，如果给的核函数参数是其他核函数，则会自动忽略该参数
    #********************************************************************************************************
    #LinearSVC()
    #是由于LinearSVC只能计算线性核，而SVC可以计算任意核,用LinearSVC的计算速度，要比用SVC且kernel传入linear参数，快很多。
    #所以如果你决定使用线性SVM，就使用LinearSVC，但如果你要是用其他核的SVM，就只能使用SVC：）
    #该算法对线性不可分的数据不能使用。

    #*************************************************************************************************8
    # svm.SVC()和svm.NuSVC(),区别仅仅在于对损失的度量方式不同，其他都一样，
    # 理论上得到的是全局最优点，解决了在神经网络方法中无法避免的局部极值问题；
    #********************************************************************************************
    #××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # 回归问题使用的svm，回归的原理：https://blog.csdn.net/qq_32742009/article/details/81435141
    #管道、松弛变量，拉格朗日乘子法，这些求解条件就是KKT条件。(1)是对拉格朗日函数取极值时候带来的一个必要条件，(2)是拉格朗日系数约束（同等式情况），(3)是不等式约束情况，(4)是互补松弛条件，(5)、(6)是原约束条件。
                                                    #对于一般的任意问题而言，KKT条件是使一组解成为最优解的必要条件，当原问题是凸问题的时候，KKT条件也是充分条件。
    # sklearn.svm.NuSVR()
    # sklearn.svm.LinearSVR()
    # sklearn.svm.SVR()
    #基本与分类的无差异
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    #什么特殊场景需要使用NuSVC分类 和 NuSVR 回归呢？如果我们对训练集训练的错误率或者说支持向量的百分比有要求的时候，可以选择NuSVC分类 和 NuSVR 。它们有一个参数来控制这个百分比。


    my_svm=svm.SVC(C=2.0,probability=True)
    #SVM用函数clf.predict_proba()时候报错如下,
    # AttributeError: predict_proba is not available
    # when probability = False
    # 解决方法：clf = SVC()默认情况probability = False，添加probability = True


    my_svm.fit(data_x,data_y)
    prediction_pob=my_svm.predict_proba(test_x)
    one_hot_text_y=label_binarize(test_y,np.arange(num_class))
    print "AUC Score macro:", (roc_auc_score(one_hot_text_y, prediction_pob, average='macro'))

    #https://blog.csdn.net/hfutdog/article/details/88085878
    #
    # Macro Average
    # 宏平均是指在计算均值时使每个类别具有相同的权重，最后结果是每个类别的指标的算术平均值。
    # Micro Average
    # 微平均是指计算多分类指标时赋予所有类别的每个样本相同的权重，将所有样本合在一起计算各个指标。


    #准确率:
    #(TP+TN)/(TP+FP+TN+FN)
    #天然适应多分类
    #实际性能是非常低下的。当不同类别样本的比例非常不均衡时，占比大的类别往往成为影响准确率的最主要因素。
    # y_pred = [0, 2, 1, 3]
    # y_true = [0, 1, 2, 3]
    # print(accuracy_score(y_true, y_pred))  # 0.5
    # print(accuracy_score(y_true, y_pred, normalize=False)) # 2

    #y_true : 一维数组，或标签指示符 / 稀疏矩阵，实际（正确的）标签.
    # y_pred : 一维数组，或标签指示符 / 稀疏矩阵，分类器返回的预测标签.
    # normalize : 布尔值, 可选的(默认为True). 如果为False，返回分类正确的样本数量，否则，返回正 确分类的得分.
    # sample_weight : 形状为[样本数量]的数组，可选. 样本权重.
    #
    #可用于多标签分类
    ## accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))



    #精确率：********************************************************************
    #TP/(TP+FP)
    #average 可选值为[None, ‘binary’ (默认), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]. 多类或 者多标签目标需要这个参数.
    # binary’: 仅报告由pos_label指定的类的结果. 这仅适用于目标（y_{true, pred}）是二进制的情况.
    # ‘micro’: 通过计算总的TP、FN和FP来全局计算指标.各类别分别计算
    # ‘macro’: 为每个标签计算指标，找到它们未加权的均值. 它不考虑标签数量不平衡的情况.先计算出每个类别的精确率，然后取平均
    # ‘weighted’: 为每个标签计算指标，并通过各类占比找到它们的加权均值（每个标签的正例数）.
    #              它解决了’macro’的标签不平衡问题；它可以产生不在精确率和召回率之间的F-score.在macro的基础上加了权重，即加权平均
    # y_true = [0, 1, 2, 0, 1, 2]
    # y_pred = [0, 2, 1, 0, 0, 1]
    # print(precision_score(y_true, y_pred, average='macro'))  # 0.2222222222222222
    # print(precision_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
    # print(precision_score(y_true, y_pred, average='weighted'))  # 0.2222222222222222
    # print(precision_score(y_true, y_pred, average=None))  # [0.66666667 0.         0.        ]


    #召回率：**************************************************************8
    #recall_score方法和precision_score方法的参数说明都是一样的，
    #TP/(TP+FN)
    # y_true = [0, 1, 2, 0, 1, 2]
    # y_pred = [0, 2, 1, 0, 0, 1]
    #print recall_score(y_true, y_pred,average='macro') # 0.3333333333333333
    # print(recall_score(y_true, y_pred, average='micro')) # 0.3333333333333333
    # print(recall_score(y_true, y_pred, average='weighted')) # 0.3333333333333333
    # print(recall_score(y_true, y_pred, average=None)) # [1. 0. 0.]
    #
    #F1***********************************************************************8
    # y_true = [0, 1, 2, 0, 1, 2]
    # y_pred = [0, 2, 1, 0, 0, 1]
    # #print(f1_score(y_true, y_pred,average='macro'))  # 0.26666666666666666
    # print(f1_score(y_true, y_pred, average='micro')) # 0.3333333333333333
    # print(f1_score(y_true, y_pred, average='weighted')) # 0.26666666666666666
    # print(f1_score(y_true, y_pred, average=None)) # [0.8 0.  0. ]
    prediction=my_svm.predict(test_x)
    print precision_score(test_y,prediction,average='weighted')
    print recall_score(test_y,prediction,average='weighted')
    print f1_score(test_y,prediction,average='weighted')


if __name__=="__main__":
    data_dict=obtain_data('文本分类语料库')
    data_x=[]
    data_y=[]
    i=0
    for key in data_dict.keys():
        for words in data_dict[key]:
            data_x.append(words)
            data_y.append(i)
        i+=1
    print data_x,'667'
    #data_x是一个二维列表，每一行代表一个文档

    #需要在此shuffle一下

    # indexs=range(len(data_y))
    # np.random.shuffle(indexs)
    # data_x=data_x(indexs)
    # data_y=data_y(indexs)


    #两个数组进行相同的shuffle
    # a = np.arange(0, 10, 1)
    # b = np.arange(10, 20, 1)
    # print(a, b)
    # # result:[0 1 2 3 4 5 6 7 8 9] [10 11 12 13 14 15 16 17 18 19]
    # state = np.random.get_state()
    # np.random.shuffle(a)
    # print(a)
    # # result:[6 4 5 3 7 2 0 1 8 9]
    # np.random.set_state(state)
    # np.random.shuffle(b)
    # print(b)
    # # result:[16 14 15 13 17 12 10 11 18 19]

    state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(state)
    np.random.shuffle(data_y)


    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)
    #调用CountVectorizer的话必须是这样的形式
    # corpus = [
    #     'This is the first document.',
    #      'This is the second second document.',
    #      'And the third one.',
    #      'Is this the first document?',
    # ]

    ##获取词袋中所有文本关键词
    #word = vectorizer.get_feature_names()

    # 将停用词表转换为list
    #stpwrdlst = stopword.splitlines()
    # 对CountVectorizer进行初始化（去除中文停用词）
    #count_vec = CountVectorizer(stop_words=stpwrdlst)  # 创建词袋数据结构 ,去停用词

    #针对处理单个字或者单个字母


    # vectorizer = CountVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b')  # token_pattern=’(?u)\b\w\w+\b’,r 表示原生字符串.转义符无效
    # word_tokenizer = vectorizer.build_tokenizer()   #分词
    # doc_terms_list_train = [word_tokenizer(doc_str) for doc_str in train_x] #二维矩阵，每一行是一个文档或句子分完词的结果列表

    term_set_fs = feature_selection(train_x, test_y, 'IG')[:2000]  # 取前2000的单词
    print term_set_fs,'987'
    # vectorizer = CountVectorizer(binary=True)#只有单词出不出现之分，不统计词频
    vectorizer = CountVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b')  # 我们要正常即可,但是还是需要处理单个字符的情况
    term_dict = dict(zip(term_set_fs, range(len(term_set_fs))))  # 构建CountVectorizer的词典，待会转化的时候以此字典为标准，
    #需要构建CountVectorizer（）适合的形式，一个一维列表，每个元素是用空格隔开的单词字符串

    vectorizer.vocabulary_ = term_dict
    print term_dict,'6yu7'
    new_train_x=[]
    for node in train_x:
        new_train_x.append(' '.join(node))
    #得到一维列表
    doc_train_vec = vectorizer.transform(new_train_x)  # 千万不要fit，因为字典标准已经建好了
    print doc_train_vec[:2],'777777999'
    new_test_x=[]
    for node in test_x:
        new_test_x.append(' '.join(node))
    doc_test_vec=vectorizer.transform(new_test_x)
    SVM_text_classify(doc_train_vec,train_y,10,doc_test_vec,test_y)










