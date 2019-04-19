#-*-coding:utf-8-*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

# corpus = [
#     'This is the first document.',
#     'This is the second second document.',
#     'And the third one.',
#     'Is this the first document?'
# ]
# vectorizer = CountVectorizer()
#
# word_tokenizer = vectorizer.build_tokenizer()
# doc_terms_list_train = [word_tokenizer(doc_str) for doc_str in corpus]
# print '***********',doc_terms_list_train,'***********'
#
#
# count = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# print(vectorizer.vocabulary_)
# print(count.toarray())
#
# transformer = TfidfTransformer()
# tfidf_matrix = transformer.fit_transform(count)
# print(tfidf_matrix.toarray())

#****************8

# from sklearn.preprocessing import  OneHotEncoder
#
# enc = OneHotEncoder()
# enc.fit([[0, 0, 3],
#          [1, 1, 0],
#          [0, 2, 1],
#          [1, 0, 2]])
#
# ans = enc.transform([[0, 1, 3]]).toarray()  # 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果
# print(ans)

#********************************
#计算auc，画ROC曲线
# !/usr/bin/python
# -*- coding:utf-8 -*-

# import numpy as np
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn import metrics
# from sklearn.preprocessing import label_binarize
# from sklearn import datasets
#
# if __name__ == '__main__':
#     np.random.seed(0)
#     datas = datasets.load_iris() # 读取数据
#     x = datas.data
#     y =datas.target # 将标签转换0,1,...
#     x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=0)
#     y_one_hot = label_binarize(y_test, np.arange(3))  # 装换成类似二进制的编码
#     alpha = np.logspace(-2, 2, 20)  # 设置超参数范围
#     model = LogisticRegressionCV(cv=3, penalty='l2')  # 使用L2正则化
#     model.fit(x_train, y_train)
#     print '超参数：', model.C_
#     # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
#     y_score = model.predict_proba(x_test)
#     print y_score.shape
#     # 1、调用函数计算micro类型的AUC
#     print '调用函数auc：', metrics.roc_auc_score(y_one_hot, y_score, average='micro')
#     # 2、手动计算micro类型的AUC
#     # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
#     fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
#     auc = metrics.auc(fpr, tpr)
#     print '手动计算auc：', auc
#     # 绘图
#     mpl.rcParams['font.sans-serif'] = u'SimHei'
#     mpl.rcParams['axes.unicode_minus'] = False
#     # FPR就是横坐标,TPR就是纵坐标
#     plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
#     plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
#     plt.xlim((-0.01, 1.02))
#     plt.ylim((-0.01, 1.02))
#     plt.xticks(np.arange(0, 1.1, 0.1))
#     plt.yticks(np.arange(0, 1.1, 0.1))
#     plt.xlabel('False Positive Rate', fontsize=13)
#     plt.ylabel('True Positive Rate', fontsize=13)
#     plt.grid(b=True, ls=':')
#     plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
#     plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
#     plt.show()

#××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
# #CountVectorizer处理中文的情况
# from sklearn.feature_extraction.text import CountVectorizer
# corpus=['我 爱 北京 天安门','北京 欢迎 您']
# #corpus=['i a b','is y o']
# # tm=[]
# # for node in corpus:
# #     tm.append(' '.join(node))
# vectorizer=CountVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b') # token_pattern=’(?u)\b\w\w+\b’,r 表示原生字符串.转义符无效
# # for n in tm:
# #     print n
#
# X = vectorizer.fit_transform(corpus)
# print ' '.join(vectorizer.get_feature_names())
#
#
# print X.toarray()

#控制台编码方式是gbk
#中文的utf-8 编码 \xe4\xb8\xad\xe6\x96\x87 强制转换为 GBK 就会乱码了，
# GBK 是两个字节存储一个中文字符，所以 \xe4\xb8\xad\xe6\x96\x87 会解码成三个字，很不幸这三个字涓枃

# 文件存储为utf - 8格式，编码声明为utf - 8，  # encoding:utf-8
# 出现汉字的地方前面加u
# 不同编码之间不能直接转换，要经过unicode中间跳转
# cmd下不支持utf - 8
# 编码raw_input提示字符串只能为gbk编码
#*************************************************************************************
#使用自身的单词列表建立CountVectorizer的词典
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b')  # 我们要正常即可
# term_set_fs=[u'我',u'爱',u'北京',u'天安门',u'欢迎',u'您']
# term_dict = dict(zip(term_set_fs, range(len(term_set_fs))))
#
# vectorizer.vocabulary_ = term_dict
# print ' '.join(vectorizer.get_feature_names())
# train_x=['您 爱 北京 爱']
# #train_x=[['您', '爱', '北京']]
# doc_train_vec = vectorizer.transform(train_x)
# print doc_train_vec.toarray(),'88'


#*************************************************************************************
#读入文件并进行分词，去停用词
# import jieba
# import codecs
# def stopwordslist(file_name):
#     stopwords = [line.strip().replace('\n','') for line in codecs.open(file_name,'r','utf-8').readlines()]
#     return stopwords
# with codecs.open('fencilianxi.txt','r','utf-8') as f:
#     datas=f.readlines()
#     print datas
#     for node in datas:
#         data_list=jieba.cut(node.strip().replace('\n',''))
#         print data_list
#         print ','.join(data_list)
#
# stopwords = stopwordslist('stop_word.txt')
# for node in stopwords:
#     print node.replace('\n','')
#     if '风雨无阻'.encode('utf8') in node:
#         print '6666'
#     else:
#         print "&&&&"
#
#*************************************************************************************
#读入文件格式是gbk，是否还能按上述步骤处理
# 字符串是str对象，中文字符串。存储方式是字节码。字节码是怎么存的：
#
# 如果这行代码在python解释器中输入&运行，那么s的格式就是解释器的编码格式；
#
# 如果这行代码是在源码文件中写入、保存然后执行，那么解释器载入代码时就将s初始化为文件指定编码(比如py文件开头那行的utf-8)；
#coding:utf-8
import codecs
import jieba
def stopwordslist(file_name):
    stopwords = [line.strip().replace('\n','') for line in codecs.open(file_name,'r','utf-8').readlines()]
    return stopwords
with codecs.open('410.TXT','r','utf-8') as f:
    datas=f.readlines()
    sentence=''
    for node in datas:
        node_new=node.replace(' ','').replace('\n','')
        if '日本工人' in node_new:  #这是可以的?
            print '6666'
        else:
            print '1111'
        sentence+=node_new

    data_list=list(jieba.cut(sentence)) #转化成list，不然只能遍历一次

    cc='/'.join(data_list)
    print cc,'tt'
    print type(data_list)
    tmp=[]
    stopwords = stopwordslist('stop_word.txt')
    for node in data_list:
        print 'ok'
        print node,'bebe'
        if node not in stopwords:
            print node,'afaf'
            tmp.append(node)
    print tmp
    print '/'.join(data_list),'9090'



