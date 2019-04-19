#!/usr/bin/env python
# coding=gbk
#https://www.cnblogs.com/wangbogong/p/3251132.html

import os
import sys

import numpy as np

import os
import sys
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from collections import Counter

def get_term_dict(doc_terms_list):  #获得该语料下的所有单词的字典
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())  # term set 排序后，按照索引做出字典
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict


def get_class_dict(doc_class_list): #获得该语料下的类的字典
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return class_dict



def stats_class_df(doc_class_list, class_dict):#得到每一个类别出现的次数，以列表的形式存储
    #doc_class_list：文档对应的一维标签列表
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list


def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict): #得到一个新的二维矩阵，行代表每一个单词，列代表每一个类别，
    # 元素代表该行对应的词，在该类（对应列）下出现的次数
    # doc_terms_list：原始语料分词之后的二维列表
    #doc_class_list：文档对应的一维标签列表
    #term_dict：单词字典
    #class_dict：类（标签）字典
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
    for k in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[k]]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] += 1
    return term_class_df_mat

#词频法,缺乏特异性。可以这样做：将那些在这5类中出现类别>=3类且频次超过100次的词作为停止词去掉后的结果如下：
def DF_dict(data):   #词条的文档频率（document frequency）,得到的数据是各个类别包含的词的词典，词典的键值是该类的语料，一个二维列表，每行为一个文档
    #由于词频不考虑特异性，所以其考虑的时候只需在各类中单独考虑出现次数多的，（会出现每个类选出来的词有一样的）
    #词频法只能每类单独选取一定数量的单词
    dict_DF={}
    for key in data.keys():
        tmp=[]
        for sentence in data[key]:
             tmp.extend(sentence)
        counter=Counter(tmp)    #把所有元素出现的次数统计下来了,形成一个词典，键值为键的出现次数
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0])) #负号代表降序排序，小括号代表元组,x[1],x[0]的顺序是有玄机的
        dict_DF[key]=count_pairs
    return dict_DF  #返回的是各个类别出现的词的频数统计，是一个词典，键是类别，键值是该类别的单词词频（排好序后的）
def DF_list(data_x,data_y): #r如果给的是打好标签的数据，既给了语料，又给标签，现将类表转换为词典，只有训练集需要
    data_dict={}
    for i in range(len(data_y)):
        data_dict.get(data_y[i], []).append(data_x[i])
    result=DF_dict(data_dict)
    return result   ##返回的是各个类别出现的词的频数统计，是一个词典



#tf-idf
#Scikit-Learn中TF-IDF权重计算方法主要用到两个类：CountVectorizer和TfidfTransformer。
#！！！！中文用结巴分词的精确模式分词，然后我用空格连接这些分词得到的句子是，在作为CountVectorizer()的输入！！！

# vectorizer = CountVectorizer()
# corpus = [
#     'This is the first document.',
#      'This is the second second document.',
#      'And the third one.',
#      'Is this the first document?',
# ]
# X = vectorizer.fit_transform(corpus)
# X.toarray()
# :
# array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
#        [0, 1, 0, 1, 0, 2, 1, 0, 1],
#        [1, 0, 0, 0, 1, 0, 1, 1, 0],
#        [0, 1, 1, 1, 0, 0, 1, 0, 1]])
# vectorizer.get_feature_names()
# :
# (['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this'])

# transformer = TfidfTransformer()
# counts = [[3, 0, 1],
#           [2, 0, 0],
#            [3, 0, 0],
#            [4, 0, 0],
#            [3, 2, 0],
#            [3, 0, 2]]
# tfidf = transformer.fit_transform(counts)
# tfidf.toarray()：
#数据结构tfidf[i][j]表示i类文本中第j个词的tf-idf权重。一个二维数组
#     array([[ 0.85...,  0.  ...,  0.52...],
#            [ 1.  ...,  0.  ...,  0.  ...],
#            [ 1.  ...,  0.  ...,  0.  ...],
#            [ 1.  ...,  0.  ...,  0.  ...],
#            [ 0.55...,  0.83...,  0.  ...],
#            [ 0.63...,  0.  ...,  0.77...]])

def tf_idf_dict(data):
    all_tmp=[]
    for key in data.keys():
        tmp = []
        for sentence in data[key]:
            tmp.extend(sentence)        #要针对一类文档的df,idf
        all_tmp.append(tmp)     #这是一个二维列表，每一行代表每一类文件下的所有语料
    vectorizer = CountVectorizer()  #申明对象
    #它通过fit_transform函数计算各个词语出现的次数，通过get_feature_names()可获取词袋中所有文本的关键字，
    # 通过toarray()可看到词频矩阵的结果
    count = vectorizer.fit_transform(all_tmp)   #把一个语料转化为对应的数字列表，数字代表出现的次数，既fit又transform，词频向量化
    transformer = TfidfTransformer()    #申明对象
    tfidf_matrix = transformer.fit_transform(count) #数据结构tfidf[i][j]表示i类文本中第j个词的tf-idf权重，一个二维数组
    #tfidf还是针对每一类语料出该类别下的tfidf，所以给类下的单词也要分别选取
    return tfidf_matrix #返回的是二维矩阵，哪行对应哪个类别不重要，在特征构造的时候，每个类都会选择一定数量的单词（如每个类别选20个单词）
def tf_idf_list(data_x,data_y):
    data_dict = {}
    for i in range(len(data_y)):
        data_dict.get(data_y[i], []).append(data_x[i])
    result=tf_idf_dict(data_dict)
    return result



#互信息法
#https://blog.csdn.net/BlowfishKing/article/details/78298057
def feature_selection_mi(class_df_list, term_set, term_class_df_mat):#term_class_df_mat是一个行为语料单词数，列为类别数的矩阵
    #class_df_list 每一个类别出现的次数，以列表的形式存储，列表中标签对应位置的元素为该类别的数量
    #term_set得到单词集合，但是本质是列表，是有序的，按序号排序
    A = term_class_df_mat   #A的每一个元素代表该单词在该类别下的出现次数
    B = np.array([(sum(x) - x).tolist() for x in A])    #B的每一个元素代表该单词在该类别以外的类里出现的次数
    C = np.tile(class_df_list, (A.shape[0], 1)) - A     #C的每一个元素代表该该类别出现的时候，该词没有出现的情况的数目
    N = sum(class_df_list)  #N代表所有类（文档）出现的数目
    class_set_size = len(class_df_list)

    term_score_mat = np.log(((A + 1.0) * N) / ((A + C) * (A + B + class_set_size))) #这个是约等的公式维度仍然是二维，行词数，列为列别数
    term_score_max_list = [max(x) for x in term_score_mat]#取出每行（该单词）最大的互信息值
    term_score_array = np.array(term_score_max_list)    #变成一维矩阵，元素数是词数
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index] #全不单词排列的列表

    return term_set_fs

#信息增益法
#http://www.blogjava.net/zhenandaci/archive/2009/03/24/261701.html
def feature_selection_ig(class_df_list, term_set, term_class_df_mat):#term_class_df_mat是一个行为语料单词数，列为类别数的矩阵
    A = term_class_df_mat   #A的每一个元素代表该单词在该类别下的出现次数
    B = np.array([(sum(x) - x).tolist() for x in A])    #B的每一个元素代表该单词在该类别以外的类里出现的次数
    C = np.tile(class_df_list, (A.shape[0], 1)) - A #C的每一个元素代表该该类别出现的时候，该词没有出现的情况的数目
    N = sum(class_df_list)  #N代表所有类（文档）出现的数目
    D = N - A - B - C   #D的每一个元素代表该词没出现
    term_df_array = np.sum(A, axis=1) # 把每行相加，得到一维矩阵，全部单词的数目个元素
    class_set_size = len(class_df_list)

    p_t = term_df_array / N #每行取平均，就是该一维矩阵每个元素都除以N
    p_not_t = 1 - p_t #同样维度的矩阵
    p_c_t_mat = (A + 1) / (A + B + class_set_size)  #数加矩阵，等于矩阵中的每个元素都加该数，矩阵除法等于对应元素相除，矩阵维度不一样会报错
        #p_c_t_mat是一个和A一样的矩阵，行数是次数，列数是类别数
    p_c_not_t_mat = (C + 1) / (C + D + class_set_size)
        #p_c_not_t_mat也一样
    p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)   #对数是对矩阵每个元素单独取对数，这里的乘法也是对应元素相乘，乘法的结果依然和p_c_t_mat维度一样，行是词数，列是类别数
        #在axis=1维度相加，意思是把矩阵每行相加，得到一维矩阵，元素数为词数
    p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)

    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t #矩阵乘法再加法，都是对应元素相加，得到的仍然是一维矩阵，元素数为词数，每个元素代表了该词的信息增益

    sorted_term_score_index = term_score_array.argsort()[:: -1] #从大到小排序每个元素是排好序的元素，在term_score_array里的索引
    term_set_fs = [term_set[index] for index in sorted_term_score_index] #返回的是单词列表，按按信息增益排好序的单词
    print term_set_fs,'000000'
    return term_set_fs

#WLLR
#https://www.cnblogs.com/wangbogong/p/3251132.html
#http://www.doc88.com/p-2912807486871.html
def feature_selection_wllr(class_df_list, term_set, term_class_df_mat):#term_class_df_mat是一个行为语料单词数，列为类别数的矩阵
    A = term_class_df_mat   #A的每一个元素代表该单词在该类别下的出现次数
    B = np.array([(sum(x) - x).tolist() for x in A])    #B的每一个元素代表该单词在该类别以外的类里出现的次数
    C_Total = np.tile(class_df_list, (A.shape[0], 1))
    N = sum(class_df_list)
    C_Total_Not = N - C_Total
    term_set_size = len(term_set)

    p_t_c = (A + 1E-6) / (C_Total + 1E-6 * term_set_size)
    p_t_not_c = (B + 1E-6) / (C_Total_Not + 1E-6 * term_set_size)
    term_score_mat = p_t_c * np.log(p_t_c / p_t_not_c)
    #上述加乘除都是针对矩阵的元素，矩阵维度不变，行为词数，列为类别数
    term_score_max_list = [max(x) for x in term_score_mat]  #取出每行（该单词）最大的值
    term_score_array = np.array(term_score_max_list)    #变为一维矩阵
    sorted_term_score_index = term_score_array.argsort()[:: -1] #[:: -1]到排序。argsort()先按从小到大排序，返回的是排完序之后的元素，在原来的列表里的位置索引
    term_set_fs = [term_set[index] for index in sorted_term_score_index]

    print term_set_fs[:10]
    return term_set_fs  #返回的是单词列表，按按wllr排好序的单词


def feature_selection(doc_terms_list, doc_class_list, fs_method):
    #doc_terms_list：二维矩阵，每一行是一个文档或句子分完词的结果列表
    #doc_class_list：一维举证，每一个文档对应的类标签
    #fs_method：选取的方法
    class_dict = get_class_dict(doc_class_list) #得到类字典
    term_dict = get_term_dict(doc_terms_list)   #得到单词字典，range了一个序号
    class_df_list = stats_class_df(doc_class_list, class_dict)#得到每一个类别出现的次数，以列表的形式存储，列表中标签对应位置的元素为该类别的数量
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)#得到一个新的二维矩阵，行代表每一个单词，列代表每一个类别，
    # 元素代表该行对应的词，在该类（对应列）下出现的次数

    term_set = [term[0] for term in sorted(term_dict.items(), key=lambda x: x[1])]  # 得到单词集合，但是本质是列表，是有序的，按序号排序
    term_set_fs = []

    if fs_method == 'MI':
        term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'IG':
        term_set_fs = feature_selection_ig(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'WLLR':
        term_set_fs = feature_selection_wllr(class_df_list, term_set, term_class_df_mat)

    return term_set_fs

#！！！！！！！！！！！！！！！！！！！
#@！！！！！！！！！！！！！！！！
#卡方分布sklearn有现成的函数：https://blog.csdn.net/snowdroptulip/article/details/78867053
#卡方的原理：https://www.cnblogs.com/wangbogong/p/3251132.html
#如：
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
iris = load_iris()
model1 = SelectKBest(chi2, k=2)#选择k个最佳特征
model1.fit_transform(iris.data, iris.target)#iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征
#原来
# array([[ 5.1,  3.5,  1.4,  0.2],
#        [ 4.9,  3. ,  1.4,  0.2],
#        [ 4.7,  3.2,  1.3,  0.2],
#        [ 4.6,  3.1,  1.5,  0.2],
#        [ 5. ,  3.6,  1.4,  0.2],
#        [ 5.4,  3.9,  1.7,  0.4],
#        [ 4.6,  3.4,  1.4,  0.3],
#处理后：
# array([[ 1.4,  0.2],
#        [ 1.4,  0.2],
#        [ 1.3,  0.2],
#        [ 1.5,  0.2],
#        [ 1.4,  0.2],
#        [ 1.7,  0.4],
#        [ 1.4,  0.3]]


if __name__=='__main__':
    movie_reviews={}
    movie_reviews['data']=''
    movie_reviews['target']=''
    #以上是个例子
    # doc_str_list_train, doc_str_list_test, doc_class_list_train, doc_class_list_test = train_test_split(
    #     movie_reviews.data, movie_reviews.target, test_size=0.2, random_state=0)
    # vectorizer = CountVectorizer(binary=True)
    # word_tokenizer = vectorizer.build_tokenizer()   #这就相当于中文分词
    #如：
    # ** ** ** ** ** *[['This', 'is', 'the', 'first', 'document'], ['This', 'is', 'the', 'second', 'second', 'document'],
    #                  ['And', 'the', 'third', 'one'], ['Is', 'this', 'the', 'first', 'document']] ** ** ** ** ** *

    #doc_terms_list_train = [word_tokenizer(doc_str) for doc_str in doc_str_list_train]#这个是个二维列表，每个字列表是一个文档或一句话
    #每一个字列表是一个文档分词之后取，去听用词后的结果

    #使用方法：
    #term_set_fs = feature_selection(doc_terms_list_train, doc_class_list_train, fs_method)[:fs_num]#这是调用上头的特征提取函数
