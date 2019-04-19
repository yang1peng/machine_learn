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

def get_term_dict(doc_terms_list):  #��ø������µ����е��ʵ��ֵ�
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())  # term set ����󣬰������������ֵ�
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict


def get_class_dict(doc_class_list): #��ø������µ�����ֵ�
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return class_dict



def stats_class_df(doc_class_list, class_dict):#�õ�ÿһ�������ֵĴ��������б����ʽ�洢
    #doc_class_list���ĵ���Ӧ��һά��ǩ�б�
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list


def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict): #�õ�һ���µĶ�ά�����д���ÿһ�����ʣ��д���ÿһ�����
    # Ԫ�ش�����ж�Ӧ�Ĵʣ��ڸ��ࣨ��Ӧ�У��³��ֵĴ���
    # doc_terms_list��ԭʼ���Ϸִ�֮��Ķ�ά�б�
    #doc_class_list���ĵ���Ӧ��һά��ǩ�б�
    #term_dict�������ֵ�
    #class_dict���ࣨ��ǩ���ֵ�
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
    for k in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[k]]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] += 1
    return term_class_df_mat

#��Ƶ��,ȱ�������ԡ�����������������Щ����5���г������>=3����Ƶ�γ���100�εĴ���Ϊֹͣ��ȥ����Ľ�����£�
def DF_dict(data):   #�������ĵ�Ƶ�ʣ�document frequency��,�õ��������Ǹ����������ĴʵĴʵ䣬�ʵ�ļ�ֵ�Ǹ�������ϣ�һ����ά�б�ÿ��Ϊһ���ĵ�
    #���ڴ�Ƶ�����������ԣ������俼�ǵ�ʱ��ֻ���ڸ����е������ǳ��ִ�����ģ��������ÿ����ѡ�����Ĵ���һ���ģ�
    #��Ƶ��ֻ��ÿ�൥��ѡȡһ�������ĵ���
    dict_DF={}
    for key in data.keys():
        tmp=[]
        for sentence in data[key]:
             tmp.extend(sentence)
        counter=Counter(tmp)    #������Ԫ�س��ֵĴ���ͳ��������,�γ�һ���ʵ䣬��ֵΪ���ĳ��ִ���
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0])) #���Ŵ���������С���Ŵ���Ԫ��,x[1],x[0]��˳������������
        dict_DF[key]=count_pairs
    return dict_DF  #���ص��Ǹ��������ֵĴʵ�Ƶ��ͳ�ƣ���һ���ʵ䣬������𣬼�ֵ�Ǹ����ĵ��ʴ�Ƶ���ź����ģ�
def DF_list(data_x,data_y): #r��������Ǵ�ñ�ǩ�����ݣ��ȸ������ϣ��ָ���ǩ���ֽ����ת��Ϊ�ʵ䣬ֻ��ѵ������Ҫ
    data_dict={}
    for i in range(len(data_y)):
        data_dict.get(data_y[i], []).append(data_x[i])
    result=DF_dict(data_dict)
    return result   ##���ص��Ǹ��������ֵĴʵ�Ƶ��ͳ�ƣ���һ���ʵ�



#tf-idf
#Scikit-Learn��TF-IDFȨ�ؼ��㷽����Ҫ�õ������ࣺCountVectorizer��TfidfTransformer��
#�������������ý�ͷִʵľ�ȷģʽ�ִʣ�Ȼ�����ÿո�������Щ�ִʵõ��ľ����ǣ�����ΪCountVectorizer()�����룡����

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
# tfidf.toarray()��
#���ݽṹtfidf[i][j]��ʾi���ı��е�j���ʵ�tf-idfȨ�ء�һ����ά����
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
            tmp.extend(sentence)        #Ҫ���һ���ĵ���df,idf
        all_tmp.append(tmp)     #����һ����ά�б�ÿһ�д���ÿһ���ļ��µ���������
    vectorizer = CountVectorizer()  #��������
    #��ͨ��fit_transform�����������������ֵĴ�����ͨ��get_feature_names()�ɻ�ȡ�ʴ��������ı��Ĺؼ��֣�
    # ͨ��toarray()�ɿ�����Ƶ����Ľ��
    count = vectorizer.fit_transform(all_tmp)   #��һ������ת��Ϊ��Ӧ�������б����ִ�����ֵĴ�������fit��transform����Ƶ������
    transformer = TfidfTransformer()    #��������
    tfidf_matrix = transformer.fit_transform(count) #���ݽṹtfidf[i][j]��ʾi���ı��е�j���ʵ�tf-idfȨ�أ�һ����ά����
    #tfidf�������ÿһ�����ϳ�������µ�tfidf�����Ը����µĵ���ҲҪ�ֱ�ѡȡ
    return tfidf_matrix #���ص��Ƕ�ά�������ж�Ӧ�ĸ������Ҫ�������������ʱ��ÿ���඼��ѡ��һ�������ĵ��ʣ���ÿ�����ѡ20�����ʣ�
def tf_idf_list(data_x,data_y):
    data_dict = {}
    for i in range(len(data_y)):
        data_dict.get(data_y[i], []).append(data_x[i])
    result=tf_idf_dict(data_dict)
    return result



#����Ϣ��
#https://blog.csdn.net/BlowfishKing/article/details/78298057
def feature_selection_mi(class_df_list, term_set, term_class_df_mat):#term_class_df_mat��һ����Ϊ���ϵ���������Ϊ������ľ���
    #class_df_list ÿһ�������ֵĴ��������б����ʽ�洢���б��б�ǩ��Ӧλ�õ�Ԫ��Ϊ����������
    #term_set�õ����ʼ��ϣ����Ǳ������б�������ģ����������
    A = term_class_df_mat   #A��ÿһ��Ԫ�ش���õ����ڸ�����µĳ��ִ���
    B = np.array([(sum(x) - x).tolist() for x in A])    #B��ÿһ��Ԫ�ش���õ����ڸ���������������ֵĴ���
    C = np.tile(class_df_list, (A.shape[0], 1)) - A     #C��ÿһ��Ԫ�ش���ø������ֵ�ʱ�򣬸ô�û�г��ֵ��������Ŀ
    N = sum(class_df_list)  #N���������ࣨ�ĵ������ֵ���Ŀ
    class_set_size = len(class_df_list)

    term_score_mat = np.log(((A + 1.0) * N) / ((A + C) * (A + B + class_set_size))) #�����Լ�ȵĹ�ʽά����Ȼ�Ƕ�ά���д�������Ϊ�б���
    term_score_max_list = [max(x) for x in term_score_mat]#ȡ��ÿ�У��õ��ʣ����Ļ���Ϣֵ
    term_score_array = np.array(term_score_max_list)    #���һά����Ԫ�����Ǵ���
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index] #ȫ���������е��б�

    return term_set_fs

#��Ϣ���淨
#http://www.blogjava.net/zhenandaci/archive/2009/03/24/261701.html
def feature_selection_ig(class_df_list, term_set, term_class_df_mat):#term_class_df_mat��һ����Ϊ���ϵ���������Ϊ������ľ���
    A = term_class_df_mat   #A��ÿһ��Ԫ�ش���õ����ڸ�����µĳ��ִ���
    B = np.array([(sum(x) - x).tolist() for x in A])    #B��ÿһ��Ԫ�ش���õ����ڸ���������������ֵĴ���
    C = np.tile(class_df_list, (A.shape[0], 1)) - A #C��ÿһ��Ԫ�ش���ø������ֵ�ʱ�򣬸ô�û�г��ֵ��������Ŀ
    N = sum(class_df_list)  #N���������ࣨ�ĵ������ֵ���Ŀ
    D = N - A - B - C   #D��ÿһ��Ԫ�ش���ô�û����
    term_df_array = np.sum(A, axis=1) # ��ÿ����ӣ��õ�һά����ȫ�����ʵ���Ŀ��Ԫ��
    class_set_size = len(class_df_list)

    p_t = term_df_array / N #ÿ��ȡƽ�������Ǹ�һά����ÿ��Ԫ�ض�����N
    p_not_t = 1 - p_t #ͬ��ά�ȵľ���
    p_c_t_mat = (A + 1) / (A + B + class_set_size)  #���Ӿ��󣬵��ھ����е�ÿ��Ԫ�ض��Ӹ���������������ڶ�ӦԪ�����������ά�Ȳ�һ���ᱨ��
        #p_c_t_mat��һ����Aһ���ľ��������Ǵ����������������
    p_c_not_t_mat = (C + 1) / (C + D + class_set_size)
        #p_c_not_t_matҲһ��
    p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)   #�����ǶԾ���ÿ��Ԫ�ص���ȡ����������ĳ˷�Ҳ�Ƕ�ӦԪ����ˣ��˷��Ľ����Ȼ��p_c_t_matά��һ�������Ǵ��������������
        #��axis=1ά����ӣ���˼�ǰѾ���ÿ����ӣ��õ�һά����Ԫ����Ϊ����
    p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)

    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t #����˷��ټӷ������Ƕ�ӦԪ����ӣ��õ�����Ȼ��һά����Ԫ����Ϊ������ÿ��Ԫ�ش����˸ôʵ���Ϣ����

    sorted_term_score_index = term_score_array.argsort()[:: -1] #�Ӵ�С����ÿ��Ԫ�����ź����Ԫ�أ���term_score_array�������
    term_set_fs = [term_set[index] for index in sorted_term_score_index] #���ص��ǵ����б�������Ϣ�����ź���ĵ���
    print term_set_fs,'000000'
    return term_set_fs

#WLLR
#https://www.cnblogs.com/wangbogong/p/3251132.html
#http://www.doc88.com/p-2912807486871.html
def feature_selection_wllr(class_df_list, term_set, term_class_df_mat):#term_class_df_mat��һ����Ϊ���ϵ���������Ϊ������ľ���
    A = term_class_df_mat   #A��ÿһ��Ԫ�ش���õ����ڸ�����µĳ��ִ���
    B = np.array([(sum(x) - x).tolist() for x in A])    #B��ÿһ��Ԫ�ش���õ����ڸ���������������ֵĴ���
    C_Total = np.tile(class_df_list, (A.shape[0], 1))
    N = sum(class_df_list)
    C_Total_Not = N - C_Total
    term_set_size = len(term_set)

    p_t_c = (A + 1E-6) / (C_Total + 1E-6 * term_set_size)
    p_t_not_c = (B + 1E-6) / (C_Total_Not + 1E-6 * term_set_size)
    term_score_mat = p_t_c * np.log(p_t_c / p_t_not_c)
    #�����ӳ˳�������Ծ����Ԫ�أ�����ά�Ȳ��䣬��Ϊ��������Ϊ�����
    term_score_max_list = [max(x) for x in term_score_mat]  #ȡ��ÿ�У��õ��ʣ�����ֵ
    term_score_array = np.array(term_score_max_list)    #��Ϊһά����
    sorted_term_score_index = term_score_array.argsort()[:: -1] #[:: -1]������argsort()�Ȱ���С�������򣬷��ص���������֮���Ԫ�أ���ԭ�����б����λ������
    term_set_fs = [term_set[index] for index in sorted_term_score_index]

    print term_set_fs[:10]
    return term_set_fs  #���ص��ǵ����б�����wllr�ź���ĵ���


def feature_selection(doc_terms_list, doc_class_list, fs_method):
    #doc_terms_list����ά����ÿһ����һ���ĵ�����ӷ���ʵĽ���б�
    #doc_class_list��һά��֤��ÿһ���ĵ���Ӧ�����ǩ
    #fs_method��ѡȡ�ķ���
    class_dict = get_class_dict(doc_class_list) #�õ����ֵ�
    term_dict = get_term_dict(doc_terms_list)   #�õ������ֵ䣬range��һ�����
    class_df_list = stats_class_df(doc_class_list, class_dict)#�õ�ÿһ�������ֵĴ��������б����ʽ�洢���б��б�ǩ��Ӧλ�õ�Ԫ��Ϊ����������
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)#�õ�һ���µĶ�ά�����д���ÿһ�����ʣ��д���ÿһ�����
    # Ԫ�ش�����ж�Ӧ�Ĵʣ��ڸ��ࣨ��Ӧ�У��³��ֵĴ���

    term_set = [term[0] for term in sorted(term_dict.items(), key=lambda x: x[1])]  # �õ����ʼ��ϣ����Ǳ������б�������ģ����������
    term_set_fs = []

    if fs_method == 'MI':
        term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'IG':
        term_set_fs = feature_selection_ig(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'WLLR':
        term_set_fs = feature_selection_wllr(class_df_list, term_set, term_class_df_mat)

    return term_set_fs

#��������������������������������������
#@��������������������������������
#�����ֲ�sklearn���ֳɵĺ�����https://blog.csdn.net/snowdroptulip/article/details/78867053
#������ԭ��https://www.cnblogs.com/wangbogong/p/3251132.html
#�磺
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
iris = load_iris()
model1 = SelectKBest(chi2, k=2)#ѡ��k���������
model1.fit_transform(iris.data, iris.target)#iris.data���������ݣ�iris.target�Ǳ�ǩ���ݣ��ú�������ѡ���k������
#ԭ��
# array([[ 5.1,  3.5,  1.4,  0.2],
#        [ 4.9,  3. ,  1.4,  0.2],
#        [ 4.7,  3.2,  1.3,  0.2],
#        [ 4.6,  3.1,  1.5,  0.2],
#        [ 5. ,  3.6,  1.4,  0.2],
#        [ 5.4,  3.9,  1.7,  0.4],
#        [ 4.6,  3.4,  1.4,  0.3],
#�����
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
    #�����Ǹ�����
    # doc_str_list_train, doc_str_list_test, doc_class_list_train, doc_class_list_test = train_test_split(
    #     movie_reviews.data, movie_reviews.target, test_size=0.2, random_state=0)
    # vectorizer = CountVectorizer(binary=True)
    # word_tokenizer = vectorizer.build_tokenizer()   #����൱�����ķִ�
    #�磺
    # ** ** ** ** ** *[['This', 'is', 'the', 'first', 'document'], ['This', 'is', 'the', 'second', 'second', 'document'],
    #                  ['And', 'the', 'third', 'one'], ['Is', 'this', 'the', 'first', 'document']] ** ** ** ** ** *

    #doc_terms_list_train = [word_tokenizer(doc_str) for doc_str in doc_str_list_train]#����Ǹ���ά�б�ÿ�����б���һ���ĵ���һ�仰
    #ÿһ�����б���һ���ĵ��ִ�֮��ȡ��ȥ���ôʺ�Ľ��

    #ʹ�÷�����
    #term_set_fs = feature_selection(doc_terms_list_train, doc_class_list_train, fs_method)[:fs_num]#���ǵ�����ͷ��������ȡ����
