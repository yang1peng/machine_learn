#-*-coding:utf-8-*-
import jieba
# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip().replace('','\n') for line in open('chinsesstoptxt.txt').readlines()]
    return stopwords
# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    print("正在分词")
    sentence_depart = jieba.cut(sentence.strip().replace('','\n'))
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    #去停用词
    for word in sentence_depart:
        if word not in stopwords:
            outstr += word
            outstr += " "
    return outstr

# 给出文档路径
filename = "Init.txt"
outfilename = "out.txt"
outputs = open(outfilename, 'w')

# 将输出结果写入ou.txt中
with open(filename, 'r') as f:
    inputs=f.readlines()
    for line in inputs:
        line_seg = seg_depart(line)
        outputs.write(line_seg + '\n')
        print("-------------------正在分词和去停用词-----------")

outputs.close()

print("删除停用词和分词成功！！！")