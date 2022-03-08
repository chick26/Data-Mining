import numpy as np
import sklearn
import matplotlib as plt
import operator

class SimpleKNN():

    def createDataSet(self):
        group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
        labels=['A','A','B','B']
        return group,labels

    def classfy0(self,inX,dataSet,labels,k):
        dataSetSize=dataSet.shape[0]                    #n条数据
        diffMat=np.tile(inX,(dataSetSize,1))-dataSet    #将输入数据复制n条,作为矩阵求与原矩阵的差值
        sqDiffMat=diffMat**2                            #求取欧氏距离 √(x1-x0)^2+(y1-y0)^2... 平方
        distances=(sqDiffMat.sum(axis=1))**0.5          #求和开根
        sortedDistIndicied=distances.argsort()          #由小到大的索引值
        classCount={}                                   #字典
        for i in range(k):
            voteByLabel=labels[sortedDistIndicied[i]]   #第k个索引对应的label
                                                        #利用字典,记录该label出现的次数
            classCount[voteByLabel]=classCount.get(voteByLabel,0)+1 #有该字段则加一,没有则0+1
                                                        #items将字典返回列表,itemgetter根据该维排序,降序
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        print(sortedClassCount[0][0])
        return sortedClassCount[0][0]

class kNN():

    def __init__(self,filename):
        '''
        读取文件,通过归一化等转换为待处理的数据矩阵
        '''
        Data=open(filename)
        DataInLine=Data.readlines()         #转换为列表形式
        NumberOfLines=len(DataInLine)
        DataMat=zeros((NumberOfLines,3))    #初始化数据矩阵
        LabelList=[]                        #初始化标签列表
        index=0
        for line in DataInLine:
            line=line.strip()               #去除首位空格
            listFromline=line.split('\t')   #以制表符为分隔
            DataMat[index,:]=listFromline[0:3]      #获取数据
            LabelList.append(int(listFromline[-1])) #获取标签



if __name__ == "__main__":
    '''
    knn=SimpleKNN()
    group,labels=knn.createDataSet()
    knn.classfy0([0,1],group,labels,3)
    '''
    knn=SimpleKNN()
    group,labels=knn.createDataSet()
    knn.classfy0([0,1],group,labels,3)

