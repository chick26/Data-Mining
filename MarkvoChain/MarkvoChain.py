import numpy as np
import math

class MarkovChain(object):

    def __init__(self):
    # Compute the log ratio
        try:
            Inside = input("")
            InsideMat = np.loadtxt(Inside)
            Outside=input("")
            OutsideMat=np.loadtxt(Outside)
        except IOError:
            error = ['Not such file']
            return error
        LogRat=np.log2(InsideMat)-np.log2(OutsideMat)
    # Generate corresponding matrix
        GenMat=[['AA','AC','AG','AT'],
                ['CA','CC','CG','CT'],
                ['GA','GC','GG','GT'],
                ['TA','TC','TG','TT']]
        LogRatDic={}
    #Combine two matrix into dictionary
        for i in range(4):
            for j in range(4):
                LogRatDic[GenMat[i][j]]=LogRat[i][j]
        self.LOGRAT=LogRatDic
    # Try to read a txt file and return a list.Return [] if there was a mistake.
        try:
            filename=input("")
            file = open(filename,'r')
        except IOError:
            error = ['Not such file']
            return error
        Data=file.readlines()
    #Clear '\n' out of DataSet
        for line in range(len(Data)):
            Data[line]=Data[line].strip('\n')
        self.Data=Data
    
    def SplitData(self,Data):
    # Split Data two by two
        Alist=[]
        Blist=[]
        if(len(Data)%2==0):
            for i in range(0, len(Data), 2):
                Alist.append(Data[i:i+2])
            DataTemp=Data[1:-1]
            for i in range(0, len(DataTemp), 2):
                Blist.append(DataTemp[i:i+2])
        else:
            DataTemp=Data[:-1]
            for i in range(0, len(DataTemp), 2):
                Alist.append(DataTemp[i:i+2])
            DataTemp=Data[1:]
            for i in range(0, len(DataTemp), 2):
                Blist.append(DataTemp[i:i+2])
        return(Alist,Blist)

    def Counter(self,Alist,Blist):
    # Dictionary record elements
        Count_1={}
        Count_2={}
        for i in Alist: Count_1[i]=(Alist.count(i))
        for i in Blist: Count_2[i]=(Blist.count(i))
    # Combine two dictionaries
        for i,j in Count_2.items():
            if i in Count_1.keys():
                Count_1[i] += j
            else:
                Count_1.update({f'{i}' : Count_2[i]})
        return(Count_1)

    def CondPro(self,LogRatDic,Count):
        S=0
        for key in Count:
            S+=self.LOGRAT[key]*Count[key]
        return S
    
    def LOOP(self):
    #Output in a txt
        #f = open('Output.txt','w')
        for line in range(len(self.Data)):
            (Alist,Blist)=self.SplitData(self.Data[line])
            Count=self.Counter(Alist,Blist)
            Sum=self.CondPro(self.LOGRAT,Count)
            if(Sum>0):
                #f.write('%.15f'%Sum+'\tInside\n')
                print('%.15f'%Sum,'\tInside')
            else:
                #f.write('%.15f'%Sum+'\tOutside\n')
                print('%.15f'%Sum,'\tOutside') 
        #f.close()

if __name__=="__main__":
    # Set UR path of all input document
    obj=MarkovChain()
    obj.LOOP()
