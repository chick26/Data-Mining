import numpy as np
import math
import pandas

class HMM(object):

    def __init__(self,filename,InputData):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
        symbol=[]
        self.symbol={}
        try:
            file = open(filename,'r')
        except IOError:
            error = ['Not such file']
            return error
        Data=file.readlines()
        for line in range(len(Data)):
            Data[line]=Data[line].split()
        for line in range(len(Data)):
            for var in range(len(Data[line])):
                try:
                    Data[line][var]=float(Data[line][var])
                except:
                    symbol=list(Data[line][var])
                    Data[line]=Data[line][:-1]
        for i in range(int(Data[0][1])):self.symbol[symbol[i]]=i
        self.State=int(Data[0][0])
        self.SymNum=int(Data[0][1])
        self.TransMat=np.log2([Data[i][:self.State] for i in range(2,2+self.State)])
        self.EmissMat=np.log2([Data[i][self.State:] for i in range(2,2+self.State)])
        self.InitMat=np.log2(Data[1])
        try:
            file = open(InputData,'r')
        except IOError:
            error = ['Not such file']
            return error
        Data0=file.readlines()[1:]
        self.InputData=(''.join(Data0).upper()).replace('\n','')

    def Forward(self):
        Sa=[]
        Sb=[]
        for index,var in enumerate(self.InputData):
            if (index==0):
                Sa.append((self.InitMat[0]+self.EmissMat[0][self.symbol.get(var,None)],'A'))
                Sb.append((self.InitMat[1]+self.EmissMat[1][self.symbol.get(var,None)],'B'))
            else:
                Sa.append(max((Sa[index-1][0]+self.TransMat[0][0]+self.EmissMat[0][self.symbol.get(var,None)],'A'),\
                    (Sb[index-1][0]+self.TransMat[1][0]+self.EmissMat[1][self.symbol.get(var,None)],'B')))
                Sb.append(max((Sa[index-1][0]+self.TransMat[0][1]+self.EmissMat[0][self.symbol.get(var,None)],'A'),\
                    (Sb[index-1][0]+self.TransMat[1][1]+self.EmissMat[1][self.symbol.get(var,None)],'B')))
        return (Sa,Sb)

    def TraceBack(self):
        (Sa,Sb)=self.Forward()
        Trac=[]
        if(Sa[-1][0]>Sb[-1][0]):Trac.append(Sa[-1][1])
        else:Trac.append(Sb[-1][1])
        j=0
        for i in range(len(Sa)-2,-1,-1):
            if(Trac[j]=='A'):
                Trac.append(Sa[i][1])
            else:
                Trac.append(Sb[i][1])
            j+=1
        Trac.reverse()
        return Trac

    def Counter(self):
        Trac=self.TraceBack()
        i,j=0,0
        list=[]
        Index=[]
        if(Trac[0]=='A'):
            Index.append('state A')
            Index.append('state B')
        else:
            Index.append('state B')
            Index.append('state A')
        list.append([1,0,Index[0]])
        while i<len(Trac)-1:
            if(Trac[i]!=Trac[i+1]):
                list[j][1]=i+1
                list.append([i+2,0,Index[(j+1)%2]])
                j+=1
            i+=1
        list[j][1]=len(Trac)
        print(*list,sep='\n')
        return list
    
    def Print2TXT(self):
        list=self.Counter()
        output = open('HMM\data.txt','w',encoding='gbk')
        for row in list:
	        rowtxt = '{},{},{}'.format(row[0],row[1],row[2])
	        output.write(rowtxt)
	        output.write('\n')
        output.close()

if __name__=="__main__":
    obj=HMM('HMM/example.hmm.txt','HMM/example.fa')
    #obj.Counter()
    obj.Print2TXT()
    #Set ur path to file