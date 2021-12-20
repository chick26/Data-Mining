import numpy as np
import math
import pandas

class HMM(object):

    def __init__(self):
    # Try to read a txt file and return a list.Return [] if there was a mistake.

        symbol=[] # the list of symbol is to store ACGT
        self.symbol={}

        # Read the data of state numbers & probabilities & etc

        filename = input("")
        file = open(filename,'r')


        Data=file.readlines()
        # Create a list to store the input data temporarily

        for line in range(len(Data)):
            Data[line]=Data[line].split()
        for line in range(len(Data)):
            for var in range(len(Data[line])):
                try:
                # If the input is not ACGT
                    Data[line][var]=float(Data[line][var])
                except:
                # If the input is ACGT
                    symbol=list(Data[line][var])
                    Data[line]=Data[line][:-1]
        for i in range(int(Data[0][1])):self.symbol[symbol[i]]=i
        # Translate ACGT into 1234

        self.State=int(Data[0][0])
        self.SymNum=int(Data[0][1])
        self.TransMat=np.log2([Data[i][:self.State] for i in range(2,2+self.State)])# Switch to log space
        self.EmissMat=np.log2([Data[i][self.State:] for i in range(2,2+self.State)])# Switch to log space
        self.InitMat=np.log2(Data[1])# Switch to log space
        # Initialization probability matrix & filling

        # Read the data of observable state seq
        filename = input("")
        file = open(filename,'r')
        Data0=file.readlines()[1:]
        # Discard the first line which is a header starting with “>” in fasta files
        self.InputData=(''.join(Data0).upper()).replace('\n','') 
        # Translate all data into upper case & Clear '\n' out of DataSet

    def Forward(self):

        Sa=[] # Table initialization for state A
        Sb=[] # Table initialization for state B

        for index,var in enumerate(self.InputData):
            if (index==0):
                # Compute the first one
                Sa.append((self.InitMat[0]+self.EmissMat[0][self.symbol.get(var,None)],'A'))
                Sb.append((self.InitMat[1]+self.EmissMat[1][self.symbol.get(var,None)],'B'))
            else:
                Sa.append(max((Sa[index-1][0]+self.TransMat[0][0]+self.EmissMat[0][self.symbol.get(var,None)],'A'),\
                    (Sb[index-1][0]+self.TransMat[1][0]+self.EmissMat[1][self.symbol.get(var,None)],'B')))
                Sb.append(max((Sa[index-1][0]+self.TransMat[0][1]+self.EmissMat[0][self.symbol.get(var,None)],'A'),\
                    (Sb[index-1][0]+self.TransMat[1][1]+self.EmissMat[1][self.symbol.get(var,None)],'B')))
        return (Sa,Sb)
        # Table filling

    def TraceBack(self):
    # The traceback part

        (Sa,Sb)=self.Forward()
        Trac=[] # Create a list to record the hidden state seq

        if(Sa[-1][0]>Sb[-1][0]):Trac.append(Sa[-1][1])
        else:Trac.append(Sb[-1][1])
        # Pick state in the last step with highest score

        j=0
        for i in range(len(Sa)-2,-1,-1):
        # Backtrace to find the path
            if(Trac[j]=='A'):
                Trac.append(Sa[i][1])
            else:
                Trac.append(Sb[i][1])
            j+=1
        Trac.reverse() # Right put the path
        return Trac

    def Counter(self):
    # The output part 1
        Trac=self.TraceBack()
        i,j,k=0,0,0
        # i is the index of Trace seq, j is the index of output line, k is the number of segments in state B
        list=[] 

        Index=[]
        if(Trac[0]=='A'):
        # If the first state is A, the state will change by ABABA...
            Index.append('state A')
            Index.append('state B')
        else:
        # If the first state is B, the state will change by BABAB...
            Index.append('state B')
            Index.append('state A')
            k += 1

        list.append([1,0,Index[0]])
        while i<len(Trac)-1:
            if(Trac[i]!=Trac[i+1]):
                list[j][1]=i+1 # Update the last index of current state
                list.append([i+2,0,Index[(j+1)%2]])
                if(Trac[i+1]=='B'): k += 1 # count the number of state B
                j+=1
            i+=1
        list[j][1]=len(Trac) # Update the last one
        print('There are ',k,' segments of the genome are in state B')
        print(*list,sep='\n')
        return list
    
    def Print2TXT(self):
    # The output part 2
        list=self.Counter()
        output = open('HMM\data.txt','w',encoding='gbk')
        for row in list:
	        rowtxt = '{},{},{}'.format(row[0],row[1],row[2])
	        output.write(rowtxt)
	        output.write('\n')
        output.close()


if __name__=="__main__":
    obj=HMM()
    obj.Counter()
    #obj.Print2TXT()
    #Set ur path to file