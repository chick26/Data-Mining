import numpy as np

file=open('HMM/example.fa')
Data0=file.readlines()[1:]
InputData=(''.join(Data0)).replace('\n','')

i,j=0,0
list=[]
Index=[]
if(InputData[0]>='a' and InputData[0]<='z'):
    Index.append('state A')
    Index.append('state B')
else:
    Index.append('state B')
    Index.append('state A')
list.append([1,0,Index[0]])
while i<len(InputData)-1:
    if(((InputData[i]>='A'and InputData[i]<='Z')and (InputData[i+1]>='a' and InputData[i+1]<='z'))\
        or ((InputData[i+1]>='A'and InputData[i+1]<='Z')and (InputData[i]>='a' and InputData[i]<='z'))):
        list[j][1]=i+1
        list.append([i+2,0,Index[(j+1)%2]])
        j+=1
    i+=1
list[j][1]=len(InputData)
output = open('HMM\data_fa.txt','w',encoding='gbk')
for row in list:
	rowtxt = '{},{},{}'.format(row[0],row[1],row[2])
	output.write(rowtxt)
	output.write('\n')
output.close()
