# -*- coding:UTF-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Data Import
df = pd.read_excel('bank.xlsx', 'bank')


# Histogram
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['age'], bins=30)
plt.title('Age distribution')
plt.xlabel('Age')
plt.show()

# Boxplot
fig = plt.figure()
bx = fig.add_subplot(111)
bx.boxplot(df['balance'])
plt.ylabel("Balance")
plt.xlabel("BoxPlot")
plt.show() 


# Scartterplot
list_age=df.age.tolist()
list_y=df.y.tolist()
list_balance=df.balance.tolist()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('Age')
plt.ylabel('Balance')
for i in range(len(list_age)):
    if list_y[i]=='yes':
        ax.scatter(list_age[i], list_balance[i],color='blue',s= 60,linewidths=0.5,marker='x',label='yes')
    if list_y[i]=='no':
        ax.scatter(list_age[i], list_balance[i],color='red',s= 60,linewidths=0.5,marker='x',label='no')

ax.set_title('Scatter Plot')
plt.show()

# 折线图
var = df.groupby('BMI').Sales.sum()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('BMI')
ax.set_ylabel('Sum of Sales')
ax.set_title('BMI wise Sum of Sales')
var.plot(kind='line')
plt.show()



