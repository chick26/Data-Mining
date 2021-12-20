from pyecharts import Parallel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
 
data = pd.read_csv('bank0.csv')
data_0 = np.array(data[['age','day','x']]).tolist()
data = pd.read_csv('bank1.csv')
data_1= np.array(data[['age','day','x']]).tolist()
schema = ['age', 'day','y']
 
parallel = Parallel('bank')
parallel.config(schema)
parallel.add('no',data_0,is_random = True,area_color='#b399ff')
parallel.add('yes',data_1,is_random = True,area_color='#006400')
parallel.render('Bank.html')