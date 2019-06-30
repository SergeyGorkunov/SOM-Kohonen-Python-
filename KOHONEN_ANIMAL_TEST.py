import Kohonen_SOM 
import pandas as pd
import numpy as np

#Загрузим и нормируем данные о животных
data=pd.read_csv('Animal_data.csv')
X=data.drop(['name'], axis=1).values
X=(X-X.min(axis=0))/X.max(axis=0)

#Создадим карту с 6 классами
N_nodes=6
animal_SOM=Kohonen_SOM.SOM(N_nodes)

#Инициализируем карту случайными весами и выведим распределение животных до обучения
print('распределение животных по классам до обучения:')
animal_SOM.learn(X,  N_iter_max=0,  type_learn=1)
is_class=animal_SOM.separate(X)
nodes_before_learn=pd.DataFrame(np.zeros((data.shape[0],N_nodes)))
nodes_before_learn=nodes_before_learn.applymap(lambda x: "     ")
max_len=0
for i in range(N_nodes):
    ind=np.where(is_class==i)
    len_=len(ind[0])
    if max_len<len_:
        max_len=len_
    if len_>0:
        nodes_before_learn[i].iloc[:len_]=data.name.values[ind]
print (nodes_before_learn.head(max_len))

#Обучим карту и выведим распределение животных             
print('распределение животных по классам после обучения:')
animal_SOM.learn(X, N_iter_max=2000, type_learn=2)
is_class=animal_SOM.separate(X)
nodes_after_learn=pd.DataFrame(np.zeros((data.shape[0],N_nodes)))
nodes_after_learn=nodes_before_learn.applymap(lambda x: "     ")
max_len=0
for i in range(N_nodes):
    ind=np.where(is_class==i)
    len_=len(ind[0])
    if max_len<len_:
        max_len=len_
    if len_>0:
        nodes_after_learn[i].iloc[:len_]=data.name.values[ind]
print (nodes_after_learn.head(max_len))
        
