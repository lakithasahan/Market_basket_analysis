# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn import preprocessing
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import re
from sklearn.preprocessing import MinMaxScaler
import networkx as nx 

G = nx.Graph()
pd.set_option('display.width',600)
pd.set_option('display.max_columns',10)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
        
    

def plot_bar_x(data):
    data=data.head(30)
    # this is for plotting purpose
    index = np.arange(len(data.index))
    plt.bar(index, data['NAME'])
    plt.xlabel('Genre', fontsize=5)
    plt.ylabel('No of Movies', fontsize=5)
    plt.xticks(index, data.index, fontsize=5, rotation=90)
    plt.title('Market Share for Each Genre 1995-2017')
    plt.show()


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    if x=='nan':
        return 0


le = preprocessing.LabelEncoder()
df=pd.read_csv('/kaggle/input/market-basket/Market_basket_df.csv')
#print(df[df['DATENEW']=='2018-01-02'].index)
#print(df[df['DATENEW']=='2019-01-02'].index)

df=df.iloc[353810:530528,:]
print(df)

#df=df.dropna()

df['NAME'] = df['NAME'].str.strip()
df['RECEIPT'] = df['RECEIPT'].astype('str')
df['UNITS'] = df['UNITS'].astype('int64')


print(df)

item_frequency=df['NAME'].value_counts()
item_frequency=pd.DataFrame(item_frequency)
print(item_frequency)
plot_bar_x(item_frequency)

basket = (df
          .groupby(['RECEIPT', 'NAME'])['UNITS']
          .sum().unstack().reset_index().fillna(0)
          .set_index('RECEIPT'))


print(basket.head(1000))

basket_sets = basket.applymap(encode_units)
print(basket_sets)

frequent_itemsets = apriori(basket_sets, min_support=0.004, use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)


rules=pd.DataFrame(rules)

for x in range(len(rules)):
    result1=re.findall(r"\w+ \w+|\w+",str(rules['antecedents'][x]))
    result2=re.findall(r"\w+ \w+|\w+",str(rules['consequents'][x]))
    rules.at[x, 'antecedents'] = result1[1]
    rules.at[x, 'consequents'] = result2[1]



A=rules['antecedents'].tolist()
B=rules['consequents'].tolist()
details=[]
print(A[0])
for x in range(len(A)):
    txt=(A[x],B[x])
    details.append(txt)
    
    

lift_size= [i * 10000 for i in rules['lift'].tolist()]
edge_width = [i * 40 for i in rules['confidence'].tolist()] 
node_color=[i * 40 for i in rules['support'].tolist()] 
#node_colors = range(0, 42)
scaler =MinMaxScaler(feature_range=(0, 13))
node_color=scaler.fit_transform((np.array(node_color)).reshape(-1,1))
node_color=(np.around(node_color.reshape(-1))).tolist()

#print(edge_width)
#node_color=my_new_list = [i * 2000 for i in rules['lift'].tolist()]
plt.figure(2,figsize=(50,50))
G.add_edges_from(details) 
nx.draw_networkx(G, with_label = True,node_size = lift_size, width=edge_width,edge_color ='.4',cmap=plt.cm.Blues,font_weight='bold',font_size=25) 
#plt.axis('off')
plt.legend('Lift', 'Confidence')
plt.show()  

      
    
    

fig = go.Figure(data=[go.Scatter(
    x=rules['support'].tolist(),
    y=rules['confidence'].tolist(),
    mode='markers',
    text=details,
    marker=dict(
        color=rules['lift'].tolist(),
        size=np.arange(0,len(lift_size)),
        showscale=True,
       
        )
)])

fig.show()
