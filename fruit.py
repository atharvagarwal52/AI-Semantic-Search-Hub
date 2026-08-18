from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
x=[ [180,7] , [200,7.5] , [250 , 8] , [300,8.5] , [330,9] , [360,9.5]]

model = KNeighborsClassifier(n_neighbors=3)
#0 for apple and 1 for orange
y= [0,0,0,1,1,1]
model.fit(x,y)
weight = float(input('enter weight'))
size = float(input('enter size'))
prediction=model.predict([[weight , size]]) [0] # 0 so it gives either 0 or 1
if(prediction==0):
    print("apple")
else:
    print("orange")

#decision making
model1= DecisionTreeClassifier()
z=[ [6,6] , [9,2] ,[ 7,2] , [8,5] ,[9,8] , [10,9]]
model1.fit(z,y)
shade=int(input('Enter shade from 1-10'))
hel=model1.predict([[size, shade]])[0]
if hel==0:
    print('APPLE')
else:
    print('Orange')