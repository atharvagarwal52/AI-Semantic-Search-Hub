import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
data ={
    'StudyH' : [1,2,3,4,5],
    'TestS': [40,50,60,70,80]
}
df=pd.DataFrame(data)
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)
print('Standard Scaler output')
print(pd.DataFrame(standard_scaled, columns=['StudyH' , 'TestS']))


minmax= MinMaxScaler()
minmaxscaled= minmax.fit_transform(df)
print('\nMin max scaled output')
print(pd.DataFrame(minmaxscaled,columns=['StudyH' , 'TestS']))
'''
#test and train data
x= df[['StudyH']]
y=df[['TestS']]
x_test , x_train , y_train , y_test = train_test_split(x,y ,test_size=0.2 ,random_state=42)
print(x_test)
print(x_train)'''