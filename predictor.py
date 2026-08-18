import pandas as pd
from sklearn.linear_model import LinearRegression

# enter data

data={
    'StudyH': [2.0,3.0, 3.5, 2.5, 8.0, 9.0, 9.5] ,
    'marks' : [54,92,54,32,86,33,90]
}
df=pd.DataFrame(data)
x=df[['StudyH']]
y=df[['marks']]

print(df)
model = LinearRegression()
model.fit(x,y)
hours=float(input('Enter how many hours have you studied'))
predicted_score= model.predict([[hours]])
predicted_score= max(0,min(100,predicted_score))
print('Your predicted score for ' , hours , ' is = ' , predicted_score)