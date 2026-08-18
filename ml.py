"""import pandas as pd
data ={
    'Name' : [ 'rahul ', 'geeta' , 'sangeeta', 'ramesh'],
    'Age' : [12,33,None ,None],
    'gender' : [ 'male', 'male', None, None]
    
}
df=pd.DataFrame(data)
print(df)
print(df.isnull().sum())

print (df.dropna())

df['Age'].fillna(df['Age'].mean(), inplace=True)
print(df)"""


