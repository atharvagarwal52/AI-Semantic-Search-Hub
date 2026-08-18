import numpy as np
np1=np.full((3,8),8)
print(np1)
list = [1,43,423,43,3 ]
print(np.array(list))
np2=np.array([[1,2,3,4,5,0],[6,7,8,9,10,9]])
for x in np2:
    for y in x:
        print(y)
print(np2[0:2 , 1:3])
print("hello" , np1)
print(np2.reshape(3,2,2))
print(np2.reshape(-1))