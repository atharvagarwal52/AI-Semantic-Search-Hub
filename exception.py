"""try:
    file=open('/Users/a/python/yhes.txt')
    content= file.read();
    print(content) 
except FileNotFoundError:
    print('file not found')
finally:
    print (" completed") """

def check_password(password):
    if(len(password)<8):
        raise Exception("error : password is weak")
    print("strong")
try:
    password=input('input password')
    check_password(password)
except Exception as e:
    print(e)