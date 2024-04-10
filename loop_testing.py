import numpy as np

size = 7
for i in range (0,size):
    print(i)
    if i == size-1:
        print('last val',i)

x = np.array([[1,2],[3,4]])

print(x)
print(x[0][:])

ydims = (2,2)
y = np.zeros(ydims)

print(y)
print(y[:][0])