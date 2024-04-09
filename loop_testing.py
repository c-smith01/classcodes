import numpy as np

for i in range (0,5+2):
    print(i)

x = np.array([[1,2],[3,4]])

print(x)
print(x[0][:])

ydims = (2,2)
y = np.zeros(ydims)

print(y)
print(y[:][0])