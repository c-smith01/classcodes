import numpy as np

arr = np.genfromtxt(r'C:\Users\c-smith\Documents\uid_gnpwdr.txt',dtype='str',delimiter=',')
print(arr[0])

# Open the text file for reading
with open(r'C:\Users\c-smith\Documents\uid_gnpwdr.txt', 'r') as file:
    # Read the content of the file
    content = file.read()

# Split the content by spaces to get individual entries
entries = content.split()

# Print the entries as arrays
for entry in entries:
    # Convert each entry to an array
    entry_array = entry.split()
    #print(entry_array)


# Lookup spectral data
#print(entry_array)