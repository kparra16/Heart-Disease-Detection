#Numpy

import numpy as np

#Converting a list into a np array

list_int = [8, 3, 34, 111]
a_int = np.array(list_int)
a_int

print(a_int.ndim)   #Rank 
print(a_int.shape)  # Shape 

print(a_int.dtype)  #Types of each element of the array

#For matrix

list_2dim = [[1.5,2,3], [4,5,6]]
a_2dim = np.array(list_2dim)
a_2dim

print(a_2dim.ndim)   #Rank 
print(a_2dim.shape)  # Shape 

#Mixing tuples and lists

a_mix = np.array([[1, 2.0], [0, 0],(5.78, 3.)])
a_mix

print(a_mix.ndim)   #Rank 
print(a_mix.shape)  # Shape 

a_intfloat = np.array([[1, 2], [3, 4]], dtype = float)
a_intfloat

print(a_intfloat.ndim)   #Rank 
print(a_intfloat.shape)  # Shape 

#Creating numpy arranges

#The function arange() returns arrays with regularly incrementing values.

np.arange(10)

np.arange(2, 10, dtype=float)

np.arange(0, 10, 0.3)     # start, end (exclusive), step=0.3

np.linspace(0.4, 1.1, 7)    # create an array of 7 elements from 0.4 to 1.1 (inclusive)

np.random.randn(4)  # 1-dimensional array of 4 normally distributed numbers

array_zeros = np.zeros((4,7))

np.ones((2, 3, 4), dtype = np.int16)

np.empty((3, 5))

#Cast type of nparray

arr_int = np.array([1, 2, 3, 4, 5])
arr_int.dtype
arr_float = arr_int.astype(np.float32)
arr_float.dtype

#astype() method can be used to convert numbers represented as strings to numeric form`

arr_str = np.array(['3.14','4.56','7.89', '32'])
arr_str
arr_str.astype(np.float)


#Multiply in numpy with matrixes is element by element

# Multiply arrays

array1 = np.array([[5, 4, 6, 1], [2, 3, 8, 10]])




array2 = np.array([[10, 12, 5, 17], [22, 33, 88, 100]])
array1 * array2

# Divide arrays - it's element by element

array1 / array2

#MCD - Maximum Common Divisor

array1 % array2

#You can use also using NumPy functions: np.add(), np.subtract(), np.multiply(), np.divide() and np.remainder()

#Transpose

x1 = np.arange(4)        # create an array of integers from 0 to 3
x2 = x1.reshape(4,1)     # reshape() function allows us to change the shape of the array
                         # the resulting array will be a vector, 4 rows and 1 column
x2

a_3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
a_3d[1,1][0]
a_3d[1,1,0]   #Access to the inner element

a_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a_2d)

#access to rows

a_2d[:2]
a_2d[:2,:]

#access to columns
a_2d[:,:2]

#access to certain elements
#Always close to left and open to right

a_2d[:1, 1:]
a_2d[:2, 2:]

a_2d[2, :1]

a_2d[:, :2]