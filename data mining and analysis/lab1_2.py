from sklearn import preprocessing
import scipy
import numpy as np

data = np.loadtxt("iris.txt", delimiter=",", usecols=(0, 1, 2, 3), dtype=float)

m = np.dot(data, data.T)
m_k = m*m
print("核矩阵:\n",m_k)
mean = np.mean(m_k)
c_k = m_k-mean
print("中心化:\n",c_k)
n_k = preprocessing.scale(m_k)  # Numpy矩阵预处理类库
print("归一化:\n",n_k)




