#!/usr/bin/env python
# coding: utf-8

# # SVM ocn Regresión

# In[7]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[6]:


X=np.sort(5*np.random.rand(200,1), axis=0)
Y=np.sin(X).ravel()
Y[::5] += 3*(0.5 - np.random.rand(40))


# In[8]:


plt.scatter(X,Y, color="darkorange")


# In[9]:


from sklearn.svm import SVR


# In[14]:


C=1e3
svr_lin=SVR(kernel="linear", C=C)
svr_rad=SVR(kernel="rbf", C=C, gamma=0.1)
svr_pol=SVR(kernel="poly", C=C,degree=3)


# In[15]:


y_lin=svr_lin.fit(X,Y).predict(X)
y_rad=svr_rad.fit(X,Y).predict(X)
y_pol=svr_pol.fit(X,Y).predict(X)


# In[17]:


lw=1
plt.figure(figsize=(16,9))
plt.scatter(X,Y, color="darkorange", label="data")
plt.scatter(X,y_lin, color="navy",lw=lw, label="SVM Lineal")
plt.scatter(X,y_rad, color="c",lw=lw, label="SVM Radial")
plt.scatter(X,y_pol, color="cornflowerblue",lw=lw, label="SVM Polinomial")
plt.xlabel("x")
plt.ylabel("y")
plt.title("SVR")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




