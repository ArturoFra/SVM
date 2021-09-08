#!/usr/bin/env python
# coding: utf-8

# # Linear Support Vector MAchine 

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm


# In[5]:


X=[1,5,1.5,8,1,9]
Y=[2,8,1.8,8,0.6,11]


# In[6]:


plt.scatter(X,Y)
plt.show()


# In[7]:


data = np.array(list(zip(X,Y)))


# In[8]:


data


# In[9]:


target = [0,1,0,1,0,1]


# In[12]:


classifier = svm.SVC(kernel="linear", C=1.0)
classifier.fit(data, target)


# In[14]:


p=np.array([10.57,20.67]).reshape(1,2)
print(p)
classifier.predict(p)


# In[15]:


w=classifier.coef_[0]
w


# In[16]:


a= -w[0]/w[1]
a


# In[17]:


b=-classifier.intercept_[0]/w[1]
b


# In[18]:


xx=np.linspace(0,10)
yy=a*xx+b


# In[19]:


plt.plot(xx,yy, "-k", label = "Hiperplano de separación")
plt.scatter(X,Y, c= target)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




