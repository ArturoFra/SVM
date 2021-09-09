#!/usr/bin/env python
# coding: utf-8

# # Clasificación de flores Iris

# In[4]:


import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


iris = datasets.load_iris()
print(iris)


# In[7]:


X=iris.data[:, :2]
Y=iris.target


# In[10]:


x_min, x_max= X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max= X[:, 1].min()-1, X[:, 1].max()+1
h=(x_max - x_min)/ 100

xx, yy =np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

x_plot= np.c_[xx.ravel(), yy.ravel()]


# In[11]:


x_plot


# In[15]:


C=1.0
svc = svm.SVC(kernel="linear", C=C, decision_function_shape="ovr").fit(X, Y)
Ypred=svc.predict(x_plot)
Ypred=Ypred.reshape(xx.shape)


# In[23]:


plt.figure(figsize=(16,9))
plt.contourf(xx,yy,Ypred,cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:, 1], c=Y, cmap=plt.cm.tab10)
plt.xlabel("Longitud d elos pétalos")
plt.ylabel("Anchura de los pétalos")
plt.xlim(xx.min(), xx.max())
plt.title("SVC para las flores iris con kernel lineal")


# In[27]:


C=1.0
svc = svm.SVC(kernel="rbf", C=C, decision_function_shape="ovr").fit(X, Y)
Ypred=svc.predict(x_plot)
Ypred=Ypred.reshape(xx.shape)


# In[30]:


plt.figure(figsize=(16,9))
plt.contourf(xx,yy,Ypred,cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:, 1], c=Y, cmap=plt.cm.tab10)
plt.xlabel("Longitud d elos pétalos")
plt.ylabel("Anchura de los pétalos")
plt.xlim(xx.min(), xx.max())
plt.title("SVC para las flores iris con kernel radial")


# In[31]:


C=1.0
svc = svm.SVC(kernel="sigmoid", C=C, decision_function_shape="ovr").fit(X, Y)
Ypred=svc.predict(x_plot)
Ypred=Ypred.reshape(xx.shape)


# In[32]:


plt.figure(figsize=(16,9))
plt.contourf(xx,yy,Ypred,cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:, 1], c=Y, cmap=plt.cm.tab10)
plt.xlabel("Longitud d elos pétalos")
plt.ylabel("Anchura de los pétalos")
plt.xlim(xx.min(), xx.max())
plt.title("SVC para las flores iris con kernel sigmoide")


# In[33]:


C=1.0
svc = svm.SVC(kernel="poly", C=C, decision_function_shape="ovr").fit(X, Y)
Ypred=svc.predict(x_plot)
Ypred=Ypred.reshape(xx.shape)


# In[34]:


plt.figure(figsize=(16,9))
plt.contourf(xx,yy,Ypred,cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:, 1], c=Y, cmap=plt.cm.tab10)
plt.xlabel("Longitud d elos pétalos")
plt.ylabel("Anchura de los pétalos")
plt.xlim(xx.min(), xx.max())
plt.title("SVC para las flores iris con kernel polinomial")


# In[35]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


# In[36]:


X, Y =shuffle(X,Y, random_state=0)


# In[37]:


Xtrain, Xtest, Ytrain, Ytest=train_test_split(X, Y, test_size=0.25, random_state=0)


# In[38]:


parameter=[{
    "kernel": ["rbf"],
    "gamma": [1e-4,1e-3,1e-2,0.1, 0.2,0.5],
    "C":[1,10,100,1000]
    },{
    "kernel": ["linear"],
    "C":[1,10,100,1000]
    }
]


# In[41]:


clf=GridSearchCV(svm.SVC(decision_function_shape="ovr"), param_grid=parameter, cv=5)


# In[42]:


clf.fit(X,Y)


# In[43]:


clf.best_params_


# In[48]:


means=clf.cv_results_["mean_test_score"]
stds=clf.cv_results_["std_test_score"]
params = clf.cv_results_["params"]
for m, s, p in zip(means, stds, params):
    print("%0.3f (+/- %0.3f) para %r" %(m, 2*s, p))
    


# In[49]:


y_pred=clf.predict(Xtest)


# In[50]:


print(classification_report(Ytest, y_pred, target_names=["setosas", "versicolor", "virginica"]))


# In[ ]:





# In[51]:


C=10
svc = svm.SVC(kernel="rbf", C=C,gamma=0.01, decision_function_shape="ovr").fit(X, Y)
Ypred=svc.predict(x_plot)
Ypred=Ypred.reshape(xx.shape)


# In[52]:


plt.figure(figsize=(16,9))
plt.contourf(xx,yy,Ypred,cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:, 1], c=Y, cmap=plt.cm.tab10)
plt.xlabel("Longitud d elos pétalos")
plt.ylabel("Anchura de los pétalos")
plt.xlim(xx.min(), xx.max())
plt.title("SVC para las flores iris con kernel rbf mejorado")


# In[ ]:





# # Resumen final de la clasificación de Iris 

# In[53]:


def svm_iris(C=1.0, gamma=0.01, kernel="rbf"):
    import pandas as pd
    import numpy as np
    from sklearn import svm, datasets
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    iris = datasets.load_iris()
    X=iris.data[:, :2]
    Y=iris.target
    x_min, x_max= X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max= X[:, 1].min()-1, X[:, 1].max()+1
    h=(x_max - x_min)/ 100

    xx, yy =np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    x_plot= np.c_[xx.ravel(), yy.ravel()]
    
    svc = svm.SVC(kernel=kernel, C=C,gamma=gamma, decision_function_shape="ovr").fit(X, Y)
    Ypred=svc.predict(x_plot)
    Ypred=Ypred.reshape(xx.shape)
    plt.figure(figsize=(16,9))
    plt.contourf(xx,yy,Ypred,cmap=plt.cm.tab10, alpha=0.3)
    plt.scatter(X[:,0], X[:, 1], c=Y, cmap=plt.cm.tab10)
    plt.xlabel("Longitud d elos pétalos")
    plt.ylabel("Anchura de los pétalos")
    plt.xlim(xx.min(), xx.max())
    plt.title("SVC para las flores iris con kernel "+ kernel)
    
    iris = datasets.load_iris()
    X=iris.data[:, :2]
    Y=iris.target
    x_min, x_max= X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max= X[:, 1].min()-1, X[:, 1].max()+1
    h=(x_max - x_min)/ 100

    xx, yy =np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    x_plot= np.c_[xx.ravel(), yy.ravel()]


# In[55]:


from ipywidgets import interact


# In[57]:


interact(svm_iris, C=[0.01, 0.1, 1, 10, 100, 1000, 1e6, 1e10], gamma=[1e-5,1e-4,1e-3,1e-2,0.1, 0.2,0.5,0.99], 
         kernel=["rbf", "linear", "sigmoid", "poly"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




