#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import seaborn as sns; sns.set()


# In[3]:


from sklearn.datasets import make_blobs


# In[4]:


X, Y =make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)


# In[6]:


plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="autumn")
plt.show()


# In[9]:


xx=np.linspace(-1,3.5)
plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="autumn")
plt.plot([0.5], [2.1], "x", color= "blue", markeredgewidth=2, markersize=10)
for a,b in [(1,0.65), (0.5, 1.6), (-0.2, 2.9)]:
    yy=a*xx+b
    plt.plot(xx,yy, "-k")

plt.xlim(-1,3.5)    
plt.show()


# # Maximizacion del MArgen

# In[12]:


xx=np.linspace(-1,3.5)
plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="autumn")
plt.plot([0.5], [2.1], "x", color= "blue", markeredgewidth=2, markersize=10)
for a,b,d in [(1,0.65,0.33), (0.5, 1.6,0.55), (-0.2, 2.9, 0.2)]:
    yy=a*xx+b
    plt.plot(xx,yy, "-k")
    plt.fill_between(xx,yy-d,yy+d, edgecolor="none", color="#BBBBBB", alpha=0.5)

plt.xlim(-1,3.5)    
plt.show()


# # Creación del SVM

# In[13]:


from sklearn.svm import SVC


# In[14]:


model = SVC(kernel="linear", C=1E10)
model.fit(X,Y)


# In[34]:


def plt_svc(model, ax=None, plot_support = True):
    """Plot de la función de decisión para una clasificación en 2D con SVC"""
    if ax is None:
        ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    ## Generamos la parrilla para evaluar el modelo
    xx=np.linspace(xlim[0], xlim[1], 30)
    yy=np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(yy,xx)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    ## Representamos las fronteras y los margenes del SVM
    
    ax.contour(X,Y,P, colors="k", levels=[-1,0,1], alpha=0.5, linestyles=["--", "-", "--"])
    
    
    print(model.support_vectors_)
    if plot_support:
        ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=300, linewidth = 1, facecolors="green")
        
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    


# In[ ]:





# In[35]:


plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap = "autumn")
plt_svc(model, plot_support=True)


# In[39]:


def plot_svm(N=10, ax=None):
    X,Y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.6)
    
    X=X[:N]
    Y=Y[:N]
    
    model = SVC(kernel = "linear", C=1E10)
    model.fit(X,Y)
    
    ax = ax or plt.gca()
    ax.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="summer")
    ax.set_xlim(-1,4)
    ax.set_ylim(-1,6)
    plt_svc(model,ax)
    


# In[40]:


fig, ax = plt.subplots(1,2, figsize=(16,6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for ax_i, N in zip(ax, [60,120]):
    plot_svm(N,ax_i)
    ax_i.set_title("N={0}".format(N))


# In[41]:


from ipywidgets import interact, fixed


# In[48]:


interact(plot_svm, N=[10,100,200], ax=fixed(None))


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




