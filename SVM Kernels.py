#!/usr/bin/env python
# coding: utf-8

# # Identificar fronteras no lineales

# In[32]:


from sklearn.datasets import make_circles, make_blobs


# In[5]:


X, Y = make_circles(100, factor = .1, noise=.1)


# In[7]:


import matplotlib.pyplot as plt
import numpy as np


# In[13]:


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
    
    if plot_support:
        ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=300, linewidth = 1, facecolors="green")
        
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    


# In[11]:


from sklearn.svm import SVC


# In[9]:


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
    


# In[10]:


plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="autumn")


# In[16]:


plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="autumn")
plt_svc(SVC(kernel="linear").fit(X,Y), plot_support=False )


# In[17]:


r=np.exp(-(X**2).sum(1))


# In[18]:


r


# In[19]:


from mpl_toolkits import mplot3d


# In[23]:


def plot_3D(elev=30, azim=30, X=X, Y=Y, r=r):
    ax=plt.subplot(projection="3d")
    ax.scatter3D(X[:,0], X[:,1], r, c=Y, s=50, cmap="autumn")
    ax.view_init(elev = elev, azim=azim)
    
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    


# In[21]:


from ipywidgets import interact, fixed


# In[26]:


interact(plot_3D, elev=[-90,-60-30,0,30,60,90], azim=[-180,-150,-120,-90,-90,-60-30,0,30,60,90,120,150,180], X=fixed(X), Y= fixed(Y), r=fixed(r))


# # Aplicando el kernel RBF(radial basis function)

# In[27]:


rbf = SVC(kernel="rbf", C=1E6)
rbf.fit(X, Y)


# In[29]:


plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="autumn")
plt_svc(rbf)
plt.scatter(rbf.support_vectors_[:,0], rbf.support_vectors_[:,1], s=300, lw=1, facecolors="none")


# # Ajustar los parámetros del sVM

# In[33]:


X,Y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)


# In[35]:


plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="winter")


# In[36]:


X,Y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)


# In[37]:


plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="winter")


# In[41]:


fig, ax = plt.subplots(1,2, figsize=(16,6))
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

for ax_i, C in zip(ax, [1E6, 0.1]):
    model=SVC(kernel="linear", C=C)
    model.fit(X,Y)
    ax_i.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="winter")
    plt_svc(model, ax_i)
    ax_i.set_title("C={0:.1f}".format(C), size=15)


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




