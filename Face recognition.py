#!/usr/bin/env python
# coding: utf-8

# # Reconocimeinto Facial

# In[6]:


from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt


# In[3]:


faces=fetch_lfw_people(min_faces_per_person=60)


# In[4]:


print(faces.target_names)


# In[5]:


print(faces.images.shape)


# In[11]:


fig, ax=plt.subplots(5,5, figsize=(16,9))
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(faces.images[i], cmap="bone")
    ax_i.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])


# In[22]:


from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline,Pipeline


# In[28]:


pca=RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc=SVC(kernel="rbf")#, class_weight="balanced")#
model=make_pipeline(pca, svc)


# In[17]:


from sklearn.model_selection import train_test_split 


# In[18]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(faces.data, faces.target, random_state=42)


# In[19]:


from sklearn.model_selection import GridSearchCV


# In[29]:


param_grid={
    "svc__C":[0.1,1,5,10,50],
    "svc__gamma":[0.0001,0.0005,0.001,0.005,0.01],
    
}
grid=GridSearchCV(model, param_grid)

get_ipython().run_line_magic('time', 'grid.fit(Xtrain, Ytrain)')


# In[30]:


print(grid.best_params_)


# In[31]:


classifier=grid.best_estimator_
yfit=classifier.predict(Xtest)


# In[32]:


Xtest.shape


# In[37]:


fig, ax=plt.subplots(8,6, figsize=(16,9))
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(Xtest[i].reshape(62,47), cmap="bone")
    ax_i.set(xticks=[], yticks=[])
    ax_i.set_ylabel(faces.target_names[yfit[i]].split()[-1], 
                   color="black" if yfit[i]==Ytest[i] else "red")
    
fig.suptitle("Predicciones de las imágenes (incorrectas en rojo)", size=15)


# In[38]:


from sklearn.metrics import classification_report


# In[40]:


print(classification_report(Ytest, yfit, target_names=faces.target_names))


# In[41]:


from sklearn.metrics import confusion_matrix


# In[42]:


mat=confusion_matrix(Ytest, yfit)


# In[43]:


import seaborn as sns; sns.set()


# In[46]:


sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=True, xticklabels=faces.target_names, yticklabels=faces.target_names)


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




