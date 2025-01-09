#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd


# In[30]:


df = pd.read_csv("day 6 09-01-25 logistic regression.csv")


# In[31]:


print(data.head(5))


# In[32]:


import matplotlib.pyplot as plt


# In[35]:


print(df.columns)


# In[37]:


df.columns = df.columns.str.strip()  # Remove leading and trailing spaces


# In[38]:


plt.figure(figsize=(5,4))
plt.scatter(df['Age'], df ['Purchased'],cmap ='bwr', label = 'Purchased')
plt.xlabel('Age')
plt.ylabel('Purchased (0 = no, 1 = Yes)')
plt.title('jai project')
plt.legend(['No purchase','Purchase'])
plt.grid(True)
plt.show()


# In[40]:


import matplotlib.pyplot as plt

# Ensure that 'Purchased' column has numeric data
print(df['Purchased'].unique())

# Create the scatter plot with 'Age' vs 'Purchased' and colormap
plt.figure(figsize=(5, 4))
plt.scatter(df['Age'], df['Purchased'], c=df['Purchased'], cmap='bwr', label='Purchased')
plt.xlabel('Age')
plt.ylabel('Purchased (0 = no, 1 = yes)')
plt.title('jai')
plt.legend()
plt.show()


# In[41]:


from sklearn.linear_model import LogisticRegression


# In[46]:


import warnings
warnings.filterwarnings("ignore")


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


#features and target variables
x=df[['Age']]
y=df[['Purchased']]

#split the data into
#x=features y= prediction
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)

#prediction
new_age = [[18]]
prediction = model.predict(new_age)
probability = model.predict_proba(new_age)
print(f"prediction for age {new_age[0][0]}: {'will purchase' if prediction[0]==1 else 'will not purchase'}")


# In[ ]:




