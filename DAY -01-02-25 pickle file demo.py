#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load the California Housing dataset
california = fetch_california_housing()


# In[2]:


california.data.shape


# In[3]:


california.data[:5]


# In[4]:


california.feature_names


# In[5]:


california.target_names


# In[6]:


df = pd.DataFrame(california.data)


# In[7]:


df.columns = california.feature_names


# In[8]:


df['MedHouseVal'] = california.target 


# In[9]:


df 


# In[10]:


X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.DataFrame(california.target, columns=["Median_House_Value"])


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[13]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[14]:


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train.values.ravel())


# In[15]:


y_train.values.shape


# In[16]:


y_pred_rf = rf_model.predict(X_test_scaled)


# In[17]:


print("Random Forest Regression Metrics:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf)}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf)}")


# In[18]:


with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)


# In[19]:


import tensorflow as tf
from tensorflow import keras

# Build the model
dl_model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)  # Output layer for regression
])


# In[20]:


dl_model.compile(optimizer="adam", loss="mse", metrics=["mae"])


# In[21]:


dl_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, batch_size=00)


# In[22]:


dl_model.save("deep_learning_model.h5")


# In[23]:


with open("random_forest_model.pkl", "rb") as f:
    loaded_rf_model = pickle.load(f)


y_pred_loaded_rf = loaded_rf_model.predict(X_test_scaled)
print(f"Loaded RF Model R2 Score: {r2_score(y_test, y_pred_loaded_rf)}")


# In[27]:


loaded_dl_model = keras.models.load_model("deep_learning_model.h5")


y_pred_loaded_dl = loaded_dl_model.predict(X_test_scaled)
print(f"Loaded DL Model MSE: {r2_score(y_test, y_pred_loaded_dl)}")


# In[ ]:




