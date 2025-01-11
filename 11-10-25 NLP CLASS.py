#!/usr/bin/env python
# coding: utf-8

# In[11]:


from nltk.tokenize import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()
input_string = "This is an example sent The Sentence."
all_sentences = tokenizer.tokenize(input_string)
print(all_sentences)


# In[13]:


#demo on re 
import re #regular expressions
text = "Hello, world! 123"
text = re.sub(r'[^a-z\s]','',text)
print(text)


# # demo of nltk (nlp+ ml)

# In[1]:


import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = [
    ['spam', 'Had your mobiles or more? u r entitiled to update to the '],
    ['ham' "i'm gonna be home soon and i don't wang to talk about this stuff any "],
    ['spam', 'congrstulations you have won $30000'],
    ['ham', 'are we still meeting in the room!! let me know']
]

#convert data into two lists: labels and sms content

labels = [row[0] for row in data]
messages = [row[1] for row in data]

#nlp preprocessing function

def prepocess_text(text):
    """ 
    prepocess the text by:
    -lowercasing
    -removing special charcters
    -tokenization
    -removing stop  words"""
    
    text= text.lower()
    
    text = re.sub(r'[^a-z\s]','',text)
    
    words = text.split()
    
    
    from nltk.corpus import stopwords
    import nltk
    nltk.downwords('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

preprocessed_messages = [prepocess_text(msg) for msg in messages]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(preprocessed_messages)
y = label

X_train, X_text, y_train, y_text = train_test_split(
    X,y, test_size=0.5, random_state = 42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


new_sms = 'congrts! yow won free ticket to xyz'
new_sms_prepocessed = preprocess_text(new_sms)
new_sms_vectorized = vectorize.transform([new_sms_preprocessed])
prediction = model.predict(new_sms_vectorized)
print(f"Prediction for new sms: {prediction[0]}")


# In[2]:



import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk

# Sample data
data = [
    ['spam', 'Had your mobiles or more? u r entitiled to update to the '],
    ['ham', "i'm gonna be home soon and i don't wang to talk about this stuff any "],
    ['spam', 'congrstulations you have won $30000'],
    ['ham', 'are we still meeting in the room!! let me know']
]

# Convert data into two lists: labels and sms content
labels = [row[0] for row in data]
messages = [row[1] for row in data]

# NLP preprocessing function
def preprocess_text(text):
    """ 
    Preprocess the text by:
    - Lowercasing
    - Removing special characters
    - Tokenization
    - Removing stop words
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    words = text.split()
    
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

# Preprocess the messages
preprocessed_messages = [preprocess_text(msg) for msg in messages]

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_messages)
y = labels

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Output performance metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict on a new SMS
new_sms = 'congrts! you won free ticket to xyz'
new_sms_processed = preprocess_text(new_sms)
new_sms_vectorized = vectorizer.transform([new_sms_processed])
prediction = model.predict(new_sms_vectorized)

print(f"Prediction for new sms: {prediction[0]}")


# In[ ]:




