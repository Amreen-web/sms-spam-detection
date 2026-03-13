import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    'message': [
        'Congratulations you won a lottery',
        'Hi how are you',
        'Claim your free prize now',
        'Let us meet tomorrow',
        'Win cash now',
        'Are you coming to class'
    ],
    'label': ['spam','ham','spam','ham','spam','ham']
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

y = df['label']

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train,y_train)

# Test prediction
msg = ["Win free cash prize"]
msg_vec = vectorizer.transform(msg)

prediction = model.predict(msg_vec)

print("Message:",msg[0])
print("Prediction:",prediction[0])