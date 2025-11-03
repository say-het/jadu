export default function handler(req, res) {
  res.send(`

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("badminton_dataset.csv")
df
for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col])
x = df.drop("Play_Badminton", axis=1)
y = df["Play_Badminton"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# Bernoulli Naive Bayes (q^x)(p)
b = BernoulliNB()
b.fit(x_train, y_train)
y_pred_b = b.predict(x_test)
bnb = accuracy_score(y_test, y_pred_b)
print("BernoulliNB Accuracy: ", bnb)

# Multinomial Naive Bayes
m = MultinomialNB()
m.fit(x_train, y_train)
y_pred_m = m.predict(x_test)
mnb = accuracy_score(y_test, y_pred_m)
print("MultinomialNB Accuracy: ", mnb)

# Gaussian Naive Bayes
g = GaussianNB()
g.fit(x_train, y_train)
y_pred_g = g.predict(x_test)
gnb = accuracy_score(y_test, y_pred_g)
print("GaussianNB Accuracy: ", gnb)

outlook = input('Enter Outlook {"Overcast", "Sunny", "Rain"} : ')
temp = input('Enter Temperature {"Cool", "Mild", "Hot"} : ')
humidity = input('Enter humidity {"High", "Normal"} : ')
wind = input('Enter Wind {"Weak", "Strong"} : ')

data = pd.DataFrame([[outlook, temp, humidity, wind]], columns=["Outlook", "Temperature", "Humidity", "Wind"])

for col in data:
    data[col] = LabelEncoder().fit_transform(data[col])
    
pred_bnb = b.predict(data)
pred_prob = b.predict_proba(data)

print("BernoulliNB Prediction : ", "Play" if pred_bnb[0]==1 else "Don't Play")
print("BernoulliNB Probabilities : ", pred_prob)

# Multinomial Naive Bayes without sklearn

mp = {}
for col in df.columns:
    val = df[col].unique()
    mp[col] = {j: i for i, j in enumerate(val)}
    df[col] = df[col].map(mp[col])


n = len(df)
cnt = df['Play_Badminton'].value_counts()
prob = {c: cnt[c]/n for c in cnt.index}

ft = [col for col in df.columns if col != 'Play_Badminton']
fprob = {}

for f in ft:
    fprob[f]={}
    for c in cnt.index:
        arr = df[df['Play_Badminton']==c]
        count = arr[f].value_counts()
        total = len(arr)
        probs = {}
        for val in df[f].unique():
            probs[val] = (count.get(val, 0) + 1) / (total + len(df[f].unique()))
        fprob[f][c] = probs
        
def predict(row):
    probs = {}
    for c in cnt.index:
        p = prob[c]
        for f in ft:
            p *= fprob[f][c][row[f]]
        probs[c] = p
    return max(probs, key=probs.get)

df['Predicted'] = df.apply(predict, axis=1)
accuracy = (df['Predicted'] == df['Play_Badminton']).mean()
print("Accuracy:", accuracy)

# ==============================================================
# 🤖 Multinomial Naive Bayes (Using sklearn)
# ==============================================================

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df2 = pd.DataFrame(data)

# Label encode each feature
le = LabelEncoder()
for col in df2.columns:
    df2[col] = le.fit_transform(df2[col])

# Split features and labels
X = df2.drop('Play_Badminton', axis=1)
y = df2['Play_Badminton']

# Train Model
model = MultinomialNB()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Accuracy
acc = accuracy_score(y, y_pred)
print("\nSklearn MultinomialNB Accuracy:", round(acc * 100, 2), "%")

# Show comparison
decoded = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
print("\nPredictions:\n", decoded)


`);
}
