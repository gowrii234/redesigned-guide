import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


df = pd.read_csv('covid_vaccine_data.csv')
print(df.head())



print(df.isnull().sum())

 df = df.dropna() or df.fillna()

sns.countplot(df['Vaccinated'])

corr = df.corr()
sns.heatmap(corr)


X = df.drop('Vaccinated', axis=1)
y = df['Vaccinated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
