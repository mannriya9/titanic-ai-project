import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Titanic-Dataset.csv")
print("Dataset loaded! Shape:", df.shape)

# Clean the data (no warnings version)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna("S")
df = df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict a new passenger [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
new_passenger = pd.DataFrame([[3, 0, 22, 1, 0, 7.25, 0]], 
               columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"])
result = model.predict(new_passenger)
print(f"Survived: {'Yes 🟢' if result[0] == 1 else 'No 🔴'}")