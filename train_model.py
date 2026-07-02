import pandas as pd

# Load dataset
df = pd.read_csv("dataset/flood.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display shape
print("\nDataset Shape:")
print(df.shape)
# Descriptive Statistics
print("\nDescriptive Statistics")
print(df.describe())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Correlation Heatmap
# ----------------------------
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=False, cmap="Blues")
plt.title("Correlation Heatmap")
#plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Convert FloodProbability to Classification (0 or 1)
df["Flood"] = (df["FloodProbability"] >= 0.5).astype(int)

# Features and Target
X = df.drop(["FloodProbability", "Flood"], axis=1)
y = df["Flood"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing Completed Successfully")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)

# Train Model
dt_model.fit(X_train, y_train)

# Prediction
y_pred = dt_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:", accuracy)
from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)

# Train Model
rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)
from sklearn.neighbors import KNeighborsClassifier

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train Model
knn_model.fit(X_train, y_train)

# Prediction
knn_pred = knn_model.predict(X_test)

# Accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy:", knn_accuracy)
from xgboost import XGBClassifier

# XGBoost Model
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

# Train Model
xgb_model.fit(X_train, y_train)

# Prediction
xgb_pred = xgb_model.predict(X_test)

# Accuracy
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\nXGBoost Accuracy:", xgb_accuracy)   
import joblib

# Save Best Model
joblib.dump(xgb_model, "model/floods.save")
joblib.dump(scaler, "model/scaler.save")     
print("\nBest Model Saved Successfully!")   
# Accuracy Comparison Graph
models = ["Decision Tree", "Random Forest", "KNN", "XGBoost"]
accuracies = [accuracy, rf_accuracy, knn_accuracy, xgb_accuracy]

plt.figure(figsize=(8,5))
plt.bar(models, accuracies)

plt.title("Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.savefig("static/images/accuracy_comparison.png")
plt.close()

print("Accuracy graph saved successfully!")       
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, xgb_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.savefig("static/images/confusion_matrix.png")

plt.close()

print("Confusion Matrix Saved Successfully!")                                      