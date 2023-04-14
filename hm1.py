from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,scale


# 讀取資料
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
data['BMI']=data['Weight']/(data['Height']/100)**2

#data = pd.get_dummies(df, columns=['Gender'])

print(data.head())
#train = data.sample(frac=0.8, random_state=1)
#test = data.drop(train.index)
# 取出需要用到的特徵和類別
labelEncoder = LabelEncoder()
data_le = pd.DataFrame(data)
data_le['Gender'] = labelEncoder.fit_transform(data_le['Gender'])
#print(data.head())
X = data[['Gender','Height', 'Weight','BMI']]#Feature 訓練資料得屬性
y = data['Index']# label 答案


# 使用StandardScaler進行正規化
scaler = StandardScaler()
X = StandardScaler().fit_transform(X)
#X=scale(X)


# 將資料切分為訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# 使用StandardScaler進行正規化
#scaler = StandardScaler()
#X_train = StandardScaler().fit_transform(X_train)
#X_test = StandardScaler().fit_transform(X_test)


# 繪製訓練和測試數據的散點圖

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(X_train[:, 1], X_train[:, 2], c=y_train)
ax1.set_title('Training data')
ax1.set_xlabel('Height')
ax1.set_ylabel('Weight')

ax2.scatter(X_test[:, 1], X_test[:, 2], c=y_test)
ax2.set_title('Test data')
ax2.set_xlabel('Height')
ax2.set_ylabel('Weight')

plt.show()



# 建立模型
#lr = LinearRegression()
logistic = LogisticRegression()
svm = SVC()
dt = DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=10,splitter='best',min_samples_leaf=10)
rf = RandomForestClassifier(criterion='gini',n_estimators=5,n_jobs=-1,random_state=42,min_samples_leaf=10)
mlp = MLPClassifier()




# 訓練模型
#lr.fit(X_train, y_train)
logistic.fit(X_train, y_train)
svm.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# 使用訓練好的模型進行預測
#lr_pred = lr.predict(X_test)
logistic_pred = logistic.predict(X_test)
svm_pred = svm.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
mlp_pred = mlp.predict(X_test)

# 計算預測準確度、precision和recall
#lr_acc = accuracy_score(y_test, lr_pred)
#lr_pre = precision_score(y_test, lr_pred, average=None)
#lr_rec = recall_score(y_test, lr_pred, average=None)

logistic_acc = accuracy_score(y_test, logistic_pred)
logistic_pre = precision_score(y_test, logistic_pred, average=None)
logistic_rec = recall_score(y_test, logistic_pred, average=None)
print("name: Logistic Regression")
print("Accuracy:", logistic_acc)
print("Precision:", logistic_pre)
print("Recall:", logistic_rec)
print("-" * 100)  
svm_acc = accuracy_score(y_test, svm_pred)
svm_pre = precision_score(y_test, svm_pred, average=None)
svm_rec = recall_score(y_test, svm_pred, average=None)
print("name: svm")
print("Accuracy:", svm_acc)
print("Precision:", svm_pre)
print("Recall:", svm_rec)
print("-" * 100)  
dt_acc = accuracy_score(y_test, dt_pred)
dt_pre = precision_score(y_test, dt_pred, average=None)
dt_rec = recall_score(y_test, dt_pred, average=None)
print("name: Decision Tree")
print("Accuracy:", dt_acc)
print("Precision:", dt_pre)
print("Recall:", dt_rec)
print("-" * 100)  
rf_acc = accuracy_score(y_test, rf_pred)
rf_pre = precision_score(y_test, rf_pred, average=None)
rf_rec = recall_score(y_test, rf_pred, average=None)
print("name: Random Forest")
print("Accuracy:", rf_acc)
print("Precision:", rf_pre)
print("Recall:", rf_rec)
print("-" * 100)  
mlp_acc = accuracy_score(y_test, mlp_pred)
mlp_pre = precision_score(y_test, mlp_pred, average=None)
mlp_rec = recall_score(y_test, mlp_pred, average=None)
print("name: mlp")
print("Accuracy:", mlp_acc)
print("Precision:", mlp_pre)
print("Recall:", mlp_rec)
print("-" * 100)  
