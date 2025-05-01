# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
# FEATURE SCALING
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/aec35348-8404-4508-b73a-81b1d3eaf9f3)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/9bb0d7ff-c0d3-4f38-bd7d-3ceb073fe522)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/8768a442-4e0e-4a91-b9cf-562ece5e2f25)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
# This is typically used in feature scaling,
#particularly max-abs scaling, which is useful
#when you want to scale data to the range [-1, 1]
#while maintaining sparsity (often used with sparse data).
```
![438593868-ad078354-a7ec-4824-b193-abab49272d93](https://github.com/user-attachments/assets/899f6d94-71ce-46d2-a989-839014972e9b)
```
# Standard Scaling
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![438593987-c3997748-b604-4f9b-b61f-6d7a180dabab](https://github.com/user-attachments/assets/e755c3e4-db6d-4f57-9752-97ca32f45256)
```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![438594102-28fd338e-68a1-4892-80f8-e9f457fc1024](https://github.com/user-attachments/assets/f73abe4d-48c0-4e53-8f08-cb7dfdfc77dd)
```
#MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![438594243-1cb180c2-cb10-42ad-9daa-392b59cb8a5b](https://github.com/user-attachments/assets/3ffad5d4-abe4-4786-bc74-cb0e507830c5)
```
#MAXIMUM ABSOLUTE SCALING:

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![438594333-7f5672be-6108-45a7-95ec-b7bfb35ef661](https://github.com/user-attachments/assets/83587cbc-cd01-4a37-a447-6c586997c4ef)
```
#ROBUST SCALING

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
![438594437-b97e589b-33bf-4dab-80e1-3c000ce648ed](https://github.com/user-attachments/assets/a52d5b03-862b-485b-b52c-46d450f2ab68)
```
#FEATURE SELECTION:

df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![438594547-4279091f-4369-4d81-a1fd-f3371fc3fcdc](https://github.com/user-attachments/assets/92ff82ac-5e1c-4bc4-8fce-2d929ec05ba2)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![438594646-99195e69-2e38-46c3-a32c-050f6fdd408d](https://github.com/user-attachments/assets/5d3658ec-4e7c-474a-b195-9a8a4aad878a)
```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
 # process non-numeric features. It ensures the model treats these columns as categories,
  # not as continuous numerical values.
df[categorical_columns]
```
![438594739-1d878f9e-a8d1-4bb1-a6da-e0fde5ab799b](https://github.com/user-attachments/assets/881457f6-cec6-4487-8a8a-4ee2aa358eed)
```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```
![438594902-fa7c4e39-7ab5-4b1f-a0db-1b8a26a7ed71](https://github.com/user-attachments/assets/272601b0-b539-47df-bc06-49afe333adb2)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
#X contains all columns except 'SalStat' — these are the input features used to predict something.
#y contains only the 'SalStat' column — this is the target variable you want to predict.
```
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![438595091-61cd0112-10dd-4d15-a08b-547b27d8cfc8](https://github.com/user-attachments/assets/74100367-0b03-415b-a972-5dec3816420e)
```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![438595250-6627af6e-b26d-4cf8-a912-2886c50f95b2](https://github.com/user-attachments/assets/01defc81-eb11-4afd-aa41-2588de88fb48)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![438595353-bea4a1e0-e442-47ea-be3a-eebf0b090434](https://github.com/user-attachments/assets/cc044cb4-6717-4886-b7d8-02ea46937e6e)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![438595474-4060b42b-aee8-4257-b407-74810343cf1e](https://github.com/user-attachments/assets/69c01e6e-dc41-4a5f-b562-c952b8f07a21)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![438595565-a07e6b66-029c-41cd-a8d1-430bb40bf77c](https://github.com/user-attachments/assets/fe636d9d-1cbd-40ae-bd1f-33f9a1a2cf59)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![438595664-992d24b7-91ee-4dde-8983-7bc9bf28c112](https://github.com/user-attachments/assets/288cb117-9056-4462-83da-cf4e27ef11f2)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![438595763-c0783a7e-bf1b-4d8a-9bf3-b70a91d4503e](https://github.com/user-attachments/assets/03415276-abbe-4320-949f-dc2a54b03533)
```
!pip install skfeature-chappers
```
![438595871-b6371826-058d-4285-acdb-5ff9769562a0](https://github.com/user-attachments/assets/4cae151b-ad1e-44e6-97e1-0c8644c25379)
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![438596159-09b3c379-fee0-4cf2-850b-c3e4e59df005](https://github.com/user-attachments/assets/48cc84c1-562a-4c43-9ff9-3fdc5561d256)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```
```
selected_features_anova = X.columns[selector_anova.get_support()]
```
```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![438596358-f1facfde-5bf9-4645-a188-23c9e477c141](https://github.com/user-attachments/assets/240d994a-0a33-49c5-ad0b-99e3d2c4e0a4)
```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![438596497-4b2fce23-14fc-449a-9fc6-4f229d6d34d0](https://github.com/user-attachments/assets/1801e443-d9f0-432e-b7fa-6430bfc9df2a)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
logreg = LogisticRegression()
```
```
n_features_to_select =6
```
```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![438597046-cb365aa6-c949-4af8-9e77-6f937d679e69](https://github.com/user-attachments/assets/35c11f62-6a7f-4ac4-8b21-beb60f156de3)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset
