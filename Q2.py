import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

#Connect to the SQLite database and read the data
conn = sqlite3.connect('heart_disease.db')
df = pd.read_sql_query("SELECT * FROM heart_data", conn)
conn.close()

# Display the first few rows of the dataframe
print(df.head())

#Data Cleaning and Preprocessing
# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Fill missing values (if any)
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Standardize numerical variables
scaler = StandardScaler()
df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# Display the cleaned and preprocessed data
print("Cleaned and Preprocessed Data:\n", df.head())

#Plot the distribution of classes for categorical variables based on the target variable
target = 'target'  

# Select categorical columns excluding the target column
categorical_columns = df.select_dtypes(include=['int64']).columns.tolist()
if target in categorical_columns:
    categorical_columns.remove(target)

# Plot the distribution of classes for each categorical variable based on the target variable
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, hue=target)
    plt.title(f'Distribution of {column} based on {target}')
    plt.show()

#Plot the distribution of classes for numeric variables based on the target variable
numeric_columns = df.select_dtypes(include=['float64']).columns.tolist()

# Plot the distribution of classes for each numeric variable based on the target variable
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue=target, kde=True, element='step')
    plt.title(f'Distribution of {column} based on {target}')
    plt.show()
