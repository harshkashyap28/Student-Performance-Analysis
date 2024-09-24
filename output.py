import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())



print(df.isnull().sum())
print(df.dtypes)

print(df['math score'].unique())
print(df['reading score'].unique())
print(df['writing score'].unique())


# Fill missing numeric values with the mean of the column
df['math score'] = df['math score'].fillna(df['math score'].mean())
df['reading score'] = df['reading score'].fillna(df['reading score'].mean())
df['writing score'] = df['writing score'].fillna(df['writing score'].mean())

# For categorical columns, fill missing values with the mode
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
df['race/ethnicity'] = df['race/ethnicity'].fillna(df['race/ethnicity'].mode()[0])
df['parental level of education'] = df['parental level of education'].fillna(df['parental level of education'].mode()[0])
df['lunch'] = df['lunch'].fillna(df['lunch'].mode()[0])
df['test preparation course'] = df['test preparation course'].fillna(df['test preparation course'].mode()[0])


# Replace any unexpected string values or outliers that do not make sense
df['math score'] = pd.to_numeric(df['math score'], errors='coerce')
df['math score'] = df['math score'].fillna(df['math score'].mean())

# Similar replacements for reading and writing scores
df['reading score'] = pd.to_numeric(df['reading score'], errors='coerce')
df['reading score'] = df['reading score'].fillna(df['reading score'].mean())

df['writing score'] = pd.to_numeric(df['writing score'], errors='coerce')
df['writing score'] = df['writing score'].fillna(df['writing score'].mean())


# Adding a new feature for average score
df['average score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)


# One-hot encoding for categorical features
df = pd.get_dummies(df, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])


from sklearn.model_selection import train_test_split

# Assuming 'average score' is the target variable
X = df.drop('average score', axis=1)
y = df['average score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()
model

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score}")

import matplotlib.pyplot as plt
import seaborn as sns

# Plot histograms of scores
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['math score'], bins=20, kde=True, color='blue')
plt.title('Math Score Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['reading score'], bins=20, kde=True, color='green')
plt.title('Reading Score Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['writing score'], bins=20, kde=True, color='red')
plt.title('Writing Score Distribution')

plt.tight_layout()
plt.show()


# Calculate correlation matrix
corr = df[['math score', 'reading score', 'writing score', 'average score']].corr()

# Plot heatmap
plt.figure(figsize=(8, 3))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


import joblib

# Save the trained model to a file
joblib.dump(model, 'student_performance_model.pkl')








