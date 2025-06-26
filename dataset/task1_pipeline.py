import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Extract - Load CSV
data = pd.read_csv('sample_data.csv')  # make sure this file is in your folder
print("Original Data:")
print(data.head())
# 2. Transform - Preprocess the data
# Fill missing values (if any)
data.fillna(method='ffill', inplace=True)

# Encode categorical values
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Feature Scaling
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Optional: Split dataset for ML use-case
X = scaled_data.drop(columns=[scaled_data.columns[-1]])
y = scaled_data[scaled_data.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 3. Load - Export cleaned data
scaled_data.to_csv('processed_data.csv', index=False)
print("✅ Process completed. Processed data saved as 'processed_data.csv'")
