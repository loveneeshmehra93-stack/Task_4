import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

artifact_dir = "C:/Users/mkaus/.gemini/antigravity/brain/8092b149-ef07-4397-aebe-bf749b4e7730"

# 1. Generate Synthetic Retail Data
np.random.seed(42)
n_samples = 1000

dates = pd.date_range(start="2023-01-01", periods=n_samples)
store_ids = np.random.randint(1, 11, size=n_samples)
categories = np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Toys'], size=n_samples)
prices = np.random.uniform(10, 500, size=n_samples)
discounts = np.random.uniform(0, 0.5, size=n_samples)
ad_spend = np.random.uniform(50, 5000, size=n_samples)
holiday_flags = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])

# Base Sales Volume
sales_volume = np.random.poisson(lam=50, size=n_samples)
# Add relationships
sales_volume = sales_volume + (ad_spend / 100) + (holiday_flags * 30) - (prices / 10) + (discounts * 50)
sales_volume = np.maximum(sales_volume, 1) # Ensure positive

revenue = prices * (1 - discounts) * sales_volume

df = pd.DataFrame({
    'Date': dates,
    'Store_ID': store_ids,
    'Category': categories,
    'Price': prices,
    'Discount': discounts,
    'Ad_Spend': ad_spend,
    'Holiday_Flag': holiday_flags,
    'Sales_Volume': sales_volume,
    'Revenue': revenue
})

# Introduce some dirty data for cleaning
df.loc[10:20, 'Revenue'] = np.nan
df.loc[50:55, 'Price'] = -50  # Outlier/Error
df = pd.concat([df, df.iloc[0:5]], ignore_index=True) # Duplicates

# 2. Data Cleaning
# Remove duplicates
df_clean = df.drop_duplicates()

# Handle missing values (impute with median)
df_clean['Revenue'] = df_clean['Revenue'].fillna(df_clean['Revenue'].median())

# Handle outliers (negative prices)
df_clean = df_clean[df_clean['Price'] > 0]

# 3. Exploratory Data Analysis & Visualizations
sns.set_theme(style="whitegrid")

# Histogram of Revenue
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['Revenue'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Revenue')
plt.xlabel('Revenue ($)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(artifact_dir, 'revenue_histogram.png'), bbox_inches='tight')
plt.close()

# Boxplot by Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Revenue', data=df_clean, palette='Set2', hue='Category', legend=False)
plt.title('Revenue by Product Category')
plt.xlabel('Category')
plt.ylabel('Revenue ($)')
plt.savefig(os.path.join(artifact_dir, 'revenue_boxplot.png'), bbox_inches='tight')
plt.close()

# Scatterplot Ad Spend vs Revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Ad_Spend', y='Revenue', hue='Holiday_Flag', alpha=0.6, palette='coolwarm', data=df_clean)
plt.title('Advertising Spend vs. Revenue')
plt.xlabel('Advertising Spend ($)')
plt.ylabel('Revenue ($)')
plt.savefig(os.path.join(artifact_dir, 'adspend_scatter.png'), bbox_inches='tight')
plt.close()

# Correlation Heatmap
numeric_df = df_clean.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig(os.path.join(artifact_dir, 'correlation_heatmap.png'), bbox_inches='tight')
plt.close()

# 4. Predictive Modeling
features = ['Price', 'Discount', 'Ad_Spend', 'Holiday_Flag']
X = df_clean[features]
y = df_clean['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with open(os.path.join(artifact_dir, 'model_metrics.txt'), 'w') as f:
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"R2: {r2:.2f}\n")

# Summary stats
stats = df_clean.describe()
stats.to_csv(os.path.join(artifact_dir, 'summary_stats.csv'))

print("Analysis complete. Visualizations and metrics saved.")
