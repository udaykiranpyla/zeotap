#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Example: Paths to the files (replace these with your actual file paths)
customers_file = r"C:\Users\uday\Desktop\zeptap\Customers.csv"
transactions_file = r"C:\Users\uday\Desktop\zeptap\Transactions.csv"

# Load the datasets
customers_df = pd.read_csv(customers_file)
transactions_df = pd.read_csv(transactions_file)

# Convert the SignupDate in customers_df to datetime
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'], errors='coerce')

# Convert the TransactionDate in transactions_df to datetime
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'], errors='coerce')

# Display the first few rows to verify the conversion
print("Customers Dataset:")
print(customers_df.head())

print("\nTransactions Dataset:")
print(transactions_df.head())


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers_file = r"C:\Users\uday\Desktop\zeptap\Customers.csv"
products_file = r"C:\Users\uday\Desktop\zeptap\Products.csv"
transactions_file = r"C:\Users\uday\Desktop\zeptap\Transactions.csv"

customers_df = pd.read_csv(customers_file)
products_df = pd.read_csv(products_file)
transactions_df = pd.read_csv(transactions_file)

# Convert dates to datetime for consistency (optional)
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'], errors='coerce')
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'], errors='coerce')

# Univariate Analysis for Customers Dataset
print("Customer Regions Distribution:")
print(customers_df['Region'].value_counts())
plt.figure(figsize=(8, 5))
sns.countplot(x='Region', data=customers_df, palette='viridis')
plt.title("Region Distribution")
plt.xticks(rotation=45)
plt.show()

# Univariate Analysis for Products Dataset
print("\nProduct Categories Distribution:")
print(products_df['Category'].value_counts())
plt.figure(figsize=(8, 5))
sns.countplot(x='Category', data=products_df, palette='muted')
plt.title("Category Distribution")
plt.xticks(rotation=45)
plt.show()

# Univariate Analysis for Price in Products
print("\nPrice Statistics in Products Dataset:")
print(products_df['Price'].describe())
plt.figure(figsize=(8, 5))
sns.histplot(products_df['Price'], kde=True, bins=20, color='blue')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Univariate Analysis for Transactions Dataset
print("\nTransaction Quantities Distribution:")
plt.figure(figsize=(8, 5))
sns.histplot(transactions_df['Quantity'], kde=False, bins=10, color='green')
plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

print("\nTotal Value Statistics in Transactions Dataset:")
print(transactions_df['TotalValue'].describe())
plt.figure(figsize=(8, 5))
sns.boxplot(y='TotalValue', data=transactions_df, palette='coolwarm')
plt.title("Total Value Distribution")
plt.ylabel("Total Value")
plt.show()


# In[6]:


import pandas as pd

# Load datasets (update file paths as needed)
customers_file = r"C:\Users\uday\Desktop\zeptap\Customers.csv"
products_file = r"C:\Users\uday\Desktop\zeptap\Products.csv"
transactions_file = r"C:\Users\uday\Desktop\zeptap\Transactions.csv"

customers_df = pd.read_csv(customers_file)
products_df = pd.read_csv(products_file)
transactions_df = pd.read_csv(transactions_file)

# Convert date columns to datetime format (optional but recommended for consistency)
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'], errors='coerce')
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'], errors='coerce')

# 1. Verify if TotalValue is consistent with Price * Quantity
transactions_df['ComputedTotal'] = transactions_df['Price'] * transactions_df['Quantity']
inconsistent_rows = transactions_df[transactions_df['ComputedTotal'] != transactions_df['TotalValue']]

print(f"Number of inconsistent TotalValue rows: {len(inconsistent_rows)}")
if len(inconsistent_rows) > 0:
    print("Sample of inconsistent rows:")
    print(inconsistent_rows.head())

# 2. Check if all CustomerIDs in Transactions exist in Customers
missing_customers = set(transactions_df['CustomerID']) - set(customers_df['CustomerID'])
print(f"Number of missing CustomerIDs: {len(missing_customers)}")
if missing_customers:
    print(f"Missing CustomerIDs: {missing_customers}")

# 3. Check if all ProductIDs in Transactions exist in Products
missing_products = set(transactions_df['ProductID']) - set(products_df['ProductID'])
print(f"Number of missing ProductIDs: {len(missing_products)}")
if missing_products:
    print(f"Missing ProductIDs: {missing_products}")

# Optional: Summary of the datasets for context
print("\n--- Customers Dataset ---")
print(customers_df.info())
print(customers_df.head())

print("\n--- Products Dataset ---")
print(products_df.info())
print(products_df.head())

print("\n--- Transactions Dataset ---")
print(transactions_df.info())
print(transactions_df.head())


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets (update file paths as needed)
customers_file =  r"C:\Users\uday\Desktop\zeptap\Customers.csv"
products_file = r"C:\Users\uday\Desktop\zeptap\Products.csv"
transactions_file =  r"C:\Users\uday\Desktop\zeptap\Transactions.csv"

customers_df = pd.read_csv(customers_file)
products_df = pd.read_csv(products_file)
transactions_df = pd.read_csv(transactions_file)

# Convert dates to datetime format (optional)
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'], errors='coerce')
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'], errors='coerce')

# Top Customers by Transaction Volume
top_customers = (
    transactions_df.groupby('CustomerID')['TotalValue']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
top_customers = pd.merge(top_customers, customers_df[['CustomerID', 'CustomerName']], on='CustomerID')
print("\nTop 5 Customers by Transaction Volume:")
print(top_customers.head(5))

# Most Popular Products
popular_products = (
    transactions_df.groupby('ProductID')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
popular_products = pd.merge(popular_products, products_df[['ProductID', 'ProductName']], on='ProductID')
print("\nTop 5 Most Popular Products by Quantity Sold:")
print(popular_products.head(5))

# Most Popular Categories
merged_transactions = pd.merge(transactions_df, products_df, on='ProductID')
popular_categories = (
    merged_transactions.groupby('Category')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
print("\nMost Popular Categories by Quantity Sold:")
print(popular_categories)

# Revenue Analysis by Region
merged_customers = pd.merge(transactions_df, customers_df, on='CustomerID')
revenue_by_region = (
    merged_customers.groupby('Region')['TotalValue']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
print("\nRevenue by Region:")
print(revenue_by_region)
# Plot: Top 5 Customers by Transaction Volume
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_customers['TotalValue'].head(5),
    y=top_customers['CustomerName'].head(5),
    palette='viridis'
)
plt.title("Top 5 Customers by Transaction Volume")
plt.xlabel("Transaction Volume")
plt.ylabel("Customer Name")
plt.show()

# Plot: Top 5 Most Popular Products
plt.figure(figsize=(10, 6))
sns.barplot(
    x=popular_products['Quantity'].head(5),
    y=popular_products['ProductName'].head(5),
    palette='muted'
)
plt.title("Top 5 Most Popular Products by Quantity Sold")
plt.xlabel("Quantity Sold")
plt.ylabel("Product Name")
plt.show()

# Plot: Most Popular Categories
plt.figure(figsize=(10, 6))
sns.barplot(
    x=popular_categories['Quantity'],
    y=popular_categories['Category'],
    palette='coolwarm'
)
plt.title("Most Popular Categories by Quantity Sold")
plt.xlabel("Quantity Sold")
plt.ylabel("Category")
plt.show()

# Plot: Revenue by Region
plt.figure(figsize=(10, 6))
sns.barplot(
    x=revenue_by_region['TotalValue'],
    y=revenue_by_region['Region'],
    palette='plasma'
)
plt.title("Revenue by Region")
plt.xlabel("Total Revenue")
plt.ylabel("Region")
plt.show()



# In[15]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the datasets
customers_file = r"C:\Users\uday\Desktop\zeptap\Customers.csv"
products_file = r"C:\Users\uday\Desktop\zeptap\Products.csv"
transactions_file =  r"C:\Users\uday\Desktop\zeptap\Transactions.csv"

customers_df = pd.read_csv(customers_file)
products_df = pd.read_csv(products_file)
transactions_df = pd.read_csv(transactions_file)

# Convert date columns to datetime
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

# Merge datasets
merged_df = transactions_df.merge(customers_df, on='CustomerID').merge(products_df, on='ProductID')

# Feature engineering
merged_df['ProductPreferences'] = merged_df.groupby('CustomerID')['Category'].transform(lambda x: ' '.join(x))
merged_df['AverageSpending'] = merged_df.groupby('CustomerID')['TotalValue'].transform('mean')
merged_df['TotalTransactions'] = merged_df.groupby('CustomerID')['TransactionID'].transform('count')

# Select relevant features
customer_profiles = merged_df[['CustomerID', 'Region', 'ProductPreferences', 'AverageSpending', 'TotalTransactions']].drop_duplicates()

# Encoding categorical variables
tfidf = TfidfVectorizer()
region_encoded = tfidf.fit_transform(customer_profiles['Region'])
product_preferences_encoded = tfidf.fit_transform(customer_profiles['ProductPreferences'])

# Normalize numeric features
customer_profiles['AverageSpending'] = (customer_profiles['AverageSpending'] - customer_profiles['AverageSpending'].mean()) / customer_profiles['AverageSpending'].std()
customer_profiles['TotalTransactions'] = (customer_profiles['TotalTransactions'] - customer_profiles['TotalTransactions'].mean()) / customer_profiles['TotalTransactions'].std()

# Combine all features
features = np.hstack([
    region_encoded.toarray(),
    product_preferences_encoded.toarray(),
    customer_profiles[['AverageSpending', 'TotalTransactions']].values
])

# Compute pairwise similarity
similarity_matrix = cosine_similarity(features)

# Generate recommendations for the first 20 customers
customer_ids = customer_profiles['CustomerID'].values
lookalike_map = {}

for idx, cust_id in enumerate(customer_ids[:20]):  # Only for the first 20 customers
    similarities = similarity_matrix[idx]
    similar_indices = np.argsort(similarities)[::-1][1:4]  # Top 3 similar customers
    similar_customers = [(customer_ids[i], similarities[i]) for i in similar_indices]
    lookalike_map[cust_id] = similar_customers

# Save lookalike map to CSV
lookalike_df = pd.DataFrame({
    "CustomerID": lookalike_map.keys(),
    "Lookalikes": [str(v) for v in lookalike_map.values()]
})
lookalike_df.to_csv("Lookalike.csv", index=False)

# Evaluate model logic for the first customer
sample_customer_id = customer_ids[0]  # First customer (C0001)
sample_customer_features = features[0]  # Features for C0001
similarities_for_sample = similarity_matrix[0]
top_similar_indices = np.argsort(similarities_for_sample)[::-1][1:4]  # Top 3 similar customers
top_similar_customers = [(customer_ids[i], similarities_for_sample[i]) for i in top_similar_indices]

# Retrieve sample customer profile and top similar profiles
sample_customer_profile = customer_profiles[customer_profiles['CustomerID'] == sample_customer_id]
top_similar_profiles = customer_profiles.iloc[top_similar_indices]

# Calculate precision based on a shared attribute (e.g., Region)
common_region_count = sum(sample_customer_profile['Region'].values[0] == top_similar_profiles['Region'].values)
precision_on_region = common_region_count / len(top_similar_customers)

# Print evaluation results
print("Sample Customer ID:", sample_customer_id)
print("Sample Customer Profile:\n", sample_customer_profile)
print("\nTop 3 Lookalike Recommendations (CustomerID, Similarity Score):")
for cust_id, score in top_similar_customers:
    print(f"Customer ID: {cust_id}, Similarity Score: {score:.2f}")

print("\nPrecision on Region:", precision_on_region)


# In[18]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the environment variable to avoid memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Load the datasets
customers_file =  r"C:\Users\uday\Desktop\zeptap\Customers.csv"
transactions_file =  r"C:\Users\uday\Desktop\zeptap\Transactions.csv"

customers_df = pd.read_csv(customers_file)
transactions_df = pd.read_csv(transactions_file)

# Convert date columns to datetime
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

# Step 1: Data Preparation
# Aggregate transaction information at the customer level
transactions_summary = transactions_df.groupby('CustomerID').agg({
    'TotalValue': ['sum', 'mean'],
    'TransactionID': 'count'
}).reset_index()
transactions_summary.columns = ['CustomerID', 'TotalSpending', 'AverageSpending', 'TotalTransactions']

# Merge with customer profile data
customer_data = customers_df.merge(transactions_summary, on='CustomerID', how='left')
customer_data.fillna(0, inplace=True)  # Handle customers without transactions

# Add feature: Time since signup (relative to the latest date in the dataset)
latest_date = customers_df['SignupDate'].max()
customer_data['TimeSinceSignup'] = (latest_date - customer_data['SignupDate']).dt.days

# Select relevant features for clustering
features = customer_data[['Region', 'TotalSpending', 'AverageSpending', 'TotalTransactions', 'TimeSinceSignup']]

# Encode categorical features (Region)
features = pd.get_dummies(features, columns=['Region'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 2: Perform clustering with KMeans (2 to 10 clusters)
db_scores = []  # To store Davies-Bouldin scores
silhouette_scores = []  # To store silhouette scores
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)  # Explicitly set n_init=10
    cluster_labels = kmeans.fit_predict(scaled_features)
    db_scores.append(davies_bouldin_score(scaled_features, cluster_labels))
    silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))

# Step 3: Identify the optimal number of clusters based on DB Index
optimal_clusters = cluster_range[np.argmin(db_scores)]

# Perform clustering with the optimal number of clusters
final_kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)  # Explicitly set n_init=10
customer_data['Cluster'] = final_kmeans.fit_predict(scaled_features)

# Step 4: Reduce dimensions for visualization (PCA)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)
customer_data['PCA1'] = reduced_features[:, 0]
customer_data['PCA2'] = reduced_features[:, 1]

# Step 5: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=customer_data['PCA1'],
    y=customer_data['PCA2'],
    hue=customer_data['Cluster'],
    palette='tab10',
    style=customer_data['Cluster'],
    s=50
)
plt.title('Customer Clusters (PCA Visualization)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()

# Output clustering metrics
print("Clustering Results:")
print("Number of Clusters:", optimal_clusters)
print("Davies-Bouldin Index:", min(db_scores))
print("Silhouette Score:", silhouette_scores[np.argmin(db_scores)])


# In[ ]:




