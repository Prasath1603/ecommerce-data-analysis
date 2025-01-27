{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# E-Commerce Customer Segmentation\n",
    "## By [FirstName LastName]\n",
    "\n",
    "This notebook implements customer segmentation using clustering techniques."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import required libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import davies_bouldin_score, silhouette_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# Set random seed for reproducibility\n",
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load datasets\n",
    "customers_df = pd.read_csv('Customers.csv')\n",
    "products_df = pd.read_csv('Products.csv')\n",
    "transactions_df = pd.read_csv('Transactions.csv')\n",
    "\n",
    "# Convert date columns\n",
    "customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])\n",
    "transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "def create_customer_features(customers_df, transactions_df, products_df):\n",
    "    # Create customer feature matrix\n",
    "    customer_features = customers_df.copy()\n",
    "    \n",
    "    # Calculate customer lifetime\n",
    "    customer_features['account_age'] = (pd.Timestamp.now() - \n",
    "                                       customer_features['SignupDate']).dt.days\n",
    "    \n",
    "    # Transaction-based features\n",
    "    transaction_features = transactions_df.groupby('CustomerID').agg({\n",
    "        'TransactionID': 'count',\n",
    "        'TotalValue': ['sum', 'mean', 'std'],\n",
    "        'Quantity': ['sum', 'mean']\n",
    "    }).round(2)\n",
    "    \n",
    "    # Flatten column names\n",
    "    transaction_features.columns = ['_'.join(col).strip() for col in \n",
    "                                   transaction_features.columns.values]\n",
    "    \n",
    "    # Calculate recency\n",
    "    last_transaction = transactions_df.groupby('CustomerID')['TransactionDate'].max()\n",
    "    customer_features['recency'] = (pd.Timestamp.now() - \n",
    "                                   last_transaction).dt.days\n",
    "    \n",
    "    # Calculate category preferences\n",
    "    category_data = transactions_df.merge(products_df, on='ProductID')\n",
    "    category_preferences = category_data.groupby(['CustomerID', 'Category'])['Quantity'].sum()\n",
    "    category_preferences = category_preferences.unstack(fill_value=0)\n",
    "    \n",
    "    # Combine all features\n",
    "    customer_features = customer_features.join(transaction_features, on='CustomerID')\n",
    "    customer_features = customer_features.join(category_preferences, on='CustomerID')\n",
    "    \n",
    "    return customer_features\n",
    "\n",
    "# Create feature matrix\n",
    "customer_features = create_customer_features(customers_df, transactions_df, products_df)\n",
    "\n",
    "# Select numerical features for clustering\n",
    "numerical_features = customer_features.select_dtypes(include=['float64', 'int64']).columns\n",
    "X = customer_features[numerical_features].fillna(0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Scale features\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "X_scaled = pd.DataFrame(X_scaled, columns=X.columns)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Determine Optimal Number of Clusters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "def evaluate_clusters(X, max_clusters=10):\n",
    "    db_scores = []\n",
    "    silhouette_scores = []\n",
    "    inertias = []\n",
    "    \n",
    "    for n_clusters in range(2, max_clusters + 1):\n",
    "        kmeans = KMeans(n_clusters=n_clusters, random_state=42)\n",
    "        labels = kmeans.fit_predict(X)\n",
    "        \n",
    "        db_scores.append(davies_bouldin_score(X, labels))\n",
    "        silhouette_scores.append(silhouette_score(X, labels))\n",
    "        inertias.append(kmeans.inertia_)\n",
    "    \n",
    "    # Plot evaluation metrics\n",
    "    fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "    \n",
    "    # Davies-Bouldin Index\n",
    "    axes[0].plot(range(2, max_clusters + 1), db_scores)\n",
    "    axes[0].set_title('Davies-Bouldin Index')\n",
    "    axes[0].set_xlabel('Number of Clusters')\n",
    "    \n",
    "    # Silhouette Score\n",
    "    axes[1].plot(range(2, max_clusters + 1), silhouette_scores)\n",
    "    axes[1].set_title('Silhouette Score')\n",
    "    axes[1].set_xlabel('Number of Clusters')\n",
    "    \n",
    "    # Elbow Plot\n",
    "    axes[2].plot(range(2, max_clusters + 1), inertias)\n",
    "    axes[2].set_title('Elbow Plot')\n",
    "    axes[2].set_xlabel('Number of Clusters')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    return db_scores, silhouette_scores, inertias\n",
    "\n",
    "# Evaluate different numbers of clusters\n",
    "db_scores, silhouette_scores, inertias = evaluate_clusters(X_scaled)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Final Clustering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Perform final clustering with optimal number of clusters\n",
    "n_clusters = 5  # Based on evaluation metrics\n",
    "final_kmeans = KMeans(n_clusters=n_clusters, random_state=42)\n",
    "customer_features['Cluster'] = final_kmeans.fit_predict(X_scaled)\n",
    "\n",
    "# Calculate final Davies-Bouldin Index\n",
    "final_db_score = davies_bouldin_score(X_scaled, customer_features['Cluster'])\n",
    "print(f'Final Davies-Bouldin Index: {final_db_score:.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Cluster Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Perform PCA for visualization\n",
    "pca = PCA(n_components=2)\n",
    "X_pca = pca.fit_transform(X_scaled)\n",
    "\n",
    "# Create visualization\n",
    "plt.figure(figsize=(10, 6))\n",
    "scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], \n",
    "                     c=customer_features['Cluster'], \n",
    "                     cmap='viridis')\n",
    "plt.title('Customer Segments Visualization')\n",
    "plt.xlabel('First Principal Component')\n",
    "plt.ylabel('Second Principal Component')\n",
    "plt.colorbar(scatter)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Cluster Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Analyze cluster characteristics\n",
    "cluster_analysis = customer_features.groupby('Cluster').agg({\n",
    "    'TransactionID_count': 'mean',\n",
    "    'TotalValue_sum': 'mean',\n",
    "    'TotalValue_mean': 'mean',\n",
    "    'Quantity_sum': 'mean',\n",
    "    'recency': 'mean',\n",
    "    'account_age': 'mean'\n",
    "}).round(2)\n",
    "\n",
    "print(\"\\nCluster Characteristics:\")\n",
    "display(cluster_analysis)\n",
    "\n",
    "# Calculate cluster sizes\n",
    "cluster_sizes = customer_features['Cluster'].value_counts().sort_index()\n",
    "print(\"\\nCluster Sizes:\")\n",
    "print(cluster_sizes)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
