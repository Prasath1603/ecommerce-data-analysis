{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# E-Commerce Lookalike Model\n",
    "## By [FirstName LastName]\n",
    "\n",
    "This notebook implements a lookalike model for customer similarity analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from scipy.spatial.distance import cdist"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Feature Engineering"
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
    "    # Transaction-based features\n",
    "    transaction_features = transactions_df.groupby('CustomerID').agg({\n",
    "        'TransactionID': 'count',\n",
    "        'TotalValue': ['sum', 'mean'],\n",
    "        'Quantity': ['sum', 'mean']\n",
    "    }).flatten()\n",
    "    \n",
    "    # Product category preferences\n",
    "    category_preferences = transactions_df.merge(\n",
    "        products_df, on='ProductID'\n",
    "    ).groupby(['CustomerID', 'Category'])['Quantity'].sum().unstack(fill_value=0)\n",
    "    \n",
    "    # Combine features\n",
    "    customer_features = customer_features.join(transaction_features, on='CustomerID')\n",
    "    customer_features = customer_features.join(category_preferences, on='CustomerID')\n",
    "    \n",
    "    return customer_features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Similarity Calculation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "def find_lookalikes(customer_features, target_customer_id, n_recommendations=3):\n",
    "    # Prepare features\n",
    "    numeric_features = customer_features.select_dtypes(include=['float64', 'int64'])\n",
    "    scaler = StandardScaler()\n",
    "    scaled_features = scaler.fit_transform(numeric_features)\n",
    "    \n",
    "    # Calculate similarity\n",
    "    target_vector = scaled_features[customer_features.index == target_customer_id]\n",
    "    similarities = 1 / (1 + cdist(target_vector, scaled_features, metric='euclidean')[0])\n",
    "    \n",
    "    # Get top similar customers\n",
    "    similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]\n",
    "    similar_scores = similarities[similar_indices]\n",
    "    \n",
    "    return list(zip(customer_features.index[similar_indices], similar_scores))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Generate Recommendations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Generate lookalikes for first 20 customers\n",
    "lookalike_results = {}\n",
    "for cust_id in customers_df['CustomerID'][:20]:\n",
    "    lookalikes = find_lookalikes(customer_features, cust_id)\n",
    "    lookalike_results[cust_id] = [(cust, float(score)) for cust, score in lookalikes]\n",
    "\n",
    "# Save results\n",
    "lookalike_df = pd.DataFrame(\n",
    "    [(k, v) for k, v in lookalike_results.items()],\n",
    "    columns=['CustomerID', 'Lookalikes']\n",
    ")\n",
    "lookalike_df.to_csv('FirstName_LastName_Lookalike.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 }
}
