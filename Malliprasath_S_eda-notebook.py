{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# E-Commerce Data Analysis - EDA\n",
    "## By [FirstName LastName]\n",
    "\n",
    "This notebook contains exploratory data analysis of the e-commerce dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline\n",
    "\n",
    "# Set style for better visualizations\n",
    "plt.style.use('seaborn')\n",
    "sns.set_palette('Set2')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Initial Inspection"
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
    "## 2. Customer Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Customer demographics\n",
    "print(\"Customer distribution by region:\")\n",
    "customers_df['Region'].value_counts().plot(kind='bar')\n",
    "plt.title('Customer Distribution by Region')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Product Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Product category analysis\n",
    "category_stats = products_df.groupby('Category').agg({\n",
    "    'ProductID': 'count',\n",
    "    'Price': ['mean', 'min', 'max']\n",
    "}).round(2)\n",
    "print(\"Product category statistics:\")\n",
    "display(category_stats)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Transaction Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Transaction trends\n",
    "monthly_sales = transactions_df.groupby(\n",
    "    transactions_df['TransactionDate'].dt.to_period('M')\n",
    ")['TotalValue'].sum()\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "monthly_sales.plot()\n",
    "plt.title('Monthly Sales Trend')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Customer Behavior Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Customer purchase patterns\n",
    "customer_stats = transactions_df.groupby('CustomerID').agg({\n",
    "    'TransactionID': 'count',\n",
    "    'TotalValue': 'sum',\n",
    "    'Quantity': 'sum'\n",
    "}).rename(columns={\n",
    "    'TransactionID': 'total_transactions',\n",
    "    'TotalValue': 'total_spent',\n",
    "    'Quantity': 'total_items'\n",
    "})\n",
    "\n",
    "print(\"Customer purchase statistics:\")\n",
    "display(customer_stats.describe())"
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
