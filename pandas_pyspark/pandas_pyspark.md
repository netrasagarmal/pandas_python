# Problem Statement: Banking Customer Deposits and Loan Management System

A bank wants to build a **transaction analytics and financial monitoring database** to track:

* Customer profiles
* Deposit accounts (savings/current)
* Loan accounts issued to customers
* Financial activities such as deposits, withdrawals, and loan repayments

The analytics team needs this system to:

* Monitor **customer deposits and withdrawals**
* Track **loan issuance and repayments**
* Identify **high-value customers**
* Detect **loan defaults or delayed payments**
* Generate **monthly revenue and risk analytics**

The database must support:

* Transactional operations (DML)
* Analytical SQL queries (window functions, aggregates)
* Complex joins and financial reporting

---

# Core Entities

The system consists of **4 core tables**:

1. Customers
2. Accounts (deposit accounts)
3. Loans
4. Transactions

This structure allows questions around:

* **One-to-many relationships**
* **Financial analytics**
* **Time-series transaction analysis**

---

# ER Diagram (Conceptual)

```
Customers
   |
   | 1
   |
   |------< Accounts
   |           |
   |           | 1
   |           |
   |           |------< Transactions
   |
   |
   |------< Loans
```

### Relationships

| Relationship           | Description                            |
| ---------------------- | -------------------------------------- |
| Customer → Accounts    | One customer can own multiple accounts |
| Account → Transactions | Each account has many transactions     |
| Customer → Loans       | Customer can have multiple loans       |

---

# Table 1: Customers

Stores customer personal and registration information.

### Columns

| Column      | Type         | Description           |
| ----------- | ------------ | --------------------- |
| customer_id | BIGINT       | Primary key           |
| first_name  | VARCHAR(50)  | Customer first name   |
| last_name   | VARCHAR(50)  | Customer last name    |
| email       | VARCHAR(120) | Unique customer email |
| phone       | VARCHAR(20)  | Contact number        |
| city        | VARCHAR(50)  | Customer city         |
| country     | VARCHAR(50)  | Country               |
| created_at  | DATETIME     | Registration date     |

### SQL

```sql
CREATE TABLE Customers (
    customer_id BIGINT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(120) UNIQUE,
    phone VARCHAR(20),
    city VARCHAR(50),
    country VARCHAR(50),
    created_at DATETIME
);
```

### Concepts Covered

* Primary key
* Unique constraint
* String data types
* Datetime type

---

# Table 2: Accounts

Represents customer deposit accounts (savings/current).

### Columns

| Column       | Type        | Description             |
| ------------ | ----------- | ----------------------- |
| account_id   | BIGINT      | Primary key             |
| customer_id  | BIGINT      | Foreign key             |
| account_type | VARCHAR(20) | Savings/Current         |
| balance      | DOUBLE      | Current account balance |
| opened_date  | DATETIME    | Account opening date    |
| status       | VARCHAR(20) | Active/Closed           |

### SQL

```sql
CREATE TABLE Accounts (
    account_id BIGINT PRIMARY KEY,
    customer_id BIGINT,
    account_type VARCHAR(20),
    balance DOUBLE,
    opened_date DATETIME,
    status VARCHAR(20),
    
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);
```

### Concepts Covered

* Foreign keys
* Financial numeric fields
* One-to-many relationship

---

# Table 3: Loans

Stores loan accounts issued to customers.

### Columns

| Column        | Type        | Description           |
| ------------- | ----------- | --------------------- |
| loan_id       | BIGINT      | Primary key           |
| customer_id   | BIGINT      | Loan owner            |
| loan_amount   | DOUBLE      | Total loan amount     |
| interest_rate | DOUBLE      | Interest percentage   |
| loan_type     | VARCHAR(50) | Home/Car/Personal     |
| issued_date   | DATETIME    | Loan issue date       |
| loan_status   | VARCHAR(20) | Active/Closed/Default |

### SQL

```sql
CREATE TABLE Loans (
    loan_id BIGINT PRIMARY KEY,
    customer_id BIGINT,
    loan_amount DOUBLE,
    interest_rate DOUBLE,
    loan_type VARCHAR(50),
    issued_date DATETIME,
    loan_status VARCHAR(20),
    
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);
```

### Concepts Covered

* Financial metrics
* Loan analytics
* Time-based financial records

---

# Table 4: Transactions

Tracks deposits, withdrawals, transfers, and loan repayments.

### Columns

| Column           | Type         | Description                            |
| ---------------- | ------------ | -------------------------------------- |
| transaction_id   | BIGINT       | Primary key                            |
| account_id       | BIGINT       | Linked account                         |
| transaction_type | VARCHAR(30)  | Deposit/Withdraw/Transfer/Loan_Payment |
| amount           | DOUBLE       | Transaction amount                     |
| transaction_date | DATETIME     | Transaction timestamp                  |
| description      | VARCHAR(255) | Transaction description                |

### SQL

```sql
CREATE TABLE Transactions (
    transaction_id BIGINT PRIMARY KEY,
    account_id BIGINT,
    transaction_type VARCHAR(30),
    amount DOUBLE,
    transaction_date DATETIME,
    description VARCHAR(255),
    
    FOREIGN KEY (account_id) REFERENCES Accounts(account_id)
);
```

### Concepts Covered

* Transaction history
* Time-series analysis
* Financial analytics

---

# Final ER Relationship Summary

| Table     | Relationship | Table        |
| --------- | ------------ | ------------ |
| Customers | 1 → Many     | Accounts     |
| Accounts  | 1 → Many     | Transactions |
| Customers | 1 → Many     | Loans        |

---

<details>
<summary>30 Easy</summary>

# ✅ **30 Pandas + PySpark Questions (SQL Equivalent)**

---

# 🔹 1. Active accounts ordered by balance

**Pandas**

```python
accounts_df[accounts_df['status'] == 'Active'] \
    .sort_values(by='balance', ascending=False)
```

**PySpark**

```python
accounts_sdf.filter("status = 'Active'") \
    .orderBy("balance", ascending=False)
```

---

# 🔹 2. Customers from cities starting with 'New'

**Pandas**

```python
customers_df[customers_df['city'].str.startswith('New')]
```

**PySpark**

```python
from pyspark.sql.functions import col
customers_sdf.filter(col("city").startswith("New"))
```

---

# 🔹 3. Count accounts per type

**Pandas**

```python
accounts_df.groupby('account_type').size().reset_index(name='count')
```

**PySpark**

```python
accounts_sdf.groupBy("account_type").count()
```

---

# 🔹 4. Customers with more than one account

**Pandas**

```python
accounts_df.groupby('customer_id') \
    .size().reset_index(name='cnt') \
    .query('cnt > 1')
```

**PySpark**

```python
from pyspark.sql.functions import count
accounts_sdf.groupBy("customer_id") \
    .agg(count("*").alias("cnt")) \
    .filter("cnt > 1")
```

---

# 🔹 5. Top 5 customers by balance

**Pandas**

```python
accounts_df.merge(customers_df, on='customer_id') \
    .sort_values(by='balance', ascending=False).head(5)
```

**PySpark**

```python
accounts_sdf.join(customers_sdf, "customer_id") \
    .orderBy("balance", ascending=False).limit(5)
```

---

# 🔹 6. Total balance per customer

**Pandas**

```python
accounts_df.groupby('customer_id')['balance'].sum().reset_index()
```

**PySpark**

```python
from pyspark.sql.functions import sum
accounts_sdf.groupBy("customer_id").agg(sum("balance"))
```

---

# 🔹 7. Customers with or without accounts (LEFT JOIN)

**Pandas**

```python
customers_df.merge(accounts_df, on='customer_id', how='left')
```

**PySpark**

```python
customers_sdf.join(accounts_sdf, "customer_id", "left")
```

---

# 🔹 8. Transactions above average amount

**Pandas**

```python
avg_val = transactions_df['amount'].mean()
transactions_df[transactions_df['amount'] > avg_val]
```

**PySpark**

```python
avg_val = transactions_sdf.agg({"amount": "avg"}).collect()[0][0]
transactions_sdf.filter(col("amount") > avg_val)
```

---

# 🔹 9. Min & Max loan amount

**Pandas**

```python
loans_df['loan_amount'].agg(['min', 'max'])
```

**PySpark**

```python
loans_sdf.agg({"loan_amount": "min", "loan_amount": "max"})
```

---

# 🔹 10. Distinct loan types

**Pandas**

```python
loans_df['loan_type'].unique()
```

**PySpark**

```python
loans_sdf.select("loan_type").distinct()
```

---

# 🔹 11. Customers who took loans

**Pandas**

```python
customers_df.merge(loans_df, on='customer_id')
```

**PySpark**

```python
customers_sdf.join(loans_sdf, "customer_id")
```

---

# 🔹 12. Customers with accounts but no loans

**Pandas**

```python
accounts_df[~accounts_df['customer_id'].isin(loans_df['customer_id'])]
```

**PySpark**

```python
accounts_sdf.join(loans_sdf, "customer_id", "left_anti")
```

---

# 🔹 13. Customers with accounts OR loans

**Pandas**

```python
pd.concat([
    accounts_df['customer_id'],
    loans_df['customer_id']
]).drop_duplicates()
```

**PySpark**

```python
accounts_sdf.select("customer_id") \
    .union(loans_sdf.select("customer_id")).distinct()
```

---

# 🔹 14. Accounts opened after 2022

**Pandas**

```python
accounts_df[pd.to_datetime(accounts_df['opened_date']) > '2022-01-01']
```

**PySpark**

```python
accounts_sdf.filter("opened_date > '2022-01-01'")
```

---

# 🔹 15. Transaction count per account

**Pandas**

```python
transactions_df.groupby('account_id').size()
```

**PySpark**

```python
transactions_sdf.groupBy("account_id").count()
```

---

# 🔹 16. Total transaction amount per account

**Pandas**

```python
transactions_df.groupby('account_id')['amount'].sum()
```

**PySpark**

```python
transactions_sdf.groupBy("account_id").agg(sum("amount"))
```

---

# 🔹 17. Rank customers by balance

**Pandas**

```python
df = accounts_df.groupby('customer_id')['balance'].sum().reset_index()
df['rank'] = df['balance'].rank(method='dense', ascending=False)
```

**PySpark**

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import rank

w = Window.orderBy(col("total_balance").desc())
accounts_sdf.groupBy("customer_id") \
    .agg(sum("balance").alias("total_balance")) \
    .withColumn("rank", rank().over(w))
```

---

# 🔹 18. Row number per account (transactions)

**Pandas**

```python
transactions_df['rn'] = transactions_df.sort_values('transaction_date') \
    .groupby('account_id').cumcount() + 1
```

**PySpark**

```python
from pyspark.sql.functions import row_number
w = Window.partitionBy("account_id").orderBy("transaction_date")
transactions_sdf.withColumn("rn", row_number().over(w))
```

---

# 🔹 19. Previous transaction (LAG)

**Pandas**

```python
transactions_df['prev'] = transactions_df.sort_values('transaction_date') \
    .groupby('account_id')['amount'].shift(1)
```

**PySpark**

```python
from pyspark.sql.functions import lag
transactions_sdf.withColumn("prev",
    lag("amount").over(w))
```

---

# 🔹 20. Next transaction (LEAD)

**Pandas**

```python
transactions_df['next'] = transactions_df.groupby('account_id')['amount'].shift(-1)
```

**PySpark**

```python
from pyspark.sql.functions import lead
transactions_sdf.withColumn("next",
    lead("amount").over(w))
```

---

# 🔹 21. Avg balance per account type

**Pandas**

```python
accounts_df.groupby('account_type')['balance'].mean()
```

**PySpark**

```python
accounts_sdf.groupBy("account_type").avg("balance")
```

---

# 🔹 22. Accounts above avg balance

**Pandas**

```python
avg_bal = accounts_df['balance'].mean()
accounts_df[accounts_df['balance'] > avg_bal]
```

**PySpark**

```python
avg_bal = accounts_sdf.agg({"balance": "avg"}).collect()[0][0]
accounts_sdf.filter(col("balance") > avg_bal)
```

---

# 🔹 23. Latest transaction per account

**Pandas**

```python
transactions_df.sort_values('transaction_date', ascending=False) \
    .groupby('account_id').head(1)
```

**PySpark**

```python
w = Window.partitionBy("account_id").orderBy(col("transaction_date").desc())
transactions_sdf.withColumn("rn", row_number().over(w)) \
    .filter("rn = 1")
```

---

# 🔹 24. Total loan per type

**Pandas**

```python
loans_df.groupby('loan_type')['loan_amount'].sum()
```

**PySpark**

```python
loans_sdf.groupBy("loan_type").sum("loan_amount")
```

---

# 🔹 25. Top 3 loans

**Pandas**

```python
loans_df.sort_values('loan_amount', ascending=False).head(3)
```

**PySpark**

```python
loans_sdf.orderBy("loan_amount", ascending=False).limit(3)
```

---

# 🔹 26. Accounts with recent transactions

**Pandas**

```python
recent = transactions_df[
    transactions_df['transaction_date'] >= pd.Timestamp.today() - pd.Timedelta(days=30)
]['account_id'].unique()
```

**PySpark**

```python
from pyspark.sql.functions import current_date, date_sub
transactions_sdf.filter(
    col("transaction_date") >= date_sub(current_date(), 30)
).select("account_id").distinct()
```

---

# 🔹 27. Customer total balance (CTE equivalent)

**Pandas**

```python
customer_balance = accounts_df.groupby('customer_id')['balance'].sum().reset_index()
```

**PySpark**

```python
customer_balance = accounts_sdf.groupBy("customer_id").agg(sum("balance"))
```

---

# 🔹 28. Top 5 customers using above

**Pandas**

```python
customer_balance.sort_values('balance', ascending=False).head(5)
```

**PySpark**

```python
customer_balance.orderBy("sum(balance)", ascending=False).limit(5)
```

---

# 🔹 29. Loans above avg

**Pandas**

```python
avg_loan = loans_df['loan_amount'].mean()
loans_df[loans_df['loan_amount'] > avg_loan]
```

**PySpark**

```python
avg_loan = loans_sdf.agg({"loan_amount": "avg"}).collect()[0][0]
loans_sdf.filter(col("loan_amount") > avg_loan)
```

---

# 🔹 30. Transactions per day

**Pandas**

```python
transactions_df['date'] = pd.to_datetime(transactions_df['transaction_date']).dt.date
transactions_df.groupby('date').size()
```

**PySpark**

```python
from pyspark.sql.functions import to_date
transactions_sdf.withColumn("date", to_date("transaction_date")) \
    .groupBy("date").count()
```

---

# 🔥 What You Just Covered (Interview Gold)

You’ve now mapped SQL → Pandas → PySpark across:

* Joins (inner, left, anti)
* Aggregations & groupby
* Window functions (rank, lag, lead)
* Subqueries → intermediate variables
* CTE → temp DataFrames
* Set operations (union, anti join)
* Time-series analytics
* Ranking & top-N problems

---

</details>
---
<details>
<summary>30 Medium</summary>

# 🚀 **30 MEDIUM-Level Pandas + PySpark Questions (with Answers)**

---

# 🔹 1. Top 5 customers by total balance

**Pandas**

```python
df = accounts_df.groupby('customer_id')['balance'].sum().reset_index()
df.sort_values('balance', ascending=False).head(5)
```

**PySpark**

```python
from pyspark.sql.functions import sum

accounts_sdf.groupBy("customer_id") \
    .agg(sum("balance").alias("total_balance")) \
    .orderBy("total_balance", ascending=False).limit(5)
```

---

# 🔹 2. Customers whose total balance > average balance

**Pandas**

```python
df = accounts_df.groupby('customer_id')['balance'].sum().reset_index()
avg_bal = accounts_df['balance'].mean()
df[df['balance'] > avg_bal]
```

**PySpark**

```python
avg_bal = accounts_sdf.agg({"balance": "avg"}).collect()[0][0]

accounts_sdf.groupBy("customer_id") \
    .agg(sum("balance").alias("total_balance")) \
    .filter(col("total_balance") > avg_bal)
```

---

# 🔹 3. Accounts with >10 transactions

**Pandas**

```python
transactions_df.groupby('account_id').size() \
    .reset_index(name='cnt').query('cnt > 10')
```

**PySpark**

```python
from pyspark.sql.functions import count

transactions_sdf.groupBy("account_id") \
    .agg(count("*").alias("cnt")) \
    .filter("cnt > 10")
```

---

# 🔹 4. Customers having both loans and accounts

**Pandas**

```python
pd.merge(accounts_df[['customer_id']],
         loans_df[['customer_id']],
         on='customer_id').drop_duplicates()
```

**PySpark**

```python
accounts_sdf.select("customer_id") \
    .join(loans_sdf.select("customer_id"), "customer_id") \
    .distinct()
```

---

# 🔹 5. Loan count per customer (descending)

**Pandas**

```python
loans_df.groupby('customer_id').size() \
    .reset_index(name='loan_count') \
    .sort_values('loan_count', ascending=False)
```

**PySpark**

```python
loans_sdf.groupBy("customer_id") \
    .count().orderBy("count", ascending=False)
```

---

# 🔹 6. Rank customers by total transaction amount

**Pandas**

```python
df = accounts_df.merge(transactions_df, on='account_id')
df = df.groupby('customer_id')['amount'].sum().reset_index()
df['rank'] = df['amount'].rank(method='dense', ascending=False)
```

**PySpark**

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import rank

w = Window.orderBy(col("total").desc())

accounts_sdf.join(transactions_sdf, "account_id") \
    .groupBy("customer_id") \
    .agg(sum("amount").alias("total")) \
    .withColumn("rank", rank().over(w))
```

---

# 🔹 7. Top account per customer (highest balance)

**Pandas**

```python
accounts_df.sort_values('balance', ascending=False) \
    .groupby('customer_id').head(1)
```

**PySpark**

```python
from pyspark.sql.functions import row_number

w = Window.partitionBy("customer_id").orderBy(col("balance").desc())

accounts_sdf.withColumn("rn", row_number().over(w)) \
    .filter("rn = 1")
```

---

# 🔹 8. Daily transaction totals

**Pandas**

```python
transactions_df['date'] = pd.to_datetime(transactions_df['transaction_date']).dt.date
transactions_df.groupby('date')['amount'].sum()
```

**PySpark**

```python
from pyspark.sql.functions import to_date

transactions_sdf.withColumn("date", to_date("transaction_date")) \
    .groupBy("date").sum("amount")
```

---

# 🔹 9. Running transaction total per account

**Pandas**

```python
transactions_df = transactions_df.sort_values('transaction_date')
transactions_df['running'] = transactions_df.groupby('account_id')['amount'].cumsum()
```

**PySpark**

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import sum

w = Window.partitionBy("account_id").orderBy("transaction_date")

transactions_sdf.withColumn("running", sum("amount").over(w))
```

---

# 🔹 10. Previous transaction amount

**Pandas**

```python
transactions_df['prev'] = transactions_df.groupby('account_id')['amount'].shift(1)
```

**PySpark**

```python
from pyspark.sql.functions import lag

transactions_sdf.withColumn("prev",
    lag("amount").over(w))
```

---

# 🔹 11. Next transaction date

**Pandas**

```python
transactions_df['next_date'] = transactions_df.groupby('account_id')['transaction_date'].shift(-1)
```

**PySpark**

```python
from pyspark.sql.functions import lead

transactions_sdf.withColumn("next_date",
    lead("transaction_date").over(w))
```

---

# 🔹 12. Customers per city

**Pandas**

```python
customers_df.groupby('city').size().reset_index(name='count')
```

**PySpark**

```python
customers_sdf.groupBy("city").count()
```

---

# 🔹 13. Customers who transacted >10,000

**Pandas**

```python
accounts_df.merge(transactions_df, on='account_id') \
    .query('amount > 10000')['customer_id'].drop_duplicates()
```

**PySpark**

```python
accounts_sdf.join(transactions_sdf, "account_id") \
    .filter(col("amount") > 10000) \
    .select("customer_id").distinct()
```

---

# 🔹 14. Accounts opened per year

**Pandas**

```python
accounts_df['year'] = pd.to_datetime(accounts_df['opened_date']).dt.year
accounts_df.groupby('year').size()
```

**PySpark**

```python
from pyspark.sql.functions import year

accounts_sdf.withColumn("year", year("opened_date")) \
    .groupBy("year").count()
```

---

# 🔹 15. Customers with balance > city average

**Pandas**

```python
df = accounts_df.merge(customers_df, on='customer_id')
df['city_avg'] = df.groupby('city')['balance'].transform('mean')
df[df['balance'] > df['city_avg']]
```

**PySpark**

```python
from pyspark.sql.functions import avg

w = Window.partitionBy("city")

accounts_sdf.join(customers_sdf, "customer_id") \
    .withColumn("city_avg", avg("balance").over(w)) \
    .filter(col("balance") > col("city_avg"))
```

---

# 🔹 16. Customers with >1 loan

**Pandas**

```python
loans_df.groupby('customer_id').size().reset_index(name='cnt').query('cnt > 1')
```

**PySpark**

```python
loans_sdf.groupBy("customer_id").count().filter("count > 1")
```

---

# 🔹 17. Customer total transactions vs total loan

**Pandas**

```python
txn = accounts_df.merge(transactions_df).groupby('customer_id')['amount'].sum()
loan = loans_df.groupby('customer_id')['loan_amount'].sum()

df = txn.to_frame().join(loan, how='left').fillna(0)
df[df['amount'] > df['loan_amount']]
```

**PySpark**

```python
txn = accounts_sdf.join(transactions_sdf, "account_id") \
    .groupBy("customer_id").agg(sum("amount").alias("txn"))

loan = loans_sdf.groupBy("customer_id") \
    .agg(sum("loan_amount").alias("loan"))

txn.join(loan, "customer_id", "left") \
    .fillna(0) \
    .filter(col("txn") > col("loan"))
```

---

# 🔹 18. Top 5 customers by transaction value (CTE style)

**Pandas**

```python
df = accounts_df.merge(transactions_df)
df.groupby('customer_id')['amount'].sum() \
    .sort_values(ascending=False).head(5)
```

**PySpark**

```python
accounts_sdf.join(transactions_sdf, "account_id") \
    .groupBy("customer_id") \
    .agg(sum("amount")) \
    .orderBy("sum(amount)", ascending=False).limit(5)
```

---

# 🔹 19. Rank transactions per account (top 3)

**Pandas**

```python
transactions_df['rank'] = transactions_df.groupby('account_id')['amount'] \
    .rank(method='dense', ascending=False)

transactions_df[transactions_df['rank'] <= 3]
```

**PySpark**

```python
w = Window.partitionBy("account_id").orderBy(col("amount").desc())

transactions_sdf.withColumn("rank", rank().over(w)) \
    .filter("rank <= 3")
```

---

# 🔹 20. Detect duplicate transactions (same account, amount, date)

**Pandas**

```python
transactions_df[transactions_df.duplicated(
    subset=['account_id','amount','transaction_date'], keep=False)]
```

**PySpark**

```python
transactions_sdf.groupBy("account_id","amount","transaction_date") \
    .count().filter("count > 1")
```

---

# 🔹 21. Monthly transaction volume

**Pandas**

```python
transactions_df['month'] = pd.to_datetime(transactions_df['transaction_date']).dt.to_period('M')
transactions_df.groupby('month')['amount'].sum()
```

**PySpark**

```python
from pyspark.sql.functions import month

transactions_sdf.withColumn("month", month("transaction_date")) \
    .groupBy("month").sum("amount")
```

---

# 🔹 22. Pivot transaction type vs amount

**Pandas**

```python
pd.pivot_table(transactions_df, values='amount',
               index='account_id',
               columns='transaction_type',
               aggfunc='sum')
```

**PySpark**

```python
transactions_sdf.groupBy("account_id") \
    .pivot("transaction_type") \
    .sum("amount")
```

---

# 🔹 23. Customers with no transactions

**Pandas**

```python
accounts = accounts_df[['account_id','customer_id']]
txn_accounts = transactions_df['account_id'].unique()

accounts[~accounts['account_id'].isin(txn_accounts)]
```

**PySpark**

```python
accounts_sdf.join(transactions_sdf, "account_id", "left_anti")
```

---

# 🔹 24. Approx distinct customers (large data)

**Pandas**

```python
transactions_df['account_id'].nunique()
```

**PySpark**

```python
from pyspark.sql.functions import approx_count_distinct

transactions_sdf.agg(approx_count_distinct("account_id"))
```

---

# 🔹 25. Conditional column (high value txn)

**Pandas**

```python
transactions_df['flag'] = transactions_df['amount'].apply(lambda x: 'High' if x>10000 else 'Low')
```

**PySpark**

```python
from pyspark.sql.functions import when

transactions_sdf.withColumn("flag",
    when(col("amount") > 10000, "High").otherwise("Low"))
```

---

# 🔹 26. Remove duplicates

**Pandas**

```python
transactions_df.drop_duplicates()
```

**PySpark**

```python
transactions_sdf.dropDuplicates()
```

---

# 🔹 27. Broadcast join (optimization)

**PySpark only**

```python
from pyspark.sql.functions import broadcast

accounts_sdf.join(broadcast(customers_sdf), "customer_id")
```

---

# 🔹 28. Cache frequently used DataFrame

**PySpark**

```python
accounts_sdf.cache()
```

---

# 🔹 29. Save partitioned data (year)

**PySpark**

```python
accounts_sdf.write.partitionBy("opened_date").parquet("/path")
```

---

# 🔹 30. Convert to SQL (Spark SQL)

**PySpark**

```python
accounts_sdf.createOrReplaceTempView("accounts")

spark.sql("""
SELECT customer_id, SUM(balance)
FROM accounts
GROUP BY customer_id
""")
```

---

# 🔥 What makes this “Medium” (Interview Insight)

These are NOT basic because they include:

* Multi-table joins (Accounts ↔ Transactions ↔ Loans)
* Window functions (rank, lag, running total)
* Business logic (risk, high-value customers)
* Time-series aggregation (monthly/daily)
* Pivoting & reshaping
* Anti-joins & set logic
* Performance (broadcast, approx count, cache)

---

</details>
---
<details>
<summary>30 Hard </summary>


---

# 🚀 **30 HARD FAANG-Level Pandas + PySpark Questions (with Answers)**

---

## 🔥 1. Top 3 customers by deposits per city

**Pandas**

```python
df = customers_df.merge(accounts_df).merge(transactions_df)
df = df[df['transaction_type'] == 'Deposit']

res = df.groupby(['city','customer_id'])['amount'].sum().reset_index()
res['rank'] = res.groupby('city')['amount'].rank(method='first', ascending=False)

res[res['rank'] <= 3]
```

**PySpark**

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, row_number

w = Window.partitionBy("city").orderBy(col("total").desc())

customers_sdf.join(accounts_sdf, "customer_id") \
    .join(transactions_sdf, "account_id") \
    .filter(col("transaction_type") == "Deposit") \
    .groupBy("city","customer_id") \
    .agg(sum("amount").alias("total")) \
    .withColumn("rn", row_number().over(w)) \
    .filter("rn <= 3")
```

---

## 🔥 2. Running balance per account

**Pandas**

```python
transactions_df = transactions_df.sort_values('transaction_date')
transactions_df['running_balance'] = transactions_df.groupby('account_id')['amount'].cumsum()
```

**PySpark**

```python
w = Window.partitionBy("account_id").orderBy("transaction_date")
transactions_sdf.withColumn("running_balance", sum("amount").over(w))
```

---

## 🔥 3. Detect decreasing transaction pattern (fraud signal)

**Pandas**

```python
transactions_df['prev'] = transactions_df.groupby('account_id')['amount'].shift(1)
transactions_df[transactions_df['amount'] < transactions_df['prev']]
```

**PySpark**

```python
from pyspark.sql.functions import lag

transactions_sdf.withColumn("prev", lag("amount").over(w)) \
    .filter(col("amount") < col("prev"))
```

---

## 🔥 4. Loan > total deposits (risk customers)

**Pandas**

```python
dep = accounts_df.merge(transactions_df).groupby('customer_id')['amount'].sum()
loan = loans_df.groupby('customer_id')['loan_amount'].sum()

df = dep.to_frame().join(loan, how='inner')
df[df['loan_amount'] > df['amount']]
```

**PySpark**

```python
dep = accounts_sdf.join(transactions_sdf, "account_id") \
    .groupBy("customer_id").agg(sum("amount").alias("dep"))

loan = loans_sdf.groupBy("customer_id") \
    .agg(sum("loan_amount").alias("loan"))

dep.join(loan, "customer_id") \
    .filter(col("loan") > col("dep"))
```

---

## 🔥 5. Transactions within 5 minutes (fraud detection)

**Pandas**

```python
transactions_df['prev_time'] = transactions_df.groupby('account_id')['transaction_date'].shift(1)
transactions_df[
    (transactions_df['transaction_date'] - transactions_df['prev_time']).dt.total_seconds() <= 300
]
```

**PySpark**

```python
from pyspark.sql.functions import lag, unix_timestamp

df = transactions_sdf.withColumn("prev", lag("transaction_date").over(w))
df.filter((unix_timestamp("transaction_date") - unix_timestamp("prev")) <= 300)
```

---

## 🔥 6. Customers with 3 consecutive transaction days (streak problem)

**Pandas**

```python
df = accounts_df.merge(transactions_df)
df['date'] = pd.to_datetime(df['transaction_date']).dt.date

df = df.sort_values(['customer_id','date'])
df['diff'] = df.groupby('customer_id')['date'].diff().dt.days

df[df['diff'] == 1]
```

👉 (Real answer: group streaks using diff !=1 break logic)

---

**PySpark**

```python
from pyspark.sql.functions import lag, datediff

df = accounts_sdf.join(transactions_sdf, "account_id") \
    .withColumn("date", to_date("transaction_date"))

df.withColumn("prev", lag("date").over(Window.partitionBy("customer_id").orderBy("date"))) \
    .withColumn("diff", datediff("date","prev")) \
    .filter("diff = 1")
```

---

## 🔥 7. Most common transaction type per account

**Pandas**

```python
df = transactions_df.groupby(['account_id','transaction_type']).size().reset_index(name='cnt')
df['rank'] = df.groupby('account_id')['cnt'].rank(method='first', ascending=False)
df[df['rank'] == 1]
```

**PySpark**

```python
w = Window.partitionBy("account_id").orderBy(col("cnt").desc())

transactions_sdf.groupBy("account_id","transaction_type") \
    .count() \
    .withColumn("rank", rank().over(w)) \
    .filter("rank = 1")
```

---

## 🔥 8. Duplicate transactions within 10 minutes

**Pandas**

```python
df = transactions_df.sort_values('transaction_date')
df['prev'] = df.groupby(['account_id','amount'])['transaction_date'].shift(1)

df[(df['transaction_date'] - df['prev']).dt.seconds <= 600]
```

**PySpark**

```python
w = Window.partitionBy("account_id","amount").orderBy("transaction_date")

transactions_sdf.withColumn("prev", lag("transaction_date").over(w)) \
    .filter((unix_timestamp("transaction_date") - unix_timestamp("prev")) <= 600)
```

---

## 🔥 9. Top 5% customers (NTILE equivalent)

**Pandas**

```python
df = accounts_df.merge(transactions_df).groupby('customer_id')['amount'].sum().reset_index()
df['percentile'] = df['amount'].rank(pct=True)
df[df['percentile'] >= 0.95]
```

**PySpark**

```python
from pyspark.sql.functions import ntile

w = Window.orderBy(col("total").desc())

accounts_sdf.join(transactions_sdf, "account_id") \
    .groupBy("customer_id") \
    .agg(sum("amount").alias("total")) \
    .withColumn("tile", ntile(20).over(w)) \
    .filter("tile = 1")
```

---

## 🔥 10. Monthly growth (MoM)

**Pandas**

```python
df = transactions_df.copy()
df['month'] = pd.to_datetime(df['transaction_date']).dt.to_period('M')

res = df.groupby('month')['amount'].sum().reset_index()
res['growth'] = res['amount'].diff()
```

**PySpark**

```python
from pyspark.sql.functions import date_trunc

df = transactions_sdf.withColumn("month", date_trunc("month","transaction_date"))

res = df.groupBy("month").sum("amount")

w = Window.orderBy("month")
res.withColumn("growth", col("sum(amount)") - lag("sum(amount)").over(w))
```

---

## 🔥 11. Longest gap between transactions

**Pandas**

```python
df = transactions_df.sort_values('transaction_date')
df['prev'] = df.groupby('account_id')['transaction_date'].shift(1)
df['gap'] = (df['transaction_date'] - df['prev']).dt.days

df.groupby('account_id')['gap'].max()
```

**PySpark**

```python
df = transactions_sdf.withColumn("prev", lag("transaction_date").over(w))
df.withColumn("gap", datediff("transaction_date","prev")) \
  .groupBy("account_id").max("gap")
```

---

## 🔥 12. Pareto 80% customers

**Pandas**

```python
df = accounts_df.merge(transactions_df).groupby('customer_id')['amount'].sum().sort_values(ascending=False)

cum = df.cumsum()
total = df.sum()

df[cum <= 0.8 * total]
```

---

## 🔥 13. Median transaction

**Pandas**

```python
transactions_df['amount'].median()
```

**PySpark**

```python
transactions_sdf.approxQuantile("amount", [0.5], 0.01)
```

---

## 🔥 14. Inactive accounts (6 months)

**Pandas**

```python
last_txn = transactions_df.groupby('account_id')['transaction_date'].max()
last_txn[last_txn < pd.Timestamp.today() - pd.Timedelta(days=180)]
```

**PySpark**

```python
transactions_sdf.groupBy("account_id") \
    .agg({"transaction_date":"max"}) \
    .filter(col("max(transaction_date)") < date_sub(current_date(),180))
```

---

## 🔥 15. Loan-to-deposit ratio

**Pandas**

```python
dep = accounts_df.merge(transactions_df).groupby('customer_id')['amount'].sum()
loan = loans_df.groupby('customer_id')['loan_amount'].sum()

ratio = (loan / dep).sort_values(ascending=False)
```

---

## 🔥 16. First transaction per account

(window already covered)

---

## 🔥 17. Deposits but no withdrawals

**Pandas**

```python
dep = transactions_df[transactions_df['transaction_type']=='Deposit']['account_id']
wd = transactions_df[transactions_df['transaction_type']=='Withdraw']['account_id']

set(dep) - set(wd)
```

---

## 🔥 18. Sequence numbers

(window row_number same as medium)

---

## 🔥 19. Increasing deposit trend

**Pandas**

```python
df = transactions_df[transactions_df['transaction_type']=='Deposit']
df['prev'] = df.groupby('account_id')['amount'].shift(1)

df[df['amount'] > df['prev']]
```

---

## 🔥 20. Transaction > account balance

**Pandas**

```python
transactions_df.merge(accounts_df) \
    .query('amount > balance')
```

---

## 🔥 21. Customers active all months

**Pandas**

```python
df = accounts_df.merge(transactions_df)
df['month'] = pd.to_datetime(df['transaction_date']).dt.month

df.groupby('customer_id')['month'].nunique().query('month == 12')
```

---

## 🔥 22. Top transaction per account

(window reuse)

---

## 🔥 23. Loan customers with no transactions

**Pandas**

```python
txn_cust = accounts_df.merge(transactions_df)['customer_id']
loan_cust = loans_df['customer_id']

set(loan_cust) - set(txn_cust)
```

---

## 🔥 24. 7-day moving avg

**Pandas**

```python
transactions_df.sort_values('transaction_date')['amount'] \
    .rolling(7).mean()
```

---

## 🔥 25. Largest gap globally

(similar to Q11 but global max)

---

## 🔥 26. Self join customers (same city)

**Pandas**

```python
customers_df.merge(customers_df, on='city') \
    .query('customer_id_x < customer_id_y')
```

---

## 🔥 27. Intersection loans & accounts

**Pandas**

```python
set(loans_df['customer_id']) & set(accounts_df['customer_id'])
```

---

## 🔥 28. Date series generation

**Pandas**

```python
pd.date_range(start='2024-01-01', end='2024-01-31')
```

**PySpark**

```python
spark.range(31).selectExpr("date_add('2024-01-01', id)")
```

---

## 🔥 29. Highest withdrawal frequency

**Pandas**

```python
transactions_df[transactions_df['transaction_type']=='Withdraw'] \
    .groupby('account_id').size().sort_values(ascending=False)
```

---

## 🔥 30. MoM increasing deposits per customer

**Pandas**

```python
df = accounts_df.merge(transactions_df)
df['month'] = pd.to_datetime(df['transaction_date']).dt.to_period('M')

res = df.groupby(['customer_id','month'])['amount'].sum().reset_index()
res['prev'] = res.groupby('customer_id')['amount'].shift(1)

res[res['amount'] > res['prev']]
```

---

# 🧠 What You Just Mastered (REAL FAANG SIGNAL)

These are **actual hiring signals**:

### 🔥 Advanced Patterns

* Gap & island (streaks)
* Session detection
* Fraud heuristics (time proximity)
* Pareto segmentation

### 🔥 Heavy Window Usage

* Running totals
* LAG/LEAD comparisons
* Ranking within partitions

### 🔥 Multi-table reasoning

* Customer → Account → Transactions → Loans
* Derived metrics (ratio, risk)

### 🔥 Time-series analytics

* MoM growth
* Rolling windows
* Activity gaps

### 🔥 DE mindset

* Approx quantile
* Broadcast joins
* Partitioning awareness
* Avoiding full scans

---

</details>
---
<details>
<summary>50 FAANG Level </summary>

Below are **50 FAANG-level questions WITH answers** mapped to your SQL sheet, covering:

* Window functions
* Time-series
* Joins
* Fraud detection
* Ranking
* Performance thinking (PySpark)

Each question includes:
👉 Problem
👉 Pandas Solution
👉 PySpark Solution

---

# 🔥 SECTION 1: Pagination / Row Number (1–5)

---

### **Q1. Get 3rd last transaction**

**Pandas**

```python
df.sort_values('transaction_date', ascending=False).iloc[2]
```

**PySpark**

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

w = Window.orderBy(col("transaction_date").desc())
df.withColumn("rn", row_number().over(w)).filter("rn=3")
```

---

### **Q2. Rows 10–20**

**Pandas**

```python
df.sort_values('transaction_date').iloc[9:20]
```

**PySpark**

```python
w = Window.orderBy("transaction_date")
df.withColumn("rn", row_number().over(w)).filter("rn BETWEEN 10 AND 20")
```

---

### **Q3. 5th highest transaction**

**Pandas**

```python
df.nlargest(5, 'amount').iloc[-1]
```

**PySpark**

```python
w = Window.orderBy(col("amount").desc())
df.withColumn("rnk", rank().over(w)).filter("rnk=5")
```

---

### **Q4. Last 10 transactions**

**Pandas**

```python
df.sort_values('transaction_date', ascending=False).head(10)
```

**PySpark**

```python
df.orderBy(col("transaction_date").desc()).limit(10)
```

---

### **Q5. Dense rank transactions**

**Pandas**

```python
df['rank'] = df['amount'].rank(method='dense', ascending=False)
```

**PySpark**

```python
df.withColumn("rank", dense_rank().over(Window.orderBy(col("amount").desc())))
```

---

# 🔥 SECTION 2: Aggregations (6–10)

---

### **Q6. Total deposits per customer**

**Pandas**

```python
df.merge(accounts).query("transaction_type=='Deposit'") \
.groupby('customer_id')['amount'].sum()
```

**PySpark**

```python
df.join(accounts, "account_id") \
.filter("transaction_type='Deposit'") \
.groupBy("customer_id").sum("amount")
```

---

### **Q7. Monthly revenue**

**Pandas**

```python
df['month'] = pd.to_datetime(df['transaction_date']).dt.to_period('M')
df.groupby('month')['amount'].sum()
```

**PySpark**

```python
df.groupBy(month("transaction_date")).sum("amount")
```

---

### **Q8. Avg loan per type**

**Pandas**

```python
loans.groupby('loan_type')['loan_amount'].mean()
```

**PySpark**

```python
loans.groupBy("loan_type").avg("loan_amount")
```

---

### **Q9. Total per transaction type**

**Pandas**

```python
df.groupby('transaction_type')['amount'].sum()
```

**PySpark**

```python
df.groupBy("transaction_type").sum("amount")
```

---

### **Q10. Pivot transaction type vs city**

**Pandas**

```python
pd.pivot_table(df, values='amount', index='city', columns='transaction_type', aggfunc='sum')
```

**PySpark**

```python
df.groupBy("city").pivot("transaction_type").sum("amount")
```

---

# 🔥 SECTION 3: Window Functions (11–20)

---

### **Q11. Running balance**

**Pandas**

```python
df.sort_values(['account_id','transaction_date']) \
.groupby('account_id')['amount'].cumsum()
```

**PySpark**

```python
w = Window.partitionBy("account_id").orderBy("transaction_date")
df.withColumn("running", sum("amount").over(w))
```

---

### **Q12. Lag transaction**

**Pandas**

```python
df['prev'] = df.groupby('account_id')['amount'].shift(1)
```

**PySpark**

```python
df.withColumn("prev", lag("amount").over(w))
```

---

### **Q13. Lead transaction**

**Pandas**

```python
df['next'] = df.groupby('account_id')['amount'].shift(-1)
```

**PySpark**

```python
df.withColumn("next", lead("amount").over(w))
```

---

### **Q14. Moving average (7-day)**

**Pandas**

```python
df['ma'] = df['amount'].rolling(7).mean()
```

**PySpark**

```python
w = Window.orderBy("transaction_date").rowsBetween(-6, 0)
df.withColumn("ma", avg("amount").over(w))
```

---

### **Q15. Rank per city**

**Pandas**

```python
df['rank'] = df.groupby('city')['amount'].rank(ascending=False)
```

**PySpark**

```python
w = Window.partitionBy("city").orderBy(col("amount").desc())
df.withColumn("rank", rank().over(w))
```

---

### **Q16. First transaction per account**

**Pandas**

```python
df.sort_values('transaction_date').groupby('account_id').first()
```

**PySpark**

```python
df.withColumn("rn", row_number().over(w)).filter("rn=1")
```

---

### **Q17. Last transaction**

**Pandas**

```python
df.sort_values('transaction_date').groupby('account_id').last()
```

**PySpark**

```python
w = Window.partitionBy("account_id").orderBy(col("transaction_date").desc())
```

---

### **Q18. Cumulative revenue**

**Pandas**

```python
df.groupby('month')['amount'].sum().cumsum()
```

**PySpark**

```python
df.groupBy("month").sum("amount") \
.withColumn("cum", sum("sum(amount)").over(Window.orderBy("month")))
```

---

### **Q19. NTILE bucket**

**Pandas**

```python
df['bucket'] = pd.qcut(df['amount'], 10, labels=False)
```

**PySpark**

```python
df.withColumn("bucket", ntile(10).over(Window.orderBy("amount")))
```

---

### **Q20. Dense rank loan**

**Pandas**

```python
loans['rank'] = loans.groupby('loan_type')['loan_amount'].rank(method='dense', ascending=False)
```

**PySpark**

```python
loans.withColumn("rank", dense_rank().over(Window.partitionBy("loan_type").orderBy(col("loan_amount").desc())))
```

---

# 🔥 SECTION 4: Joins (21–25)

---

### **Q21. Customer → Accounts join**

**Pandas**

```python
pd.merge(customers, accounts, on='customer_id')
```

**PySpark**

```python
customers.join(accounts, "customer_id")
```

---

### **Q22. Full join**

**Pandas**

```python
pd.merge(df1, df2, on='id', how='outer')
```

**PySpark**

```python
df1.join(df2, "id", "outer")
```

---

### **Q23. Anti join (customers without accounts)**

**Pandas**

```python
customers[~customers.customer_id.isin(accounts.customer_id)]
```

**PySpark**

```python
customers.join(accounts, "customer_id", "left_anti")
```

---

### **Q24. Semi join**

**PySpark**

```python
customers.join(accounts, "customer_id", "left_semi")
```

---

### **Q25. Broadcast join**

**PySpark**

```python
from pyspark.sql.functions import broadcast
df.join(broadcast(small_df), "id")
```

---

# 🔥 SECTION 5: Time Series (26–30)

---

### **Q26. Days between transactions**

**Pandas**

```python
df['diff'] = df.groupby('account_id')['transaction_date'].diff()
```

**PySpark**

```python
df.withColumn("diff", col("transaction_date") - lag("transaction_date").over(w))
```

---

### **Q27. Monthly growth**

**Pandas**

```python
monthly['growth'] = monthly['amount'].diff()
```

**PySpark**

```python
df.withColumn("growth", col("amt") - lag("amt").over(w))
```

---

### **Q28. Growth %**

**Pandas**

```python
monthly['pct'] = monthly['amount'].pct_change()
```

---

### **Q29. Inactive accounts**

**Pandas**

```python
df.groupby('account_id')['transaction_date'].max()
```

---

### **Q30. Cohort month**

**Pandas**

```python
customers['cohort'] = customers['created_at'].dt.to_period('M')
```

---

# 🔥 SECTION 6: Fraud Detection (31–35)

---

### **Q31. Same amount within 10 min**

**Pandas**

```python
df['prev_time'] = df.groupby(['account_id','amount'])['transaction_date'].shift()
df[df['transaction_date'] - df['prev_time'] <= pd.Timedelta('10m')]
```

---

### **Q32. Duplicate rows**

**Pandas**

```python
df[df.duplicated()]
```

**PySpark**

```python
df.dropDuplicates()
```

---

### **Q33. Sudden spike detection**

**Pandas**

```python
df['flag'] = df['amount'] > df['amount'].rolling(5).mean()*3
```

---

### **Q34. High frequency transactions**

**PySpark**

```python
df.groupBy("account_id").count().filter("count > 50")
```

---

### **Q35. Outliers**

**Pandas**

```python
df[df['amount'] > df['amount'].quantile(0.99)]
```

---

# 🔥 SECTION 7: Advanced (36–50)

---

### **Q36. Median**

```python
df['amount'].median()
```

---

### **Q37. Percent contribution**

```python
df['pct'] = df['amount'] / df['amount'].sum()
```

---

### **Q38. Rolling sum**

```python
df['roll'] = df['amount'].rolling(7).sum()
```

---

### **Q39. Expanding mean**

```python
df['exp'] = df['amount'].expanding().mean()
```

---

### **Q40. Sample data**

```python
df.sample(10)
```

---

### **Q41. Distinct count**

```python
df['account_id'].nunique()
```

---

### **Q42. Approx distinct (Spark)**

```python
df.select(approx_count_distinct("account_id"))
```

---

### **Q43. Repartition**

```python
df.repartition(10)
```

---

### **Q44. Cache**

```python
df.cache()
```

---

### **Q45. Explain plan**

```python
df.explain()
```

---

### **Q46. Write parquet**

```python
df.write.parquet("path")
```

---

### **Q47. Union**

```python
df1.union(df2)
```

---

### **Q48. UDF**

```python
from pyspark.sql.functions import udf
```

---

### **Q49. Pandas apply**

```python
df['new'] = df['amount'].apply(lambda x: x*2)
```

---

### **Q50. Query optimization (important)**

👉 Use:

* **column pruning**
* **predicate pushdown**
* **broadcast joins**
* **cache only reused DF**

---

# 🚀 Final Insight (VERY IMPORTANT)

This set maps exactly:

| SQL Pattern | Pandas          | PySpark   |
| ----------- | --------------- | --------- |
| Window      | groupby + shift | Window    |
| Join        | merge           | join      |
| Rank        | rank()          | rank()    |
| CTE         | chaining        | temp view |
| Aggregation | groupby         | groupBy   |

---

</details>
---
