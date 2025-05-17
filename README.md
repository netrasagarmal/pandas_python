# pandas_python
Pandas notes and functions use example with dataset (not EDA)

---

### 🔍 **1. Data Loading / Inspection**

| Function            | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `pd.read_csv()`     | Load CSV files                             |
| `pd.read_excel()`   | Load Excel files                           |
| `pd.read_json()`    | Load JSON data                             |
| `df.head(n)`        | First `n` rows                             |
| `df.tail(n)`        | Last `n` rows                              |
| `df.info()`         | Overview of data types and non-null counts |
| `df.describe()`     | Summary statistics                         |
| `df.shape`          | Rows and columns                           |
| `df.columns`        | Column names                               |
| `df.dtypes`         | Data types of each column                  |
| `df.index`          | Index information                          |
| `df.sample(n)`      | Random sample of `n` rows                  |
| `df.memory_usage()` | Memory consumption                         |

---

### 🧹 **2. Data Cleaning**

| Function               | Purpose                            |
| ---------------------- | ---------------------------------- |
| `df.isnull()`          | Boolean DataFrame of nulls         |
| `df.notnull()`         | Boolean DataFrame of non-nulls     |
| `df.dropna()`          | Drop rows with null values         |
| `df.fillna(value)`     | Fill null values                   |
| `df.replace(old, new)` | Replace values                     |
| `df.duplicated()`      | Check duplicate rows               |
| `df.drop_duplicates()` | Remove duplicates                  |
| `df.astype(type)`      | Change data type                   |
| `pd.to_datetime()`     | Convert to datetime                |
| `df.apply(func)`       | Apply function across rows/columns |

---

### 🔄 **3. Updating / Creating Columns**

| Function                            | Purpose                                |
| ----------------------------------- | -------------------------------------- |
| `df['col'] = ...`                   | Add or modify a column                 |
| `df.rename(columns={'old': 'new'})` | Rename columns                         |
| `df.assign(new_col=lambda x: ...)`  | Create new column                      |
| `df.insert(loc, col, value)`        | Insert column at specific location     |
| `df.eval('expression')`             | Evaluate string expressions on columns |

---

### 📊 **4. Aggregation and Grouping**

| Function                       | Purpose                               |
| ------------------------------ | ------------------------------------- |
| `df.groupby('col')`            | Group by column                       |
| `grouped.mean()`               | Group-wise mean                       |
| `grouped.agg(['mean', 'sum'])` | Multiple aggregations                 |
| `df.pivot_table()`             | Advanced grouping with aggregation    |
| `df.cumsum()`                  | Cumulative sum                        |
| `df.cumprod()`                 | Cumulative product                    |
| `df.rolling(window).mean()`    | Rolling statistics                    |
| `df.expanding().mean()`        | Expanding statistics                  |
| `df.transform()`               | Return aligned result with same shape |

---

### 🔢 **5. Conditional Filtering**

| Function / Syntax                                   | Purpose                    |
| --------------------------------------------------- | -------------------------- |
| `df[df['col'] > 10]`                                | Filter by condition        |
| `df[(df['A'] > 5) & (df['B'] < 10)]`                | Combine conditions         |
| `df.query('A > 5 and B < 10')`                      | Query method for filtering |
| `df.loc[df['col'] == 'val', 'other_col'] = new_val` | Conditional update         |

---

### ✂️ **6. Slicing and Indexing**

| Function / Syntax          | Purpose                 |
| -------------------------- | ----------------------- |
| `df.loc[row, col]`         | Label-based indexing    |
| `df.iloc[row, col]`        | Position-based indexing |
| `df.set_index('col')`      | Set index to a column   |
| `df.reset_index()`         | Reset index to default  |
| `df.sort_values(by='col')` | Sort by column          |
| `df.sort_index()`          | Sort by index           |

---

### 🗑️ **7. Dropping**

| Function                            | Purpose               |
| ----------------------------------- | --------------------- |
| `df.drop('col', axis=1)`            | Drop column           |
| `df.drop([index], axis=0)`          | Drop rows             |
| `df.drop(columns=['col1', 'col2'])` | Drop multiple columns |
| `df.drop(index=[1, 2])`             | Drop multiple rows    |

---

### 🔗 **8. Merging, Joining, Concatenating**

| Function                       | Purpose                 |
| ------------------------------ | ----------------------- |
| `pd.concat([df1, df2])`        | Concatenate along axis  |
| `pd.merge(df1, df2, on='key')` | SQL-style join          |
| `df1.join(df2)`                | Join on index or column |

---

### 🔁 **9. Mapping and Encoding**

| Function                | Purpose                          |
| ----------------------- | -------------------------------- |
| `df['col'].map(func)`   | Map values using a function/dict |
| `df['col'].apply(func)` | Apply custom function            |
| `pd.get_dummies(df)`    | One-hot encoding                 |

---

### 📈 **10. Visualization**

| Function                | Purpose            |
| ----------------------- | ------------------ |
| `df.plot()`             | Basic plotting     |
| `df.hist()`             | Histogram          |
| `df.boxplot()`          | Box plot           |
| `df.plot.scatter(x, y)` | Scatter plot       |
| `df.corr()`             | Correlation matrix |

---
## Data Loading and Basic Information

### `pd.read_csv()`
**Use**: Loads data from CSV files into a DataFrame
**Syntax**: `pd.read_csv(filepath_or_buffer, sep=',', delimiter=None, header='infer', names=None, index_col=None, usecols=None, dtype=None, engine=None, skiprows=None, nrows=None, na_values=None)`

### `pd.read_excel()`
**Use**: Loads data from Excel files into a DataFrame
**Syntax**: `pd.read_excel(io, sheet_name=0, header=0, names=None, index_col=None, usecols=None, dtype=None, engine=None, skiprows=None, nrows=None)`

### `df.info()`
**Use**: Prints concise summary of DataFrame
**Syntax**: `df.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None)`

### `df.describe()`
**Use**: Generates descriptive statistics of numerical columns
**Syntax**: `df.describe(percentiles=None, include=None, exclude=None)`

### `df.dtypes`
**Use**: Returns Series with data types of each column
**Syntax**: `df.dtypes`

### `df.shape`
**Use**: Returns tuple of DataFrame dimensions (rows, columns)
**Syntax**: `df.shape`

### `df.columns`
**Use**: Returns Index of column labels
**Syntax**: `df.columns`

## Exploratory Data Analysis

### `df.head()`
**Use**: Returns first n rows
**Syntax**: `df.head(n=5)`

### `df.tail()`
**Use**: Returns last n rows
**Syntax**: `df.tail(n=5)`

### `df.sample()`
**Use**: Returns random sample of rows
**Syntax**: `df.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)`

### `df.value_counts()`
**Use**: Returns counts of unique values
**Syntax**: `df.value_counts(subset=None, normalize=False, sort=True, ascending=False, dropna=True)`

### `df.nunique()`
**Use**: Count distinct observations over requested axis
**Syntax**: `df.nunique(axis=0, dropna=True)`

### `df.duplicated()`
**Use**: Returns boolean Series denoting duplicate rows
**Syntax**: `df.duplicated(subset=None, keep='first')`

### `df.isnull()`
**Use**: Detects missing values (returns boolean)
**Syntax**: `df.isnull()`

### `df.notnull()`
**Use**: Detects non-missing values (returns boolean)
**Syntax**: `df.notnull()`

### `df.isna()`
**Use**: Alias for isnull()
**Syntax**: `df.isna()`

### `df.notna()`
**Use**: Alias for notnull() 
**Syntax**: `df.notna()`

## Selection and Slicing

### `df[column_name]`
**Use**: Selects a single column as Series
**Syntax**: `df[column_name]`

### `df[[column_list]]`
**Use**: Selects multiple columns as DataFrame
**Syntax**: `df[[col1, col2, ...]]`

### `df.loc[]`
**Use**: Label-based indexing for rows and columns
**Syntax**: `df.loc[row_label, column_label]`

### `df.iloc[]`
**Use**: Integer position-based indexing
**Syntax**: `df.iloc[row_position, column_position]`

### `df.at[]`
**Use**: Label-based access to a single value
**Syntax**: `df.at[row_label, column_label]`

### `df.iat[]`
**Use**: Integer position-based access to a single value
**Syntax**: `df.iat[row_position, column_position]`

### `df.xs()`
**Use**: Returns cross-section from DataFrame
**Syntax**: `df.xs(key, axis=0, level=None, drop_level=True)`

## Filtering and Conditional Selection

### `df[condition]`
**Use**: Filters rows based on condition
**Syntax**: `df[df[column] > value]`

### `df.query()`
**Use**: Queries DataFrame with a boolean expression
**Syntax**: `df.query(expr, inplace=False, **kwargs)`

### `df.where()`
**Use**: Replace values where condition is False
**Syntax**: `df.where(cond, other=nan, inplace=False, axis=None, level=None)`

### `df.mask()`
**Use**: Replace values where condition is True
**Syntax**: `df.mask(cond, other=nan, inplace=False, axis=None, level=None)`

### `df.nlargest()`
**Use**: Return the n largest values
**Syntax**: `df.nlargest(n, columns, keep='first')`

### `df.nsmallest()`
**Use**: Return the n smallest values
**Syntax**: `df.nsmallest(n, columns, keep='first')`

## Grouping and Aggregation

### `df.groupby()`
**Use**: Groups DataFrame using a mapper or by a Series of columns
**Syntax**: `df.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=<no_default>, observed=False, dropna=True)`

### `df.agg()` / `df.aggregate()`
**Use**: Aggregates using one or more operations
**Syntax**: `df.agg(func=None, axis=0, *args, **kwargs)`

### `df.transform()`
**Use**: Calls function on self producing a DataFrame with transformed values
**Syntax**: `df.transform(func, axis=0, *args, **kwargs)`

### `df.apply()`
**Use**: Applies function along an axis of DataFrame
**Syntax**: `df.apply(func, axis=0, raw=False, result_type=None, args=(), **kwargs)`

### `df.pivot_table()`
**Use**: Creates spreadsheet-style pivot table
**Syntax**: `df.pivot_table(values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False, sort=True)`

### `df.crosstab()`
**Use**: Computes a cross-tabulation of two or more factors
**Syntax**: `pd.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False)`

### `df.cut()`
**Use**: Bins values into discrete intervals
**Syntax**: `pd.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True)`

### `df.qcut()`
**Use**: Bins values into equal-sized buckets based on rank or sample quantiles
**Syntax**: `pd.qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise')`

## Handling Missing Data

### `df.dropna()`
**Use**: Removes missing values
**Syntax**: `df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)`

### `df.fillna()`
**Use**: Fills missing values
**Syntax**: `df.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)`

### `df.interpolate()`
**Use**: Fills NaN values using interpolation methods
**Syntax**: `df.interpolate(method='linear', axis=0, limit=None, inplace=False, limit_direction=None, limit_area=None, downcast=None, **kwargs)`

### `df.replace()`
**Use**: Replaces values
**Syntax**: `df.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')`

## Data Transformation

### `df.rename()`
**Use**: Renames labels
**Syntax**: `df.rename(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore')`

### `df.set_index()`
**Use**: Sets DataFrame index using existing columns
**Syntax**: `df.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)`

### `df.reset_index()`
**Use**: Resets index to default integer index
**Syntax**: `df.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')`

### `df.sort_values()`
**Use**: Sorts by values
**Syntax**: `df.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)`

### `df.sort_index()`
**Use**: Sorts by index
**Syntax**: `df.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None)`

### `df.drop()`
**Use**: Drops specified labels
**Syntax**: `df.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')`

### `df.drop_duplicates()`
**Use**: Removes duplicate rows
**Syntax**: `df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)`

## Adding and Updating Data

### `df.assign()`
**Use**: Assigns new columns to DataFrame
**Syntax**: `df.assign(**kwargs)`

### `df[new_column] = values`
**Use**: Creates a new column
**Syntax**: `df[new_column_name] = value_or_array`

### `pd.concat()`
**Use**: Concatenates DataFrames along an axis
**Syntax**: `pd.concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True)`

### `df.append()`
**Use**: Appends rows of other to end of caller
**Syntax**: `df.append(other, ignore_index=False, verify_integrity=False, sort=False)`

### `df.merge()`
**Use**: Merges DataFrames
**Syntax**: `df.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)`

### `pd.merge()`
**Use**: Merges DataFrames
**Syntax**: `pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)`

### `df.join()`
**Use**: Joins columns of another DataFrame
**Syntax**: `df.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)`

## Data Summarization

### `df.sum()`
**Use**: Returns sum of values
**Syntax**: `df.sum(axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs)`

### `df.mean()`
**Use**: Returns mean of values
**Syntax**: `df.mean(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)`

### `df.median()`
**Use**: Returns median of values
**Syntax**: `df.median(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)`

### `df.min()`
**Use**: Returns minimum of values
**Syntax**: `df.min(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)`

### `df.max()`
**Use**: Returns maximum of values
**Syntax**: `df.max(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)`

### `df.count()`
**Use**: Counts non-NA cells for each column or row
**Syntax**: `df.count(axis=0, level=None, numeric_only=False)`

### `df.std()`
**Use**: Returns standard deviation
**Syntax**: `df.std(axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs)`

### `df.var()`
**Use**: Returns unbiased variance
**Syntax**: `df.var(axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs)`

### `df.corr()`
**Use**: Computes pairwise correlation of columns
**Syntax**: `df.corr(method='pearson', min_periods=1)`

### `df.cov()`
**Use**: Computes pairwise covariance of columns
**Syntax**: `df.cov(min_periods=None, ddof=1)`

### `df.pct_change()`
**Use**: Calculates percentage change between elements
**Syntax**: `df.pct_change(periods=1, fill_method='pad', limit=None, freq=None, **kwargs)`

## Data Type Conversion

### `df.astype()`
**Use**: Casts to a specified dtype
**Syntax**: `df.astype(dtype, copy=True, errors='raise')`

### `pd.to_numeric()`
**Use**: Converts argument to a numeric type
**Syntax**: `pd.to_numeric(arg, errors='raise', downcast=None)`

### `pd.to_datetime()`
**Use**: Converts argument to datetime
**Syntax**: `pd.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)`

### `pd.to_timedelta()`
**Use**: Converts argument to timedelta
**Syntax**: `pd.to_timedelta(arg, unit=None, errors='raise')`

## Data Export

### `df.to_csv()`
**Use**: Writes to CSV file
**Syntax**: `df.to_csv(path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.')`

### `df.to_excel()`
**Use**: Writes to Excel file
**Syntax**: `df.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True, freeze_panes=None, storage_options=None)`

### `df.to_dict()`
**Use**: Converts DataFrame to dictionary
**Syntax**: `df.to_dict(orient='dict', into=dict)`

### `df.to_json()`
**Use**: Converts DataFrame to JSON string
**Syntax**: `df.to_json(path_or_buf=None, orient=None, date_format=None, double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', index=True, indent=None)`

---

## 🔍 1. **Data Loading / Inspection**

```python
import pandas as pd

# Load data
df = pd.read_csv("data.csv")  # Read CSV
df = pd.read_excel("data.xlsx")  # Read Excel
df = pd.read_json("data.json")  # Read JSON

# Inspect
df.head(5)             # First 5 rows
df.tail(5)             # Last 5 rows
df.info()              # Data types, non-null info
df.describe()          # Summary statistics (numerical)
df.shape               # (rows, columns)
df.columns             # Column names
df.dtypes              # Data types of each column
df.index               # Index of the dataframe
df.sample(5)           # Random 5 rows
df.memory_usage()      # Memory usage
```

---

## 🧹 2. **Data Cleaning**

```python
df.isnull()                    # Check for missing values (boolean)
df.notnull()                  # Opposite of isnull
df.dropna()                   # Drop rows with any NaNs
df.dropna(axis=1)             # Drop columns with any NaNs
df.fillna(0)                  # Fill NaNs with 0
df.fillna(method='ffill')    # Forward fill
df.replace("old", "new")     # Replace values

df.duplicated()              # Check duplicate rows (boolean)
df.drop_duplicates()         # Drop duplicate rows

df.astype(int)               # Convert type
df["date"] = pd.to_datetime(df["date"])  # Convert to datetime
df["col"].apply(lambda x: x.upper())     # Apply function to column
```

---

## 🔄 3. **Updating / Creating Columns**

```python
df["new_col"] = df["old_col"] * 2     # Create/modify column
df.rename(columns={"old": "new"})     # Rename columns

# Insert at specific position
df.insert(loc=1, column="inserted", value=df["col1"] + 1)

# Assign multiple columns
df = df.assign(double_A=df["A"]*2, triple_B=df["B"]*3)

# Evaluate string expressions
df.eval("C = A + B", inplace=True)
```

---

## 📊 4. **Grouping and Aggregation**

```python
grouped = df.groupby("col")           # Group by a column
grouped.mean()                        # Mean for each group
grouped.agg({"col1": "sum", "col2": "mean"})  # Multiple aggregations

# Pivot table
pd.pivot_table(df, index="A", columns="B", values="C", aggfunc="sum")

df.cumsum()        # Cumulative sum
df.cumprod()       # Cumulative product

# Rolling window (e.g., 3)
df["rolling_mean"] = df["col"].rolling(window=3).mean()

# Expanding window
df["expanding_sum"] = df["col"].expanding().sum()

# Transform group values (broadcast result back)
df["normalized"] = df.groupby("category")["value"].transform(lambda x: x / x.sum())
```

---

## 🔢 5. **Conditional Filtering**

```python
df[df["col"] > 10]                       # Filter rows where col > 10
df[(df["A"] > 5) & (df["B"] < 10)]      # Multiple conditions with & and |
df.query("A > 5 and B < 10")            # Query string filter

# Update conditionally
df.loc[df["col"] > 10, "col"] = 100
```

---

## ✂️ 6. **Slicing and Indexing**

```python
df.loc[0, "col"]          # Row by label
df.iloc[0, 0]             # Row by index
df.loc[:, ["A", "B"]]     # All rows, selected columns
df.iloc[1:4]              # Rows 1 to 3

df.set_index("id", inplace=True)   # Set column as index
df.reset_index(inplace=True)       # Reset index

df.sort_values("col")              # Sort by column ascending
df.sort_values("col", ascending=False)  # Descending
df.sort_index()                    # Sort by index
```

---

## 🗑️ 7. **Dropping Rows/Columns**

```python
df.drop("col1", axis=1, inplace=True)            # Drop column
df.drop(columns=["col1", "col2"], inplace=True)  # Drop multiple columns
df.drop([0, 1], axis=0, inplace=True)            # Drop rows by index
df.drop(index=[2, 3], inplace=True)              # Drop rows using index keyword
```

---

## 🔗 8. **Merging / Joining / Concatenation**

```python
# Concatenation
pd.concat([df1, df2], axis=0)        # Stack rows
pd.concat([df1, df2], axis=1)        # Stack columns

# Merge
pd.merge(df1, df2, on="key")                   # Inner join
pd.merge(df1, df2, on="key", how="left")       # Left join
pd.merge(df1, df2, on="key", how="outer")      # Outer join

# Join on index
df1.join(df2, lsuffix='_left', rsuffix='_right')  # Index-based join
```

---

## 🔁 9. **Mapping / Encoding / Functions**

```python
df["col"].map({"A": 1, "B": 2})        # Map values via dict
df["col"].apply(lambda x: x**2)        # Apply custom function
df["category_encoded"] = pd.get_dummies(df["category"])  # One-hot (not recommended like this for full encoding)
pd.get_dummies(df, columns=["category", "gender"])       # One-hot encode multiple columns
```

---

## 📈 10. **Basic Visualization (Optional Pandas API)**

```python
import matplotlib.pyplot as plt

df["col"].plot()                # Line plot
df["col"].hist()                # Histogram
df.boxplot(column=["col1", "col2"])  # Box plot
df.plot.scatter(x="A", y="B")   # Scatter plot
plt.show()

# Correlation heatmap
df.corr()                       # Get correlation matrix
```