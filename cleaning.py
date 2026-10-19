import pandas as pd
df=pd.read_csv("amazon.csv")
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
df["rating_count"] = pd.to_numeric(
    df["rating_count"].str.replace(",", "", regex=False),
    errors="coerce",
)
df["rating_count"] = (
    df["rating_count"]
    .fillna(df.groupby("category")["rating_count"].transform("median"))
    .round()
    .astype("Int64")
)
print(df.isnull().sum())

print(df["product_name"].head(10))
df["brand"] = df["product_name"].str.split().str[0]
print(df["brand"].head(10))
df[["product_name", "category"]].head(10)
print(df["category"].head(10))
df["subcategory"] = df["category"].str.split("|").str[-1]
print(df["subcategory"].head(10))
df["main_category"] = df["category"].str.split("|").str[0]
print(df["main_category"].head(10))
df["short_name"] = df["product_name"].str.split().str[:5].str.join(" ")
print(df["short_name"].head(10))
print(df[["main_category", "subcategory", "short_name"]].head(10))

# 1. Check most common brands
print("\nTop 15 Brands:")
print(df["brand"].value_counts().head(15))

# 2. Count products in each main category
print("\nProducts by Main Category:")
print(df["main_category"].value_counts().head(15))

# 3. Check most common subcategories
print("\nTop 15 Subcategories:")
print(df["subcategory"].value_counts().head(15))

# 4. Find longest product names
df["name_length"] = df["product_name"].str.len()

print("\nLongest Product Names:")
print(
    df[["product_name", "name_length"]]
    .sort_values("name_length", ascending=False)
    .head(10)
)

# 5. Find most repeated product names
print("\nMost Common Product Names:")
print(df["product_name"].value_counts().head(10))

# Convert rating to numeric
df["rating"] = pd.to_numeric(
    df["rating"].astype(str).str.strip(),
    errors="coerce"
)

# Convert discount_percentage to numeric
df["discount_percentage"] = pd.to_numeric(
    df["discount_percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip(),
    errors="coerce"
)

# Check data types
print(df[["rating", "discount_percentage"]].dtypes)

# Check missing values
print(df[["rating", "discount_percentage"]].isnull().sum())
df["rating"] = df["rating"].fillna(
    df.groupby("category")["rating"].transform("median")
)
print(df["rating"].isnull().sum())
# Drop rows where rating is null
df = df.dropna(subset=["rating"])

# Check missing values
print(df["rating"].isnull().sum())
df["discounted_price"] = pd.to_numeric(
    df["discounted_price"].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip(),
    errors="coerce"
)
print(df["discounted_price"].isnull().sum())
df["actual_price"] = pd.to_numeric(
    df["actual_price"].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip(),
    errors="coerce"
)
print(df["actual_price"].isnull().sum())
print(df[["discounted_price", "actual_price"]].dtypes)
print(df[["discounted_price", "actual_price"]].isnull().sum())

df = df.drop(columns=["img_link", "product_link"])

df.to_csv("amazon_cleaned.csv", index=False)