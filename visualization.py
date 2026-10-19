import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("amazon_cleaned.csv")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Top 10 Brands
df["brand"].value_counts().head(10).plot(
    kind="bar", ax=axes[0, 0], color="teal"
)
axes[0, 0].set_title("Top 10 Brands")
axes[0, 0].set_ylabel("Product Count")
axes[0, 0].tick_params(axis="x", rotation=45)

# 2. Category Distribution
df["main_category"].value_counts().head(6).plot(
    kind="pie", ax=axes[0, 1], autopct="%1.1f%%",
    colormap="Set3", ylabel=""
)
axes[0, 1].set_title("Category Distribution")

# 3. Rating Distribution
df["rating"].plot(
    kind="hist", bins=10, ax=axes[1, 0],
    color="cornflowerblue", edgecolor="black"
)
axes[1, 0].set_title("Customer Rating Distribution")
axes[1, 0].set_xlabel("Rating")

# 4. Top 10 Popular Products
df.nlargest(10, "rating_count").set_index(
    "short_name"
)["rating_count"].sort_values().plot(
    kind="barh", ax=axes[1, 1], color="coral"
)
axes[1, 1].set_title("Top 10 Products by Rating Count")
axes[1, 1].set_xlabel("Rating Count")

plt.suptitle("Amazon Product Analysis Dashboard", fontsize=16)
plt.tight_layout()
plt.show()