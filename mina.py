import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ===============================
# 1️⃣ قراءة الداتا
# ===============================
df = pd.read_excel("Online Retail.xlsx")
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Quantity'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# ===============================
# 2️⃣ Streamlit واجهة
# ===============================
st.title("Online Retail Data Visualization")

# إجمالي المبيعات لكل دولة
sales_by_country = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False)

st.subheader("Total Sales by Country")
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.bar(sales_by_country.index, sales_by_country.values, color='skyblue')
ax1.set_ylabel("Total Sales")
ax1.set_xlabel("Country")
ax1.set_title("Total Sales per Country")
plt.xticks(rotation=45)
st.pyplot(fig1)

# متوسط الكمية لكل منتج
st.subheader("Top 20 Products by Quantity Sold")
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(20)
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.barh(top_products.index[::-1], top_products.values[::-1], color='orange')
ax2.set_xlabel("Quantity Sold")
ax2.set_ylabel("Product")
ax2.set_title("Top 20 Products by Quantity Sold")
st.pyplot(fig2)

# المبيعات الشهرية
st.subheader("Monthly Sales Trend")
df['Month'] = df['InvoiceDate'].dt.to_period('M')
monthly_sales = df.groupby('Month')['TotalAmount'].sum()

fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o', linestyle='-', color='green')
ax3.set_xlabel("Month")
ax3.set_ylabel("Total Sales")
ax3.set_title("Monthly Sales Trend")
plt.xticks(rotation=45)
st.pyplot(fig3)
