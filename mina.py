import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

st.set_page_config(page_title="Online Retail Segmentation", layout="wide")

# ===============================
# 1️⃣ تجهيز البيانات (Excel → CSV)
# ===============================
excel_file = "Online Retail.xlsx"
csv_file = "online_retail_small.csv"

if os.path.exists(excel_file) and not os.path.exists(csv_file):
    df_excel = pd.read_excel(excel_file)
    df_small = df_excel[['CustomerID','InvoiceDate','Quantity','UnitPrice']]
    df_small.to_csv(csv_file, index=False)
    st.success(f"✅ CSV file '{csv_file}' created from Excel")

# ===============================
# 2️⃣ تحميل CSV
# ===============================
if not os.path.exists(csv_file):
    st.error(f"{csv_file} not found. Please upload Excel file first.")
    st.stop()

df = pd.read_csv(csv_file)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

st.title("Online Retail Customer Segmentation")

# ===============================
# 3️⃣ حساب RFM
# ===============================
NOW = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (NOW - x.max()).days,  # Recency
    'InvoiceNo': 'count' if 'InvoiceNo' in df.columns else 'count',  # Frequency
    'TotalAmount': lambda x: (df.loc[x.index,'Quantity']*df.loc[x.index,'UnitPrice']).sum()  # Monetary
})

rfm.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalAmount':'Monetary'}, inplace=True)

# ===============================
# 4️⃣ تدريب K-Means
# ===============================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# ===============================
# 5️⃣ الرسومات
# ===============================
st.subheader("Monetary Distribution")
fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.hist(rfm['Monetary'], bins=20, color='skyblue', edgecolor='black')
ax1.set_xlabel("Monetary")
ax1.set_ylabel("Number of Customers")
ax1.set_title("Customer Monetary Distribution")
st.pyplot(fig1)

st.subheader("Recency vs Frequency Clustered")
fig2, ax2 = plt.subplots(figsize=(10,4))
scatter = ax2.scatter(rfm['Recency'], rfm['Frequency'], c=rfm['Cluster'], cmap='viridis', s=50)
ax2.set_xlabel("Recency (days)")
ax2.set_ylabel("Frequency")
ax2.set_title("Recency vs Frequency")
st.pyplot(fig2)

# ===============================
# 6️⃣ واجهة لتوقع Cluster العملاء الجدد
# ===============================
st.subheader("Predict Cluster for New Customer")
recency = st.number_input("Recency (days since last purchase)", min_value=0)
frequency = st.number_input("Frequency (number of orders)", min_value=1)
monetary = st.number_input("Monetary (total spending)", min_value=0.0)

if st.button("Predict Cluster"):
    new_data = np.array([[recency, frequency, monetary]])
    new_scaled = scaler.transform(new_data)
    cluster = kmeans.predict(new_scaled)
    st.success(f"This customer belongs to Cluster {cluster[0]}")

# ===============================
# 7️⃣ حفظ الموديلات
# ===============================
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
st.info("✅ KMeans model and scaler saved for future use.")
