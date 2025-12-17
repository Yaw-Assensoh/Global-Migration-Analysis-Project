# pages/9_Economic_Impact.py - SUPER SIMPLE
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Economic Impact", page_icon="💰")
st.title(" How Migration Affects Economies")

# Simple data table
st.subheader("Country Data")
data = pd.DataFrame({
    'Country': ['USA', 'Germany', 'UK', 'Canada', 'India', 'Nigeria'],
    'Income Level': ['High', 'High', 'High', 'High', 'Low', 'Low'],
    'GDP per Person': ['$65,000', '$48,000', '$42,000', '$52,000', '$2,100', '$2,300'],
    'Migration': ['+3.2 (Immigration)', '+1.8 (Immigration)', '+2.1 (Immigration)', 
                  '+5.6 (Immigration)', '-1.2 (Emigration)', '-3.1 (Emigration)']
})

st.dataframe(data, use_container_width=True)

# Simple explanation
st.subheader("What This Means")

st.markdown("""
###  The Pattern:
**Richer countries** (USA, Germany, Canada) → **Get immigrants** (+ numbers)  
**Poorer countries** (India, Nigeria) → **Lose emigrants** (- numbers)

###  Economic Impact:
1. **Immigration helps rich countries:**
   - More workers
   - More taxpayers
   - More consumers

2. **Emigration hurts poor countries:**
   - Lose skilled workers ("brain drain")
   - Get money from abroad ("remittances")

###  Quick Example:
**USA** gets immigrants → Economy grows faster  
**Nigeria** loses emigrants → Harder to develop

###  Simple Conclusion:
Migration flows from **poor → rich** countries. This helps rich countries grow but can make it harder for poor countries to develop.
""")

# Interactive part
st.subheader("Try It Yourself")

country = st.selectbox("Pick a country:", data['Country'].tolist())
row = data[data['Country'] == country].iloc[0]

st.markdown(f"""
**{country} Statistics:**
- Income: {row['Income Level']}
- GDP per person: {row['GDP per Person']}
- Migration: {row['Migration']}
""")

# Simple prediction
if row['Income Level'] == 'High':
    st.success(f"✅ {country} is rich → Gets immigrants → Economy benefits")
else:
    st.warning(f"⚠️ {country} is poor → Loses emigrants → Development challenge")

# Footer
st.markdown("---")
st.write("*Simple explanation of migration economics*")