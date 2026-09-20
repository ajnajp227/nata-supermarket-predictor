import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nata Supermarkets - Customer Predictor",
    page_icon="🛒",
    layout="centered",
)

st.title("🛒 Nata Supermarkets: Customer Spending Predictor")
st.markdown(
    "Enter customer demographic and shopping channel details below to get instant predictions on high-value purchasing likelihood."
)

st.divider()


# --- 2. TRAIN BASELINE MODEL ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 1000

    incomes = np.random.normal(52000, 20000, n_samples).clip(10000, 150000)
    educations = np.random.choice(
        ["Graduation", "PhD", "Master", "Basic", "2n Cycle"], n_samples
    )
    store_purchases = np.random.randint(1, 15, n_samples)
    web_purchases = np.random.randint(0, 12, n_samples)
    catalog_purchases = np.random.randint(0, 10, n_samples)

    high_wine_spender = (
        (incomes > 45000)
        & (np.isin(educations, ["PhD", "Master", "Graduation"]))
    ) | (catalog_purchases > 3)
    target = high_wine_spender.astype(int)

    df_train = pd.DataFrame(
        {
            "Income": incomes,
            "Education": educations,
            "NumStorePurchases": store_purchases,
            "NumWebPurchases": web_purchases,
            "NumCatalogPurchases": catalog_purchases,
            "Target": target,
        }
    )

    le = LabelEncoder()
    df_train["Education_Enc"] = le.fit_transform(df_train["Education"])

    X = df_train[
        [
            "Income",
            "Education_Enc",
            "NumStorePurchases",
            "NumWebPurchases",
            "NumCatalogPurchases",
        ]
    ]
    y = df_train["Target"]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    return model, le


model, le_education = train_model()

# --- 3. USER INPUT FORM ---
st.header("📋 Input Customer Profile")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income ($)",
        min_value=10000,
        max_value=200000,
        value=55000,
        step=2500,
    )
    education = st.selectbox(
        "Education Level",
        options=["Graduation", "PhD", "Master", "Basic", "2n Cycle"],
    )

with col2:
    store_purchases = st.slider("In-Store Purchases / Year", 0, 25, 6)
    web_purchases = st.slider("Web Purchases / Year", 0, 20, 4)
    catalog_purchases = st.slider("Catalog Purchases / Year", 0, 15, 2)

st.divider()

# --- 4. PREDICTION LOGIC & DISPLAY ---
if st.button("🔮 Predict Customer Behavior", use_container_width=True):
    edu_encoded = le_education.transform([education])[0]
    input_data = np.array(
        [[income, edu_encoded, store_purchases, web_purchases, catalog_purchases]]
    )

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("🎯 Prediction Results")

    if prediction == 1:
        st.success("🍷 **High-Value Wine & Meat Buyer Target**")
        st.write(
            f"**Likelihood:** `{probabilities[1]*100:.1f}%` probability of contributing over 50% spend to high-margin categories."
        )
        st.info(
            "**Strategic Recommendation:** Target this customer with exclusive Wine Tasting invitations, Direct Mail Catalogs, and Premium Meat Pairing bundles."
        )
    else:
        st.warning("🛒 **Standard Value / Value-Focused Customer**")
        st.write(
            f"**Likelihood:** `{probabilities[0]*100:.1f}%` probability of basic/essential category basket spend."
        )
        st.info(
            "**Strategic Recommendation:** Target with In-Store discount vouchers, essential basket deals, and digital web promotions to build basket size."
        )
