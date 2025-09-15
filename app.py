import streamlit as st
import pandas as pd
import joblib

model, feature_cols = joblib.load("instagram_reach_model.pkl")

st.title("Instagram Reach Predictor")

# Input fields for features
impressions_home = st.number_input("From Home", min_value=0, value=100)
impressions_hashtags = st.number_input("From Hashtags", min_value=0, value=50)
impressions_explore = st.number_input("From Explore", min_value=0, value=30)
impressions_other = st.number_input("From Other", min_value=0, value=10)
saves = st.number_input("Saves", min_value=0, value=5)
comments = st.number_input("Comments", min_value=0, value=2)
shares = st.number_input("Shares", min_value=0, value=1)
likes = st.number_input("Likes", min_value=0, value=20)
profile_visits = st.number_input("Profile Visits", min_value=0, value=10)
follows = st.number_input("Follows", min_value=0, value=1)

# Create input DataFrame
input_df = pd.DataFrame([[
    impressions_home, impressions_hashtags, impressions_explore, impressions_other,
    saves, comments, shares, likes, profile_visits, follows
]], columns=feature_cols)

# Predict reach
if st.button("Predict Reach"):
    predicted_reach = model.predict(input_df)
    st.success(f"Predicted Impressions (Reach): {int(predicted_reach[0])}")