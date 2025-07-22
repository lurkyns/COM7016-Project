import streamlit as st

def welcome():
    # Set full-width layout
    st.set_page_config(layout="wide")

    # Centered title with emojis
    st.markdown("<h1 style='text-align: center; color: #3A3A3A;'> Can I Eat This? </h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #6C6C6C;'>Your smart diabetes-friendly food checker</h4>", unsafe_allow_html=True)

    # Description with better formatting
    st.markdown("""
    <div style='padding: 20px; background-color: #F9F9F9; border-radius: 10px;'>
        <p style='font-size: 18px; color: #333333;'>
         Welcome! This app helps you find out if a food is suitable for people living with diabetes.  
        Just type the food name and get instant feedback, alternatives, and smart nutrition insights.  
        <br><br>
        <b> Features:</b>  
        - AI-powered analysis of common foods  
        - Suggestions for healthier alternatives  
        - Explanation of why a food may or may not be suitable  
        - Your feedback helps us grow the database!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<p style='text-align: center; font-size: 16px;'>👇 Ready? Use the search bar in the menu or click below to begin!</p>", unsafe_allow_html=True)

    # Button to go to next page 
    if st.button(" Start Checking Food"):
        st.session_state.page = "search"
