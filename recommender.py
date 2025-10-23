import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Set page configuration
st.set_page_config(
    page_title="Investment Portfolio Recommender",
    page_icon="💰",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .recommendation-card {
        background-color: #d21034;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .message-card {
        background-color: #d21034;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .customer-select {
        text-align: center;
        padding: 20px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load existing customer models and data
@st.cache_resource
def load_existing_customer_models():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('tfidf.pkl', 'rb') as file:
            tfidf = pickle.load(file)
        df = pd.read_csv('investment_member.csv')
        return model, tfidf, df
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Load new customer market data
@st.cache_data
def load_new_customer_data():
    mmf_data = {
        'Fund': [
            'Cytonn Money Market Fund', 
            'Lofty Corban Money Market Fund', 
            'Etica Money Market Fund', 
            'ArvoCap Money Market Fund', 
            'Kuza Money Market Fund', 
            'GenAfrica Money Market Fund', 
            'Nabo Africa Money Market Fund', 
            'Jubilee Money Market Fund', 
            'Madison Money Market Fund', 
            'Co-op Money Market Fund', 
            'KCB Money Market Fund', 
            'Sanlam Money Market Fund', 
            'ABSA Shilling MMF', 
            'Enwealth Money Market Fund', 
            'Mail Money Market Fund', 
            'Mayfair Money Market Fund', 
            'Faulu Money Market Fund', 
            'Orient Kasha Money Market Fund', 
            'Genghis Money Market Fund', 
            'African Alliance Kenya Money Market Fund', 
            'Dry Associates Money Market Fund', 
            'Old Mutual Money Market Fund', 
            'Apollo Money Market Fund', 
            'CIC Money Market Fund', 
            'ICEA Lion Money Market Fund', 
            'British-American Money Market Fund', 
            'Equity Money Market Fund'
        ],
        'Return': [
            18.07, 17.96, 17.11, 17.09, 16.88, 16.51, 15.66, 15.61, 15.43, 15.36, 15.17, 
            15.13, 15.03, 15.00, 15.01, 15.00, 14.86, 14.81, 14.74, 14.56, 14.39, 13.97, 
            13.91, 13.75, 13.61, 13.29, 13.23
        ]
    }
    
    sacco_data = {
        'Name': [
            'MWALIMU NATIONAL', 'STIMA DT', 'KENYA NATIONAL POLICE DT', 'HARAMBEE', 'TOWER',
            'AFYA', 'UNAITAS', 'IMARISHA', 'UNITED NATIONS DT', 'UKULIMA',
            'HAZINA', 'GUSII MWALIMU', 'INVEST AND GROW', 'MENTOR', 'IMARIKA',
            'BANDARI DT', 'TRANSNATION', 'SAFARICOM', 'BORESHA', 'WINAS',
            'NEWFORTIS', 'KIMISITU'
        ],
        'Total_Assets': [
            66.43, 59.15, 54.24, 38.57, 23.23,
            22.79, 22.70, 21.78, 18.21, 15.18,
            14.76, 14.30, 14.06, 13.47, 13.11,
            12.68, 12.02, 11.72, 11.25, 11.28,
            10.67, 11.07
        ]
    }
    dollar_funds = {
        'Provider': ['NCBA', 'CIC', 'Jubilee'],
        'Rate': [3.98, 5.0, 5.79]
    }
    
    fixed_deposits = {
        'Provider': ['NCBA', 'CIC', 'Jubilee', 'Madison', 'Sanlam'],
        'Rate': [11.65, 12.0, 15.56, 13.0, 17.6]
    }
    
    return pd.DataFrame(mmf_data), pd.DataFrame(sacco_data),pd.DataFrame(dollar_funds),pd.DataFrame(fixed_deposits)
# Retrieve the data for new customer market
mmf_data, sacco_data,dollar_funds, fixed_deposits = load_new_customer_data()

def get_recommendations_with_messages(member_features, df, member_data, n=5):
    """
    Get recommendations with personalized messages for existing customers
    """
    recommended_products = []
    messages = []
    
    member_beneficiery_age = member_data.get('beneficiary_age')
    member_age_group = member_data.get('age_group')
    member_town = member_data.get('town')
    member_gender = member_data.get('gender')
    member_current_products = set(member_data.get('current_products', []))

    # Rule 1: Beneficiary age recommendations
    if member_beneficiery_age is not None:
        if 18 <= member_beneficiery_age <= 25:
            recommended_products.append("Student Account")
            messages.append(
                "Planning for your child's future? Our Student Account is perfect for "
                "young adults aged 18-25. Start securing their educational journey today!"
            )
        elif member_beneficiery_age < 18:
            recommended_products.append("Junior Account")
            messages.append(
                "Give your child a head start with our Junior Account! It's specially designed "
                "for children under 18 to help them develop good financial habits early."
            )

    # Rule 2: Age group recommendations
    age_group_products = (
        df[df['age_group'] == member_age_group]
        .portfolio_map.value_counts()
        .index
        .tolist()
    )
    for product in age_group_products:
        if product not in member_current_products and product not in recommended_products:
            recommended_products.append(product)
            if len(recommended_products) >= n:
                break

    if age_group_products:
        messages.append(
            f"Members in your age group are enjoying these popular products: "
            f"{', '.join(age_group_products[:3])}. Join them in making smart financial choices!"
        )

    # Rule 3: Location-based recommendations
    town_products = (
        df[df['town'] == member_town]
        .portfolio_map.value_counts()
        .index
        .tolist()
    )
    for product in town_products:
        if product not in member_current_products and product not in recommended_products:
            recommended_products.append(product)
            if len(recommended_products) >= n:
                break

    if town_products:
        messages.append(
            f"Trending in {member_town}! Your neighbors are choosing "
            f"{', '.join(town_products[:3])}. Discover why these products are popular in your community!"
        )

    # Final personalized message
    if recommended_products:
        messages.append(
            f"💡 Pro tip: Adding {', '.join(recommended_products[:n])} to your portfolio "
            f"could help you achieve your financial goals faster!"
        )

    return recommended_products[:n], messages

def calculate_risk_score(answers):
    """
    Calculate risk score for new customers based on their answers
    """
    score = 0
    
    duration_weights = {
        'Less than 1 year': 1,
        '1-3 years': 2,
        '3-5 years': 3,
        'More than 5 years': 4
    }
    score += duration_weights.get(answers['investment_duration'], 0)
    
    if answers['emergency_fund'] == 'Yes':
        score -= 2
    
    withdrawal_weights = {
        'Very frequently (weekly)': 1,
        'Frequently (monthly)': 2,
        'Occasionally (quarterly)': 3,
        'Rarely (yearly or less)': 4
    }
    score += withdrawal_weights.get(answers['withdrawal_frequency'], 0)
    
    risk_weights = {
        'Very Low': 1,
        'Low': 2,
        'Medium': 3,
        'High': 4,
        'Very High': 5
    }
    score += risk_weights.get(answers['risk_appetite'], 0) * 2
    
    return score

# def get_investment_recommendations(risk_score, investment_amount,currency, loan_access):
    """
    Get investment recommendations for new customers based on risk score
    """
    recommendations = []
    # Dynamic selection of top 2 MMFs and SACCOs
    top_2_mmf = mmf_data.nlargest(2, 'Return')
    top_2_saccos = sacco_data.nlargest(2, 'Total_Assets')
    
    # Currency-specific recommendations
    if currency == 'USD':
        top_2_dollar_funds = dollar_funds.nlargest(2, 'Rate')
        recommendations.append({
            'product': 'Dollar Funds',
            'allocation': 50,
            'description': 'USD Investment with top-performing providers',
            'recommended_providers': top_2_dollar_funds['Provider'].tolist()
        })
    
    # Loan access recommendations
    if loan_access:
        recommendations.append({
            'product': 'SACCOs',
            'allocation': 30,
            'description': 'Loan access with top SACCOs',
            'recommended_providers': top_2_saccos['Name'].tolist()
        })
    if risk_score <= 5:  # Very Conservative
        recommendations.append({
            'product': 'Money Market Funds',
            'allocation': 70,
            'description': 'Low risk, high liquidity, suitable for emergency funds',
            # 'recommended_providers': ['Cytonn MMF', 'Lofty Goshan MMF']
            'recommended_providers': top_2_mmf['Fund'].tolist()
        })
        top_2_fixed_deposits = fixed_deposits.nlargest(2, 'Rate')
        recommendations.append({
            'product': 'Fixed Deposits',
            'allocation': 30,
            'description': 'Low risk, stable returns, limited liquidity',
            #'recommended_providers': ['Top tier banks']
            'recommended_providers': top_2_fixed_deposits['Provider'].tolist()
        })
    
    elif risk_score <= 8:  # Conservative
        recommendations.append({
            'product': 'Money Market Funds',
            'allocation': 50,
            'description': 'Low risk, high liquidity',
            'recommended_providers': ['Cytonn MMF', 'Etica MMF']
        })
        recommendations.append({
            'product': 'SACCOs',
            'allocation': 30,
            'description': 'Moderate risk, good for loans',
            'recommended_providers': ['Mwalimu National', 'Stima DT']
        })
        recommendations.append({
            'product': 'Government Bonds',
            'allocation': 20,
            'description': 'Low risk, fixed income',
            'recommended_providers': ['Treasury Direct']
        })
    
    elif risk_score <= 12:  # Balanced
        recommendations.append({
            'product': 'Money Market Funds',
            'allocation': 30,
            'description': 'Low risk, emergency fund',
            'recommended_providers': ['Cytonn MMF', 'Kuza MMF']
        })
        recommendations.append({
            'product': 'SACCOs',
            'allocation': 30,
            'description': 'Moderate risk, loan access',
            'recommended_providers': ['Mwalimu National', 'Kenya Police DT']
        })
        recommendations.append({
            'product': 'Equity Funds',
            'allocation': 40,
            'description': 'Higher risk, growth potential',
            'recommended_providers': ['Top performing equity funds']
        })
    
    else:  # Aggressive
        recommendations.append({
            'product': 'Equity Funds',
            'allocation': 60,
            'description': 'High risk, high potential returns',
            'recommended_providers': ['Leading equity funds']
        })
        recommendations.append({
            'product': 'Money Market Funds',
            'allocation': 20,
            'description': 'Liquidity buffer',
            'recommended_providers': ['Cytonn MMF']
        })
        recommendations.append({
            'product': 'Dollar Funds',
            'allocation': 20,
            'description': 'Currency diversification',
            'recommended_providers': ['Top dollar funds']
        })
    
    return recommendations

# def get_investment_recommendations(risk_score, investment_amount, currency, loan_access):
    """
    Get investment recommendations based on risk score, investment amount, currency, and loan needs
    """
    recommendations = []
    
    # Dynamic selection of top 2 MMFs and SACCOs
    top_2_mmf = mmf_data.nlargest(2, 'Return')
    top_2_saccos = sacco_data.nlargest(2, 'Total_Assets')
    top_2_fixed_deposits = fixed_deposits.nlargest(2, 'Rate')
    top_2_dollar_funds = dollar_funds.nlargest(2, 'Rate')

    # Currency-specific recommendations
    if currency == 'USD':
        recommendations.append({
            'product': 'Dollar Funds',
            'allocation': 50,
            'description': 'USD Investment with top-performing providers',
            'recommended_providers': top_2_dollar_funds['Provider'].tolist()
        })
    
    # Loan access recommendations
    if loan_access:
        recommendations.append({
            'product': 'SACCOs',
            'allocation': 30,
            'description': 'Loan access with top SACCOs',
            'recommended_providers': top_2_saccos['Name'].tolist()
        })
    
    # Very Conservative Risk Profile (risk_score <= 5)
    if risk_score <= 5:
        recommendations.append({
            'product': 'Money Market Funds',
            'allocation': 70,
            'description': 'Low risk, high liquidity with top-performing funds',
            'recommended_providers': top_2_mmf['Fund'].tolist()
        })
        recommendations.append({
            'product': 'Fixed Deposits',
            'allocation': 30,
            'description': 'Low risk, stable returns',
            'recommended_providers': top_2_fixed_deposits['Provider'].tolist()
        })
    
    # Conservative Risk Profile (5 < risk_score <= 8)
    elif risk_score <= 8:
        recommendations.append({
            'product': 'Money Market Funds',
            'allocation': 50,
            'description': 'Low risk, high liquidity',
            'recommended_providers': top_2_mmf['Fund'].tolist()
        })
        recommendations.append({
            'product': 'SACCOs',
            'allocation': 30,
            'description': 'Moderate risk, community-based investments',
            'recommended_providers': top_2_saccos['Name'].tolist()
        })
        recommendations.append({
            'product': 'Government Bonds',
            'allocation': 20,
            'description': 'Low risk, fixed income',
            'recommended_providers': ['Treasury Direct']
        })
    
    # Balanced Risk Profile (8 < risk_score <= 12)
    elif risk_score <= 12:
        recommendations.append({
            'product': 'Money Market Funds',
            'allocation': 30,
            'description': 'Low risk emergency fund allocation',
            'recommended_providers': top_2_mmf['Fund'].tolist()
        })
        recommendations.append({
            'product': 'SACCOs',
            'allocation': 30,
            'description': 'Moderate risk community investments, loan access',
            'recommended_providers': top_2_saccos['Name'].tolist()
        })
        recommendations.append({
            'product': 'Equity Funds',
            'allocation': 40,
            'description': 'Higher risk, growth potential',
            'recommended_providers': ['Top performing equity funds']
        })
    
    # Aggressive Risk Profile (risk_score > 12)
    else:
        recommendations.append({
            'product': 'Equity Funds',
            'allocation': 60,
            'description': 'High risk, high potential returns',
            'recommended_providers': ['Leading equity funds']
        })
        recommendations.append({
            'product': 'Fixed Deposits',
            'allocation': 30,
            'description': 'Liquidity buffer',
            'recommended_providers': top_2_fixed_deposits['Provider'].tolist()
        })
        recommendations.append({
            'product': 'Dollar Funds',
            'allocation': 10,
            'description': 'Currency diversification',
            'recommended_providers': top_2_dollar_funds['Provider'].tolist()
        })
    
    return recommendations


def get_investment_recommendations(risk_score, investment_amount, currency, loan_access):
    """
    Get investment recommendations based on risk score, investment amount, currency, and loan needs
    """
    recommendations = []
    products_recommended = set()
    
    # Dynamic selection of top 2 MMFs and SACCOs
    top_2_mmf = mmf_data.nlargest(2, 'Return')
    top_2_saccos = sacco_data.nlargest(2, 'Total_Assets')
    top_2_fixed_deposits = fixed_deposits.nlargest(2, 'Rate')
    top_2_dollar_funds = dollar_funds.nlargest(2, 'Rate')

    # Currency-specific recommendations
    if currency == 'USD' and 'Dollar Funds' not in products_recommended:
        recommendations.append({
            'product': 'Dollar Funds',
            'allocation': 50,
            'description': 'USD Investment with top-performing providers',
            'recommended_providers': top_2_dollar_funds['Provider'].tolist()
        })
        products_recommended.add('Dollar Funds')
    
    # Loan access recommendations
    if loan_access and 'SACCOs' not in products_recommended:
        recommendations.append({
            'product': 'SACCOs',
            'allocation': 30,
            'description': 'Loan access with top SACCOs',
            'recommended_providers': top_2_saccos['Name'].tolist()
        })
        products_recommended.add('SACCOs')
    
    # Very Conservative Risk Profile (risk_score <= 5)
    if risk_score <= 5:
        if 'Money Market Funds' not in products_recommended:
            recommendations.append({
                'product': 'Money Market Funds',
                'allocation': 70,
                'description': 'Low risk, high liquidity with top-performing funds',
                'recommended_providers': top_2_mmf['Fund'].tolist()
            })
            products_recommended.add('Money Market Funds')
        if 'Fixed Deposits' not in products_recommended:
            recommendations.append({
                'product': 'Fixed Deposits',
                'allocation': 30,
                'description': 'Low risk, stable returns',
                'recommended_providers': top_2_fixed_deposits['Provider'].tolist()
            })
            products_recommended.add('Fixed Deposits')
    
    # Conservative Risk Profile (5 < risk_score <= 8)
    elif risk_score <= 8:
        if 'Money Market Funds' not in products_recommended:
            recommendations.append({
                'product': 'Money Market Funds',
                'allocation': 50,
                'description': 'Low risk, high liquidity',
                'recommended_providers': top_2_mmf['Fund'].tolist()
            })
            products_recommended.add('Money Market Funds')
        if 'SACCOs' not in products_recommended:
            recommendations.append({
                'product': 'SACCOs',
                'allocation': 30,
                'description': 'Moderate risk, community-based investments',
                'recommended_providers': top_2_saccos['Name'].tolist()
            })
            products_recommended.add('SACCOs')
        if 'Government Bonds' not in products_recommended:
            recommendations.append({
                'product': 'Government Bonds',
                'allocation': 20,
                'description': 'Low risk, fixed income',
                'recommended_providers': ['Treasury Direct']
            })
            products_recommended.add('Government Bonds')
    
    # Balanced Risk Profile (8 < risk_score <= 12)
    elif risk_score <= 12:
        if 'Money Market Funds' not in products_recommended:
            recommendations.append({
                'product': 'Money Market Funds',
                'allocation': 30,
                'description': 'Low risk emergency fund allocation',
                'recommended_providers': top_2_mmf['Fund'].tolist()
            })
            products_recommended.add('Money Market Funds')
        if 'SACCOs' not in products_recommended:
            recommendations.append({
                'product': 'SACCOs',
                'allocation': 30,
                'description': 'Moderate risk community investments, loan access',
                'recommended_providers': top_2_saccos['Name'].tolist()
            })
            products_recommended.add('SACCOs')
        if 'Equity Funds' not in products_recommended:
            recommendations.append({
                'product': 'Equity Funds',
                'allocation': 40,
                'description': 'Higher risk, growth potential',
                'recommended_providers': ['Top performing equity funds']
            })
            products_recommended.add('Equity Funds')
    
    # Aggressive Risk Profile (risk_score > 12)
    else:
        if 'Equity Funds' not in products_recommended:
            recommendations.append({
                'product': 'Equity Funds',
                'allocation': 60,
                'description': 'High risk, high potential returns',
                'recommended_providers': ['Leading equity funds']
            })
            products_recommended.add('Equity Funds')
        if 'Fixed Deposits' not in products_recommended:
            recommendations.append({
                'product': 'Fixed Deposits',
                'allocation': 30,
                'description': 'Liquidity buffer',
                'recommended_providers': top_2_fixed_deposits['Provider'].tolist()
            })
            products_recommended.add('Fixed Deposits')
        if 'Dollar Funds' not in products_recommended:
            recommendations.append({
                'product': 'Dollar Funds',
                'allocation': 10,
                'description': 'Currency diversification',
                'recommended_providers': top_2_dollar_funds['Provider'].tolist()
            })
            products_recommended.add('Dollar Funds')
    
    return recommendations
def show_existing_customer_interface():
    """
    Display interface for existing customers
    """
    st.title("📊 Existing Customer Investment Recommendations")
    
    with st.form("recommendation_form"):
        st.subheader("📝 Enter Member Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_group = st.selectbox(
                "Age Group",
                options=['0-18', '19-30', '31-45', '46-60', '60+']
            )
            
            town = st.selectbox(
                "Town",
                options=['NAIROBI', 'MOMBASA', 'KISUMU', 'ELDORET', 'NAKURU', 'JUJA', 'KERICHO']
            )
            
            gender = st.radio(
                "Gender",
                options=['Male', 'Female']
            )

        with col2:
            has_beneficiary = st.checkbox("Has Beneficiary")
            
            beneficiary_age = None
            if has_beneficiary:
                beneficiary_age = st.number_input(
                    "Beneficiary Age",
                    min_value=0,
                    max_value=100,
                    value=18
                )
            
            current_products = st.multiselect(
                "Current Products",
                options=[
                    'Money Market',
                    'Fixed Income',
                    'Equity Fund',
                    'Balanced Fund',
                    'Dollar Fund',
                    'Student Account',
                    'Junior Account'
                ]
            )

        n_recommendations = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=5,
            value=3
        )
        
        submit_button = st.form_submit_button("Get Recommendations")

    if submit_button:
        model, tfidf, df = load_existing_customer_models()
        
        if model is not None and tfidf is not None and df is not None and not df.empty:
            try:
                member_data = {
                    'age_group': age_group,
                    'beneficiary_age': beneficiary_age if has_beneficiary else None,
                    'town': town,
                    'gender': gender,
                    'current_products': current_products
                }
                
                member_features = pd.DataFrame({
                    'member_age': [age_group],
                    'beneficiery_age': [beneficiary_age if has_beneficiary else np.nan],
                    'age_group': [age_group],
                    'gender_mapped': [gender]
                })
                
                member_features['features'] = member_features.astype(str).sum(axis=1)
                features_tfidf = tfidf.transform(member_features['features'])
                
                recommendations, messages = get_recommendations_with_messages(
                    features_tfidf,
                    df,
                    member_data,
                    n=n_recommendations
                )
                
                st.markdown("---")
                st.subheader("🎯 Recommended Products")
                
                rec_col, msg_col = st.columns([1, 2])
                
                with rec_col:
                    for i, product in enumerate(recommendations, 1):
                        st.markdown(
                            f"""
                            <div class="recommendation-card">
                                <h4>{i}. {product}</h4>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                
                with msg_col:
                    for message in messages:
                        st.markdown(
                            f"""
                            <div class="message-card">
                                <p>{message}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
                # Display current portfolio if any
                if current_products:
                    st.markdown("---")
                    st.subheader("📂 Current Portfolio")
                    st.write(", ".join(current_products))
                
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
        else:
            st.error("Unable to load the required models and data. Please check if all files are present.")

def show_new_customer_interface():
    st.title("🌟 New Customer Investment Advisory")
    
    tab1, tab2, tab3 = st.tabs(["Investment Profile", "Market Data", "About"])
    
    with tab1:
        st.header("Investment Profile Questionnaire")
        
        with st.form("investment_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                investment_amount = st.number_input(
                    "How much do you plan to invest? (KES)", 
                    min_value=1000, 
                    value=10000
                )
                
                investment_duration = st.selectbox(
                    "How long do you plan to invest?",
                    ["Less than 1 year", "1-3 years", "3-5 years", "More than 5 years"]
                )
                currency = st.selectbox(
                    "Which currency do you want to invest in?",
                    ["KES", "USD"]
                )
                emergency_fund = st.radio(
                    "Is this an emergency fund account?",
                    ["Yes", "No"]
                )
            
            with col2:
                withdrawal_frequency = st.selectbox(
                    "How often do you plan to withdraw?",
                    ["Very frequently (weekly)", "Frequently (monthly)", 
                     "Occasionally (quarterly)", "Rarely (yearly or less)"]
                )
                
                risk_appetite = st.select_slider(
                    "What is your risk appetite?",
                    options=["Very Low", "Low", "Medium", "High", "Very High"]
                )

                loan_access = st.checkbox(
                    "Do you want access to loans?"
                )
                
                investment_knowledge = st.select_slider(
                    "How would you rate your investment knowledge?",
                    options=["Beginner", "Intermediate", "Advanced"]
                )
            
            submitted = st.form_submit_button("Get Recommendations")
            
            if submitted:
                answers = {
                    'investment_duration': investment_duration,
                    'emergency_fund': emergency_fund,
                    'withdrawal_frequency': withdrawal_frequency,
                    'risk_appetite': risk_appetite
                }
                
                risk_score = calculate_risk_score(answers)
                recommendations = get_investment_recommendations(risk_score, investment_amount,currency,loan_access)
                
                st.success("Based on your profile, here are our recommendations:")
                
                # Display recommendations
                cols = st.columns(len(recommendations))
                for idx, (col, rec) in enumerate(zip(cols, recommendations)):
                    with col:
                        st.markdown(f"### {rec['product']}")
                        st.markdown(f"**Allocation: {rec['allocation']}%**")
                        st.markdown(f"Amount: KES {investment_amount * rec['allocation']/100:,.2f}")
                        st.markdown(f"*{rec['description']}*")
                        st.markdown("**Recommended Providers:**")
                        for provider in rec['recommended_providers']:
                            st.markdown(f"- {provider}")
                
                # Create and display allocation pie chart
                allocation_data = pd.DataFrame([
                    {'Product': r['product'], 'Allocation': r['allocation']}
                    for r in recommendations
                ])
                
                fig = px.pie(allocation_data, 
                           values='Allocation', 
                           names='Product',
                           title='Recommended Portfolio Allocation')
                st.plotly_chart(fig)
    
    with tab2:
        st.header("Current Market Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Money Market Funds")
            st.dataframe(mmf_data)
            
            # Create bar chart for MMF returns
            fig = px.bar(mmf_data, 
                        x='Fund', 
                        y='Return',
                        title='Money Market Fund Returns (%)')
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Top SACCOs by Total Assets")
            st.dataframe(sacco_data)
            
            # Create bar chart for SACCO assets
            fig = px.bar(sacco_data, 
                        x='Name', 
                        y='Total_Assets',
                        title='SACCO Total Assets (Billion KES)')
            st.plotly_chart(fig)
    
    with tab3:
        st.header("About the Investment Advisor")
        st.write("""
        This investment advisory system helps you make informed investment decisions based on your:
        - Investment timeline
        - Risk tolerance
        - Liquidity needs
        - Emergency fund requirements
        
        The system provides personalized recommendations across various investment products including:
        - Money Market Funds
        - SACCOs
        - Fixed Deposits
        - Government Bonds
        - Equity Funds
        - Dollar Funds
        
        All recommendations are based on current market data and best practices in financial planning.
        """)

def main():
    st.title("Welcome to Investment Portfolio Recommender")
    st.markdown("---")
    
    # Customer type selection
    st.markdown("""
        <div class='customer-select'>
            <h2>Are you an existing customer?</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        customer_type = st.radio(
            "Select your customer status:",
            options=["Existing Customer", "New Customer"],
            horizontal=True,
            key="customer_type"
        )
    
    st.markdown("---")
    
    # Show appropriate interface based on selection
    if customer_type == "Existing Customer":
        show_existing_customer_interface()
    else:
        show_new_customer_interface()

if __name__ == "__main__":
    main()