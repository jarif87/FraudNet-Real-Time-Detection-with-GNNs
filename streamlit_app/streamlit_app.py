import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import os

# Define FraudGNN class
class FraudGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FraudGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and threshold
try:
    # Try root directory first (Hugging Face Spaces working directory)
    model_path = 'fraud_gnn_model.pth'
    threshold_path = 'optimal_threshold.txt'

    # Fallback: Try relative to src/ (if files are misplaced)
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), 'fraud_gnn_model.pth')
    if not os.path.exists(threshold_path):
        threshold_path = os.path.join(os.path.dirname(__file__), 'optimal_threshold.txt')

    # Alternative: If files are in a 'models/' folder (uncomment if applicable)
    # model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_gnn_model.pth')
    # threshold_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'optimal_threshold.txt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please upload fraud_gnn_model.pth to the repository root.")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Threshold file not found at {threshold_path}. Please upload optimal_threshold.txt to the repository root.")

    model = FraudGNN(input_dim=7, hidden_dim=16, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(threshold_path, 'r') as f:
        threshold = float(f.read())
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or threshold: {e}")
    st.stop()

# City and state mappings
city_mapping = {
    'Atlanta': 0, 'Bronx': 1, 'Brooklyn': 2, 'Chicago': 3, 'Dallas': 4, 'Houston': 5,
    'Indianapolis': 6, 'Las Vegas': 7, 'Los Angeles': 8, 'Louisville': 9, 'Miami': 10,
    'Minneapolis': 11, 'New York': 12, 'ONLINE': 13, 'Orlando': 14, 'Philadelphia': 15,
    'San Antonio': 16, 'San Diego': 17, 'San Francisco': 18, 'Tucson': 19, 'other': 20
}
state_mapping = {
    'AK': 0, 'AL': 1, 'AR': 2, 'AZ': 3, 'Algeria': 4, 'Antigua and Barbuda': 5, 'Argentina': 6,
    'Aruba': 7, 'Australia': 8, 'Austria': 9, 'Azerbaijan': 10, 'Bahrain': 11, 'Bangladesh': 12,
    'Barbados': 13, 'Belarus': 14, 'Belgium': 15, 'Belize': 16, 'Bosnia and Herzegovina': 17,
    'Brazil': 18, 'CA': 19, 'CO': 20, 'CT': 21, 'Cabo Verde': 22, 'Cambodia': 23, 'Canada': 24,
    'Central African Republic': 25, 'Chile': 26, 'China': 27, 'Colombia': 28, 'Costa Rica': 29,
    "Cote d'Ivoire": 30, 'Croatia': 31, 'Czech Republic': 32, 'DC': 33, 'DE': 34, 'Denmark': 35,
    'Dominica': 36, 'Dominican Republic': 37, 'East Timor (Timor-Leste)': 38, 'Ecuador': 39,
    'Egypt': 40, 'Eritrea': 41, 'Estonia': 42, 'FL': 43, 'Fiji': 44, 'Finland': 45, 'France': 46,
    'GA': 47, 'Georgia': 48, 'Germany': 49, 'Ghana': 50, 'Greece': 51, 'Guatemala': 52,
    'Guyana': 53, 'HI': 54, 'Haiti': 55, 'Honduras': 56, 'Hong Kong': 57, 'Hungary': 58,
    'IA': 59, 'ID': 60, 'IL': 61, 'IN': 62, 'Iceland': 63, 'India': 64, 'Indonesia': 65,
    'Ireland': 66, 'Israel': 67, 'Italy': 68, 'Jamaica': 69, 'Japan': 70, 'Jordan': 71,
    'KS': 72, 'KY': 73, 'Kenya': 74, 'Kosovo': 75, 'Kuwait': 76, 'LA': 77, 'Latvia': 78,
    'Lebanon': 79, 'Liberia': 80, 'Lithuania': 81, 'Luxembourg': 82, 'MA': 83, 'MD': 84,
    'ME': 85, 'MI': 86, 'MN': 87, 'MO': 88, 'MS': 89, 'MT': 90, 'Macedonia': 91,
    'Malaysia': 92, 'Malta': 93, 'Mexico': 94, 'Moldova': 95, 'Monaco': 96, 'Morocco': 97,
    'Mozambique': 98, 'Myanmar (Burma)': 99, 'NC': 100, 'ND': 101, 'NE': 102, 'NH': 103,
    'NJ': 104, 'NM': 105, 'NV': 106, 'NY': 107, 'Nauru': 108, 'Netherlands': 109,
    'New Zealand': 110, 'Nicaragua': 111, 'Niger': 112, 'Nigeria': 113, 'Norway': 114,
    'OH': 115, 'OK': 116, 'OR': 117, 'Oman': 118, 'PA': 119, 'Pakistan': 120, 'Panama': 121,
    'Peru': 122, 'Philippines': 123, 'Poland': 124, 'Portugal': 125, 'RI': 126, 'Romania': 127,
    'Russia': 128, 'SC': 129, 'SD': 130, 'Saudi Arabia': 131, 'Senegal': 132, 'Serbia': 133,
    'Seychelles': 134, 'Singapore': 135, 'Slovakia': 136, 'Slovenia': 137, 'Somalia': 138,
    'South Africa': 139, 'South Korea': 140, 'Spain': 141, 'Sri Lanka': 142, 'Sudan': 143,
    'Suriname': 144, 'Sweden': 145, 'Switzerland': 146, 'Syria': 147, 'TN': 148, 'TX': 149,
    'Taiwan': 150, 'Thailand': 151, 'The Bahamas': 152, 'Tunisia': 153, 'Turkey': 154,
    'Tuvalu': 155, 'UT': 156, 'Uganda': 157, 'Ukraine': 158, 'United Arab Emirates': 159,
    'United Kingdom': 160, 'Uruguay': 161, 'Uzbekistan': 162, 'VA': 163, 'VT': 164,
    'Vatican City': 165, 'Vietnam': 166, 'WA': 167, 'WI': 168, 'WV': 169, 'WY': 170,
    'Yemen': 171, 'Zimbabwe': 172
}

def predict_fraud(transactions):
    try:
        df = pd.DataFrame(transactions, columns=[
            'Zipcode', 'Merchant_State_Code', 'User_Frequency_Per_Day',
            'Time_Difference_Hours', 'Merchant_Category_Code',
            'Merchant_City_Code', 'Transaction_Amount'
        ])
        node_features = torch.tensor(df.values, dtype=torch.float).to(device)
        
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        
        if len(df) > 1:
            zipcodes = node_features[:, 0].cpu().numpy()
            edge_list = []
            zipcode_threshold = 1000
            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    if abs(zipcodes[i] - zipcodes[j]) < zipcode_threshold:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
        
        graph_data = Data(x=node_features, edge_index=edge_index).to(device)
        
        if model is None:
            raise ValueError("Model not loaded. Check if fraud_gnn_model.pth exists.")
        
        with torch.no_grad():
            out = model(graph_data)
            out = torch.atleast_1d(out)
            pred_binary = (out > threshold).float().cpu().numpy()
            pred_proba = out.cpu().numpy()
            pred_binary = np.atleast_1d(pred_binary)
            pred_proba = np.atleast_1d(pred_proba)
        
        return pred_binary, pred_proba
    except Exception as e:
        st.error(f"Error in predict_fraud: {e}")
        return None, None

# Custom CSS for highly compact, eye-catching design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(52, 152, 219, 0.5); }
        50% { box-shadow: 0 0 15px rgba(52, 152, 219, 0.8); }
        100% { box-shadow: 0 0 5px rgba(52, 152, 219, 0.5); }
    }
    @keyframes icon-pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .stApp {
        background: #ffffff;
        max-width: 400px;
        margin: 10px auto;
        padding: 10px;
        font-family: 'Poppins', sans-serif;
        border-radius: 10px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border: 2px solid transparent;
        animation: glow 3s infinite;
    }
    /* Alternative Pastel Gradient Design (uncomment to use) */
    /*
    .stApp {
        background: linear-gradient(135deg, #e6f0fa, #f3e5f5);
        max-width: 400px;
        margin: 10px auto;
        padding: 10px;
        font-family: 'Poppins', sans-serif;
        border-radius: 10px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border: 2px solid transparent;
        animation: glow 3s infinite;
    }
    */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > select {
        padding: 5px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 0.8rem;
        background: #f9f9f9;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus, .stSelectbox > div > div > select:focus {
        outline: none;
        border-color: #3498db;
        box-shadow: 0 0 6px rgba(52, 152, 219, 0.7);
    }
    .stSelectbox > div > div > select {
        appearance: none;
        background: #f9f9f9 url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24"><path fill="%23333" d="M7 10l5 5 5-5z"/></svg>') no-repeat right 8px center;
    }
    .stButton > button {
        padding: 6px;
        background: linear-gradient(45deg, #3498db, #ff6f61);
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 0.85rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 111, 97, 0.5);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    .result-box {
        background: #f1f3f5;
        padding: 8px;
        border-radius: 6px;
        text-align: center;
        margin-top: 8px;
        border: 1px solid #ddd;
        animation: glow 3s infinite;
    }
    .result-box h2 {
        font-size: 1rem;
        color: #2c3e50;
        margin-bottom: 4px;
    }
    .result-box p {
        font-size: 0.8rem;
        color: #7f8c8d;
    }
    .fa-shield-alt {
        animation: icon-pulse 2s infinite;
    }
    .form-label {
        font-weight: 600;
        font-size: 0.75rem;
        color: #2c3e50;
        margin-bottom: 3px;
        display: flex;
        align-items: center;
    }
    .form-label i {
        color: #ff6f61;
        margin-right: 5px;
        transition: color 0.3s;
    }
    .form-label i:hover {
        color: #3498db;
    }
    .stForm {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50; font-size: 1.5rem; margin-bottom: 8px;'>
        <i class='fas fa-shield-alt' style='color: #ff6f61; margin-right: 8px;'></i>
        FraudShield
    </h1>
    <p style='text-align: center; font-size: 0.8rem; color: #555; margin-bottom: 8px; line-height: 1.4;'>
        Enter transaction details to detect fraud. Provide accurate zip code, merchant details, and amount.
    </p>
""", unsafe_allow_html=True)

with st.form(key="fraud_form"):
    st.markdown("<div class='form-label'><i class='fas fa-map-marker-alt'></i>Zipcode</div>", unsafe_allow_html=True)
    zipcode = st.number_input("", value=91750.0, step=0.01, format="%.2f", key="zipcode")
    
    st.markdown("<div class='form-label'><i class='fas fa-globe'></i>Merchant State</div>", unsafe_allow_html=True)
    merchant_state = st.selectbox("", sorted(state_mapping.keys()), index=sorted(state_mapping.keys()).index("TX"), key="state")
    
    st.markdown("<div class='form-label'><i class='fas fa-user-clock'></i>User Frequency Per Day</div>", unsafe_allow_html=True)
    user_freq = st.number_input("", value=1.0, step=0.01, format="%.2f", key="freq")
    
    st.markdown("<div class='form-label'><i class='fas fa-hourglass-half'></i>Time Difference (Hours)</div>", unsafe_allow_html=True)
    time_diff = st.number_input("", value=16601.95, step=0.01, format="%.2f", key="time")
    
    st.markdown("<div class='form-label'><i class='fas fa-store'></i>Merchant Category Code</div>", unsafe_allow_html=True)
    merchant_category = st.number_input("", value=5912.0, step=0.01, format="%.2f", key="category")
    
    st.markdown("<div class='form-label'><i class='fas fa-city'></i>Merchant City</div>", unsafe_allow_html=True)
    merchant_city = st.selectbox("", sorted(city_mapping.keys()), index=sorted(city_mapping.keys()).index("Houston"), key="city")
    
    st.markdown("<div class='form-label'><i class='fas fa-dollar-sign'></i>Transaction Amount</div>", unsafe_allow_html=True)
    transaction_amount = st.number_input("", value=128.35, step=0.01, format="%.2f", key="amount")
    
    submit_button = st.form_submit_button("Predict Fraud", use_container_width=True)

    if submit_button:
        try:
            if not all([zipcode, user_freq, time_diff, merchant_category, transaction_amount]):
                st.error("All fields are required.")
            elif merchant_state not in state_mapping:
                st.error(f"Invalid Merchant State: {merchant_state}")
            elif merchant_city not in city_mapping:
                st.error(f"Invalid Merchant City: {merchant_city}")
            else:
                transaction = {
                    'Zipcode': float(zipcode),
                    'Merchant_State_Code': int(state_mapping[merchant_state]),
                    'User_Frequency_Per_Day': float(user_freq),
                    'Time_Difference_Hours': float(time_diff),
                    'Merchant_Category_Code': float(merchant_category),
                    'Merchant_City_Code': int(city_mapping[merchant_city]),
                    'Transaction_Amount': float(transaction_amount)
                }
                transactions = [list(transaction.values())]
                predictions, probabilities = predict_fraud(transactions)
                
                if predictions is None or probabilities is None:
                    st.error("Prediction failed. Check server logs for details.")
                else:
                    result = 'Fraud' if predictions[0] == 1 else 'Not Fraud'
                    st.markdown(f"""
                        <div class='result-box'>
                            <h2>Transaction: {result}</h2>
                            <p>Probability of Fraud: {probabilities[0]:.4f}</p>
                        </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: Invalid input - {str(e)}")