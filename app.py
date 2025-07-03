from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import os

app = Flask(__name__)

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

# Load model and threshold with error handling
try:
    model_path = 'fraud_gnn_model.pth'
    threshold_path = 'optimal_threshold.txt'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Ensure it exists in the project directory.")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Threshold file {threshold_path} not found. Ensure it exists in the project directory.")
        
    model = FraudGNN(input_dim=7, hidden_dim=16, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    with open(threshold_path, 'r') as f:
        threshold = float(f.read())
except Exception as e:
    print(f"Error loading model or threshold: {e}")
    model = None
    threshold = 0.5  # Fallback threshold

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
        
        # For a single transaction, create an empty edge index
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        
        # If multiple transactions, create edges based on zipcode proximity
        if len(df) > 1:
            zipcodes = node_features[:, 0].cpu().numpy()
            edge_list = []
            zipcode_threshold = 1000  # Proximity threshold (unscaled zip codes)
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
            # Ensure output is 1D for consistency
            out = torch.atleast_1d(out)  # Convert scalar to 1D tensor if needed
            pred_binary = (out > threshold).float().cpu().numpy()
            pred_proba = out.cpu().numpy()  # Already sigmoid-applied in model
            # Ensure 1D NumPy arrays
            pred_binary = np.atleast_1d(pred_binary)
            pred_proba = np.atleast_1d(pred_proba)
        
        return pred_binary, pred_proba
    except Exception as e:
        print(f"Error in predict_fraud: {e}")
        return None, None

# Route to serve JavaScript files from the 'js' folder
@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Validate form inputs
            required_fields = [
                'Zipcode', 'Merchant_State_Code', 'User_Frequency_Per_Day',
                'Time_Difference_Hours', 'Merchant_Category_Code',
                'Merchant_City_Code', 'Transaction_Amount'
            ]
            for field in required_fields:
                if field not in request.form or not request.form[field]:
                    raise ValueError(f"Missing or empty field: {field}")
            
            # Validate numeric inputs
            numeric_fields = [
                'Zipcode', 'User_Frequency_Per_Day', 'Time_Difference_Hours',
                'Merchant_Category_Code', 'Transaction_Amount'
            ]
            for field in numeric_fields:
                try:
                    float(request.form[field])
                except ValueError:
                    raise ValueError(f"Invalid number for {field}")
            
            # Validate Merchant_State_Code
            state = request.form['Merchant_State_Code']
            if state not in state_mapping:
                raise ValueError(f"Invalid Merchant State: {state}")
            
            # Validate Merchant_City_Code
            city = request.form['Merchant_City_Code']
            if city not in city_mapping:
                raise ValueError(f"Invalid Merchant City: {city}")
            
            transaction = {
                'Zipcode': float(request.form['Zipcode']),
                'Merchant_State_Code': int(state_mapping[request.form['Merchant_State_Code']]),
                'User_Frequency_Per_Day': float(request.form['User_Frequency_Per_Day']),
                'Time_Difference_Hours': float(request.form['Time_Difference_Hours']),
                'Merchant_Category_Code': float(request.form['Merchant_Category_Code']),
                'Merchant_City_Code': int(city_mapping[request.form['Merchant_City_Code']]),
                'Transaction_Amount': float(request.form['Transaction_Amount'])
            }
            transactions = [list(transaction.values())]
            predictions, probabilities = predict_fraud(transactions)
            
            if predictions is None or probabilities is None:
                raise ValueError("Prediction failed. Check server logs for details.")
            
            result = 'Fraud' if predictions[0] == 1 else 'Not Fraud'
            return render_template('index.html', result=f'Transaction: {result}', 
                                 probability=f'{probabilities[0]:.4f}', 
                                 cities=city_mapping.keys(), 
                                 states=state_mapping.keys())
        except Exception as e:
            return render_template('index.html', result=f'Error: Invalid input - {str(e)}', 
                                 cities=city_mapping.keys(), 
                                 states=state_mapping.keys())
    return render_template('index.html', result=None, 
                         cities=city_mapping.keys(), 
                         states=state_mapping.keys())

if __name__ == '__main__':
    app.run(debug=True)  # Disable debug=True in production