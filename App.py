import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model


# Load all models
model_dnn_ctgan_aug = load_model('C:/Users/Nimish Bhatt/Downloads/DNN_CTGAN_AUG.keras')
model_dnn_ctgan_real = load_model('C:/Users/Nimish Bhatt/Downloads/DNN_CTGAN_REAL.keras')
model_lstm = load_model('C:/Users/Nimish Bhatt/Downloads/lstm.keras')
model_lstmgan = load_model('C:/Users/Nimish Bhatt/Downloads/lstmgan.keras')

# Model mapping for selection
models = {
    "DNN CTGAN Augmented": model_dnn_ctgan_aug,
    "DNN Real": model_dnn_ctgan_real,
    "LSTM Real": model_lstm,
    "LSTM CTGAN Augmented": model_lstmgan
}

# Preprocessing functions
std_scaler = StandardScaler()

def standardization(df, col):
    """Standardize the numeric columns."""
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr), 1))
    return df

def preprocess_lstm(df):
    """Preprocessing specific to LSTM models."""
    numeric_col = df.select_dtypes(include='number').columns
    df = standardization(df, numeric_col)
    
    # One-hot-encoding attack label
    df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'], prefix="", prefix_sep="")
    
    # Replace True/False with 1/0
    df.replace({True: 1, False: 0}, inplace=True)
    
    return df

def preprocess_dnn(df):
    """Preprocessing specific to DNN models."""
    # Use Label Encoding for categorical features
    categorical_columns = ['protocol_type', 'service', 'flag']
    label_encoder = LabelEncoder()
    
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    
    # Drop 'num_outbound_cmds' if it exists
    if 'num_outbound_cmds' in df.columns:
        df.drop("num_outbound_cmds", axis=1, inplace=True)
    
    # Define columns to scale
    columns_to_scale = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
                        'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 
                        'num_shells', 'num_access_files', 'count', 'srv_count', 'dst_host_count', 
                        'dst_host_srv_count']
    
    # Scale numerical columns using MinMax
    scaler = MinMaxScaler()
    for column in columns_to_scale:
        if column in df.columns:
            df[column] = scaler.fit_transform(df[[column]])
    
    return df

# Streamlit frontend
st.title("Intrusion Detection Model Testing")

# File uploader for test CSV
uploaded_file = st.file_uploader("Upload your test CSV file", type="csv")

# Model selection
model_choice = st.selectbox("Choose a model to test", options=list(models.keys()))

# Process file and make predictions
if uploaded_file is not None:
    # Read the uploaded CSV
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", data.head())
    
    # Apply the appropriate preprocessing
    if 'LSTM' in model_choice:
        st.write("Applying LSTM-specific preprocessing...")
        data = preprocess_lstm(data)
    else:
        st.write("Applying DNN-specific preprocessing...")
        data = preprocess_dnn(data)
    
    st.write("Preview of preprocessed data:", data.head())
    
    # Convert DataFrame to NumPy array
    features = data.to_numpy()

    # Select the chosen model
    chosen_model = models[model_choice]

    # Make predictions
    predictions = chosen_model.predict(features)

    # Display predictions
    st.write(f"Predictions using the {model_choice} model:")
    st.write(predictions)