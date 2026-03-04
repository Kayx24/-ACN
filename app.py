import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

class CNN1D(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 96):
        super().__init__()
        # Enhanced 1D branch với deeper architecture và residual connections
        self.conv1 = nn.Conv1d(in_channels, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(96)
        self.conv2 = nn.Conv1d(96, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(192, 384, kernel_size=3, padding=1)  # Thêm layer thứ 3
        self.bn3 = nn.BatchNorm1d(384)
        
        # Residual connection
        self.shortcut = nn.Conv1d(in_channels, 384, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm1d(384)
        
        self.dropout = nn.Dropout(0.3)  # Giảm dropout
        self.fc = nn.Linear(384, out_dim)
        
    def forward(self, x):                # x: (B, C=in_channels, L)
        identity = x
        
        # Main path
        h = F.relu(self.bn1(self.conv1(x)))
        if h.size(-1) > 1:
            h = F.max_pool1d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        # Residual connection
        identity = self.bn_shortcut(self.shortcut(identity))
        identity = F.adaptive_max_pool1d(identity, h.size(-1))
        
        h = h + identity  # Residual connection
        h = F.adaptive_max_pool1d(h, 1).squeeze(-1)  # (B, 512)
        h = self.dropout(h)
        return self.fc(h)                # (B, out_dim)

class CNN2D(nn.Module):
    def __init__(self, out_dim: int = 96):
        super().__init__()
        # Main path
        self.conv1 = nn.Conv2d(1, 48, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, 384, 3, padding=1)  # thêm layer sâu
        self.bn4 = nn.BatchNorm2d(384)

        # Residual (shortcut) path
        self.shortcut = nn.Conv2d(1, 384, kernel_size=1)  # map từ 1 channel → 512 channel
        self.bn_shortcut = nn.BatchNorm2d(384)

        self.dropout = nn.Dropout(0.2)  # Giảm dropout
        self.fc = nn.Linear(384, out_dim)

    def forward(self, x):                 # x: (B, 1, H, W)
        identity = x

        # Main path
        h = F.relu(self.bn1(self.conv1(x)))
        if h.size(-1) > 1:
            h = F.max_pool2d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        # if h.size(-1) > 1:
        #     h = F.max_pool2d(h, 2)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))

        # Residual
        identity = self.bn_shortcut(self.shortcut(identity))
        identity = F.adaptive_max_pool2d(identity, h.shape[-2:])  # resize to match h

        # Add residual
        h = h + identity

        # Global pooling
        h = F.adaptive_max_pool2d(h, 1).squeeze(-1).squeeze(-1)  # (B, 512)
        h = self.dropout(h)
        return self.fc(h)                  # (B, out_dim)

class AttentionFusion(nn.Module):
    """Attention-based feature fusion thay vì simple concatenation"""
    def __init__(self, feature_dim: int = 96):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1)
        )
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
        
    def forward(self, h1, h2):
        # h1, h2: (B, feature_dim)
        concat_features = torch.cat([h1, h2], dim=-1)  # (B, feature_dim*2)
        attention_weights = self.attention(concat_features)  # (B, 2)
        
        # Apply attention weights
        weighted_h1 = h1 * attention_weights[:, 0:1]
        weighted_h2 = h2 * attention_weights[:, 1:2]
        
        # Combine weighted features
        fused = torch.cat([weighted_h1, weighted_h2], dim=-1)
        return self.fc(fused)  # (B, feature_dim)

class DVRCNN(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.branch1d = CNN1D(in_channels=d_in, out_dim=96)
        self.branch2d = CNN2D(out_dim=96)
        
        # Enhanced fusion với attention mechanism
        self.fusion = AttentionFusion(feature_dim=96)
        
        # Enhanced classifier với GELU, giảm dropout (Note: Trong training, dùng optimizer với weight_decay=0.01 cho L2 reg)
        self.classifier = nn.Sequential(
            nn.Linear(96, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),  # Thay ReLU bằng GELU
            nn.Dropout(0.3),  # Giảm dropout
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.2),  # Giảm dropout
            nn.Linear(96, n_classes),
        )
        
    def forward(self, x1d, x2d):
        h1 = self.branch1d(x1d)  # (B, 128)
        h2 = self.branch2d(x2d)  # (B, 128)
        h_fused = self.fusion(h1, h2)  # (B, 128)

        return self.classifier(h_fused)

#Load Model

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "last_checkpoint_14.pt",
        map_location="cpu",
        weights_only=False
    )

    n_classes = len(checkpoint["label_names"])
    d_in = len(feature_columns)  

    model = DVRCNN(d_in, n_classes) 
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model

try:
    model = load_model()
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("MODEL LOAD ERROR:", e)
    raise e

#Streamlit UI

st.title("NSL-KDD Intrusion Detection System")
st.write("Upload a CSV file with raw NSL-KDD features.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
    # Preprocessing
    df = df.drop(columns=["difficulty", "num_outbound_cmds"], errors="ignore")
    
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    X_scaled = scaler.transform(df)
    
    # Convert to tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Create window_len = 2 (duplicate for demo)
    d_in = len(feature_columns)

    # 1D input đúng shape: (B, d_in, 1)
    X1d = X_tensor.unsqueeze(-1)

    # 2D input reshape về 11x11 (vì 121 = 11x11)
    X2d = X_tensor.view(-1, 1, 11, 11)
    
    with torch.no_grad():
        outputs = model(X1d, X2d)
        preds = torch.argmax(outputs, dim=1).numpy()
    
    labels = label_encoder.inverse_transform(preds)
    df["Prediction"] = labels
    
    st.success("Prediction completed!")
    st.dataframe(df.head())
    
    st.dataframe(df[["Prediction"]].head())
    st.write("Prediction distribution:")
    st.write(df["Prediction"].value_counts())
    
class CNN1D(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 96):
        super().__init__()
        # Enhanced 1D branch với deeper architecture và residual connections
        self.conv1 = nn.Conv1d(in_channels, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(96)
        self.conv2 = nn.Conv1d(96, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(192, 384, kernel_size=3, padding=1)  # Thêm layer thứ 3
        self.bn3 = nn.BatchNorm1d(384)
        
        # Residual connection
        self.shortcut = nn.Conv1d(in_channels, 384, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm1d(384)
        
        self.dropout = nn.Dropout(0.3)  # Giảm dropout
        self.fc = nn.Linear(384, out_dim)
        
    def forward(self, x):                # x: (B, C=in_channels, L)
        identity = x
        
        # Main path
        h = F.relu(self.bn1(self.conv1(x)))
        if h.size(-1) > 1:
            h = F.max_pool1d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        # Residual connection
        identity = self.bn_shortcut(self.shortcut(identity))
        identity = F.adaptive_max_pool1d(identity, h.size(-1))
        
        h = h + identity  # Residual connection
        h = F.adaptive_max_pool1d(h, 1).squeeze(-1)  # (B, 512)
        h = self.dropout(h)
        return self.fc(h)                # (B, out_dim)

class CNN2D(nn.Module):
    def __init__(self, out_dim: int = 96):
        super().__init__()
        # Main path
        self.conv1 = nn.Conv2d(1, 48, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, 384, 3, padding=1)  # thêm layer sâu
        self.bn4 = nn.BatchNorm2d(384)

        # Residual (shortcut) path
        self.shortcut = nn.Conv2d(1, 384, kernel_size=1)  # map từ 1 channel → 512 channel
        self.bn_shortcut = nn.BatchNorm2d(384)

        self.dropout = nn.Dropout(0.2)  # Giảm dropout
        self.fc = nn.Linear(384, out_dim)

    def forward(self, x):                 # x: (B, 1, H, W)
        identity = x

        # Main path
        h = F.relu(self.bn1(self.conv1(x)))
        if h.size(-1) > 1:
            h = F.max_pool2d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        # if h.size(-1) > 1:
        #     h = F.max_pool2d(h, 2)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))

        # Residual
        identity = self.bn_shortcut(self.shortcut(identity))
        identity = F.adaptive_max_pool2d(identity, h.shape[-2:])  # resize to match h

        # Add residual
        h = h + identity

        # Global pooling
        h = F.adaptive_max_pool2d(h, 1).squeeze(-1).squeeze(-1)  # (B, 512)
        h = self.dropout(h)
        return self.fc(h)                  # (B, out_dim)

class AttentionFusion(nn.Module):
    """Attention-based feature fusion thay vì simple concatenation"""
    def __init__(self, feature_dim: int = 96):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1)
        )
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
        
    def forward(self, h1, h2):
        # h1, h2: (B, feature_dim)
        concat_features = torch.cat([h1, h2], dim=-1)  # (B, feature_dim*2)
        attention_weights = self.attention(concat_features)  # (B, 2)
        
        # Apply attention weights
        weighted_h1 = h1 * attention_weights[:, 0:1]
        weighted_h2 = h2 * attention_weights[:, 1:2]
        
        # Combine weighted features
        fused = torch.cat([weighted_h1, weighted_h2], dim=-1)
        return self.fc(fused)  # (B, feature_dim)

class DVRCNN(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.branch1d = CNN1D(in_channels=d_in, out_dim=96)
        self.branch2d = CNN2D(out_dim=96)
        
        # Enhanced fusion với attention mechanism
        self.fusion = AttentionFusion(feature_dim=96)
        
        # Enhanced classifier với GELU, giảm dropout (Note: Trong training, dùng optimizer với weight_decay=0.01 cho L2 reg)
        self.classifier = nn.Sequential(
            nn.Linear(96, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),  # Thay ReLU bằng GELU
            nn.Dropout(0.3),  # Giảm dropout
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.2),  # Giảm dropout
            nn.Linear(96, n_classes),
        )
        
    def forward(self, x1d, x2d):
        h1 = self.branch1d(x1d)  # (B, 128)
        h2 = self.branch2d(x2d)  # (B, 128)
        h_fused = self.fusion(h1, h2)  # (B, 128)

        return self.classifier(h_fused)