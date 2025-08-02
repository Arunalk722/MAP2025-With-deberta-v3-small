import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import re
import gradio as gr
from transformers import DebertaV2Model, DebertaV2Tokenizer
from sklearn.preprocessing import StandardScaler

# ==========================
# Configuration
# ==========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 256
MODELS_DIR = './models/'
CAT_ENCODER_PATH = os.path.join(MODELS_DIR, 'cat_encoder.pkl')
MISC_ENCODER_PATH = os.path.join(MODELS_DIR, 'misc_encoder.pkl')
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, 'feature_cols.pkl')
TRAIN_DATA_PATH = './dataset/train.csv'
DEFAULT_MODEL = 'map_2025_best_model_fold7.pt'

# ==========================
# Feature Extraction (from training script)
# ==========================
def extract_math_features(text):
    if not isinstance(text, str):
        return {
            'frac_count': 0, 'number_count': 0, 'operator_count': 0,
            'decimal_count': 0, 'question_mark': 0, 'math_keyword_count': 0
        }
    features = {
        'frac_count': len(re.findall(r'FRAC_\d+_\d+|\\frac', text)),
        'number_count': len(re.findall(r'\b\d+\b', text)),
        'operator_count': len(re.findall(r'[\+\-\*\/\=]', text)),
        'decimal_count': len(re.findall(r'\d+\.\d+', text)),
        'question_mark': int('?' in text),
        'math_keyword_count': len(re.findall(r'solve|calculate|equation|fraction|decimal', text.lower()))
    }
    return features

def create_features(df):
    for col in ['QuestionText', 'MC_Answer', 'StudentExplanation']:
        df[col] = df[col].fillna('')
    df['mc_answer_len'] = df['MC_Answer'].str.len()
    df['explanation_len'] = df['StudentExplanation'].str.len()
    df['question_len'] = df['QuestionText'].str.len()
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)
    for col in ['QuestionText', 'MC_Answer', 'StudentExplanation']:
        mf = df[col].apply(extract_math_features).apply(pd.Series)
        prefix = 'mc_' if col == 'MC_Answer' else 'exp_' if col == 'StudentExplanation' else ''
        mf.columns = [f'{prefix}{c}' for c in mf.columns]
        df = pd.concat([df, mf], axis=1)
    df['sentence'] = (
        "Question: " + df['QuestionText'] +
        " Answer: " + df['MC_Answer'] +
        " Explanation: " + df['StudentExplanation']
    )
    return df

# ==========================
# Deep Learning Model (from training script)
# ==========================
class MathMisconceptionModel(nn.Module):
    def __init__(self, n_categories, n_misconceptions, feature_dim):
        super().__init__()
        self.bert = DebertaV2Model.from_pretrained('microsoft/deberta-v3-small')
        self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-small')
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.category_head = nn.Sequential(
            nn.Linear(768 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_categories)
        )
        self.misconception_head = nn.Sequential(
            nn.Linear(768 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_misconceptions)
        )

    def forward(self, input_texts, features):
        tokens = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)
        outputs = self.bert(**tokens)
        text_emb = outputs.last_hidden_state[:, 0, :]
        feat_emb = self.feature_processor(features)
        combined = torch.cat([text_emb, feat_emb], dim=1)
        return self.category_head(combined), self.misconception_head(combined)

# ==========================
# Load Resources
# ==========================
try:
    with open(CAT_ENCODER_PATH, 'rb') as f:
        cat_enc = pickle.load(f)
    with open(MISC_ENCODER_PATH, 'rb') as f:
        misc_enc = pickle.load(f)
    with open(FEATURE_COLS_PATH, 'rb') as f:
        feature_cols = pickle.load(f)

    # Fit scaler on the original training data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    processed_train_df = create_features(train_df.copy())
    for col in feature_cols:
        if col not in processed_train_df.columns:
            processed_train_df[col] = 0
    train_features = processed_train_df[feature_cols].fillna(0).values
    scaler = StandardScaler().fit(train_features)

except FileNotFoundError as e:
    print(f"Error loading resources: {e}")
    exit()

# ==========================
# Prediction Logic
# ==========================
def predict(model_name, question, mc_answer, explanation, export_csv):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        return "❌ Model not found.", None

    # Create DataFrame for prediction
    data = {
        'QuestionText': [question],
        'MC_Answer': [mc_answer],
        'StudentExplanation': [explanation]
    }
    df = pd.DataFrame(data)

    # Feature engineering
    processed_df = create_features(df.copy())
    for col in feature_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0
    features = processed_df[feature_cols].fillna(0).values
    features_scaled = scaler.transform(features)

    # Load model
    model = MathMisconceptionModel(
        n_categories=len(cat_enc.classes_),
        n_misconceptions=len(misc_enc.classes_),
        feature_dim=features_scaled.shape[1]
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Prediction
    text = processed_df['sentence'].tolist()
    features_tensor = torch.tensor(features_scaled, dtype=torch.float).to(DEVICE)

    with torch.no_grad():
        cat_logits, misc_logits = model(text, features_tensor)
        cat_pred = torch.argmax(cat_logits, 1).cpu().item()
        misc_pred = torch.argmax(misc_logits, 1).cpu().item()

    predicted_category = cat_enc.inverse_transform([cat_pred])[0]
    predicted_misconception = misc_enc.inverse_transform([misc_pred])[0]

    result_text = (
        f"Predicted Category: {predicted_category}\n"
        f"Predicted Misconception: {predicted_misconception}"
    )

    csv_path = None
    if export_csv:
        export_df = pd.DataFrame([{
            "Question": question,
            "MC_Answer": mc_answer,
            "Student_Explanation": explanation,
            "Predicted_Category": predicted_category,
            "Predicted_Misconception": predicted_misconception,
            "Model_Used": model_name
        }])
        csv_path = "predictions.csv"
        file_exists = os.path.isfile(csv_path)
        export_df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

    return result_text, csv_path

# ==========================
# Gradio UI
# ==========================
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(model_files, value=DEFAULT_MODEL, label="Select Model"),
        gr.Textbox(label="Enter Question", lines=3),
        gr.Textbox(label="Enter Correct Answer (MC_Answer)", lines=1),
        gr.Textbox(label="Enter Student's Explanation", lines=5),
        gr.Checkbox(label="Export Prediction to CSV")
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.File(label="CSV File (if exported)")
    ],
    title="Math Misconception Predictor",
    description="Select a model and provide the question, correct answer, and student's explanation to get a prediction."
)

if __name__ == "__main__":
    iface.launch()
