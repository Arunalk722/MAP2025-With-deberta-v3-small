# ==========================
# Setup for Kaggle
# ==========================

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import re
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from transformers import DebertaV2Model, DebertaV2Tokenizer


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

# ==========================
# Feature Extraction
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
# Deep Learning Model
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
# Evaluation Logic
# ==========================
def evaluate_model(model_path, processed_df, features_scaled, cat_enc, misc_enc):
    print(f"\n--- Evaluating model: {model_path} ---")

    model = MathMisconceptionModel(
        n_categories=len(cat_enc.classes_),
        n_misconceptions=len(misc_enc.classes_),
        feature_dim=features_scaled.shape[1]
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    texts = processed_df['sentence'].tolist()
    features_tensor = torch.tensor(features_scaled, dtype=torch.float)

    batch_size = 32
    cat_probs_list, misc_probs_list = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Predicting with {os.path.basename(model_path)}"):
            batch_texts = texts[i:i+batch_size]
            batch_features = features_tensor[i:i+batch_size].to(DEVICE)
            cat_logits, misc_logits = model(batch_texts, batch_features)
            cat_probs_list.append(torch.softmax(cat_logits, 1).cpu().numpy())
            misc_probs_list.append(torch.softmax(misc_logits, 1).cpu().numpy())

    cat_probs = np.vstack(cat_probs_list)
    misc_probs = np.vstack(misc_probs_list)
    cat_preds = np.argmax(cat_probs, axis=1)
    misc_preds = np.argmax(misc_probs, axis=1)

    results_df = processed_df[['row_id', 'QuestionText', 'MC_Answer', 'StudentExplanation']].copy()
    results_df['true_category'] = processed_df['Category'].values
    results_df['predicted_category'] = cat_enc.inverse_transform(cat_preds)
    results_df['true_misconception'] = processed_df['Misconception'].values
    results_df['predicted_misconception'] = misc_enc.inverse_transform(misc_preds)

    cat_probs_df = pd.DataFrame(cat_probs, columns=[f"prob_cat_{c}" for c in cat_enc.classes_])
    misc_probs_df = pd.DataFrame(misc_probs, columns=[f"prob_misc_{c}" for c in misc_enc.classes_])
    
    final_df = pd.concat([results_df.reset_index(drop=True), cat_probs_df, misc_probs_df], axis=1)

    output_filename = f"predictions_{os.path.basename(model_path).replace('.pt', '.csv')}"
    final_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

# =========================
# Main Execution
# ==========================
def main():
    print("Loading shared encoders, data, and features...")
    with open(CAT_ENCODER_PATH, 'rb') as f:
        cat_enc = pickle.load(f)
    with open(MISC_ENCODER_PATH, 'rb') as f:
        misc_enc = pickle.load(f)
    with open(FEATURE_COLS_PATH, 'rb') as f:
        feature_cols = pickle.load(f)

    df = pd.read_csv(TRAIN_DATA_PATH)
    df['Misconception'] = df['Misconception'].fillna('NA')

    print("Creating features for the dataset (one time)...")
    processed_df = create_features(df.copy())
    for col in feature_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0
    features = processed_df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
    if not model_files:
        print("No model files (.pt) found in the 'models' directory.")
        return

    print(f"Found {len(model_files)} models to evaluate: {model_files}")
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        evaluate_model(model_path, processed_df, features_scaled, cat_enc, misc_enc)

    print("\nAll models evaluated.")

if __name__ == '__main__':
    main()
