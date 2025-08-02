# 🧐 MAP2025 with DeBERTa-v3-small

> A Gradio-based web app for evaluating student explanations using a fine-tuned [DeBERTa-v3-small](https://huggingface.co/microsoft/deberta-v3-small) model.

&#x20;&#x20;

---

## 📅 Clone the Repository

```bash
git clone https://github.com/Arunalk722/MAP2025-With-deberta-v3-small.git
cd MAP2025-With-deberta-v3-small
```

---

## 📆 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔄 Download the Pretrained Model

Download the pretrained model files from the link below:

👉 [Download from OneDrive](https://outlookuwicac-my.sharepoint.com/\:f:/g/personal/st20274122_outlook_cardiffmet_ac_uk/EiVVvp1G2gNMqaRg-U5za2wBPRKAO1fkNgYL_mq4bgTmWQ?e=wUd0pE)

> 🔓 **Note:** no need to sign in to download.

- Unzip the downloaded model files.
- Place the extracted folder inside the `models/` directory, like so:

```
MAP2025-With-deberta-v3-small/
├── app.py
├── models/
│   └── map_2025_best_model_fold7.pt
```

---

## 🚀 Run the App Locally

```bash
python app.py
```

- The terminal will show a local URL (e.g., [http://127.0.0.1:7860/](http://127.0.0.1:7860/))
- Open this URL in your browser to access the web interface.

---

## 🌐 Web Interface Guide

### 🔘 Model Selection

- **Default Model**: `map_2025_best_model_fold7.pt`
- If multiple `.pt` files are present, use the dropdown to select your preferred model.

### ✏️ Input Fields

| Field                 | Description                                |
| --------------------- | ------------------------------------------ |
| `Question`            | Enter the question text                    |
| `Correct Answer`      | Provide the correct multiple-choice answer |
| `Student Explanation` | Paste the student's written response       |

### 📁 Export

- Optionally, check `Export Prediction to CSV` to save outputs.

### 🧪 Submit

- Click `Submit` to generate predictions.
- Results will appear in the `Prediction Result` textbox.

---

## 💡 Try It Online (No Setup Required)

You can try the model directly in your browser via Hugging Face Spaces:

👉 [**Live Demo on Hugging Face**](https://huggingface.co/spaces/arunalk722/map2025)



---

## 🧰 Tech Stack

- Python 3.8+
- Hugging Face Transformers
- Gradio
- PyTorch

---

## 🧑‍💼 Author

**Aruna Shantha**\
📧 [GitHub](https://github.com/Arunalk722)\
📬 [Hugging Face](https://huggingface.co/arunalk722)

*if you need to help*\
feel fee to contact me

![TiredGroggyGIF](https://github.com/user-attachments/assets/b17bd2d8-1e65-49a4-b8c0-9a0e395ad989) \

📩 arunalk722@hotmail.com


