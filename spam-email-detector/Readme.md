# 📧 Spam Email Detector

A simple machine learning web app that detects whether an email is spam or not, built using the **Spambase dataset** and a **Naive Bayes** classifier, with a **Streamlit** frontend.

## What it does

- Paste any email text into the app
- The app extracts 57 features from the text (word frequency, character frequency, capital letter patterns) — the same features used in the classic UCI Spambase dataset
- A trained Naive Bayes model predicts whether the email is spam or not
- Shows spam probability and which spam-related words were found in the text



## Tech Stack

- Python
- Streamlit (frontend)
- scikit-learn (Naive Bayes model)
- pandas / numpy
- joblib (model saving/loading)

## Project Structure

```
spam-email-detector/
├── app.py                  # Streamlit frontend
├── spam_email_detector.py  # model training script
├── spam_model.pkl          # trained model (saved)
├── spam.csv                # dataset
├── requirements.txt
└── README.md
```

## How to run locally

1. Clone this repo
```bash
git clone https://github.com/faridkhan02/spam-email-detector.git
cd spam-email-detector
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. (Optional) Retrain the model
```bash
python spam_email_detector.py
```
This will regenerate `spam_model.pkl`.

4. Run the app
```bash
streamlit run app.py
```

5. Open the local URL Streamlit shows in your browser (usually `http://localhost:8501`)

## Dataset

This project uses the [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) — 4601 emails labeled as spam or not spam, with 57 pre-computed features per email.

## Model

- Algorithm: Gaussian Naive Bayes
- Train/test split: 80/20
- Evaluated using accuracy score and confusion matrix

## Author

**MD Farid Khan**
[GitHub](https://github.com/faridkhan02) • [LinkedIn](https://www.linkedin.com/in/farid-khan)

## Feedback

If you find any bugs or have suggestions, feel free to open an issue or reach out. Still learning and improving this project!