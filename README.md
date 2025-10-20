# Email-Spam-Classifier
# 📩 SMS / Email Spam Classifier

> 🚫 A Machine Learning–powered NLP app that detects whether a given message is **Spam** or **Not Spam**, built using **Streamlit**, **Scikit-learn**, and **NLTK**.

---

## 🧠 Overview

Spam messages are everywhere — from fake bank alerts to “You’ve won a prize!” scams.  
This project aims to automatically classify text messages (SMS or emails) as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and **Machine Learning**.

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 🐍 |
| **Libraries** | Scikit-learn, NLTK, Streamlit, Pandas, NumPy |
| **Model** | Multinomial Naive Bayes |
| **Feature Extraction** | TF-IDF Vectorization |
| **Frontend** | Streamlit Web App |

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading
- Used the `spam.csv` dataset (from Kaggle) containing labeled SMS messages.
- Cleaned and retained only relevant columns: `label` and `message`.

### 2️⃣ Data Preprocessing
- **Lowercasing** → Convert text to lowercase.  
- **Tokenization** → Split sentences into words using `nltk.word_tokenize`.  
- **Stopword & Punctuation Removal** → Remove unnecessary words like “is”, “the”, etc.  
- **Stemming** → Reduce words to their root form using `PorterStemmer`.  

> 🧹 Function used: `transform_text(text)` — ensures consistent text processing.

### 3️⃣ Feature Extraction
- Transformed text into numeric form using **TF-IDF Vectorizer**:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  tfidf = TfidfVectorizer(max_features=3000)


Dataset : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset



https://github.com/user-attachments/assets/5ce5fa4b-b024-44fc-81b9-39085b8ea1a4

https://github.com/user-attachments/assets/60eb30f2-69bb-4238-b6d2-fbf1ec7470cf

https://github.com/user-attachments/assets/e84f87fe-2314-4695-9215-ea8527737995

