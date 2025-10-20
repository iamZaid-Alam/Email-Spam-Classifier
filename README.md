# 📩 SMS / Email Spam Classifier

> 🚫 A Machine Learning–powered NLP app that detects whether a given message is **Spam** or **Not Spam**, built using **Streamlit**, **Scikit-learn**, and **NLTK**.

url : https://iamzaid-alam-email-spam-classifier-app-dotrvr.streamlit.app/
---

## 🧠 Overview

Spam messages are everywhere — from fake bank alerts to “You’ve won a prize!” scams.  
This project aims to automatically classify text messages (SMS or emails) as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and **Machine Learning**.

---

📊 Dataset : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

🚀 Demo : 

https://github.com/user-attachments/assets/5ce5fa4b-b024-44fc-81b9-39085b8ea1a4


https://github.com/user-attachments/assets/60eb30f2-69bb-4238-b6d2-fbf1ec7470cf



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

### EDA
- Shape of data : (5572, 5)
- How data looks :
  <img width="1187" height="558" alt="image" src="https://github.com/user-attachments/assets/7c6f5395-b3aa-450f-af9e-3f32b57437b7" />
  
- info :
 <img width="569" height="345" alt="image" src="https://github.com/user-attachments/assets/03f93edf-f46a-4826-9df0-49d6b1420e78" />
 
- Columns dropped : ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
  ```python
  df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

- After this, the sample of data looks :
 <img width="720" height="322" alt="image" src="https://github.com/user-attachments/assets/5e702879-9140-4bd0-908c-77716753f830" />
  
- Change the column name:
   ```python
  df.rename(columns={'v1':'target','v2':'text'},inplace=True)
<img width="699" height="375" alt="image" src="https://github.com/user-attachments/assets/635bde6f-d0f7-4334-9fdc-b1177ea56076" />

- Apply LabelEncoder on Target.

- There are no null values but some duplicated records (around 403 records). Drop them :
  ```python
  df.drop_duplicates(keep='first',inplace=True)

- ```python
  from enum import auto
  plt.pie(df.target.value_counts(), labels=["ham","spam"],autopct="%0.2f")   
 <img width="633" height="588" alt="image" src="https://github.com/user-attachments/assets/45ccdc32-d3b8-431c-a588-e422103b224b" />
 
> 🪄 Dataset is Imbalanced.

- Create three features/columns : number_of_characters, number_of_words, number_of_sentences
  ```python
   df['number_of_characters'] = df['text'].apply(len)
   df['number_of_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
   df['number_of_sentences'] = df['text'].apply(lambda x : len(nltk.sent_tokenize(x)))

- Describe of new features:
<img width="963" height="533" alt="image" src="https://github.com/user-attachments/assets/a78da68f-fbdc-4ad1-a3d3-7d5109b94c88" />

- sns.pairplot(df,hue='target') :
<img width="640" height="577" alt="image" src="https://github.com/user-attachments/assets/cffac0a7-d52b-42ce-bbfc-7f012c05dc35" />
 


### 2️⃣ Data Preprocessing
- **Lowercasing** → Convert text to lowercase.  
- **Tokenization** → Split sentences into words using `nltk.word_tokenize`.  
- **Stopword & Punctuation Removal** → Remove unnecessary words like “is”, “the”, etc.  
- **Stemming** → Reduce words to their root form using `PorterStemmer`.  

> 🧹 Function used: create a function `transform_text(text)` — which will take care of preprocessing.
    ```python

🚨 Most frequent spam words : 
<img width="641" height="629" alt="image" src="https://github.com/user-attachments/assets/0c43680b-5470-4d59-a3a4-57a35bad40df" />

✅ Non spam/Ham words : 
<img width="659" height="640" alt="image" src="https://github.com/user-attachments/assets/b6ed06df-1745-46b2-9ab2-c2c2d23b1f27" />

<img width="508" height="378" alt="image" src="https://github.com/user-attachments/assets/479cb012-609c-45f5-b512-d4a178b3eb30" />




### 🧠 Model Building
- Did the feature extraction with  CountVectorizer,TfidfVectorizer, TfidfVectorizer(max_features=3000).
  Basically we have different ways of Changing the words to vector :
   • Bag of Words (BoW)
   • Term Frequency-Inverse Document Frequency (TF-IDF)
   • Word2Vec
  
- In Naive Bayes, MultinomialNB was giving good accuracy and precision.

- Also tried with others models :
- <img width="730" height="366" alt="image" src="https://github.com/user-attachments/assets/448dea7e-13b5-478a-ad04-7a533cac0ca0" />

- Models Accuracy and Precision:
- <img width="554" height="546" alt="image" src="https://github.com/user-attachments/assets/8aa2a64c-7c7c-46e0-9639-77f6eab74fc8" />

<img width="702" height="636" alt="image" src="https://github.com/user-attachments/assets/30e50ee4-3d80-4c98-93fc-831de1a55983" />

- Also tried with scaling, max_features=3000, then the final performance df :
<img width="1510" height="447" alt="image" src="https://github.com/user-attachments/assets/c9a0fb13-275f-4496-b59b-e08ff27b3736" />

- Gave a shot to other ensemble techniques:
<img width="1254" height="327" alt="image" src="https://github.com/user-attachments/assets/e379759d-f5d6-482a-acf4-d68950221b34" />


<img width="1250" height="422" alt="image" src="https://github.com/user-attachments/assets/1c923f16-e2b9-45af-b96f-d74b8ced7913" />


---

👨‍💻 Author

Zaid Alam
_Data Analyst & Machine Learning Enthusiast_
📧 zaidalam49@gmail.com
🌐 https://www.linkedin.com/in/zaid-alam98/


https://github.com/user-attachments/assets/e84f87fe-2314-4695-9215-ea8527737995


❤️ **Thank you for checking out this project!**


