# Task5_Kaiburr
Consumer Complaint Text Classification
This project classifies consumer complaints into four categories:

Credit Reporting

Debt Collection

Consumer Loan

Mortgage

The classification is done using Natural Language Processing (NLP) techniques and Machine Learning models.

Project Steps
Explanatory Data Analysis (EDA) and Feature Engineering

Text Preprocessing

Selection of Multi-Class Classification Models

Comparison of Model Performance

Model Evaluation

Prediction on New Complaints

1️⃣ How to Run the Project
🔹 Prerequisites
Python 3.x

Jupyter Notebook or any Python IDE (VS Code, PyCharm, etc.)

Install dependencies using:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn nltk scikit-learn
2️⃣ Dataset Information
The dataset is downloaded from Consumer Complaint Database

It contains customer complaints and their respective product categories.

3️⃣ Steps to Execute the Project
Step 1: Load the Dataset
Run Consumer_Complaint_Classification.py

The dataset is loaded using:

python
Copy
Edit
df = pd.read_csv("Consumer_Complaints.csv")
df = df[['Product', 'Consumer complaint narrative']].dropna()
Feature Engineering: Mapping product names to labels.

Step 2: Text Preprocessing
Convert text to lowercase.

Remove special characters and stopwords.

Tokenization using NLTK.

Example:

python
Copy
Edit
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = word_tokenize(text)
    return " ".join(text)
Step 3: Train Multi-Class Classification Models
Convert text to TF-IDF Vector Representation.

Split data into Train/Test sets.

Train different models:

Naïve Bayes

Logistic Regression

Random Forest

Example:

python
Copy
Edit
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
Step 4: Model Comparison & Evaluation
Performance Metrics:

Accuracy

Precision

Recall

F1-Score

python
Copy
Edit
print(classification_report(y_test, y_pred))
Model Comparison Plot:

python
Copy
Edit
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
Step 5: Prediction
The trained model can classify new complaints.

Example:

python
Copy
Edit
sample_text = "I have issues with my mortgage payment."
print("Predicted Category:", predict_category(sample_text))
4️⃣ Screenshots
📌 Screenshots of Model Performance, Training Results, and Sample Predictions
(Attach the screenshots here as images in your repository.)

5️⃣ Deployment Instructions
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-repo.git
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python Consumer_Complaint_Classification.py
6️⃣ Conclusion
This project successfully classifies consumer complaints into predefined categories using NLP and machine learning. Future improvements can include deep learning models like LSTMs and Transformers.

