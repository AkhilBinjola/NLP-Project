# Natural Language Processing: IMDb Sentiment Analysis and News Classification

Two text classification projects built with TF-IDF features and classical machine learning models.

| Part | Task | Data | Best model | Test accuracy |
|---|---|---|---|---|
| A | Movie review sentiment (positive / negative) | 50,000 IMDb reviews | Logistic Regression | 88.9% |
| B | News article categorisation (10 categories) | News articles dataset | Logistic Regression | 79.4% |

---

## Part A: IMDb Sentiment Analysis

**Approach**
1. Cleaned the review text: lower-casing, removing punctuation, tokenising and removing stop words with NLTK.
2. Converted text to vectors with TF-IDF (`max_features=5000`).
3. 80/20 train–test split (`random_state=42`), so 10,000 reviews were held out for testing.
4. Trained and compared three classifiers.

| Model | Accuracy |
|---|---|
| **Logistic Regression** | **88.9%** |
| Linear SVM | 88.5% |
| Multinomial Naive Bayes | 85.3% |

Logistic Regression is balanced across both classes (precision, recall and F1 all about 0.88–0.90).

---

## Part B: News Article Classification

**Approach**
1. Built the text input from each article's headline and short description, then lower-cased it and cleaned non-word characters with regular expressions.
2. TF-IDF features (`max_features=5000`, English stop words removed).
3. 80/20 train–test split (`random_state=42`) and 5-fold cross-validation.
4. Compared Logistic Regression, Linear SVM and Naive Bayes.
5. Saved the best model and the vectorizer with `joblib` so the classifier can be reused without retraining.

**Results:** Logistic Regression was best, with 79.4% test accuracy and 78.3% mean accuracy in 5-fold cross-validation. The gap between the two is small, which suggests the model generalises rather than overfits.

The 10 categories include Politics, Sports, Business, Wellness and Travel. Because there are ten classes, 79% accuracy is a stronger result than it looks next to the two-class sentiment task.

---

## Tech stack

Python, Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, Seaborn, Joblib.

## How to run

```bash
git clone https://github.com/AkhilBinjola/NLP-Project
cd NLP-Project
pip install -r requirements.txt
jupyter notebook
```

The notebooks read their CSV files from `/content/` (Google Colab). To run locally, change the `pd.read_csv(...)` paths to where you saved the datasets. Part A needs the NLTK stop-word and tokenizer data, which you can download once with `nltk.download('stopwords')` and `nltk.download('punkt')`.

## Ideas for improvement

- Try n-grams and larger TF-IDF vocabularies.
- Compare against a transformer model such as DistilBERT.
- Add a per-class confusion matrix for the news classifier to see which categories get confused.

## Author

Akhil Binjola · [LinkedIn](https://linkedin.com/in/akhilbinjola) · [GitHub](https://github.com/AkhilBinjola)
