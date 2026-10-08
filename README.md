# NLP Project

By Akhil Binjola

Two text-classification projects with TF-IDF + classical ML (scikit-learn, NLTK).

## Part A – IMDb Movie Review Sentiment Analysis
File: `NLP_partA.ipynb`
Dataset: `Imdb - data_imdb.csv` (review, sentiment – ~50k balanced pos/neg)

- Preprocessing: lowercase, punctuation/special-char removal, stopword removal, word_tokenize
- Features: TF-IDF (max_features=5000) + word_count, char_count, avg_word_length
- Split: 80/20 train-test
- Models:
  - LogisticRegression – ~88.88% accuracy
  - MultinomialNB – ~85.32%
  - LinearSVC – ~88.58%, best speed/accuracy trade-off
- Evaluation: accuracy, precision/recall/F1, confusion-matrix heatmaps

Video: https://drive.google.com/file/d/1ePAW0Qjv_zAgNVUytilGZD61NEyl1sv5/view?usp=sharing

## Part B – News Article Classification
File: `NLP_partB.ipynb`
Dataset: `data_news - data_news.csv` (category, headline, short_description)

- Cleaning: lowercase, non-word to space, whitespace normalize, dropna, combine headline + short_description to `text`
- Features: TfidfVectorizer (english stopwords, max_features=5000), EDA category distribution
- Models: LogisticRegression, MultinomialNB, LinearSVC with 80/20 split + 5-fold cross-validation
- Evaluation: accuracy, CV accuracy, classification_report, confusion matrices

Video: https://drive.google.com/file/d/1K8Mh_UGRz7vcR8f41wbx_dcAz0R7otHw/view?usp=sharing

## How to run
1. `pip install pandas numpy nltk scikit-learn matplotlib seaborn`
2. `python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"`
3. Open notebooks in Jupyter/Colab, point CSV paths to local files, run all.
