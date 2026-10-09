# UrbanTech Feedback Classifier

Automatically routes transit app feedback to the right department using Multinomial Naive Bayes.

## Business Problem
UrbanTech's navigation app receives hundreds of feedback messages daily, and they are sorted by hand. This project classifies each message into one of three departments:

- **Navigation Issues**: directions, routes, maps
- **Service Updates**: accuracy of transit service information
- **App Experience**: app functionality and interface

## Dataset
`urban_feedback.csv`: 197 messages with columns `feedback_id`, `feedback_text` and `department` (classes are balanced: 65 to 66 per class).

## Approach
1. Data loading and exploration (class balance, message length by department)
2. Text preprocessing: lowercase, remove punctuation and numbers, tokenize, remove stopwords, POS-aware lemmatization
3. Feature extraction: Bag of Words (`CountVectorizer`) and `TfidfVectorizer`, using a stratified 80/20 split
4. Multinomial Naive Bayes with top words per class
5. Evaluation: accuracy, classification report, confusion matrices
6. Tuning with `GridSearchCV` over a pipeline (vectorizer type, `alpha`, `max_features`, `min_df`, `max_df`, n-grams)
7. `classify_feedback()` prediction function returning the department and confidence scores

## Results
| Model | Test accuracy |
|---|---|
| Bag of Words + NB | 0.750 |
| TF-IDF + NB | 0.725 |
| Tuned TF-IDF + NB | 0.700 (5-fold CV: 0.827) |

The test set is only 40 messages, so the cross-validation score is the more reliable estimate. `service_updates` is the hardest class because those messages often mention the app itself.

## Repository Contents
- `text_modeling_lab.ipynb`: full analysis
- `urban_feedback.csv`: dataset

NLTK data (punkt, stopwords, wordnet) downloads automatically in the first cell.