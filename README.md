# Customer Review Sentiment Analyzer

An end-to-end ML pipeline that classifies customer reviews as **Positive**,
**Negative**, or **Neutral**, built for the AI/ML Intern project brief.

## Project Structure
```
sentiment_analyzer/
├── data/
│   ├── generate_dataset.py     # builds the labeled dataset (offline, template-based)
│   └── reviews_dataset.csv     # generated dataset (label, review)
├── src/
│   ├── preprocess.py           # text cleaning (lowercase, remove punctuation/stopwords)
│   └── train.py                # feature extraction, training, evaluation, model export
├── models/                     # saved vectorizer + trained models (.pkl)
├── outputs/                    # confusion matrices, comparison chart, word clouds
├── streamlit_app.py            # bonus interactive web UI
└── README.md
```

## About the Dataset
The dataset (`data/reviews_dataset.csv`, ~1,500 reviews) is generated offline
with `generate_dataset.py` using realistic review patterns across 20 product
categories (electronics, clothing, home goods, etc.), split evenly across
Positive / Negative / Neutral. It includes deliberately **ambiguous "hard"
examples** (e.g. "not bad at all, works really well") plus a small amount of
label noise (~6%) to mimic real-world annotator disagreement, so the task
isn't trivially easy.

> If you have internet access, you can swap this out for a real-world
> dataset such as an Amazon or Yelp reviews dataset on Kaggle — just replace
> `reviews_dataset.csv` with a CSV containing `label`
> (positive/negative/neutral) and `review` columns; the rest of the pipeline
> works unchanged.

## Pipeline Steps
1. **Dataset preparation** — realistic labeled review data
2. **Text cleaning** — lowercase, strip punctuation/digits, remove stopwords
3. **NLP preprocessing** — tokenization + stopword removal
4. **Feature extraction** — TF-IDF (unigrams, top 2,000 features)
5. **Models trained**:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM (bonus 3rd model)
6. **Evaluation** — Accuracy, macro-averaged Precision/Recall/F1 (3-class), Confusion Matrix
7. **Testing** — each model run on brand-new example reviews not seen during training
8. **Bonus** — Streamlit app + per-sentiment word cloud visualizations

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib

# 2. Generate the dataset
python data/generate_dataset.py

# 3. Train models, evaluate, and export artifacts
python src/train.py

# 4. (Bonus) Launch the interactive web app
pip install streamlit
streamlit run streamlit_app.py
```

## Results

| Model                | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-----------------------|----------|--------------------|-----------------|------------|
| Naive Bayes           | 0.943    | 0.943              | 0.943           | 0.943      |
| Logistic Regression   | 0.943    | 0.943              | 0.943           | 0.943      |
| Linear SVM            | 0.943    | 0.943              | 0.943           | 0.943      |

(Numbers regenerate each run and are saved to `outputs/model_comparison.csv`;
exact values can vary slightly depending on the random train/test split.)

See `outputs/` for:
- `confusion_matrix_naive_bayes.png`
- `confusion_matrix_logistic_regression.png`
- `confusion_matrix_svm.png`
- `model_comparison.png` (bar chart)
- `model_comparison.csv` (raw metrics table)
- `wordcloud_positive.png`, `wordcloud_negative.png`, `wordcloud_neutral.png`

## Example Predictions on New Reviews
```
[POSITIVE] This laptop stand is fantastic, sturdy and easy to set up!
[NEGATIVE] Terrible experience, the blender stopped working after one use.
[NEUTRAL ] It's an okay backpack, does the job but nothing special.
[POSITIVE] Absolutely love this coffee maker, best purchase this year!
[NEGATIVE] The jacket ripped on the first wear, very disappointed.
[NEUTRAL ] The mouse works fine, average quality for the price.
```

## Notes / Possible Extensions
- Swap in a real-world dataset (Amazon/Yelp reviews) for production use
- Add a Random Forest or a fine-tuned transformer (e.g. DistilBERT) for comparison
- Add k-fold cross-validation instead of a single train/test split
- Use a real `wordcloud` package for nicer visuals (this project uses a
  dependency-free matplotlib-based word cloud since the pipeline is designed
  to run fully offline)
