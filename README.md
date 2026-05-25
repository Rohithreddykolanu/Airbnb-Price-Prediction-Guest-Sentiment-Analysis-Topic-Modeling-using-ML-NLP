<h1 align="center">Airbnb Price Prediction, Guest Sentiment Analysis & Topic Modeling</h1>

<p align="center"><i>Using Machine Learning, NLP, and Neural Networks</i></p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-2.0-013243?logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%2FKeras-NN-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-Regressor-006ACC">
  <img alt="NLTK" src="https://img.shields.io/badge/NLTK-NLP-154F5B">
  <img alt="Transformers" src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E">
  <img alt="BERTopic" src="https://img.shields.io/badge/BERTopic-Topic%20Modeling-4C72B0">
</p>

## Overview

This project analyzes the Chicago short term rental market using guest reviews and structured listing data from Inside Airbnb, based on the snapshot dated September 22, 2025. It pursues two connected goals. The first is a guest experience analysis that turns unstructured review text into measurable signals through sentiment analysis, zero shot classification, and topic modeling. The second is a price prediction study that combines structured property attributes with the review based signals to model nightly listing prices using neural networks and gradient boosting. Together these objectives test whether qualitative guest perceptions carry pricing information beyond what structured listing attributes alone can capture.

## Objectives

1. **Guest experience analysis.** Apply natural language processing to Chicago Airbnb reviews to quantify sentiment, score guest experience themes, and surface the topics guests discuss most often, producing actionable insight for hosts.
2. **Price prediction.** Build and compare regression models that predict listing price from property characteristics, host quality signals, and review derived features, and identify the key drivers of listing valuation.

## Data

The data was sourced from Inside Airbnb for Chicago, Illinois.

| Dataset | Initial shape | Role |
| --- | --- | --- |
| Reviews | 492,465 records across 6 columns | Text corpus for all NLP tasks |
| Listings | 8,660 records across 79 columns | Structured input for price prediction |

The reviews were filtered to the 2023 to 2025 period, aggregated to the most recent 20 reviews per listing, and reduced to one consolidated entry for each of 6,705 unique listings. After language filtering, 5,982 English listings remained for NLP. The listings data was filtered to Chicago based hosts, merged with the NLP features, cleaned, and reduced to a final modeling set of 3,840 listings.

## Part 1: Guest Experience Analysis (NLP)

**Text preprocessing.** A sequential pipeline removed URLs and special characters, lowercased text, tokenized on whitespace, removed standard NLTK stopwords alongside a custom Airbnb and Chicago stopword set, and lemmatized tokens with the WordNet lemmatizer. Embedded HTML was decoded and stripped with BeautifulSoup, and a transformer language detector (`papluca/xlm-roberta-base-language-detection`) retained only English reviews, which accounted for 89.22 percent of listings.

**Sentiment analysis.** Two complementary approaches were compared. The VADER lexicon analyzer classified 96.44 percent of listings as positive, while the Hugging Face transformer pipeline (`distilbert-base-uncased-finetuned-sst-2-english`) classified 67.95 percent as positive. The gap reflects how the custom stopword removal stripped common positive adjectives, leaving the transformer to work with more neutral vocabulary.

**Zero shot classification.** The `typeform/distilbert-base-uncased-mnli` model scored each listing across five guest experience categories of Cleanliness, Safety, Amenities, Location, and Service using a multi label setup, producing thematic relevance scores that later served as price prediction features.

**Topic modeling.** Three classical methods were implemented and compared, namely Latent Dirichlet Allocation on a Bag of Words matrix, Non negative Matrix Factorization on a TF IDF matrix, and Latent Semantic Analysis through Truncated SVD. NMF produced the most interpretable and structurally balanced topics, surfacing themes such as walkability, upscale property features and views, group and family stays, and property issues. BERTopic was added as a transformer based method that learns topics from contextual sentence embeddings.

## Part 2: Price Prediction

The filtered listings were merged with the Hugging Face sentiment scores, the five zero shot category scores, and the five LDA topic scores. A focused set of 24 columns was selected across property characteristics, location, host quality, review signals, availability, and NLP derived features. After type conversion, missing value treatment, selective outlier handling, top one percent price trimming, and categorical encoding, the data was split 70 to 30 into 2,660 training and 1,141 test records with 23 features.

Three models were built and compared:

- **Neural Network Model 1 (baseline).** A Keras Sequential network with a normalization layer and three hidden layers of 128, 64, and 32 ReLU units, trained with the Adam optimizer and MSE loss.
- **Neural Network Model 2 (regularized).** A deeper four layer network adding L2 regularization, Batch Normalization, Dropout, and Early Stopping to control the overfitting seen in Model 1.
- **XGBoost Regressor.** A gradient boosting model tuned with 5 fold Grid Search Cross Validation over depth, learning rate, estimators, subsampling, and regularization parameters.

## Results

| Model | MAE | MAPE | RMSE | R² | Adj. R² |
| --- | --- | --- | --- | --- | --- |
| NN Model 1 (Test) | 57.04 | 36.13% | 89.40 | 0.5356 | 0.5260 |
| NN Model 2 (Test) | 57.25 | 38.67% | 86.85 | 0.5617 | 0.5526 |
| XGBoost (Test) | 53.71 | 35.91% | 81.23 | 0.6166 | 0.6087 |

The XGBoost model delivered the best test performance with an R² of 0.6166 and the lowest test RMSE of 81.23 dollars, outperforming both neural networks on every error metric. The regularized neural network achieved the smallest gap between training and test scores, confirming the value of its regularization strategy even though its overall accuracy was lower.

## Key Findings

- **Property size dominates price.** Accommodation capacity, bedrooms, and bathrooms were the strongest predictors in both the correlation analysis and the XGBoost feature importance, with bathrooms correlating with price at 0.63, accommodates at 0.61, and bedrooms at 0.59.
- **Room type matters.** Entire home listings commanded the highest prices while private rooms were substantially cheaper, and room type ranked among the top four model features.
- **Review signals add real value.** Among the NLP features, the LDA topic associated with upscale property features and views (topic_2) was the most influential, confirming that qualitative guest experience captured from text is linked to listing valuation.
- **Reviews skew strongly positive.** Both sentiment methods and the high review ratings reflect a market where guests who complete stays tend to leave favourable feedback.
- **NMF was the preferred topic model** for generating interpretable, well separated themes used as features.

## Recommendations

- For the platform, build a dynamic pricing guidance tool that blends structured attributes with review derived indicators, and feed sentiment and topic scores into search ranking so consistently strong listings gain visibility.
- For hosts, prioritize accommodation capacity as the strongest revenue lever, deliver consistently high quality experiences across cleanliness, service, and location, and consider whether converting private rooms to full home access can unlock a higher price tier.
- For guests, use sentiment, cleanliness, and service signals alongside star ratings to judge value, and weigh private room options carefully against their review based quality scores.

## Repository Structure

```
airbnb-price-sentiment-nlp/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── airbnb_analysis.ipynb
├── reports/
│   └── Project_Final_Report.docx
└── presentation/
    └── Project_Presentation.pptx
```

## Getting Started

```bash
# clone
git clone https://github.com/<your-username>/airbnb-price-sentiment-nlp.git
cd airbnb-price-sentiment-nlp

# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# launch the notebook
jupyter notebook notebooks/airbnb_analysis.ipynb
```

The datasets are not committed to the repository. They can be downloaded directly from Inside Airbnb at https://insideairbnb.com/get-the-data/ using the Chicago snapshot.

## Tech Stack

Python, pandas, NumPy, Matplotlib, seaborn, Plotly, scikit-learn, TensorFlow and Keras, XGBoost, NLTK, vaderSentiment, Hugging Face Transformers, BeautifulSoup, WordCloud, BERTopic, Jupyter

## Team

Group 5: Rohith Reddy Kolanu (technical lead), Allen Joe Winny Tharigopala, Ruthwik Reddy Kolanu, and Gianni Colonero.
