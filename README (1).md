# Financial AI: News Sentiment & Stock Movement Predictor

A Streamlit web application combining financial-news sentiment analysis
with experimental stock-movement classification.

> **Disclaimer:** This project is for educational purposes only.
> Predictions are experimental, not financial advice, and do not
> guarantee returns.

## Project Overview

The app has three main sections:

1.  **News Sentiment** --- uses the pre-trained FinBERT model to
    classify financial text as Positive, Negative, or Neutral.
2.  **Stock Prediction** --- downloads historical data for a
    user-selected ticker, engineers technical features, and trains an
    XGBoost classifier to estimate the next session's movement as Down,
    Flat, or Up.
3.  **Model Evaluation** --- presents evaluation results for the
    available model components.

**Important:** FinBERT and the XGBoost stock classifier are separate
components. FinBERT sentiment is not currently an input feature for the
stock classifier.

## Features

-   Financial-news sentiment classification.
-   User-selectable ticker symbols, such as `RVNL.NS`, `RELIANCE.NS`,
    `TCS.NS`, `INFY.NS`, `AAPL`, or `NVDA`.
-   Historical market data retrieved with `yfinance`.
-   Technical features including returns, moving-average ratios, RSI,
    volatility, and volume ratios.
-   XGBoost multi-class stock movement classification.
-   Chronological train/validation/test split (70% / 15% / 15%).

## Models and Datasets

### FinBERT --- pre-trained sentiment model

**Link:** https://huggingface.co/ProsusAI/finbert

FinBERT is a pre-trained language model designed for financial sentiment
analysis. It predicts Positive, Negative, or Neutral sentiment. Unless a
separate fine-tuning step is added, describe it as a pre-trained model,
not a model trained from scratch in this project.

### Financial PhraseBank --- labelled evaluation dataset

**Link:** https://huggingface.co/datasets/takala/financial_phrasebank

**Configuration:** `sentences_allagree`

Financial PhraseBank contains labelled financial-news sentences. The
`sentences_allagree` configuration contains examples for which
annotators agreed on the sentiment label. In this project, the dataset
is used to evaluate sentiment predictions; it is not the stock-price
dataset.

### Stock market data

Historical price and volume data are retrieved through `yfinance` from
Yahoo Finance data. Availability and coverage depend on the selected
ticker and data provider.

## How It Works

### Sentiment analysis

1.  The user enters a financial headline, news sentence, or statement.
2.  FinBERT predicts its sentiment.
3.  The app displays the predicted class and confidence information
    where implemented.
4.  For evaluation, predictions are compared with the labelled Financial
    PhraseBank examples.

### Stock movement prediction

1.  The user enters a stock ticker.
2.  The app fetches historical market data.
3.  Technical features are calculated from historical prices and
    volumes.
4.  The next-session return is assigned a class:
    -   **Down:** return below `-0.2%`
    -   **Flat:** return between `-0.2%` and `+0.2%`
    -   **Up:** return above `+0.2%`
5.  Data is split chronologically into training, validation, and test
    sets.
6.  XGBoost is trained on earlier observations and evaluated on later
    observations.

Stock prices are difficult to predict. Accuracy may vary by ticker and
period; good historical metrics do not prove that a strategy will be
profitable. Earlier experiments showed modest stock-classification
performance, and the Flat class was particularly difficult to predict.

## Technology Stack

-   Python
-   Streamlit
-   Hugging Face Transformers
-   FinBERT (`ProsusAI/finbert`)
-   Financial PhraseBank
-   yfinance
-   XGBoost
-   scikit-learn
-   pandas and NumPy
-   PyTorch

## Project Structure

``` text
financial_ai_streamlit/
├── app.py
├── requirements.txt
└── README.md
```

-   `app.py` --- main Streamlit application.
-   `requirements.txt` --- Python dependencies.
-   `README.md` --- project overview and instructions.

## Run Locally

### 1. Install Python

Use a Python version compatible with the packages listed in
`requirements.txt`.

### 2. Open a terminal in the project folder

Make sure `app.py` and `requirements.txt` are in the same folder.

### 3. (Recommended) Create a virtual environment

**Windows:**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

### 5. Start the application

``` bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually
`http://localhost:8501`. The first run may take longer because the
FinBERT model and dependencies may need to download. Internet access is
required for model downloads and market-data retrieval.

## Deploy on Streamlit Community Cloud

1.  Create a GitHub repository.
2.  Upload `app.py`, `requirements.txt`, and `README.md` to the
    repository root.
3.  Visit https://share.streamlit.io/ and sign in with GitHub.
4.  Create a new app and select the repository and branch.
5.  Set the main file path to `app.py`.
6.  Deploy and inspect the app logs if installation or startup fails.

Large machine-learning dependencies can make initial deployment slower
and may require sufficient memory.

## Evaluation and Limitations

-   Sentiment evaluation uses human-labelled financial sentences from
    Financial PhraseBank.
-   Stock evaluation uses a chronological holdout test set and
    classification metrics implemented in the app.
-   Market behaviour changes, so historical patterns may not continue.
-   Some tickers may return insufficient data or fail to download.
-   Transaction costs, slippage, taxes, and liquidity are not
    necessarily included in model predictions.
-   The app does not guarantee returns and does not automatically place
    trades.

## Responsible Use

Use this project as a learning demonstration, not as the sole basis for
investment decisions. Verify data and results independently and apply
appropriate risk management.

## References

-   FinBERT model: https://huggingface.co/ProsusAI/finbert
-   Financial PhraseBank dataset:
    https://huggingface.co/datasets/takala/financial_phrasebank
-   Streamlit documentation: https://docs.streamlit.io/
-   yfinance: https://github.com/ranaroussi/yfinance
-   XGBoost documentation: https://xgboost.readthedocs.io/

## Author

Add your name, course, institution, and project guide here if required
for submission.
