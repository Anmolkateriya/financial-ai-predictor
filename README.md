# Financial AI Dashboard

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose your repository, branch, and `app.py` as the main file.
4. Click **Deploy**.

First startup can take several minutes because FinBERT model weights are downloaded when needed. Market data comes from Yahoo Finance. Predictions are experimental, not financial advice.
