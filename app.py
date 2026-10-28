import io, zipfile, warnings
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yfinance as yf
from huggingface_hub import hf_hub_download
from transformers import pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')
st.set_page_config(page_title='Financial AI Dashboard', page_icon='📈', layout='wide')

@st.cache_resource(show_spinner='Loading FinBERT...')
def load_finbert():
    return pipeline('text-classification', model='ProsusAI/finbert', tokenizer='ProsusAI/finbert', device=0 if torch.cuda.is_available() else -1)

@st.cache_data(show_spinner='Evaluating FinBERT...')
def finbert_evaluation():
    zpath = hf_hub_download(repo_id='takala/financial_phrasebank', filename='data/FinancialPhraseBank-v1.0.zip', repo_type='dataset')
    with zipfile.ZipFile(zpath) as zf:
        name = next(n for n in zf.namelist() if n.endswith('Sentences_AllAgree.txt'))
        with zf.open(name) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding='latin-1'), sep='@', header=None, names=['sentence','sentiment'])
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
    df = df.dropna()
    tx, vx, ty, vy = train_test_split(df.sentence.tolist(), df.sentiment.tolist(), test_size=.2, random_state=42, stratify=df.sentiment)
    model = load_finbert()
    pred = []
    for i in range(0, len(vx), 16):
        pred.extend([x['label'].lower() for x in model(vx[i:i+16], truncation=True, batch_size=16)])
    return accuracy_score(vy, pred), f1_score(vy, pred, average='macro', zero_division=0), classification_report(vy, pred, labels=['negative','neutral','positive'], zero_division=0)

def analyze_stock(ticker, period):
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError('Enter a stock ticker.')
    raw = yf.download(ticker, period=period, interval='1d', auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f'No data found for {ticker}. Indian tickers often use .NS (NSE) or .BO (BSE).')
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    raw = raw[['Close','Volume']].dropna().sort_index()
    if len(raw) < 150: raise ValueError('Not enough historical data; try a longer period or another ticker.')
    f = raw.copy()
    for lag in [1,2,3,5,10,20]: f[f'Return_{lag}'] = f.Close.pct_change(lag)
    for w in [5,10,20,50]: f[f'MA_{w}'] = f.Close / f.Close.rolling(w).mean() - 1
    d = f.Close.diff(); gain = d.clip(lower=0).rolling(14).mean(); loss = (-d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan); f['RSI'] = 100 - 100/(1+rs)
    for w in [5,10,20]: f[f'Volatility_{w}'] = f.Close.pct_change().rolling(w).std()
    f['Volume_Ratio'] = f.Volume / f.Volume.rolling(20).mean()
    cols = [c for c in f.columns if c not in ['Close','Volume']]
    nr = raw.Close.shift(-1) / raw.Close - 1
    f['Target'] = np.select([nr < -.002, nr > .002], [0,2], default=1).astype(float)
    f.loc[nr.isna(), 'Target'] = np.nan
    train_data = f.replace([np.inf,-np.inf], np.nan).dropna().copy()
    train_data.Target = train_data.Target.astype(int)
    if len(train_data) < 120: raise ValueError('Not enough usable data after feature creation.')
    X, y = train_data[cols], train_data.Target
    n = len(X); a, b = int(n*.70), int(n*.85)
    Xtr,ytr = X.iloc[:a],y.iloc[:a]; Xv,yv = X.iloc[a:b],y.iloc[a:b]; Xt,yt = X.iloc[b:],y.iloc[b:]
    if len(Xv)==0 or len(Xt)==0: raise ValueError('Not enough rows for train/validation/test split.')
    if set(ytr.unique()) != {0,1,2}: raise ValueError('Training data lacks one of Down/Flat/Up classes; try another ticker or longer period.')
    model = XGBClassifier(n_estimators=250, max_depth=2, learning_rate=.03, subsample=.8, colsample_bytree=.8, reg_lambda=5, objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42, verbosity=0)
    model.fit(Xtr,ytr)
    vp = model.predict(Xv); tp = model.predict(Xt)
    acc = accuracy_score(yt,tp)*100; base = accuracy_score(yt, np.full(len(yt), int(ytr.mode().iloc[0])))*100
    mf1 = f1_score(yt,tp,labels=[0,1,2],average='macro',zero_division=0)
    # Features for latest available bar (whose next-day target is unknown)
    latest = raw.copy()
    for lag in [1,2,3,5,10,20]: latest[f'Return_{lag}'] = latest.Close.pct_change(lag)
    for w in [5,10,20,50]: latest[f'MA_{w}'] = latest.Close/latest.Close.rolling(w).mean()-1
    d=latest.Close.diff(); gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean(); rs=gain/loss.replace(0,np.nan); latest['RSI']=100-100/(1+rs)
    for w in [5,10,20]: latest[f'Volatility_{w}']=latest.Close.pct_change().rolling(w).std()
    latest['Volume_Ratio']=latest.Volume/latest.Volume.rolling(20).mean()
    lx=latest[cols].replace([np.inf,-np.inf],np.nan).dropna().iloc[[-1]]
    pr=model.predict_proba(lx)[0]; p={int(c):float(v) for c,v in zip(model.classes_,pr)}; names={0:'Down',1:'Flat',2:'Up'}; choice=max([0,1,2],key=lambda c:p.get(c,0))
    return {'ticker':ticker,'date':lx.index[-1],'prediction':names[choice], 'prob':{names[k]:p.get(k,0) for k in [0,1,2]},'accuracy':acc,'baseline':base,'f1':mf1,'val':accuracy_score(yv,vp)*100,'report':classification_report(yt,tp,labels=[0,1,2],target_names=['Down','Flat','Up'],zero_division=0),'matrix':confusion_matrix(yt,tp,labels=[0,1,2]),'n':(len(Xtr),len(Xv),len(Xt))}

st.title('📈 Financial AI Dashboard')
st.caption('FinBERT financial-news sentiment + XGBoost stock movement research tool')
st.warning('Educational research only. Predictions are uncertain and are not financial advice. News sentiment is not an input to the stock model.')
t_news, t_stock, t_eval = st.tabs(['📰 News Sentiment','📊 Stock Prediction','🧪 Model Evaluation'])
with t_news:
    st.subheader('Financial news sentiment — FinBERT')
    news = st.text_area('Enter a financial headline', placeholder='The company reported strong quarterly profits.')
    if st.button('Analyze sentiment'):
        if not news.strip(): st.error('Enter a headline first.')
        else:
            try:
                with st.spinner('Analyzing with FinBERT...'): result = load_finbert()(news.strip(), truncation=True)[0]
                c1,c2 = st.columns(2); c1.metric('Sentiment',result['label'].upper()); c2.metric('Text confidence',f"{result['score']*100:.2f}%")
                st.caption('Confidence refers to sentiment classification, not stock price direction.')
            except Exception as e: st.error(str(e))
with t_stock:
    st.subheader('Analyze any supported stock ticker')
    c1,c2 = st.columns([2,1])
    ticker = c1.text_input('Stock ticker',value='RVNL.NS',help='Examples: RVNL.NS, RELIANCE.NS, TCS.NS, INFY.NS, AAPL, NVDA')
    period = c2.selectbox('Historical period',['5y','3y','2y'],index=0)
    if st.button('Analyze stock',type='primary'):
        try:
            with st.spinner(f'Downloading history and training XGBoost for {ticker.upper()}...'): r=analyze_stock(ticker,period)
            st.success(f"Analysis complete: {r['ticker']}"); st.caption(f"Latest feature date: {r['date']}")
            st.metric('Next-session movement estimate',r['prediction'])
            p1,p2,p3=st.columns(3); p1.metric('Up model probability',f"{r['prob']['Up']*100:.2f}%"); p2.metric('Down model probability',f"{r['prob']['Down']*100:.2f}%"); p3.metric('Flat model probability',f"{r['prob']['Flat']*100:.2f}%")
            st.subheader('Unseen historical test evaluation')
            a,b,c=st.columns(3); a.metric('Test accuracy',f"{r['accuracy']:.2f}%"); b.metric('Majority baseline',f"{r['baseline']:.2f}%"); c.metric('Macro F1',f"{r['f1']:.4f}")
            st.write(f"Validation accuracy: **{r['val']:.2f}%** · Samples (train/validation/test): {r['n']}")
            st.text('Classification report'); st.code(r['report']); st.text('Confusion matrix (rows=actual, columns=predicted; Down, Flat, Up)'); st.write(r['matrix'])
            st.info('The model is trained separately for each selected stock. Estimates are not guaranteed trading signals.')
        except Exception as e: st.error(f'Could not analyze ticker: {e}')
with t_eval:
    st.subheader('FinBERT held-out evaluation')
    st.write('Evaluation uses a held-out split of Financial PhraseBank. First run may take several minutes.')
    if st.button('Run FinBERT evaluation'):
        try:
            with st.spinner('Evaluating FinBERT...'): acc,f1,report=finbert_evaluation()
            a,b=st.columns(2); a.metric('Held-out accuracy',f'{acc*100:.2f}%'); b.metric('Macro F1',f'{f1:.4f}'); st.code(report)
        except Exception as e: st.error(f'Evaluation failed: {e}')
    st.markdown('FinBERT and stock movement metrics measure different tasks and should not be compared directly.')
