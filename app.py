import re
import time
from collections import Counter
from html import escape
from io import StringIO
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

APP_NAME = 'SMS Shield AI'
BASE = Path(__file__).parent
PAGES = ['Live Predictor', 'Model Comparison', 'Dataset & EDA', 'Preprocessing Pipeline', 'Batch SMS Tester']
GREEN, RED, ORANGE, GREY = '#10b981', '#ef4444', '#f59e0b', '#9ca3af'
MODEL_COLORS = {'Naive Bayes': GREEN, 'SVM': RED, 'Logistic Regression': ORANGE}
DEFAULT_SMS = 'URGENT! Your mobile number has been awarded £2,000 cash prize! Claim now by calling 09061701461. T&C apply.'
EXAMPLES = {
    '🎁 Prize scam': 'Congratulations! You have won a free iPhone. Click the link to claim your prize now!',
    '🏦 Bank scam': 'URGENT: your bank account is blocked. Call 09061234567 to verify.',
    '💬 Normal chat': 'Hey, are we meeting at 6 for dinner today?',
    '🔐 OTP message': 'Your OTP for login is 483920. Do not share it with anyone.',
}
SAMPLE_BATCH = [
    'URGENT! Call 09061701461 to claim your £2000 cash prize. T&C apply.',
    'Hey, are you free for lunch tomorrow around 1? Let me know.',
    'WINNER! You have won a free gift card! Text WIN to 82222 now.',
    'Can you send me the project guidelines when you have time?',
    'Free entry in a weekly competition. Text WIN to 80086 now',
    "I'll be there in 10 minutes, traffic is a bit heavy.",
    'FreeMsg: Urgent! Reply STOP to 80488 for weekly cash prizes. T&C apply',
    'Are we still meeting in the campus library at 3 PM today?',
    'Your OTP for login is 483920. Do not share it with anyone.',
    'You have been selected for a £500 supermarket voucher! Claim now at www.prizes-uk.com',
]

st.set_page_config(page_title=APP_NAME, page_icon='🛡️', layout='wide')

# ---------------------------------------------------------------- styling
st.markdown("""<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}
header[data-testid="stHeader"] {background: transparent;}
h1.page-title {font-size: 1.9rem; font-weight: 800; margin: 0; padding: 0; color: #111827;}
p.page-sub {color: #6b7280; margin: 2px 0 18px 0; font-size: 0.95rem;}
.cards {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px;}
.card {background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 18px;}
.c-label {font-size: 0.82rem; color: #6b7280;}
.c-value {font-size: 1.7rem; font-weight: 800; color: #111827; line-height: 1.25; margin-top: 2px;}
.c-sub {font-size: 0.78rem; color: #6b7280; margin-top: 2px;}
.sec {font-size: 1.05rem; font-weight: 700; color: #111827; margin: 0 0 6px 0;}
.hint {font-size: 0.88rem; color: #6b7280; margin-bottom: 8px;}
div[data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border-radius: 12px;}
.banner {display: flex; align-items: center; justify-content: space-between; gap: 24px; border-radius: 10px; padding: 14px 20px; margin: 14px 0 12px 0;}
.banner.spam {background: #fee2e2; border: 1px solid #fecaca; color: #b91c1c;}
.banner.ham {background: #dcfce7; border: 1px solid #bbf7d0; color: #15803d;}
.banner .b-left {font-size: 1.35rem; font-weight: 800; white-space: nowrap;}
.banner .b-right {flex: 1; max-width: 360px; font-size: 0.85rem; font-weight: 600;}
.banner .b-top {display: flex; justify-content: space-between; margin-bottom: 4px;}
.track {background: rgba(0,0,0,0.10); border-radius: 8px; height: 10px; overflow: hidden;}
.fill {height: 100%; border-radius: 8px;}
.banner.spam .fill {background: #ef4444;} .banner.ham .fill {background: #22c55e;}
.box {background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px; height: 100%; color: #1f2937; font-size: 0.9rem;}
.box h4 {margin: 0 0 8px 0; font-size: 0.95rem; color: #111827;}
.mono {font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.82rem; color: #374151; word-break: break-word;}
.sig {display: flex; align-items: center; gap: 8px; margin: 5px 0; font-size: 0.85rem;}
.sig .t {width: 64px; text-align: right; color: #374151;} .sig .v {width: 44px; color: #6b7280; font-size: 0.78rem;}
.sig .bar {flex: 1; background: #f3f4f6; border-radius: 6px; height: 9px;} .sig .bar div {height: 100%; border-radius: 6px; background: #ef4444;}
.chip {display: inline-block; padding: 2px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; color: #fff;}
.chip.spam {background: #ef4444;} .chip.ham {background: #10b981;}
.pill {display: inline-block; padding: 5px 12px; border-radius: 8px; background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; font-size: 0.85rem; margin: 0 8px 6px 0;}
table.t {width: 100%; border-collapse: collapse; font-size: 0.86rem; color: #1f2937;}
table.t th {text-align: left; padding: 9px 10px; border-bottom: 1px solid #e5e7eb; background: #fafafa; font-weight: 600; color: #374151;}
table.t td {padding: 9px 10px; border-bottom: 1px solid #f3f4f6;}
.stage-row {display: grid; grid-template-columns: 1fr 24px 1fr 24px 1fr 24px 1fr; align-items: stretch; gap: 6px;}
.arrow {display: flex; align-items: center; justify-content: center; color: #9ca3af; font-size: 1.2rem;}
.stage {background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 14px;}
.stage h4 {margin: 0 0 8px 0; font-size: 0.9rem; color: #111827;}
.stage .inner {background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; min-height: 96px;}
.stage .foot {display: inline-block; margin-top: 8px; font-size: 0.75rem; background: #f3f4f6; border-radius: 6px; padding: 2px 8px; color: #4b5563;}
/* sidebar */
section[data-testid="stSidebar"] {background: #0f172a; min-width: 270px;}
section[data-testid="stSidebar"] * {color: #e2e8f0;}
.brand {font-size: 1.25rem; font-weight: 800; color: #fff; margin: 4px 0 14px 0;}
.side-h {font-size: 0.95rem; font-weight: 700; color: #fff; margin: 18px 0 8px 0; padding-top: 14px; border-top: 1px solid #1e293b;}
.side-box {background: #1e293b; border-radius: 8px; padding: 10px 12px; font-size: 0.82rem; margin-top: 12px; color: #cbd5e1;}
section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 4px;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {padding: 8px 12px; border-radius: 8px; width: 100%; cursor: pointer;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background: #166534;}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {display: none;}
section[data-testid="stSidebar"] [data-baseweb="select"] *, section[data-testid="stSidebar"] [data-testid="stSelectbox"] input,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] button {color: #111827 !important; -webkit-text-fill-color: #111827 !important;}
.stButton > button, .stDownloadButton > button {border-radius: 8px; font-weight: 600;}
div[data-testid="stTextArea"] textarea {border-radius: 10px; font-size: 0.95rem;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- text processing
def tokenize(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)   # links
    text = re.sub(r'\d+', ' num ', text)                 # numbers
    text = re.sub(r'[^a-z\s]', ' ', text)                # punctuation / symbols
    return text.split()                                   # tokenization


def preprocess(text, stop_words, stemmer):
    # same logic as preprocess() in the training notebook
    clean_tokens = []
    for w in tokenize(text):
        if w not in stop_words and len(w) > 1:
            clean_tokens.append(stemmer.stem(w))
    return ' '.join(clean_tokens)


def spam_weights(name, model):
    """Per-feature weight toward spam, used to show which words pushed the decision."""
    if name == 'Naive Bayes':
        return model.feature_log_prob_[1] - model.feature_log_prob_[0]
    if name == 'Logistic Regression':
        return model.coef_[0]
    weights = []
    for cc in model.calibrated_classifiers_:
        est = getattr(cc, 'estimator', None) or getattr(cc, 'base_estimator')
        weights.append(est.coef_[0])
    return np.mean(weights, axis=0)


@st.cache_resource(show_spinner='Training the models on the SMS dataset (first run only)...')
def build():
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    path = next((p for p in (BASE / 'data' / 'SMSSpamCollection', BASE / 'SMSSpamCollection') if p.exists()), None)
    if path is None:
        return None
    raw = pd.read_csv(path, sep='\t', names=['label', 'text'])
    df = raw.drop_duplicates().reset_index(drop=True)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_text'] = df['text'].apply(lambda t: preprocess(t, stop_words, stemmer))
    df['length'] = df['text'].str.len()

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label_num'], test_size=0.2, stratify=df['label_num'], random_state=42)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    Xtr, Xte = tfidf.fit_transform(X_train), tfidf.transform(X_test)

    models = {
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'SVM': CalibratedClassifierCV(LinearSVC(class_weight='balanced'), cv=5),
    }
    probs, speed, weights = {}, {}, {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        probs[name] = model.predict_proba(Xte)[:, 1]
        sample_texts = list(X_test[:50])          # time one-message-at-a-time, like the live app
        t0 = time.perf_counter()
        for t in sample_texts:
            model.predict_proba(tfidf.transform([t]))
        speed[name] = (time.perf_counter() - t0) / len(sample_texts) * 1000
        weights[name] = spam_weights(name, model)

    vocab = set()
    for t in df['text']:
        vocab.update(tokenize(t))
    return dict(raw_n=len(raw), df=df, tfidf=tfidf, models=models, probs=probs, speed=speed, weights=weights,
                y_test=y_test.values, n_test=len(y_test), stop_words=stop_words, stemmer=stemmer,
                names=tfidf.get_feature_names_out(), doc_freq=np.asarray((Xtr > 0).sum(axis=0)).ravel(),
                raw_vocab=len(vocab))


A = build()
if A is None:
    st.error('Dataset not found. Put the SMSSpamCollection file in a data/ folder next to app.py.')
    st.stop()


def evaluate(thr):
    rows, cms = [], {}
    y = A['y_test']
    for name in A['models']:
        pred = (A['probs'][name] >= thr).astype(int)
        rows.append({'Model': name, 'Accuracy': accuracy_score(y, pred),
                     'Precision': precision_score(y, pred, zero_division=0),
                     'Recall': recall_score(y, pred, zero_division=0), 'F1': f1_score(y, pred, zero_division=0),
                     'Speed': A['speed'][name]})
        cms[name] = confusion_matrix(y, pred)
    return pd.DataFrame(rows).set_index('Model'), cms


def top_terms(indices, values, k):
    """Strongest words pushing toward spam (positive contribution only)."""
    out = []
    for i in np.argsort(values)[::-1]:
        name = A['names'][indices[i]]
        if values[i] <= 0:
            break
        if set(name.split()) == {'num'}:      # the number placeholder says nothing on its own
            continue
        out.append((name, float(values[i])))
        if len(out) == k:
            break
    return out


def predict_text(message, model_name):
    clean = preprocess(message, A['stop_words'], A['stemmer'])
    vec = A['tfidf'].transform([clean])
    prob = float(A['models'][model_name].predict_proba(vec)[0, 1])
    contrib = vec.multiply(A['weights'][model_name].reshape(1, -1)).tocsr()
    return clean, prob, top_terms(contrib.indices, contrib.data, 5)


# ---------------------------------------------------------------- html helpers
def page_header(title, sub='NLP Capstone Project • Project 2 • UCI SMS Spam Collection'):
    st.markdown(f'<h1 class="page-title">{title}</h1><p class="page-sub">{sub}</p>', unsafe_allow_html=True)


def cards(items):
    html = ''.join(f'<div class="card"><div class="c-label">{l}</div><div class="c-value">{v}</div>'
                   f'<div class="c-sub">{s}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="cards">{html}</div>', unsafe_allow_html=True)


def sec(title, hint=''):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="hint">{hint}</div>' if hint else ''),
                unsafe_allow_html=True)


def table(headers, rows):
    head = ''.join(f'<th>{h}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    st.markdown(f'<table class="t"><tr>{head}</tr>{body}</table>', unsafe_allow_html=True)


def fmt_speed(ms):
    return f'{ms * 1000:.0f} µs' if ms < 0.1 else f'{ms:.2f} ms'


def chip(label):
    kind = 'spam' if str(label).lower() == 'spam' else 'ham'
    return f'<span class="chip {kind}">{kind.upper()}</span>'


def show(fig, height=330, legend_top=False):
    legend = dict(orientation='h', yanchor='bottom', y=1.02 if legend_top else -0.25, x=0.5, xanchor='center')
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#374151', size=12), legend=legend)
    try:
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def perf_fig(res):
    fig = go.Figure()
    for name in res.index:
        vals = [res.loc[name, c] * 100 for c in ('Accuracy', 'Precision', 'Recall', 'F1')]
        fig.add_bar(name=name, x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], y=vals,
                    marker_color=MODEL_COLORS[name], text=[f'{v:.1f}%' for v in vals], textposition='outside')
    fig.update_yaxes(range=[0, 112], title='Score (%)', gridcolor='#eef0f3')
    fig.update_layout(barmode='group')
    return fig


def cm_heat(cm, short=False):
    z = cm / cm.sum(axis=1, keepdims=True)
    return go.Heatmap(z=z, text=cm, texttemplate='%{text}', textfont=dict(size=18), showscale=False,
                      x=['Ham', 'Spam'] if short else ['Predicted Ham', 'Predicted Spam'],
                      y=['Ham', 'Spam'] if short else ['Actual Ham', 'Actual Spam'],
                      colorscale=[[0, '#f3f4f6'], [1, '#10b981']], zmin=0, zmax=1, xgap=3, ygap=3)


# ---------------------------------------------------------------- sidebar
st.session_state.setdefault('sms', DEFAULT_SMS)
st.session_state.setdefault('analyzed', DEFAULT_SMS)
res0, _ = evaluate(0.5)
best = res0['F1'].idxmax()

with st.sidebar:
    st.markdown(f'<div class="brand">🛡️ {APP_NAME}</div>', unsafe_allow_html=True)
    page = st.radio('Navigation', PAGES, label_visibility='collapsed')
    st.markdown('<div class="side-h">⚙️ Configuration & Model Settings</div>', unsafe_allow_html=True)
    names = list(A['models'])
    model_name = st.selectbox('Select Model', names, index=names.index(best),
                              format_func=lambda n: n + (' (Best)' if n == best else ''))
    threshold = st.slider('Spam Decision Threshold', 0.05, 0.95, 0.50, 0.05)
    st.markdown(f'<div class="side-box">Vectorizer: TF-IDF (1-2 grams, 5,000 features)</div>'
                f'<div class="side-box">Dataset: UCI SMS Spam Collection ({len(A["df"]):,} unique SMS)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-box">NLP Capstone Project</div>', unsafe_allow_html=True)

res, cms = evaluate(threshold)
df = A['df']
n_spam, n_ham = int(df['label_num'].sum()), int((df['label_num'] == 0).sum())

# ================================================================ page 1
if page == 'Live Predictor':
    page_header('💬 SMS Spam Detection Dashboard', 'NLP Capstone Project • Project 2 • UCI SMS Spam Collection Dataset')
    r = res.loc[model_name]
    cards([('Total Dataset Messages', f'{len(df):,}', f'{n_ham:,} Ham | {n_spam:,} Spam'),
           ('Model Accuracy', f'{r.Accuracy:.1%}', f'{model_name} classifier'),
           ('Spam Precision', f'{r.Precision:.1%}', 'When it says spam, how often it is right'),
           ('F1-Score', f'{r.F1:.1%}', 'Balanced harmonic mean')])

    with st.container(border=True):
        sec('✉️ Live SMS Classifier', 'Enter any SMS message below to analyze in real-time:')

        def use_example(text):
            st.session_state.sms = text
            st.session_state.analyzed = text

        ex_cols = st.columns(len(EXAMPLES))
        for col, (label, text) in zip(ex_cols, EXAMPLES.items()):
            col.button(label, on_click=use_example, args=(text,), key='ex_' + label)
        st.text_area('SMS message', key='sms', height=110, label_visibility='collapsed')
        if st.button('Analyze Message 🔍', type='primary'):
            if st.session_state.sms.strip():
                st.session_state.analyzed = st.session_state.sms.strip()
            else:
                st.warning('Please enter a message first.')

        message = st.session_state.analyzed
        clean, prob, signals = predict_text(message, model_name)
        is_spam = prob >= threshold
        conf = prob if is_spam else 1 - prob
        kind, title = ('spam', '⚠️ SPAM DETECTED') if is_spam else ('ham', '✅ NOT SPAM (SAFE)')
        st.markdown(f'<div class="banner {kind}"><div class="b-left">{title}</div><div class="b-right">'
                    f'<div class="b-top"><span>Prediction Confidence:</span><span>{conf:.1%}</span></div>'
                    f'<div class="track"><div class="fill" style="width:{conf * 100:.1f}%"></div></div></div></div>',
                    unsafe_allow_html=True)

        left, right = st.columns([1.5, 1])
        with left:
            st.markdown(f'<div class="box"><h4>Text Processing Insights</h4>'
                        f'<b>Original Length:</b> {len(message)} characters | <b>Cleaned Tokens:</b> {len(clean.split())}<br><br>'
                        f'<span class="mono"><b>Pipeline:</b> [Lowercase, Links/Numbers Replaced, Stopwords Removed, Stemmed]<br>'
                        f'<b>Cleaned text:</b> {escape(clean) if clean else "(nothing left after cleaning)"}</span></div>',
                        unsafe_allow_html=True)
        with right:
            top = max([s for _, s in signals], default=1)
            bars = ''.join(f'<div class="sig"><span class="t">{escape(t)}</span><div class="bar"><div style="width:{s / top * 100:.0f}%"></div></div>'
                           f'<span class="v">({s:.2f})</span></div>' for t, s in signals)
            st.markdown(f'<div class="box"><h4>Top Spam Signal Terms</h4>'
                        f'{bars or "No spam-leaning words found in this message."}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        sec('📊 Model Performance Comparison')
        show(perf_fig(res), 340)
    with c2, st.container(border=True):
        sec(f'📈 Confusion Matrix ({model_name})')
        fig = go.Figure(cm_heat(cms[model_name]))
        fig.update_yaxes(autorange='reversed')
        show(fig, 340)

# ================================================================ page 2
elif page == 'Model Comparison':
    page_header('📊 Model Evaluation & Comparison')
    best_t = res['F1'].idxmax()
    others = [n for n in res.index if n != best_t]
    cards([('Best Model', best_t, f'{res.loc[best_t, "Accuracy"]:.1%} accuracy • highest F1-score'),
           (f'{others[0]} Accuracy', f'{res.loc[others[0], "Accuracy"]:.1%}', f'Precision: {res.loc[others[0], "Precision"]:.1%}'),
           (f'{others[1]} Accuracy', f'{res.loc[others[1], "Accuracy"]:.1%}', f'Precision: {res.loc[others[1], "Precision"]:.1%}'),
           ('TF-IDF Vectorizer', f'{len(A["names"]):,} Features', 'N-grams: (1, 2) | Stopwords removed')])
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        sec('📊 Model Performance Comparison')
        show(perf_fig(res), 380)
    with c2, st.container(border=True):
        sec('📈 Side-by-Side Confusion Matrices')
        fig = make_subplots(rows=1, cols=3, subplot_titles=list(cms), horizontal_spacing=0.08)
        for i, name in enumerate(cms, start=1):
            fig.add_trace(cm_heat(cms[name], short=True), row=1, col=i)
            fig.update_yaxes(autorange='reversed', showticklabels=(i == 1), title_text='Actual' if i == 1 else None, row=1, col=i)
            fig.update_xaxes(title_text='Predicted', row=1, col=i)
        show(fig, 380)
    fastest = res['Speed'].idxmin()
    rows = []
    for name, rr in res.iterrows():
        status = '🏆 Best Overall' if name == best_t else ('⚡ Fastest' if name == fastest else 'Baseline')
        rows.append([f'<b>{name}</b>', f'{rr.Accuracy:.1%}', f'{rr.Precision:.1%}', f'{rr.Recall:.1%}',
                     f'{rr.F1:.1%}', fmt_speed(rr.Speed), status])
    with st.container(border=True):
        sec(f'📋 Detailed Model Evaluation Table (Test Set: {A["n_test"]:,} SMS)')
        table(['Model Name', 'Accuracy', 'Precision (Spam)', 'Recall (Spam)', 'F1-Score (Spam)', 'Inference per SMS', 'Status'], rows)

# ================================================================ page 3
elif page == 'Dataset & EDA':
    page_header('📊 Exploratory Data Analysis — UCI SMS Spam Collection')
    ham_len, spam_len = df[df.label == 'ham']['length'], df[df.label == 'spam']['length']
    cards([('Total SMS Records', f'{len(df):,}', f'{A["raw_n"]:,} raw • {A["raw_n"] - len(df)} duplicates removed'),
           ('Ham Messages', f'{n_ham:,} ({n_ham / len(df):.1%})', 'Regular legitimate text'),
           ('Spam Messages', f'{n_spam:,} ({n_spam / len(df):.1%})', 'Unsolicited spam texts'),
           ('Avg Message Length', f'{spam_len.mean():.0f} chars vs {ham_len.mean():.0f}',
            f'Spam: {spam_len.mean():.0f} ch | Ham: {ham_len.mean():.0f} ch')])
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        sec('📈 Class Distribution (Ham vs. Spam)')
        fig = go.Figure(go.Pie(labels=['Ham', 'Spam'], values=[n_ham, n_spam], hole=0.6,
                               marker_colors=[GREEN, RED], textinfo='percent'))
        show(fig, 340)
    with c2, st.container(border=True):
        sec('📊 Message Length Distribution (Characters)')
        xs = np.linspace(0, 320, 240)
        fig = go.Figure()
        for lab, series, colr in (('Ham', ham_len, GREEN), ('Spam', spam_len, RED)):
            fig.add_scatter(x=xs, y=gaussian_kde(series)(xs), name=lab, fill='tozeroy', line_color=colr)
        fig.update_xaxes(title='Character Count', gridcolor='#eef0f3')
        fig.update_yaxes(title='Density', gridcolor='#eef0f3')
        show(fig, 340, legend_top=True)
    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        sec('🔤 Top 10 Most Frequent Spam Tokens', 'Stemmed words from spam messages. The num and url placeholders are left out.')
        words = ' '.join(df[df.label == 'spam']['clean_text']).split()
        common = [(w, c) for w, c in Counter(words).most_common(14) if w not in ('num', 'url')][:10]
        fig = go.Figure(go.Bar(x=[c for _, c in common][::-1], y=[w for w, _ in common][::-1], orientation='h',
                               marker_color=RED, text=[c for _, c in common][::-1], textposition='outside'))
        fig.update_xaxes(gridcolor='#eef0f3', range=[0, max(c for _, c in common) * 1.18])
        show(fig, 360)
    with c4, st.container(border=True):
        sec('📋 Sample Dataset Preview (UCI SMS Spam Collection)')
        rows = []
        for _, rr in df.sample(8, random_state=3).iterrows():
            txt = rr.text if len(rr.text) <= 70 else rr.text[:67] + '...'
            rows.append([chip(rr.label), escape(txt), rr.length])
        table(['Label', 'SMS Message Text', 'Length'], rows)

# ================================================================ page 4
elif page == 'Preprocessing Pipeline':
    page_header('⚙️ NLP Preprocessing & Feature Vectorization Pipeline')
    cards([('Raw Vocabulary', f'{A["raw_vocab"]:,} words', 'Distinct tokens in the corpus'),
           ('TF-IDF Features', f'{len(A["names"]):,}', 'Vocabulary limit: 5,000 (1-2 grams)'),
           ('Stopwords Removed', f'{len(A["stop_words"])}', 'NLTK English stopword list'),
           ('Normalization', 'Stemming', 'Porter Stemmer (NLTK)')])
    with st.container(border=True):
        sec('🔄 Preprocessing Pipeline Steps')
        steps = ['Lowercasing', 'URL & Number Replacement', 'Punctuation Removal', 'Tokenization', 'Stopword Removal', 'Stemming']
        st.markdown(''.join(f'<span class="pill">✅ {s}</span>' for s in steps), unsafe_allow_html=True)
        sample = st.text_input('Try your own SMS', value=DEFAULT_SMS)
        toks = tokenize(sample)
        filtered = [w for w in toks if w not in A['stop_words'] and len(w) > 1]
        stemmed = [A['stemmer'].stem(w) for w in filtered]
        clean = ' '.join(stemmed)
        vec = A['tfidf'].transform([clean])
        lst = lambda xs: escape('[' + ', '.join(f'"{x}"' for x in xs) + ']')
        st.markdown(
            '<div class="stage-row">'
            f'<div class="stage"><h4>Stage 1: Raw SMS Input</h4><div class="inner mono">{escape(sample)}</div><span class="foot">{len(sample)} characters</span></div>'
            '<div class="arrow">→</div>'
            f'<div class="stage"><h4>Stage 2: Cleaned & Tokenized</h4><div class="inner mono">{lst(toks)}</div><span class="foot">{len(toks)} tokens</span></div>'
            '<div class="arrow">→</div>'
            f'<div class="stage"><h4>Stage 3: Stopwords Removed</h4><div class="inner mono">{lst(filtered)}</div><span class="foot">{len(filtered)} tokens</span></div>'
            '<div class="arrow">→</div>'
            f'<div class="stage"><h4>Stage 4: Stemmed & TF-IDF</h4><div class="inner mono">{lst(stemmed)}</div><span class="foot">Sparse vector • (1, {len(A["names"]):,}) • {vec.nnz} non-zero</span></div>'
            '</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    nz = vec.tocsr()
    order = np.argsort(nz.data)[::-1][:8]
    feat = [(A['names'][nz.indices[i]], nz.data[i], nz.indices[i]) for i in order]
    with c1, st.container(border=True):
        sec('🔍 Feature Vector Inspection & TF-IDF Weights')
        rows = [[f'<b>{t}</b>', f'{w:.4f}', f'{A["tfidf"].idf_[ix]:.2f}', int(A['doc_freq'][ix])] for t, w, ix in feat]
        if rows:
            table(['Token (stemmed)', 'TF-IDF Weight', 'IDF Score', 'Messages containing it'], rows)
        else:
            st.info('No words left after cleaning.')
    with c2, st.container(border=True):
        sec('📊 Top TF-IDF Weights in Sample SMS', 'Red words push toward spam, green toward ham (selected model).')
        if feat:
            w_model = A['weights'][model_name]
            fig = go.Figure(go.Bar(x=[w for _, w, _ in feat][::-1], y=[t for t, _, _ in feat][::-1], orientation='h',
                                   marker_color=[RED if w_model[ix] > 0 else GREEN for _, _, ix in feat][::-1],
                                   text=[f'{w:.2f}' for _, w, _ in feat][::-1], textposition='outside'))
            fig.update_xaxes(gridcolor='#eef0f3', range=[0, max(w for _, w, _ in feat) * 1.2])
            show(fig, 300)
    with st.container(border=True):
        sec('🧾 Vectorization Settings')
        st.markdown('**Vectorizer:** TF-IDF  •  **N-gram range:** (1, 2)  •  **Max features:** 5,000  •  '
                    '**Train/test split:** 80/20, stratified  •  **Duplicates removed before splitting:** yes')

# ================================================================ page 5
else:
    page_header('📥 Batch SMS Tester & Export')

    def read_messages(file):
        raw = file.getvalue().decode('utf-8', errors='ignore')
        if file.name.lower().endswith('.csv'):
            data = pd.read_csv(StringIO(raw))
            cols = {c.lower(): c for c in data.columns}
            tcol = next((cols[c] for c in ('text', 'message', 'sms', 'msg', 'v2') if c in cols), None)
            if tcol is None:
                obj = [c for c in data.columns if data[c].dtype == object]
                if not obj:
                    return None
                tcol = max(obj, key=lambda c: data[c].astype(str).str.len().mean())
            out = pd.DataFrame({'text': data[tcol].astype(str)})
            lcol = next((cols[c] for c in ('label', 'v1', 'category', 'class') if c in cols), None)
            if lcol:
                out['true'] = data[lcol].astype(str).str.lower().str.strip()
            return out
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        parts = [l.partition('\t') for l in lines]
        if lines and all(h.lower() in ('ham', 'spam') and s for h, s, _ in parts):
            return pd.DataFrame({'text': [t for _, _, t in parts], 'true': [h.lower() for h, _, _ in parts]})
        return pd.DataFrame({'text': lines})

    with st.container(border=True):
        sec('📂 Upload SMS Batch File (.csv, .txt)', 'A .csv needs a text column (text, message or sms). A .txt file takes one message per line.')
        up = st.file_uploader('Upload', type=['csv', 'txt'], label_visibility='collapsed')
        b1, b2, b3 = st.columns([1, 1, 1])
        if up is not None and st.session_state.get('batch_name') != up.name:
            parsed = read_messages(up)
            if parsed is None or parsed.empty:
                st.error('Could not find any messages in that file.')
            else:
                st.session_state.batch_in, st.session_state.batch_name = parsed.head(20000), up.name
                st.session_state.batch_ran = False
        if b1.button('Use sample messages'):
            st.session_state.batch_in, st.session_state.batch_name = pd.DataFrame({'text': SAMPLE_BATCH}), 'sample'
            st.session_state.batch_ran = True
        if b2.button('▶ Run Batch Prediction', type='primary'):
            st.session_state.batch_ran = True

    batch = st.session_state.get('batch_in')
    if batch is None or not st.session_state.get('batch_ran'):
        st.info('Upload a file (or use the sample messages), then press Run Batch Prediction.')
    else:
        clean = batch['text'].apply(lambda t: preprocess(t, A['stop_words'], A['stemmer']))
        X = A['tfidf'].transform(clean)
        prob = A['models'][model_name].predict_proba(X)[:, 1]
        is_spam = prob >= threshold
        conf = np.where(is_spam, prob, 1 - prob)
        C = X.multiply(A['weights'][model_name].reshape(1, -1)).tocsr()
        terms = []
        for i in range(X.shape[0]):
            s, e = C.indptr[i], C.indptr[i + 1]
            terms.append([t for t, _ in top_terms(C.indices[s:e], C.data[s:e], 3)])
        out = pd.DataFrame({'SMS Message': batch['text'], 'Predicted Class': np.where(is_spam, 'SPAM', 'HAM'),
                            'Confidence': np.round(conf * 100, 1), 'Flagged Trigger Terms': [', '.join(t) for t in terms]})
        with b3:
            st.download_button('⬇ Download Results CSV', out.to_csv(index=False).encode('utf-8'),
                               file_name='spam_predictions.csv', mime='text/csv')
        n, ns = len(out), int(is_spam.sum())
        sub_acc = ''
        if 'true' in batch and set(batch['true']) <= {'ham', 'spam'}:
            sub_acc = f'Accuracy vs file labels: {np.mean(out["Predicted Class"].str.lower() == batch["true"]):.1%}'
        cards([('Total Tested', f'{n:,}', 'Messages analyzed in batch'),
               ('Spam Flagged', f'<span style="color:#ea580c">{ns:,} ({ns / n:.1%})</span>', sub_acc or 'Detected unsolicited spam'),
               ('Ham Verified', f'<span style="color:#15803d">{n - ns:,} ({(n - ns) / n:.1%})</span>', 'Legitimate messages'),
               ('Avg Confidence', f'{conf.mean():.1%}', 'Model prediction certainty')])
        with st.container(border=True):
            sec(f'📋 Batch Prediction Results (Previewing first {min(n, 10)} of {n:,} rows)')
            rows = []
            for i, rr in out.head(10).iterrows():
                txt = rr['SMS Message'] if len(rr['SMS Message']) <= 80 else rr['SMS Message'][:77] + '...'
                rows.append([i + 1, escape(txt), chip(rr['Predicted Class']), f'{rr["Confidence"]:.1f}%', escape(rr['Flagged Trigger Terms'])])
            table(['#', 'SMS Message', 'Predicted Class', 'Confidence', 'Flagged Trigger Terms'], rows)
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            sec('📊 Batch Class Breakdown')
            fig = go.Figure()
            fig.add_bar(y=['Batch'], x=[n - ns], name=f'Ham: {n - ns:,}', orientation='h', marker_color=GREEN)
            fig.add_bar(y=['Batch'], x=[ns], name=f'Spam: {ns:,}', orientation='h', marker_color=RED)
            fig.update_layout(barmode='stack')
            fig.update_yaxes(visible=False)
            fig.update_xaxes(visible=False)
            show(fig, 200, legend_top=True)
        with c2, st.container(border=True):
            sec('🔑 Most Frequent Spam Triggers in Batch')
            cnt = Counter(t for flag, ts in zip(is_spam, terms) if flag for t in ts).most_common(8)
            if cnt:
                fig = go.Figure(go.Bar(x=[c for _, c in cnt][::-1], y=[t for t, _ in cnt][::-1], orientation='h', marker_color=RED))
                fig.update_xaxes(gridcolor='#eef0f3', title='Times flagged')
                show(fig, 260)
            else:
                st.caption('No spam was flagged in this batch.')
