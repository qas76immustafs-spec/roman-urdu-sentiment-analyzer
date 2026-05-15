"""
Roman Urdu Sentiment Analyzer — Streamlit Web App
Classifies Roman Urdu social media text as Positive, Negative, or Neutral
with confidence scores and real-time prediction.

Auto-trains model on first run if model file is not found.
"""

import streamlit as st
import pickle
import json
import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Roman Urdu Sentiment Analyzer",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREPROCESSING
# ============================================================

ROMAN_URDU_STOPWORDS = {
    'hai', 'hain', 'ho', 'hota', 'hoti', 'hua', 'hui', 'hun',
    'ka', 'ke', 'ki', 'ko', 'kya', 'se', 'mein', 'pe', 'par',
    'ne', 'aur', 'ya', 'bhi', 'toh', 'to', 'ab', 'is', 'us',
    'ek', 'yeh', 'woh', 'ye', 'wo', 'jo', 'jab', 'tab',
    'apna', 'apni', 'apne', 'mera', 'meri', 'mere',
    'tera', 'teri', 'tere', 'uska', 'uski', 'uske',
    'hum', 'tum', 'woh', 'main', 'tu',
    'the', 'tha', 'thi', 'raha', 'rahi', 'rahe',
    'kar', 'karna', 'karni', 'karte', 'karti',
    'ja', 'jana', 'jata', 'jati', 'jao',
    'le', 'lena', 'leta', 'leti', 'lo',
    'de', 'dena', 'deta', 'deti', 'do',
    'aa', 'aana', 'aata', 'aati', 'aao',
    'nahi', 'na', 'mat', 'nah',
    'agar', 'lekin', 'magar', 'phir', 'warna',
    'abhi', 'kabhi', 'koi', 'kuch', 'sab', 'sirf',
    'per', 'wala', 'wali', 'wale',
}

NORMALIZATION_MAP = {
    'bohat': 'bohot', 'bhot': 'bohot', 'buht': 'bohot', 'bot': 'bohot',
    'acha': 'acha', 'achha': 'acha', 'accha': 'acha', 'aacha': 'acha',
    'achi': 'achi', 'achhi': 'achi', 'acchi': 'achi',
    'bura': 'bura', 'bora': 'bura', 'burra': 'bura',
    'buri': 'buri', 'burri': 'buri',
    'kharab': 'kharab', 'kharb': 'kharab', 'khrab': 'kharab',
    'pasand': 'pasand', 'psand': 'pasand', 'pasnd': 'pasand',
    'zabardast': 'zabardast', 'zbrdst': 'zabardast', 'zabrdast': 'zabardast',
    'shandar': 'shandar', 'shandaar': 'shandar', 'shandr': 'shandar',
    'ghatiya': 'ghatiya', 'ghtiya': 'ghatiya', 'ghatya': 'ghatiya',
    'khush': 'khush', 'khsh': 'khush', 'kush': 'khush',
    'dukh': 'dukh', 'dukhi': 'dukhi', 'dkh': 'dukh',
    'lajawab': 'lajawab', 'lajwab': 'lajawab', 'ljwab': 'lajawab',
    'behtareen': 'behtareen', 'behtreen': 'behtareen', 'behtereen': 'behtareen',
    'bekaar': 'bekaar', 'bekar': 'bekaar', 'bekr': 'bekaar',
    'kamaal': 'kamaal', 'kamal': 'kamaal', 'kmal': 'kamaal',
    'mazaa': 'maza', 'mza': 'maza', 'maaza': 'maza',
    'lazeez': 'lazeez', 'laziz': 'lazeez', 'lzeez': 'lazeez',
    'pareshan': 'pareshan', 'preshan': 'pareshan', 'prshan': 'pareshan',
}


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    tokens = text.split()
    tokens = [NORMALIZATION_MAP.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in ROMAN_URDU_STOPWORDS and len(t) > 1]
    return ' '.join(tokens)


# ============================================================
# DATASET
# ============================================================

def get_dataset():
    """Return the labeled Roman Urdu sentiment dataset."""

    positive_samples = [
        "yeh bohot acha product hai mujhe pasand aya",
        "main bohot khush hun aaj ka din bohot acha raha",
        "kya zabardast match tha bilkul kamaal",
        "bohot maza aya aaj party mein",
        "yeh movie bohot achi thi must watch",
        "Allah ka shukar hai sab theek hai",
        "best restaurant hai yeh try karo zaroor",
        "bohot pyara hai yeh bachcha mashallah",
        "result acha aya alhamdulillah",
        "yeh game bohot fun hai khelo zaroor",
        "kya baat hai bhai kamaal kar diya",
        "bohot talented hai yeh banda respect",
        "mujhe yeh jagah bohot pasand hai",
        "aaj ka khana bohot tasty tha",
        "bohot helpful log hain yahan ke",
        "exam acha gaya mera inshallah pass",
        "yeh gaana bohot hit hai sunna chahiye",
        "dil khush hogaya yeh dekh ke",
        "amazing quality hai is product ki",
        "bohot supportive hai yeh teacher",
        "kya shandar performance thi waah",
        "acha laga sun ke bohot khushi hui",
        "yeh phone bohot fast hai love it",
        "great work bhai keep it up",
        "mujhe aaj promotion mili alhamdulillah",
        "bohot pyari jagah hai visit karo",
        "yeh course bohot informative tha",
        "best decision tha yeh lena",
        "bohot acchi service hai recommend karta hun",
        "mashallah bohot talented lagte ho",
        "is dukan ka khana bohot fresh hai",
        "yeh laptop bohot smooth chalta hai",
        "kya zabardast goal tha shandar",
        "mujhe bohot maza aya concert mein",
        "bohot acha insaan hai dil ka saaf",
        "yeh deal bohot sasti hai grab karo",
        "main bohot motivated feel kar raha hun",
        "camera quality outstanding hai is phone ki",
        "bohot pyara design hai love it",
        "bilkul perfect hai yeh mere liye",
        "kya lajawab taste hai yeh",
        "family ke saath acha waqt guzara",
        "yeh app bohot useful hai download karo",
        "sab log bohot cooperative the",
        "mujhe yeh rang bohot pasand hai",
        "interview acha gaya positive vibes",
        "kya behtareen quality hai is brand ki",
        "bohot lucky feel kar raha hun aaj",
        "yeh jagah bohot peaceful hai",
        "zabardast batting ki usne kya player hai",
        "bohat acha experience tha overall",
        "khana bohat lazeez tha maza aa gaya",
        "is hotel ki service best hai",
        "aaj bohat productive din tha",
        "yeh dress bohat pyari lag rahi hai",
        "staff bohat friendly tha",
        "delivery bohat fast aayi impressed",
        "kya kamaal ki painting hai wow",
        "bohat achi book hai must read",
        "aaj mood bohat acha hai",
        "yeh teacher bohat well explain karte hain",
        "meri team ne bohat acha kiya proud",
        "yeh perfume ki smell bohat achi hai",
        "hospital mein treatment bohat acha mila",
        "bohat acha lagta hai yeh sunke",
        "party mein bohat enjoy kiya",
        "yeh scheme bohat faydemand hai",
        "salary increase hogayi bohot khush hun",
        "yeh shoes bohot comfortable hain",
        "exam clear hogaya first attempt mein",
        "bohot mast weather hai aaj",
        "is university ka environment bohot acha hai",
        "coding seekh raha hun bohot maza aa raha",
        "mera best friend bohot loyal hai",
        "yeh series bohot interesting hai binge watch karo",
        "prize jeet liya bohot proud feel",
        "yeh car bohot smooth chalti hai",
        "biryani bohot lajawab thi aaj",
        "sunset bohot beautiful tha aaj",
        "gym jaana start kiya feeling great",
        "project complete hogaya on time relief",
        "teacher ne appreciate kiya feel good",
        "naya phone liya bohot khush hun",
        "dost se milke bohot acha laga",
        "marks ache aaye family khush hai",
        "morning walk pe gaya fresh feel",
        "yeh chai bohot zabardast hai",
        "flight on time aayi smooth journey",
        "internship mil gayi excited hun",
        "aaj cricket khela bohot maza aya",
        "yeh course free hai aur bohot acha",
        "sab theek hogaya tension khatam",
        "bohot pyari smile hai tumhari",
        "naya ghar liya bohot khushi hai",
        "health improve horahi hai daily exercise se",
        "yeh painting bohot creative hai",
        "apna business start kiya feeling proud",
        "aaj bohot positive energy mil rahi hai",
        "yeh vlog bohot entertaining tha",
        "new year new goals excited hun",
    ]

    negative_samples = [
        "yeh product bekaar hai waste of money",
        "bohot bura experience tha never again",
        "service bohot ghatiya hai complain karunga",
        "kya bakwas movie thi time waste",
        "bohot disappoint hua main yeh dekh ke",
        "paise barbaad hogaye is pe",
        "yeh dukan wala fraud hai dhoka deta hai",
        "bohot boring tha yeh program",
        "quality bohot kharab hai mat lena",
        "bohot ganda khana tha yahan ka",
        "delivery mein bohot delay hua fed up",
        "customer service worst hai bilkul",
        "yeh phone hang hota hai bara",
        "bohot rude staff hai is jagah ka",
        "result kharab aya bohot tension hai",
        "yeh cheez toot gayi ek din mein",
        "pura din barbaad hogaya is ki wajah se",
        "bohot mehnga hai aur quality zero",
        "koi madad nahi karta yahan",
        "bohot ghatiya management hai",
        "exam mein fail hogaya bohot bura laga",
        "yeh app crash hota hai baar baar",
        "refund nahi milta fraud company hai",
        "traffic mein phanse hue hain bohot irritation",
        "bohot unfair decision tha yeh",
        "health kharab hai bohot pareshan hun",
        "is jagah jaana galti thi",
        "yeh banda bohot fake hai trust mat karo",
        "packaging toot ke aayi bohot careless",
        "internet speed bohot slow hai worst isp",
        "yeh scam hai paisa mat lagao",
        "team haar gayi bohot dukh hua",
        "puri raat neend nahi aayi tension ki wajah se",
        "bohot ganda mausam hai bahar mat jao",
        "is company mein kaam karna mushkil hai",
        "bohot mehnga bill aaya bijli ka",
        "yeh dawai kaam nahi kar rahi",
        "transport bohot kharab hai yahan",
        "bhai ne dhoka diya vishwas toot gaya",
        "salary delay ho rahi hai bohot frustration",
        "exam bohot mushkil tha kuch samajh nahi aya",
        "yeh brand overrated hai bilkul",
        "bohot thak gaya hun aaj kaam karke",
        "rent bohot zyada hai afford nahi hota",
        "yeh plan change kiya unhone bina bataye",
        "fever aur sir dard bohot zyada hai",
        "parking nahi milti yahan kabhi",
        "bohot ghatiya attitude hai is ki",
        "paise doob gaye investment mein",
        "yeh course scam hai kuch nahi sikhate",
        "bohat buri halat hai sadak ki",
        "yeh network hamesha down rehta hai",
        "complaint ki lekin koi response nahi",
        "bohat mushkil hai yeh sab handle karna",
        "manager bohat rude hai kaam nahi kar sakte",
        "ac kharab hogaya garmi mein mar rahe hain",
        "paani ka masla bohat serious hai yahan",
        "load shedding se tang aa gaye",
        "mobile chori hogaya bohat nuqsaan hua",
        "doctor ne galat medicine di worse hogaya",
        "assignment reject hogaya dobara karna padega",
        "dost ne paise wapas nahi kiye",
        "interview mein reject hogaya hopeless feel",
        "yeh restaurant overpriced hai bohot",
        "presentation mein bohot nervous tha flop hogaya",
        "laptop kharab hogaya data chala gaya",
        "barish mein ghar mein paani aa gaya",
        "bus late aayi class miss hogayi",
        "wifi bohot slow hai kuch load nahi hota",
        "pet kharab hai khana hazam nahi hota",
        "marks bohot kam aaye regret hai",
        "fine lag gaya traffic police ne",
        "project mein bugs bohot hain fix nahi hote",
        "bohot akela feel ho raha hai",
        "ghar walon se jhagda hogaya",
        "yeh sim ki calls drop hoti hain",
        "mood bohot kharab hai aaj",
        "headphones toot gaye ek hafte mein",
        "parhai mein dil nahi lagta",
        "loan ka installment miss hogaya",
        "yeh warranty fake nikli scam",
        "uber ne zyada charge kiya",
        "sardi mein heater kaam nahi karta",
        "cricket team ki bowling bohot weak hai",
        "yeh game pay to win hai unfair",
        "friend zone hogaya bohot sad",
        "is area mein bijli bohot jati hai",
        "coding mein error aa raha fix nahi ho raha",
        "yeh movie ki story predictable thi boring",
        "subah se sir dard hai medicine bhi nahi",
        "bohot time waste hua is meeting mein",
        "ghar ki safai karne ka mann nahi",
        "rent increase hogaya dobara shift hona padega",
        "aaj bura din tha sab galat hua",
        "scholarship nahi mili bohot mehnat ki thi",
        "freelancing mein client ne payment nahi ki",
        "yeh scheme pagal banati hai logon ko",
        "anxiety bohot zyada hai kal exam hai",
        "order cancel hogaya bina wajah",
        "yeh teacher partial hai unfair marking",
    ]

    neutral_samples = [
        "kal university jana hai class hai",
        "aaj mausam thanda hai bahar",
        "meeting 3 baje hai office mein",
        "market se saman lana hai ghar ke liye",
        "parso exam hai preparation karni hai",
        "aaj monday hai kaam pe jana hai",
        "yeh kitne ki hai puchna hai",
        "kal doctor ke paas jana hai checkup",
        "bus abhi aayi hai chalo chalte hain",
        "yeh form fill karna hai deadline hai",
        "ghar pe khana banana hai aaj",
        "library mein study karni hai",
        "assignment submit karna hai friday tak",
        "office se 6 baje niklunga",
        "kal flight hai islamabad ki",
        "yeh link share karo group mein",
        "parking yahan available hai",
        "registration date 15 may hai",
        "yeh road construction chal rahi hai",
        "school ke result aaj aaye hain",
        "petrol ki qeemat badli hai",
        "yeh naya store khula hai area mein",
        "kal sunday hai chutti hai",
        "pharmacy se medicine leni hai",
        "phone charge pe lagao battery low hai",
        "rent due date agle hafte hai",
        "cv update karna hai job ke liye",
        "aaj raat ko match hai 8 baje",
        "uber book karo jaana hai",
        "yeh file download ho rahi hai",
        "semester finals june mein hain",
        "naya update aaya hai phone mein",
        "traffic signal pe ruko abhi red hai",
        "passport renew karwana hai",
        "gym ka timing kya hai",
        "aaj grocery shopping karni hai",
        "project ki deadline next week hai",
        "interview kal subah 10 baje hai",
        "bill pay karna hai last date hai",
        "yeh formula yaad karna hai exam ke liye",
        "documents ready rakhna office ke liye",
        "metro station yahan se paas hai",
        "wifi ka password kya hai",
        "class cancel hogayi aaj teacher nahi aye",
        "courier aaya hai neeche jaake lelo",
        "sim ki validity check karo",
        "meeting reschedule hogayi kal pe",
        "paper pattern change hua hai is baar",
        "canteen mein chai available hai",
        "library 9 se 5 tak khuli hai",
        "yeh software install karna hai laptop mein",
        "kal se ramzan shuru hai",
        "election ki date announce hogayi",
        "admission last date 30 june hai",
        "parking charges 50 rupee hain",
        "ac ka remote kahan hai dhundo",
        "kal convocation hai university mein",
        "naya semester august mein shuru hoga",
        "lab mein experiment karna hai aaj",
        "presentation kal deni hai slides ready hain",
        "hostel mein mess ka khana milta hai",
        "sports day next month hai",
        "marksheet collect karni hai office se",
        "yeh software ka trial version hai",
        "attendance kam hai warning aayi",
        "campus mein wifi available hai",
        "kal holiday hai eid ki chutti",
        "practical exam next week hai",
        "thesis submission date december hai",
        "canteen mein paratha milta hai",
        "shuttle service 8 baje hai",
        "mid term exams october mein hain",
        "society mein event plan ho raha hai",
        "admission form online bharna hai",
        "transport fee 5000 hai semester ki",
        "orientation week mein session hoga",
        "id card banwana hai naya",
        "hostel room allot hua hai",
        "semester break july mein hai",
        "certificate verify karwana hai HEC se",
        "bank account kholna hai scholarship ke liye",
        "teacher ne assignment de diya hai",
        "lecture notes share karo group mein",
        "yeh book library mein available hai",
        "campus interview next friday hai",
        "summer classes mein enroll karna hai",
        "fyp topic select karna hai",
        "lab report submit karni hai kal",
        "grading policy change hui hai is semester",
        "student portal pe result upload hua",
        "society ka budget approve hua",
        "hostel mess timing change hui hai",
        "library card renew karwana hai",
        "scholarship form ki deadline parso hai",
        "exam hall mein seating arrangement lagi hai",
        "new timetable issue hua hai",
        "campus mein construction chal rahi hai",
        "convocation rehearsal kal hai",
        "clearance form submit karna hai",
        "alumni meetup next sunday hai",
    ]

    data = []
    for text in positive_samples:
        data.append({"text": text, "sentiment": "Positive"})
    for text in negative_samples:
        data.append({"text": text, "sentiment": "Negative"})
    for text in neutral_samples:
        data.append({"text": text, "sentiment": "Neutral"})

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ============================================================
# MODEL TRAINING (runs once, cached)
# ============================================================

@st.cache_resource
def load_or_train_model():
    """Load model from disk if available, otherwise train from scratch."""

    model_path = "model/sentiment_model.pkl"
    metadata_path = "model/metadata.json"

    # Try loading pre-trained model
    if os.path.exists(model_path) and os.path.exists(metadata_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return model, metadata
        except Exception:
            pass

    # Train from scratch
    df = get_dataset()
    df['cleaned'] = df['text'].apply(preprocess_text)
    df = df[df['cleaned'].str.strip() != ''].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['sentiment'],
        test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    # Naive Bayes
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2, max_df=0.95, sublinear_tf=True)),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    nb_pipeline.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb_pipeline.predict(X_test))
    nb_cv = cross_val_score(nb_pipeline, df['cleaned'], df['sentiment'], cv=5, scoring='accuracy')

    # Logistic Regression
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2, max_df=0.95, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_pipeline.predict(X_test))
    lr_cv = cross_val_score(lr_pipeline, df['cleaned'], df['sentiment'], cv=5, scoring='accuracy')

    # Select best
    if lr_cv.mean() >= nb_cv.mean():
        best_model = lr_pipeline
        best_name = "Logistic Regression"
    else:
        best_model = nb_pipeline
        best_name = "Naive Bayes"

    metadata = {
        "best_model": best_name,
        "test_accuracy": round(lr_acc if best_name == "Logistic Regression" else nb_acc, 4),
        "cv_accuracy": round(lr_cv.mean() if best_name == "Logistic Regression" else nb_cv.mean(), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": len(df),
        "classes": list(df['sentiment'].unique()),
        "features": "TF-IDF (unigrams + bigrams)",
        "nb_test_acc": round(nb_acc, 4),
        "nb_cv_acc": round(nb_cv.mean(), 4),
        "lr_test_acc": round(lr_acc, 4),
        "lr_cv_acc": round(lr_cv.mean(), 4),
    }

    # Save for next time
    os.makedirs("model", exist_ok=True)
    try:
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass

    return best_model, metadata


model, metadata = load_or_train_model()


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .stTextArea textarea {
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.markdown(f"**Model:** {metadata['best_model']}")
    st.markdown(f"**Test Accuracy:** {metadata['test_accuracy']*100:.1f}%")
    st.markdown(f"**CV Accuracy:** {metadata['cv_accuracy']*100:.1f}%")
    st.markdown(f"**Training Samples:** {metadata['total_samples']}")
    st.markdown(f"**Features:** {metadata['features']}")

    st.markdown("---")
    st.markdown("### 🔬 Model Comparison")
    comparison_df = pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression"],
        "Test Acc": [f"{metadata['nb_test_acc']*100:.1f}%", f"{metadata['lr_test_acc']*100:.1f}%"],
        "CV Acc": [f"{metadata['nb_cv_acc']*100:.1f}%", f"{metadata['lr_cv_acc']*100:.1f}%"],
    })
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool classifies **Roman Urdu** (Urdu written in Latin script)
    social media text into three sentiment categories.

    **Pipeline:**
    - Custom tokenizer
    - Stop-word removal (80+ Roman Urdu words)
    - Spelling normalization
    - TF-IDF + N-gram features
    - ML classification

    Built as an NLP research project for low-resource language sentiment analysis.
    """)


# ============================================================
# MAIN APP
# ============================================================
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🔍 Roman Urdu Sentiment Analyzer")
st.markdown("*Analyze sentiment in Roman Urdu social media text*")
st.markdown("</div>", unsafe_allow_html=True)

# Input
st.markdown("### Enter Roman Urdu Text")
user_input = st.text_area(
    label="Type or paste Roman Urdu text below:",
    placeholder="e.g., yeh bohot acha product hai mujhe pasand aya",
    height=120,
    label_visibility="collapsed"
)

# Example texts
st.markdown("**Try these examples:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😊 Positive", use_container_width=True):
        st.session_state['example_text'] = "bohot maza aya aaj party mein kya zabardast time tha"
with col2:
    if st.button("😠 Negative", use_container_width=True):
        st.session_state['example_text'] = "yeh product bekaar hai bohot ghatiya quality waste of money"
with col3:
    if st.button("😐 Neutral", use_container_width=True):
        st.session_state['example_text'] = "kal university jana hai class hai subah 9 baje"

if 'example_text' in st.session_state:
    user_input = st.session_state['example_text']
    del st.session_state['example_text']
    st.rerun()

# Analyze button
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if user_input.strip():
        cleaned = preprocess_text(user_input)

        if cleaned.strip():
            prediction = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]
            classes = model.classes_

            emoji_map = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}
            color_map = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}

            st.markdown(f"""
            <div class='sentiment-{color_map[prediction]}'>
                <h2 style='margin:0; font-size:28px;'>{emoji_map[prediction]} {prediction}</h2>
                <p style='margin:5px 0 0 0; opacity:0.8;'>
                    Confidence: {max(probabilities)*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Confidence Scores")
            for cls, prob in sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True):
                emoji = emoji_map.get(cls, "")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(float(prob))
                with col_b:
                    st.markdown(f"**{emoji} {cls}**: {prob*100:.1f}%")

            with st.expander("🔧 Preprocessing Details"):
                st.markdown(f"**Original text:** {user_input}")
                st.markdown(f"**Cleaned text:** {cleaned}")
                original_tokens = user_input.lower().split()
                cleaned_tokens = cleaned.split()
                removed = set(original_tokens) - set(cleaned_tokens)
                if removed:
                    st.markdown(f"**Removed tokens:** {', '.join(removed)}")
        else:
            st.warning("After preprocessing, no meaningful tokens remained. Try a longer or more descriptive text.")
    else:
        st.warning("Please enter some text to analyze.")


# ============================================================
# BATCH ANALYSIS
# ============================================================
st.markdown("---")
st.markdown("### 📋 Batch Analysis")
st.markdown("Enter multiple texts (one per line) to analyze in bulk:")

batch_input = st.text_area(
    label="Batch input",
    placeholder="yeh bohot acha hai\nyeh bekaar hai\nkal meeting hai",
    height=100,
    key="batch",
    label_visibility="collapsed"
)

if st.button("📊 Analyze Batch", use_container_width=True):
    if batch_input.strip():
        lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]

        results = []
        for line in lines:
            cleaned = preprocess_text(line)
            if cleaned.strip():
                pred = model.predict([cleaned])[0]
                probs = model.predict_proba([cleaned])[0]
                confidence = max(probs) * 100
                emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}
                results.append({
                    "Text": line[:60] + ("..." if len(line) > 60 else ""),
                    "Sentiment": f"{emoji.get(pred, '')} {pred}",
                    "Confidence": f"{confidence:.1f}%"
                })

        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, hide_index=True, use_container_width=True)

            sentiments = [r["Sentiment"].split()[-1] for r in results]
            counts = pd.Series(sentiments).value_counts()
            st.markdown(f"**Summary:** {len(results)} texts analyzed — "
                       f"Positive: {counts.get('Positive', 0)}, "
                       f"Negative: {counts.get('Negative', 0)}, "
                       f"Neutral: {counts.get('Neutral', 0)}")
    else:
        st.warning("Please enter at least one text line.")


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; opacity:0.5; font-size:12px;'>"
    "Roman Urdu Sentiment Analyzer — NLP Research Project | Built with Scikit-learn & Streamlit"
    "</p>",
    unsafe_allow_html=True
)
