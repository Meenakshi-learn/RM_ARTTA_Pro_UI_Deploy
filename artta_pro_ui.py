import streamlit as st
import requests
import feedparser
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')

st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🚀 Research Analyzer Portal</h1>", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/DSU-logo.png", width=150)
    st.title("📌 About ARTTA")
    st.markdown("""
    **Academic Research Trend Topic Analyzer**  
    👩‍💻 Developed by:  
    - R Ankitha  ENG24CSE0002
    - Meenakshi  ENG24CSE0013

    🧑‍🏫 Supervised by:  
    - Dr. Prabhakar M  

    🎓 M.Tech - Data Science  
    Dayananda Sagar University

    📂 [GitHub Repo](https://github.com/Meenakshi-learn)
    🌐 [Live App](https://streamlit.io/cloud)
    """)

# Hero header
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>📚 ARTTA v2</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #34495E;'>Academic Research Trend Topic Analyzer</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("ARTTA helps students and researchers explore trending academic topics by analyzing real-time abstracts from arXiv. Just type a topic and discover key terms, keyword importance, and topic clusters.")

# Input
query = st.text_input("🔍 Enter a research topic (e.g., 'deep learning', 'blockchain')")

# Core functions
def clean_corpus(abstracts):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned = []
    for abstract in abstracts:
        text = re.sub(r'[^a-zA-Z\s]', '', abstract.lower())
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        cleaned.append(" ".join(tokens))
    return cleaned

def fetch_arxiv(query, max_results=30):
    base_url = "http://export.arxiv.org/api/query?"
    full_url = f"{base_url}search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate"
    feed = feedparser.parse(requests.get(full_url).text)
    abstracts = [entry.summary.replace('\n', ' ').strip() for entry in feed.entries]
    return abstracts

def compute_tfidf(corpus, top_n=20):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(corpus)
    scores = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_scores[:top_n], columns=['Keyword', 'TF-IDF Score'])

def lda_topic_modeling(corpus, n_topics=5, n_words=8):
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append((f"Topic {idx+1}", top_words))
    return topics

def show_wordcloud(corpus):
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(corpus))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())

def plot_bar_chart(df_keywords):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='TF-IDF Score', y='Keyword', data=df_keywords)
    plt.title("Top Keywords by TF-IDF Score")
    plt.tight_layout()
    st.pyplot(plt.gcf())

# Main processing
if query and st.button("🚀 Analyze Now"):
    abstracts = fetch_arxiv(query)
    if not abstracts:
        st.warning("No abstracts found. Try another topic.")
    else:
        st.success(f"Fetched {len(abstracts)} abstracts from arXiv.")
        cleaned = clean_corpus(abstracts)

        tabs = st.tabs(["☁️ Word Cloud", "📈 Top Keywords", "🧠 Topic Clusters"])

        with tabs[0]:
            show_wordcloud(cleaned)

        with tabs[1]:
            tfidf_df = compute_tfidf(cleaned)
            st.dataframe(tfidf_df)
            plot_bar_chart(tfidf_df)

        with tabs[2]:
            lda_topics = lda_topic_modeling(cleaned)
            for i, words in lda_topics:
                st.markdown(f"**{i}:** {' | '.join(words)}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>© 2025 R Ankitha & Meenakshi | DSU | Research Methodology Project</p>", unsafe_allow_html=True)
