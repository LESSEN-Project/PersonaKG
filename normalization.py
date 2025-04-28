import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer, util

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

canonical_attributes = [
    "cycling", "photography", "cooking", "hiking", "running", "painting", "traveling"
]

vectorizer = TfidfVectorizer().fit(canonical_attributes)
canonical_embeddings = vectorizer.transform(canonical_attributes)

model = SentenceTransformer('all-MiniLM-L6-v2')
canonical_embeddings_st = model.encode(canonical_attributes, convert_to_tensor=True)

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)

def normalize_attribute(raw_attr):
    preprocessed = preprocess(raw_attr)

    synsets = wn.synsets(preprocessed)
    if synsets:
        lemma = synsets[0].lemmas()[0].name().replace('_', ' ')
        print(f"[WordNet Match] '{raw_attr}' normalized to '{lemma}'")
        return lemma

    attr_embedding = vectorizer.transform([preprocessed])
    cos_scores = cosine_similarity(attr_embedding, canonical_embeddings)
    best_idx = np.argmax(cos_scores)
    best_match = canonical_attributes[best_idx]
    best_score = cos_scores[0, best_idx]

    if best_score > 0.3:
        print(f"[TF-IDF Match] '{raw_attr}' matched to '{best_match}' (score: {best_score:.2f})")
        return best_match

    # Sentence Transformer fallback
    attr_embedding_st = model.encode(preprocessed, convert_to_tensor=True)
    cos_scores_st = util.cos_sim(attr_embedding_st, canonical_embeddings_st)
    best_idx_st = cos_scores_st.argmax()
    best_match_st = canonical_attributes[best_idx_st]
    best_score_st = cos_scores_st[0, best_idx_st].item()

    if best_score_st > 0.75:
        print(f"[Sentence Transformer Match] '{raw_attr}' matched to '{best_match_st}' (score: {best_score_st:.2f})")
        return best_match_st

    print(f"[No Good Match] '{raw_attr}' kept as '{preprocessed}'")
    return preprocessed

# Example raw attributes from LLM
raw_attributes = [
    "likes to ride bikes",
    "bike riding",
    "loves mountain biking",
    "enjoys taking photos",
    "cooks gourmet meals",
    "likes going to the gym"
]

# Process and normalize
normalized_attributes = [normalize_attribute(attr) for attr in raw_attributes]

print("\nFinal Normalized Attributes:")
print(normalized_attributes)
