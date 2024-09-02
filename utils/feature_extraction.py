from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np

distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def extract_features(texts, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(texts)
    elif method == 'bow':
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(texts)
    elif method == 'bert':
        def vectorize_text(text):
            inputs = distilbert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = distilbert_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        
        feature_matrix = np.array([vectorize_text(text) for text in texts])
        return feature_matrix
    
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")
