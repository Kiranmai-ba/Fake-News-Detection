import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

try:
    # Sample data
    data = pd.DataFrame({
        'text': ['This is real news.', 'Fake news alert!', 'Another real article.', 'Fake news here.'],
        'label': ['Real', 'Fake', 'Real', 'Fake']
    })

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(model, 'lr_model.joblib')

    print("Model and vectorizer saved successfully! 🎉")
except Exception as e:
    print(f"Error: {e}")
