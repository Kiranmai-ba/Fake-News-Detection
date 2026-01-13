 from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.linear_model import LogisticRegression
   import joblib
   
   # Train
   vectorizer = TfidfVectorizer()
   X_train = vectorizer.fit_transform(train_text)
   model = LogisticRegression()
   model.fit(X_train, train_labels)
   
   # Save
   joblib.dump(vectorizer, 'vectorizer.jb')
   joblib.dump(model, 'lr_model.jb')
