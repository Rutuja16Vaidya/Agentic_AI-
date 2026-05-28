import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("sarojini_naidu_poetry_dataset.csv")

X = df["Themes"]
y = df["Tone"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))

sample_poem = ["""
You flaunt your beauty in the rose, your glory in the dawn,

Your sweetness in the nightingale, your whiteness in the swan.

You haunt my waking like a dream, my slumber like a moon,

Pervade me like a musky scent, possess me like a tune.

Yet, when I crave of you, my sweet, one tender moment's grace,

You cry, "I sit behind the veil, I cannot show my face."

Shall any foolish veil divide my longing from my bliss?

Shall any fragile curtain hide your beauty from my kiss?

What war is this of Thee and Me? Give o'er the wanton strife,

You are the heart within my heart, the life within my life.
"""]

prediction = model.predict(sample_poem)

print("Predicted Tone:", prediction[0])