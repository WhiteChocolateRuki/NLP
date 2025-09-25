# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import string
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from sklearn.metrics import confusion_matrix, classification_report

# % load dataset
clinical_text_df = pd.read_csv("mtsamples.csv")

clinical_text_df = clinical_text_df[clinical_text_df["transcription"].notna()]
data_categories = clinical_text_df.groupby(clinical_text_df["medical_specialty"])

c = 1
for cat_name, cat_data in data_categories:
    print(f"Category_{c}: {cat_name}: {len(cat_data)}")
    c = c + 1

filtered_data_categories = data_categories.filter(lambda x: x.shape[0] > 50)
final_data_categories = filtered_data_categories.groupby(filtered_data_categories["medical_specialty"])

c = 1
for cat_name, cat_data in final_data_categories:
    print(f"Category_{c}: {cat_name}: {len(cat_data)}")
    c = c + 1

plt.figure()
sns.countplot(y = "medical_specialty", data = filtered_data_categories)
plt.show()

data = filtered_data_categories[["transcription", "medical_specialty"]]
data = data.drop(data[data["transcription"].isna()].index)


import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

#  NLP: text cleaning: low cha, lemmatization
def clean_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation)) # noktalama isaretlerini kaldir.
    text1 = "".join([w for w in text if not w.isdigit()])
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    
    text2 = text1.lower()
    text2 = replace_by_space_re.sub(" ", text2)
    
    return text2


def lemmatize_text(text):
    wordlist = []
    lemmatizer = WordNetLemmatizer()

    sentences = sent_tokenize(text)
    initial_sentence = sentences[0:1]
    final_sentence = sentences[len(sentences)-2:len(sentences)-1]

    for sentence in initial_sentence:
        words = word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))

    for sentence in final_sentence:
        words = word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))

    return " ".join(wordlist)

data["transcription"] = data["transcription"].apply(lemmatize_text)
data["transcription"] = data["transcription"].apply(clean_text)


# metin temsili: metinlerin -> sayisal olarak ifade edilmesi: tf idf
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
tfidf = vectorizer.fit_transform(data["transcription"].tolist())
feature_names = sorted(vectorizer.get_feature_names_out())

labels = data["medical_specialty"].tolist()
tsne = TSNE(n_components=2, init='random')
tsne_result = tsne.fit_transform(tfidf)

sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue = labels)



# %% PCA analizi dimension reduction, training: logistic regression
pca = PCA()
tfidf_reduced = pca.fit_transform(tfidf.toarray())
labels = data["medical_specialty"].tolist()
category_list = data.medical_specialty.unique()

X_train, X_test, y_train, y_test = train_test_split(tfidf_reduced, labels, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels = category_list)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")

print(classification_report(y_test, y_pred))



#save
import pickle

with open("NLP.pkl", "wb") as f:
    pickle.dump(lr, f)