import streamlit as st
import pandas as pd
import nltk
import gensim
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, silhouette_score

nltk.download('stopwords')
nltk.download('punkt')


st.title('Topic Modeling Skripsi Portal Tugas Akhir Universitas Trunojoyo Madura')
load_data, preprocessing, feature_extraction, clasification = st.tabs(
    ['Load Data', 'Preprocessing', 'Feature Extraction', 'Clasification'])


with load_data:
    uploaded_file = st.file_uploader('Upload file CSV')
    if uploaded_file is not None:
        # df = pd.read_csv("https://raw.githubusercontent.com/syayidalaziz10/cobu/main/data/ptaa.csv")
        df = pd.read_csv(uploaded_file)
        st.header('Dataset')
        st.dataframe(df)
    else:
        st.warning('Mohon upload file CSV terlebih dahulu.')


with preprocessing:
    if 'df' in locals():
        df = df.astype(str)
        abstrak_column = df['abstrak'].apply(lambda x: x.lower())

        def process_tokenize(text):
            text = text.split()
            return text
        
        def process_punctuation(texts):
            cleaned_text = [re.sub(r'[.,()=%:&-]', '', text) for text in texts]
            cleaned_text = [re.sub(r'\d+', '', text) for text in cleaned_text]
            return cleaned_text
        
        def process_stopword(tokens):
            stop_words = set(stopwords.words('indonesian'))
            custom_stop_words = ['masingmasing','tiaptiap','satusatunya', 'intinya', 'seiring']
            stop_words.update(custom_stop_words)
            filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
            return filtered_tokens

        def process_stemming(tokens):
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            return " " .join(stemmed_tokens)

        tokenize_abstrak = abstrak_column.apply(process_tokenize)
        st.write('1. Tokenisasi:', tokenize_abstrak)

        punctuation_abstrak = tokenize_abstrak.apply(process_punctuation)
        st.write('2. Punctuation:', punctuation_abstrak)

        stopword_abstrak = punctuation_abstrak.apply(process_stopword)
        st.write('3. Stopword:', stopword_abstrak)

        steeming_abstrak = stopword_abstrak.apply(process_stemming)
        st.write('4. Steeming:', steeming_abstrak)

    else:
        st.warning('Mohon upload file CSV terlebih dahulu.')

with feature_extraction:
    if 'df' in locals():

        st.write('1. Feature Extraction (Local Weighting)')
        countvectorizer = CountVectorizer(analyzer= 'word')
        term_matrix = countvectorizer.fit_transform(steeming_abstrak)
        count_tokens = countvectorizer.get_feature_names_out()
        df_countvect = pd.DataFrame(data = term_matrix.toarray(),columns = count_tokens)
        df_countvect = pd.concat([df_countvect, df['label-topic']], axis=1)
        st.write('Term Frequency', df_countvect)

        countvectorizer = CountVectorizer(analyzer='word')
        log_matrix = countvectorizer.fit_transform(steeming_abstrak)
        count_tokens = countvectorizer.get_feature_names_out()
        count_log_matrix = np.log1p(log_matrix)
        df_log_countvect = pd.DataFrame(data=count_log_matrix.toarray(), columns=count_tokens)
        df_log_countvect = pd.concat([df_log_countvect, df['label-topic']], axis=1)
        st.write('Log Frequency', df_log_countvect)

        countvectorizer = CountVectorizer(analyzer='word', binary=True)
        binary_matrix = countvectorizer.fit_transform(steeming_abstrak)
        df_binary = pd.DataFrame(binary_matrix.toarray(), columns=countvectorizer.get_feature_names_out())
        df_binary = pd.concat([df_binary, df['label-topic']], axis=1)
        st.write('Binary', df_binary)
        
        st.write('2. Feature Extraction (Global Weighting)')
        tfidfvectorizer = TfidfVectorizer(analyzer='word')
        tfidf = tfidfvectorizer.fit_transform(steeming_abstrak)
        tfidf_tokens = tfidfvectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(data = tfidf.toarray(), columns = tfidf_tokens)
        tfidf_label = pd.concat([tfidf_df, df['label-topic']], axis=1)
        st.write('TF-IDF Vectorizer',tfidf_label)



        st.write('3. LDA (Latent Dirchlect Allocation)')
        document = stopword_abstrak
        dictionary = corpora.Dictionary(document)
        corpus = [dictionary.doc2bow(tokens) for tokens in document]

        num_topics = 2
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=dictionary,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    passes=10,
                                                    per_word_topics=True)
        
        st.write('LDA Build Model Topics', lda_model.print_topics())


        topic_proportions_list = []

        for index, doc in enumerate(corpus):
            topic_prop = lda_model.get_document_topics(doc)
            proportions = {f'Topic {i+1}': 0.0 for i in range(num_topics)}

            for topic in topic_prop:
                proportions[f'Topic {topic[0] + 1}'] = topic[1]
            topic_proportions_list.append(proportions)

        topic_proportions = pd.DataFrame(topic_proportions_list)
        topic_proportions_df = pd.DataFrame(topic_proportions_list)
        topic_proportions_df.insert(0, 'judul', df['judul'])
        topic_proportions_df.insert(1, 'abstrak', df['abstrak'])
        topic_proportions_df.insert(4, 'label-topic', df['label-topic'])
        st.write('LDA Document Topic Proportion',topic_proportions_df)

        st.write('4. Clustering Feature Extraction')

        def kmeans_clustering(data):
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            num_clusters = 3
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            return clusters

        topic_proportions_clusters = kmeans_clustering(topic_proportions.values)
        tfidf_clusters = kmeans_clustering(tfidf_df.values)

        topic_proportions_df = pd.DataFrame({'cluster':topic_proportions_clusters})
        topic_proportions_df.insert(0, 'judul', df['judul'])
        topic_proportions_df.insert(1, 'abstrak', df['abstrak'])
        topic_proportions_df.insert(3, 'label-topic', df['label-topic'])
        
        
        tfidf_clusters_df = pd.DataFrame({'cluster':tfidf_clusters})
        tfidf_clusters_df.insert(0, 'judul', df['judul'])
        tfidf_clusters_df.insert(1, 'abstrak', df['abstrak'])
        tfidf_clusters_df.insert(3, 'label-topic', df['label-topic'])

        # Menghitung Silhouette Score
        silhouette_topic_proportions = silhouette_score(topic_proportions.values, topic_proportions_clusters)
        silhouette_tfidf = silhouette_score(tfidf_df.values, tfidf_clusters)

        # Menampilkan hasil clustering dan Silhouette Score
        st.write('Clustering Results (TF-IDF)')
        st.write(tfidf_clusters_df)
        st.write('Silhouette Score (TF-IDF):', silhouette_tfidf)
        st.write('Clustering Results (Topic Proportions)')
        st.write(topic_proportions_df)
        st.write('Silhouette Score (Topic Proportions):', silhouette_topic_proportions)


    else:
        st.warning('Mohon upload file CSV terlebih dahulu.')

with clasification:
    if 'df' in locals():
        st.write('1. Klasifikasi kNN')

        X = tfidf_df
        y = df['label-topic']
        y = y.replace({'komputasi': 1, 'rpl': 0})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        k = 3
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues")
        st.write('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)

        # Evaluasi metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)     

        # Tampilkan tabel evaluasi
        st.write('Tabel Evaluasi:')
        eval_metrics = pd.DataFrame({
            'Akurasi': [accuracy],
            'Presisi': [precision],
            'Recall': [recall],
            'F1-Score': [f1]
        })
        st.write(eval_metrics)

    else:
        st.warning('Mohon upload file CSV terlebih dahulu.')


