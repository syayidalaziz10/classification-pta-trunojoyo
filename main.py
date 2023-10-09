import streamlit as st
import pandas as pd
import nltk
import gensim
import re
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')


st.title('Topic Modeling Skripsi Portal Tugas Akhir Universitas Trunojoyo Madura')
load_data, preprocessing, feature_extraction = st.tabs(
    ['Load Data', 'Preprocessing', 'Feature Extraction'])


with load_data:
    uploaded_file = st.file_uploader('Upload file CSV')
    if uploaded_file is not None:
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
        st.write('Term Frequency', df_countvect)

        countvectorizer = CountVectorizer(analyzer='word')
        log_matrix = countvectorizer.fit_transform(steeming_abstrak)
        count_tokens = countvectorizer.get_feature_names_out()
        count_log_matrix = np.log1p(log_matrix)
        df_log_countvect = pd.DataFrame(data=count_log_matrix.toarray(), columns=count_tokens)
        st.write('Log Frequency', df_log_countvect)

        countvectorizer = CountVectorizer(analyzer='word', binary=True)
        binary_matrix = countvectorizer.fit_transform(steeming_abstrak)
        df_binary = pd.DataFrame(binary_matrix.toarray(), columns=countvectorizer.get_feature_names_out())
        st.write('Binary', df_binary)
        
        st.write('2. Feature Extraction (Global Weighting)')
        tfidfvectorizer = TfidfVectorizer(analyzer='word')
        tfidf = tfidfvectorizer.fit_transform(steeming_abstrak)
        tfidf_tokens = tfidfvectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(data = tfidf.toarray(), columns = tfidf_tokens)
        st.write('TF-IDF Vectorizer',tfidf_df)



        st.write('3. LDA (Latent Dirchlect Allocation)')
        document = stopword_abstrak
        dictionary = corpora.Dictionary(document)
        corpus = [dictionary.doc2bow(tokens) for tokens in document]

        num_topics = 3
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=dictionary,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    passes=10,
                                                    per_word_topics=True)
        
        # st.write('LDA Build Model Topics', lda_model.print_topics())


        topic_proportions_list = []

        for index, doc in enumerate(corpus):
            topic_prop = lda_model.get_document_topics(doc)
            proportions = {f'Topic {i+1}': 0.0 for i in range(num_topics)}

            for topic in topic_prop:
                proportions[f'Topic {topic[0] + 1}'] = topic[1]
            topic_proportions_list.append(proportions)

        topic_proportions_df = pd.DataFrame(topic_proportions_list)
        topic_proportions_df.insert(0, 'Dokumen', range(1, len(topic_proportions_df) + 1))
        st.write('LDA Document Topic Proportion',topic_proportions_df)

        st.write('4. Clustering Feature Extraction')
        

    else:
        st.warning('Mohon upload file CSV terlebih dahulu.')
