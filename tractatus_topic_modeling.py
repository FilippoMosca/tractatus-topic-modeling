
import re
import random
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from HanTa import HanoverTagger as ht
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim
from IPython.display import display


# Load and preprocess text data
file_path = '/Users/filippomosca/Desktop/Tractatus Logico-Philosophicus.md'
with open(file_path, 'r', encoding='utf-8') as file:
    tractatus_text = file.read()

# Segment the text based on the structure of the Tractatus
pattern = re.compile(r'^[\t\s\-]*([\d]+(?:\.\d+)*)\s', re.MULTILINE)
matches = list(pattern.finditer(tractatus_text))
segments = []

for i in range(len(matches)):
    start = matches[i].start(1)
    end = matches[i + 1].start(1) if i + 1 < len(matches) else len(tractatus_text)
    segments.append(tractatus_text[start:end].strip())

# Print a random segment to verify segmentation
if segments:
    print("Random segment after segmentation:", random.choice(segments))

# Store original segments for later comparison
original_segments = segments.copy()



##################################################



# Text Cleaning

nltk.download('stopwords')
german_stopwords = set(stopwords.words('german'))

# Remove punctuation and special characters
segments_no_punctuation = [re.sub(r'[,.!?();\[\]“”–:„|«»=_⊃\-∃^~*>]', '', text) for text in segments]

# Remove image references
segments_no_images = [re.sub(r'\bimagepng[^\s]+', '', text) for text in segments_no_punctuation]

# Remove occurrences of "MATHEMATICAL FORMULA"
segments_no_math = [text.replace("MATHEMATICAL FORMULA", "") for text in segments_no_images]

# Remove all numbers
segments_no_numbers = [re.sub(r'\b\d+\b', '', text) for text in segments_no_math]

# Remove words containing "overline"
def remove_words_with_overline(text):
    return ' '.join([word for word in text.split() if 'overline' not in word])

segments_no_overline = [remove_words_with_overline(text) for text in segments_no_numbers]

# Remove words containing "\text{c}"
def remove_words_with_textc(text):
    return ' '.join([word for word in text.split() if '\\text{c}' not in word])

segments_no_textc = [remove_words_with_textc(text) for text in segments_no_overline]

# Convert to lowercase
segments_lower = [text.lower() for text in segments_no_textc]

# Remove stopwords
segments_no_stopwords = [' '.join([word for word in text.split() if word not in german_stopwords]) for text in segments_lower]

# Remove custom stopwords
def remove_custom_stopwords(texts, custom_stopwords):
    return [' '.join([word for word in text.split() if word not in custom_stopwords]) for text in texts]

custom_stopwords = {"xi", "ja", "bzw", "zb", "etc", "screenshot", "png", "usw", "wr", "wrs", "dh", "e"}
segments_no_custom_stopwords = remove_custom_stopwords(segments_no_stopwords, custom_stopwords)



#################################################


# Lemmatization


# Lemmatization using HanoverTagger (Only Nouns and Adjectives)
tagger = ht.HanoverTagger('morphmodel_ger.pgz')
lemmatized_segments = []

for text in segments_no_custom_stopwords:  # Directly using cleaned text
    tokenized_sentence = word_tokenize(text)  # Tokenization using NLTK
    lemmas = [lemma for (word, lemma, pos) in tagger.tag_sent(tokenized_sentence, taglevel=1) if pos.startswith(('N', 'ADJ'))]
    lemmatized_segments.append(lemmas)

# Custom Lemma Mapping AFTER Lemmatization
custom_lemma_mapping = {
    "Elementarsatz": ["elementarsatz", "elementarsätze", "elementarsatzes"],
    "Form": ["form", "formen"],
    "Wahrheitsmöglichkeit": ["wahrheitsmöglichkeiten", "wahrheitsmöglichkeit"],
    "Gegenstand": ["gegenstand", "gegenstandes", "gegenstands", "gegenstände", "gegenständen"],
    "Philosophie": ["philosophie", "philosophien"],
    "Kausalitätsgesetz": ["kausalitätsgesetz", "kausalitätsgesetzes"],
    "Naturwissenschaft": ["naturwissenschaft", "naturwissenschaften"]
}

# Apply custom lemma mapping, checking original tokens
for idx, (original_tokens, lemmas) in enumerate(zip(segments_no_custom_stopwords, lemmatized_segments)):
    for base_lemma, forms in custom_lemma_mapping.items():
        if any(form in original_tokens for form in forms):
            if base_lemma not in lemmas:  # Avoid duplicates
                lemmas.append(base_lemma)

# Custom Lemma Removal
lemmas_to_remove = {"gebildet", "sagen", "benützen", "ersehen"}  # Add more unwanted lemmas here
lemmatized_segments_filtered = [[lemma for lemma in segment if lemma not in lemmas_to_remove] for segment in lemmatized_segments]

# Dictionary of lemma corrections
lemma_corrections = {
    "Urzeiche": "Urzeichen",
    "variabl": "variable",
    "satzvariabl": "Satzvariable",
    "saztvariable": "Satzvariable",
    "urzeich": "Urzeichen",
    "Tautologia": "Tautologie",
    "Tautologien": "Tautologie",
    "tautologi": "tautologisch",
    "Solipsismu": "Solipsismus",
    "Freges": "Frege",
    "Symbolismu": "Symbolismus",
    "elementarsätze": "Elementarsatz",
    "elementarsätz": "Elementarsatz"
}

# Apply corrections directly to lemmatized segments
lemmatized_segments_corrected = [[lemma_corrections.get(word, word) for word in segment] for segment in lemmatized_segments_filtered]

# Create Gensim Dictionary and Corpus
dictionary = corpora.Dictionary(lemmatized_segments_corrected)
corpus = [dictionary.doc2bow(text) for text in lemmatized_segments_corrected]

# Print verification sample
if lemmatized_segments_corrected:
    random_index = random.randint(0, len(lemmatized_segments_corrected) - 1)
    print("Original segment before preprocessing:", original_segments[random_index])
    print("Processed segment after full preprocessing:", lemmatized_segments_corrected[random_index])



################################################


# Topic Modeling


# Topic Modeling using LDA (Gensim)
n_topics = 7
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, passes=10, random_state=0)

# Print top words for each topic
def print_top_words(model, num_words=10):
    for topic_idx in range(n_topics):
        words = [word for word, _ in model.show_topic(topic_idx, num_words)]
        print(f"Topic #{topic_idx}: {', '.join(words)}")
    print()

print_top_words(lda_model, num_words=10)

# Topic Visualization
prepared_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(prepared_data)

# Extract most representative documents per topic
def extract_top_documents(lda_model, corpus, dictionary, original_docs, n_top_words=10, n_top_documents=10, output_file_path="/Users/filippomosca/Desktop/tractatus_most_paradigmatic_segments_by_topic.txt"):
    try:
        doc_topic_dist = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
        topics_words = [(topic_idx, [word for word, _ in lda_model.show_topic(topic_idx, n_top_words)]) for topic_idx in range(n_topics)]
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for topic_number, words in topics_words:
                f.write(f"Topic {topic_number}:\n")
                f.write(", ".join(words) + "\n\n")
                doc_indices = np.argsort([doc[topic_number][1] for doc in doc_topic_dist])[::-1][:n_top_documents]
                for doc_index in doc_indices:
                    f.write(f"Document segment:\n{original_docs[doc_index]}\n\n")
                    f.write("-" * 80 + "\n")
                f.write("= " * 40 + "\n\n")
        print(f"File successfully saved at {output_file_path}")
    except Exception as e:
        print(f"Error while saving the file: {e}")

extract_top_documents(lda_model, corpus, dictionary, original_segments)








