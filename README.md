# tractatus-topic-modeling
Topic modeling applied to Wittgenstein’s Tractatus Logico-Philosophicus using LDA and NLP techniques




README


Overview
The script tractatus_topic_modeling.py performs topic modeling on Tractatus Logico-Philosophicus by Ludwig Wittgenstein. The pipeline consists of four main steps:
1. Data Loading & Segmentation: The text is loaded and split into structured segments based on the numbering in Tractatus.
2. Preprocessing & Cleaning: Unnecessary elements such as punctuation, images, numbers, and stopwords are removed.
3. Lemmatization: Text is exclusively lemmatized for nouns and adjectives, and custom lemma mappings are applied.
4. Topic Modeling: An LDA model is trained to extract topics, visualize results, and identify key segments per topic.


Code Breakdown
Data Loading & Segmentation
* The structured numbering of the Tractatus is identified using regex patterns. This ensures that each segment is correctly parsed based on its hierarchical numbering.
* Once the numbering patterns are detected, the text is split in a way that corresponds to the structure of the Tractatus.
* The extracted segments are stored in a list, preserving their order for further analysis.

Preprocessing & Cleaning
* Punctuation, special characters, and formatting symbols are removed to ensure uniformity in text processing.
* References to images and metadata, such as filenames or mathematical formula placeholders, are identified and deleted.
* Mathematical formulae are removed from the text.
* Numbers are filtered out, as they do not contribute to the linguistic content of the analysis.
* Words containing specific substrings (e.g., "overline" or "\text{c}") are identified and removed.
* Text is converted to lowercase to standardize formatting.
* A predefined list of German stopwords is loaded and applied to eliminate common, non-informative words.
* A custom list of domain-specific unwanted words, such as abbreviations and formatting markers, is applied to refine the dataset further.

Lemmatization
* The text is lemmatized with an exclusive focus on nouns and adjectives (the lemmatizer had significant issues handling German particle verbs, leading to distorted results when verbs were included).
* Additional checks and manual interventions are performed to correct errors introduced by the lemmatization process. This includes:
    * Custom Lemma Mapping: Some words were skipped during lemmatization, particularly uncommon compound words in German (e.g., "Wahrheitsmöglichkeit" and its inflected forms). The process revisits original tokens and explicitly assigns the correct lemma where necessary.
    * Manual Lemma Corrections: Some words were incorrectly lemmatized (e.g., all inflected forms of "Tautologie" were mapped to "Tautologi"). Manual corrections ensure that such errors are resolved by explicitly setting the correct lemma.
    * Unwanted Lemma Removal: Some words were incorrectly lemmatized when they should not have been lemmatized at all. For instance, certain verbs were mistakenly processed. These unnecessary lemmas are identified and removed.

Topic Modeling
* A dictionary is created to map words to unique IDs, which is essential for training the topic modeling algorithm.
* The preprocessed text segments are converted into a bag-of-words format, representing each segment as a numerical vector of word occurrences.
* An LDA (Latent Dirichlet Allocation) model is trained with a user-defined number of topics.
* The most representative words for each topic are identified and displayed.
* pyLDAvis is used to generate an interactive visualization of topics and their distribution.
* The most representative text segments for each topic are extracted and saved in a file. Examining these segments might help in better understanding and identifying what each topic is. 



