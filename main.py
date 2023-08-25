import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

# 加载数据
labeled_train = pd.read_csv("word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv('word2vec-nlp-tutorial/unlabeledTrainData.tsv', delimiter='\t', quoting=3)
test = pd.read_csv('word2vec-nlp-tutorial/testData.tsv', delimiter='\t', quoting=3)

# data preprocessing
def preprocess_review(review):
    # Remove HTML tags
    review = re.sub(r'<.*?>', ' ', review)

    # Remove non-letter characters
    review = re.sub(r'[^a-zA-Z]', ' ', review)

    # Convert to lowercase and tokenize
    words = review.lower().split()

    # Remove stopwords
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if w not in stops]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]

    return lemmatized_words


labeled_train['review'] = labeled_train['review'].apply(preprocess_review)
unlabeled_train['review'] = unlabeled_train['review'].apply(preprocess_review)
test['review'] = test['review'].apply(preprocess_review)


# Train Word2Vec Model:
# We'll train the Word2Vec model using both labeled and unlabeled reviews to get richer embeddings:
from gensim.models import Word2Vec

# Combine the reviews for Word2Vec training
all_reviews = list(labeled_train['review']) + list(unlabeled_train['review'])

# Train Word2Vec model
model = Word2Vec(sentences=all_reviews, vector_size=100, window=5, min_count=1, workers=4)

# Vectorize reviews



def review_to_vector(review):
    vector = np.zeros(model.vector_size)
    num_words = 0
    for word in review:
        if word in model.wv:
            vector += model.wv[word]
            num_words += 1
    if num_words:
        vector /= num_words
    return vector

labeled_train_vectors = np.array([review_to_vector(review) for review in labeled_train['review']])
test_vectors = np.array([review_to_vector(review) for review in test['review']])

# Distribution of Sentiments:
# Visualize the distribution of positive and negative sentiments in the training data.
labeled_train['sentiment'].value_counts().plot(kind='bar')
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Word Clouds:
# Display the most frequent words in positive and negative reviews.
from wordcloud import WordCloud

# For positive reviews
positive_words = ' '.join(labeled_train[labeled_train['sentiment'] == 1]['review'].apply(' '.join))
wordcloud = WordCloud(background_color="white", max_words=100).generate(positive_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in positive reviews')
plt.show()

# For negative reviews
negative_words = ' '.join(labeled_train[labeled_train['sentiment'] == 0]['review'].apply(' '.join))
wordcloud = WordCloud(background_color="white", max_words=100).generate(negative_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in negative reviews')
plt.show()

# Visualize Word2Vec Embeddings using t-SNE:
# t-SNE is a tool to visualize high-dimensional data.
# It can be used to visualize Word2Vec embeddings to see if
# words with similar meanings are clustered together.
from sklearn.manifold import TSNE

# Get the embeddings
words = list(model.wv.index_to_key)
embeddings = np.array([model.wv[word] for word in words])  # Convert list to numpy array

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings[:300])  # Taking only the first 300 for visualization

# Plot
plt.figure(figsize=(10, 10))
for i, word in enumerate(words[:300]):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
plt.title('t-SNE visualization of Word2Vec embeddings')
plt.show()

