import numpy as np

embedding_path1 = "embeddings/crawl-300d-2M.vec" #FastText
embedding_path2 = "embeddings/glove.42B.300d.txt" #Glove 300d
embed_size = 300

def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

def build_matrix(embedding_path, tk, max_features):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding = "utf-8"))

    word_index = tk.word_index
    nb_words = max_features
    embedding_matrix = np.zeros((nb_words + 1, 300))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
def create_embedding_matrix(embed, tk, max_features):
    if embed == 1:
        print("Please download and put this embeddings to the folder ethos/experiments/embeddings: ")
        print('https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')
        return build_matrix(embedding_path1, tk, max_features)
    elif embed == 2:
        print("Please download and put this embeddings to the folder ethos/experiments/embeddings: ")
        print('http://nlp.stanford.edu/data/glove.42B.300d.zip')
        return build_matrix(embedding_path2, tk, max_features)
    else:
        print("Please download and put this embeddings to the folder ethos/experiments/embeddings: ")
        print('https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')
        print('http://nlp.stanford.edu/data/glove.42B.300d.zip')
        return np.concatenate([build_matrix(embedding_path1, tk, max_features), build_matrix(embedding_path2, tk, max_features)], axis=-1)
  