import math


class SimpleTfidfVectorizer:
    """
    A simplified TF-IDF vectorizer that tokenizes documents and generates TF-IDF vectors.

    Args:
        analyzer (str): Determines the type of tokenization ('word' or 'char').
        ngram_range (tuple): The range of n-grams to consider during tokenization.
    """

    def __init__(self, analyzer='word', ngram_range=(1, 1)):
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_ = {}
        self.feature_names_ = []

    def _tokenize(self, text):
        """
        Tokenizes the input text based on the analyzer type and n-gram range.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokens or n-grams from the input text.
        """
        if self.analyzer == 'char':
            ngrams = []
            n_min, n_max = self.ngram_range
            for n in range(n_min, n_max + 1):
                ngrams.extend([text[i:i + n]
                              for i in range(len(text) - n + 1)])
            return ngrams
        else:
            return text.split()

    def fit(self, raw_documents):
        """
        Fits the vectorizer on the provided raw documents and calculates the IDF values.

        Args:
            raw_documents (List[str]): A list of documents to fit the vectorizer on.
        """
        df = {}
        total_docs = len(raw_documents)

        for doc in raw_documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        self.vocabulary_ = {term: idx for idx,
                            term in enumerate(sorted(df.keys()))}
        self.feature_names_ = sorted(df.keys())

        self.idf_ = {term: math.log(
            (1 + total_docs) / (1 + df[term])) + 1.0 for term in df}

    def transform(self, raw_documents):
        """
        Transforms the input documents into TF-IDF vectors.

        Args:
            raw_documents (List[str]): A list of documents to transform.

        Returns:
            List[List[float]]: A list of TF-IDF vectors for the input documents.
        """
        X = []
        for doc in raw_documents:
            tokens = self._tokenize(doc)
            tf = {token: tokens.count(token)
                  for token in tokens if token in self.vocabulary_}

            max_tf = max(tf.values()) if tf else 1

            vector = [0.0] * len(self.vocabulary_)
            for token, count in tf.items():
                idx = self.vocabulary_[token]
                vector[idx] = (count / max_tf) * self.idf_[token]

            X.append(vector)

        return X

    def fit_transform(self, raw_documents):
        """
        Fits the vectorizer and transforms the documents in one step.

        Args:
            raw_documents (List[str]): A list of documents to fit and transform.

        Returns:
            List[List[float]]: A list of TF-IDF vectors for the input documents.
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)
