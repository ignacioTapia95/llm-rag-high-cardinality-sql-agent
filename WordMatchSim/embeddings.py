from langchain.embeddings.base import Embeddings


class CustomTfidfEmbeddings(Embeddings):
    """
    Custom TF-IDF embedding class that integrates a vectorizer and converts embeddings to float.

    Args:
        vectorizer: A vectorizer instance that transforms text into TF-IDF vectors.
    """

    def __init__(self, vectorizer):
        """
        Initializes the CustomTfidfEmbeddings with the given vectorizer.
        """
        self.vectorizer = vectorizer

    def embed_documents(self, texts):
        """
        Embeds a list of documents by transforming them into TF-IDF vectors.

        Args:
            texts (List[str]): A list of documents to embed.

        Returns:
            List[List[float]]: A list of embedded document vectors as lists of floats.
        """
        embeddings = self.vectorizer.transform(texts)
        return [list(map(float, vector)) for vector in embeddings]

    def embed_query(self, text):
        """
        Embeds a single query by transforming it into a TF-IDF vector.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: The embedded query vector as a list of floats.
        """
        embedding = self.vectorizer.transform([text])[0]
        return list(map(float, embedding))
