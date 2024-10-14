# Manejo de Categorías de Alta Cardinalidad en Bases de Conocimiento SQL para Agentes LLM tipo RAG

Este proyecto propone una solución para manejar tablas SQL con variables categóricas de alta cardinalidad, que sirven como fuente de conocimiento para agentes LLM (Large Language Models) tipo RAG (Retrieval-Augmented Generation). La solución se enfoca en identificar similitudes de escritura en lugar de relaciones semánticas, lo que es crucial cuando se trabaja con datos categóricos, como marcas o productos, en los que el usuario puede cometer errores tipográficos o usar términos aproximados.

### Uso en Agentes LLM RAG
Este sistema de embeddings basado en TF-IDF puede integrarse en un agente LLM tipo RAG para mejorar la precisión en la recuperación de categorías, incluso si el usuario comete errores ortográficos o proporciona términos aproximados. Esto facilita la interacción con bases de datos SQL de alta cardinalidad, sin depender de las relaciones semánticas típicas que utilizan otros embeddings.

En lugar de usar modelos de embeddings tradicionales que identifican relaciones semánticas, esta solución emplea TF-IDF con n-gramas de caracteres para capturar las similitudes de escritura. El uso de embeddings almacenados en una base de datos vectorial permite realizar búsquedas eficientes y rápidas, devolviendo la categoría más cercana en términos de escritura.



### Descripción del Problema
En tablas SQL con variables categóricas de alta cardinalidad (por ejemplo, listas de marcas o productos), el gran número de valores posibles genera desafíos a la hora de realizar consultas, especialmente si el nombre de la categoría no coincide exactamente con el valor almacenado. Los usuarios pueden cometer errores tipográficos o usar variantes del nombre.

### Por qué no usamos modelos de embeddings preentrenados
Los modelos de embeddings proporcionados por desarrolladores de LLMs (como OpenAI, Google, etc.) están diseñados principalmente para capturar relaciones semánticas entre textos, es decir, el significado de las palabras. Sin embargo, en este caso, no buscamos comprender el significado, sino que simplemente queremos comparar similitudes en la escritura (ortografía) entre un término de entrada y las categorías almacenadas.

Por esta razón, utilizamos un enfoque basado en TF-IDF (Term Frequency-Inverse Document Frequency) con análisis a nivel de caracteres (n-gramas). Este método es más adecuado para identificar similitudes en la escritura entre diferentes cadenas de texto.

### Solución Propuesta
#### Flujo de Trabajo:
1. Vectorización de Palabras Categóricas: Se utiliza TF-IDF con un análisis a nivel de caracteres para generar embeddings de las palabras categóricas (por ejemplo, nombres de marcas). Esto captura patrones en la escritura y permite comparar términos de forma precisa, independientemente de su significado.

2. Cálculo de Similitud: Para realizar una consulta, se calcula la similitud coseno entre el vector del término de entrada y los vectores de las categorías almacenadas. La similitud coseno mide qué tan parecidos son los vectores en términos de escritura, sin importar su significado semántico.

3. Almacenamiento en Firestore: Los embeddings generados se almacenan en Firestore, que actúa como una base de datos vectorial. Esto permite realizar búsquedas rápidas y eficientes sobre los embeddings de las categorías.

4. Consultas de Similitud: Cuando un usuario realiza una consulta con un término aproximado (por ejemplo, "esquetchers" en lugar de "Skechers"), el sistema compara este término con los nombres de las categorías en Firestore y devuelve la más similar en términos de escritura.

### Estructura del Código
* NumPy: Utilizado para cálculos matemáticos y de vectores.
* Google Cloud Firestore: Base de datos utilizada para almacenar embeddings y realizar búsquedas vectoriales.
* LangChain: Framework que maneja la creación de embeddings y búsquedas.
* CustomTFIDFEmbeddings: Clase personalizada para generar y manejar embeddings de TF-IDF.
* SimpleTfidfVectorizer: Clase que tokeniza textos y genera embeddings de TF-IDF personalizados utilizando n-gramas de caracteres.

### Cómo Funciona
1. Generación de Embeddings: El vectorizador tokeniza los textos categóricos utilizando n-gramas de caracteres, lo que permite capturar similitudes en la escritura, como pequeñas variaciones o errores tipográficos.

2. Similitud Coseno: Se utiliza la similitud coseno para comparar el vector de la consulta del usuario con los vectores de las categorías almacenadas. La similitud coseno mide la similitud en la estructura de las palabras, devolviendo un valor entre 0 y 1, donde 1 indica una coincidencia perfecta.

3. Consulta Aproximada: Cuando un usuario realiza una consulta con un término que no coincide exactamente con los valores almacenados en la base de datos (por ejemplo, "esquetchers" en lugar de "Skechers"), el sistema encuentra el término más similar basándose en la similitud de escritura.