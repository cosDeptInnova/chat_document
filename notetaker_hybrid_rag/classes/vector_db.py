# Clase para Qdrant (Donde crearemos la colección y subiremos los vectores. La función es buscar porr significado)

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from sentence_transformers import SentenceTransformer

# Cargamos las variables del archivo .env al inicio
load_dotenv()

class VectorDB:
    def __init__(self):
        # Configuración desde variables de entorno (.env)
        host = os.getenv("QDRANT_HOST")
        port = int(os.getenv("QDRANT_PORT"))
        self.collection_name = os.getenv("QDRANT_COLLECTION")

        # 1.- Cargamos el modelo de Embeddings local
        model_path = os.getenv("LOCAL_EMBED_DIR")

        if not model_path:
            raise ValueError("ERROR CRÍTICO: No se ha definido LOCAL_EMBED_DIR en el archivo .env")

        print(f"Cargando modelo de embeddings desde: {model_path} ...")

        # 2. Carga del Modelo
        try:
            self.model = SentenceTransformer(model_path)
            
            # Detectamos el tamaño del vector del modelo automáticamente
            self.vector_size = self.model.get_sentence_embedding_dimension()
            print(f"Modelo cargado. Dimensión de vectores: {self.vector_size}")
        except Exception as e:
            print(f"Error cargando el modelo local: {e}. Asegúrate de que la ruta es correcta.")
            # Si falla el modelo, paramos todo, ya que no podemos vectorizar.
            raise e
        
        # 3.- Conectamos con Qdrant
        self.client = QdrantClient(host=host, port=port)
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """Crea la colección en Qdrant si no existe todavía."""
        try:
            # Intentamos obtener la colección
            self.client.get_collection(collection_name=self.collection_name)
            print(f"Conectado a la colección: '{self.collection_name}'")
        except Exception:
            # Si da error (404, porque no existe), la creamos
            print(f"Creando colección '{self.collection_name}' con tamaño {self.vector_size}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            print(f"Colección '{self.collection_name}' creada con éxito.")
    
    # Modificado para recibir la lista de textos y la lista de payloads
    def upsert_chunks(self, texts, payloads):
        if not self.model: raise Exception("Modelo no cargado.")
        if not texts: return []
        
        print(f"Vectorizando {len(texts)} fragmentos...")
        embeddings = self.model.encode(texts)

        points = []
        chunk_metadata_list = [] # Para devolver a Neo4j

        for i, (vector, payload) in enumerate(zip(embeddings, payloads)):
            point_id = str(uuid.uuid4())

            chunk_metadata_list.append({
                "chunk_id": point_id,
                "index": payload.get("chunk_index", i)
            })

            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload # Aquí va toda la metadata rica (doc_id, datetime, topics, etc.)
            ))
        
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Subidos {len(texts)} chunks vectorizados a Qdrant.")

        return chunk_metadata_list

    def search(self, query_text, limit=5):
        """
        Busca los fragmentos más parecidos a una pregunta.
        """
        if not self.model:
            return []

        # Convertimos la pregunta del usuario en vector
        query_vector = self.model.encode(query_text).tolist()
        
        # Usamos query_points
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        
        # Devuelve la lista de fragmentos encontrados
        return response.points