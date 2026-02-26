# Clase para Neo4j (Aquí crearemos los nodos como Reunión, Participante, Tema, para poder buscar por "conexión" entre ellos)

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Cargamos variables de entorno
load_dotenv()

class GraphDB:
    def __init__(self):
        # 1. Configuración desde .env
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verificamos conectividad rápida
            self.driver.verify_connectivity()
            print("Conexión exitosa a Neo4j.")
        except Exception as e:
            print(f"Error conectando a Neo4j: {e}")
            raise e

    def close(self):
        self.driver.close()

    # Control de Duplicados
    def check_meeting_exists(self, meeting_id):
        """Devuelve True si la reunión ya está en el grafo."""
        with self.driver.session() as session:
            result = session.run("MATCH (m:Meeting {id: $id}) RETURN m LIMIT 1", id=meeting_id)
            return result.single() is not None

    def create_meeting_node(self, meeting_id, raw_transcript):
        """
        Crea el nodo central de la reunión y conecta a los participantes.
        """
        with self.driver.session() as session:
            # 1. Crear el nodo central de la reunión
            session.run("MERGE (m:Meeting {id: $m_id})", m_id=meeting_id)

            # 2. Extraer participantes únicos y conectarlos
            participantes = set()
            for entry in raw_transcript:
                # Ahora leemos 'speaker' del bloque normalized
                name = entry.get("speaker")
                if name: participantes.add(name)

            # 3. Insertar participantes
            # Usamos UNWIND para optimizar y crearlos todos de una vez, no en un bucle for/foreach de Python
            if participantes:
                query = """
                UNWIND $names AS p_name
                MERGE (p:Participant {name: p_name})
                WITH p
                MATCH (m:Meeting {id: $m_id})
                MERGE (p)-[:ATTENDED]->(m)
                """
                session.run(query, names=list(participantes), m_id=meeting_id)
        
        print(f"Grafo: Reunión {meeting_id} creada con {len(participantes)} participantes.")

    def link_topic_to_meeting(self, meeting_id, temas):
        """
        Conecta un tema extraído con la reunión.
        """
        if not temas:
            return
        
        # Adaptamos para leer diccionarios si vienen con peso, o strings si vienen solos
        nombres_temas = [t["name"] if isinstance(t, dict) else t for t in temas]

        with self.driver.session() as session:
            # Usamos otra vez UNWIND para evitar múltiples llamadas y optimizarlo
            query = """
            MATCH (m:Meeting {id: $m_id})
            UNWIND $temas_list AS t_name
            MERGE (t:Topic {name: t_name})
            MERGE (m)-[:DISCUSSED]->(t)
            """
            session.run(query, m_id=meeting_id, temas_list=nombres_temas)
        
        print(f"Grafo: {len(temas)} temas vinculados a la reunión.")
    

    def create_chunk_nodes(self, meeting_id, chunk_metadata_list):
        """
        Crea nodos 'Chunk' en Neo4j que contienen el ID de Qdrant.
        Conecta: (Meeting) -> (Chunk) -> (Topic si pudiéramos, de momento Meeting)
        """
        if not chunk_metadata_list:
            return
        
        with self.driver.session() as session:
            # Pasamos la lista de diccionarios completam con UNWIND
            # y Neo4j la recorre internamente (mucho más rápido que Python)
            query = """
            MATCH (m:Meeting {id: $m_id})
            UNWIND $chunks_data AS chunk
            MERGE (c:Chunk {qdrant_id: chunk.chunk_id})
            SET c.index = chunk.index, 
                c.preview = chunk.text_preview
            MERGE (m)-[:HAS_TRANSCRIPT_CHUNK]->(c)
            """
            session.run(query, m_id=meeting_id, chunks_data=chunk_metadata_list)
            
        print(f"Grafo: Nodos de Chunk creados y conectados para reunión {meeting_id}")

    # Función para la consulta de nodos en el grafo con los ids de los chunks sacados por la pregunta en Qdrant
    def get_chunk_context(self, qdrant_id):
        """
        Dado el ID de un vector de Qdrant, busca en Neo4j a qué reunión pertenece,
        quién estaba en ella y de qué temas se hablaron.
        """
        with self.driver.session() as session:
            query = """
            MATCH (c:Chunk {qdrant_id: $q_id})<-[:HAS_TRANSCRIPT_CHUNK]-(m:Meeting)
            OPTIONAL MATCH (p:Participant)-[:ATTENDED]->(m)
            OPTIONAL MATCH (m)-[:DISCUSSED]->(t:Topic)
            RETURN m.id AS meeting_id,
                   collect(DISTINCT p.name) AS participants,
                   collect(DISTINCT t.name) AS topics
            """
            result = session.run(query, q_id=qdrant_id)
            record = result.single()
            
            if record:
                return {
                    "meeting_id": record["meeting_id"],
                    "participants": record["participants"],
                    "topics": record["topics"]
                }
            return None