from typing import List

async def get_collection_tags_impl(self) -> List[str]:
    """
    Lee todas las tags de la colección del usuario desde Redis.
    """
    redis_key = f"tags:{self.collection_name}"
    tags = await self.redis_client.smembers(redis_key)
    return list(tags) if tags else []

async def get_matched_tags_impl(self, query: str) -> List[str]:
    """
    Devuelve las tags que aparecen dentro del texto de la query (case-insensitive).
    """
    all_tags = await get_collection_tags_impl(self)
    q_low = (query or "").lower()
    return [t for t in all_tags if t.lower() in q_low]
