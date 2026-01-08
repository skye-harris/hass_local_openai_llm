from datetime import datetime

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from custom_components.local_openai import LOGGER


class WeaviateError(Exception):
    pass


class WeaviateClient:
    """Weaviate API Client."""

    def __init__(self, hass: HomeAssistant, host: str, api_key: str | None) -> None:
        """Initialize the weaviate client."""
        self._aiohttp = async_get_clientsession(hass=hass)
        self._host = host
        self._api_key = api_key

    @staticmethod
    def prepare_class_name(class_name: str) -> str:
        """Prepare our class name for use with Weaviate."""
        return class_name.lower().capitalize().replace(" ", "")

    def _api_headers(self) -> dict:
        """Prepare headers for API calls."""
        headers = {
            "Content-Type": "application/json",
        }

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        return headers

    async def near_text(
        self, class_name: str, query: str, threshold: float, limit: int
    ):
        """Query weaviate for vector similarity."""
        class_name = self.prepare_class_name(class_name)
        query_obj = {
            "query": f"""
            {{
              Get {{
                {class_name}(
                  nearText: {{
                    concepts: ["{query}"],
                    certainty: {threshold},
                  }},
                  limit: {limit},
                ) {{
                  content
                  _additional {{
                    certainty
                  }}
                }}
              }}
            }}
            """
        }
        try:
            async with self._aiohttp.post(
                f"{self._host}/v1/graphql", json=query_obj, headers=self._api_headers()
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result.get("data", {}).get("Get", {}).get(class_name, [])
        except aiohttp.ClientError as err:
            LOGGER.warning("Error communicating with Weaviate API: %s", err)
            raise WeaviateError("Unable to query Weaviate") from err

    async def hybrid_search(
        self, class_name: str, query: str, alpha: float, threshold: float, limit: int
    ):
        """Query Weaviate Hybrid search."""
        class_name = self.prepare_class_name(class_name)
        query_obj = {
            "query": f"""
            {{
              Get {{
                {class_name}(
                  hybrid: {{
                    query: "{query}",
                    properties: ["query"],
                    alpha: {alpha}
                    bm25SearchOperator: {{
                      operator: Or
                      minimumOrTokensMatch: 2
                    }}
                  }},
                  limit: {limit},
                ) {{
                  query
                  content
                  _additional {{
                    score
                    rerank(
                      property: "query"
                      query: "Reorder the results so that the most relevant query is first, using the following request to guide you: ${query}"
                    ) {{
                      score
                    }}
                  }}
                }}
              }}
            }}
            """
        }
        try:
            start_time = datetime.now()
            async with self._aiohttp.post(
                url=f"{self._host}/v1/graphql",
                json=query_obj,
                headers=self._api_headers(),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                time_diff = datetime.now() - start_time
                millis = time_diff.microseconds / 1000

                LOGGER.debug(f"Weaviate query took {millis} milliseconds")

                results = result.get("data", {}).get("Get", {}).get(class_name, [])
                return [
                    res
                    for res in results
                    if float(res.get("_additional", {}).get("score", 0)) >= threshold
                ]
        except aiohttp.ClientError as err:
            LOGGER.warning("Error communicating with Weaviate API: %s", err)
            raise WeaviateError("Unable to query Weaviate") from err

    async def create_class(self, class_name: str):
        """Create our object class in Weaviate."""
        class_name = self.prepare_class_name(class_name)

        query_obj = {
            "class": class_name,
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {},
                "reranker-transformers": {},
            },
            "properties": [
                {
                    "name": "query",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "vectorizePropertyName": False,
                        },
                    },
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "vectorizePropertyName": False,
                        },
                    },
                },
            ],
        }

        try:
            async with self._aiohttp.post(
                f"{self._host}/v1/schema", json=query_obj, headers=self._api_headers()
            ) as resp:
                resp.raise_for_status()
                await self.seed_sample_data(class_name)
                return
        except aiohttp.ClientError as err:
            LOGGER.warning(
                "Error communicating with Weaviate API: %s, request: %s", err, query_obj
            )
            raise WeaviateError("Unable to create object class in Weaviate") from err

    async def does_class_exist(self, class_name: str) -> bool:
        """Check if an object class exists in Weaviate."""
        class_name = self.prepare_class_name(class_name)

        try:
            async with self._aiohttp.get(
                f"{self._host}/v1/schema/{class_name}", headers=self._api_headers()
            ) as resp:
                if resp.status == 404:
                    return False

                resp.raise_for_status()
                return True
        except aiohttp.ClientResponseError as err:
            LOGGER.warning("Error communicating with Weaviate API: %s", err)
            raise WeaviateError("Unable to lookup object class in Weaviate") from err
        except aiohttp.ClientError as err:
            LOGGER.warning("Error communicating with Weaviate API: %s", err)
            raise WeaviateError("Unable to lookup object class in Weaviate") from err

    async def does_object_exist(self, class_name: str, object_uuid: str) -> bool:
        """Check if an object exists in Weaviate."""
        class_name = self.prepare_class_name(class_name)

        try:
            async with self._aiohttp.get(
                f"{self._host}/v1/objects/{class_name}/{object_uuid}",
                headers=self._api_headers(),
            ) as resp:
                if resp.status == 404:
                    return False

                resp.raise_for_status()
                return True
        except aiohttp.ClientResponseError as err:
            LOGGER.warning("Error communicating with Weaviate API: %s", err)
            raise WeaviateError("Unable to lookup object in Weaviate") from err
        except aiohttp.ClientError as err:
            LOGGER.warning("Error communicating with Weaviate API: %s", err)
            raise WeaviateError("Unable to lookup object in Weaviate") from err

    async def add_object(
        self, class_name: str, query: str, content: str | None, object_uuid: str | None
    ) -> object:
        """Add an object to Weaviate."""
        class_name = self.prepare_class_name(class_name)

        query_obj = {
            "class": class_name,
            "properties": {
                "query": query,
                "content": content,
            },
        }

        if object_uuid:
            query_obj["id"] = object_uuid

        try:
            async with self._aiohttp.post(
                url=f"{self._host}/v1/objects",
                json=query_obj,
                headers=self._api_headers(),
            ) as resp:
                resp.raise_for_status()

                LOGGER.info(f"Added object of class '{class_name}' into Weaviate")
                return
        except aiohttp.ClientError as err:
            LOGGER.warning(
                "Error communicating with Weaviate API: %s, request: %s", err, query_obj
            )
            raise WeaviateError("Unable to insert new object into Weaviate") from err

    async def replace_object(
        self, class_name: str, query: str, content: str | None, object_uuid: str
    ) -> object:
        """Replace an object in Weaviate."""
        class_name = self.prepare_class_name(class_name)

        query_obj = {
            "id": object_uuid,
            "class": class_name,
            "properties": {
                "query": query,
                "content": content,
            },
        }

        try:
            async with self._aiohttp.put(
                f"{self._host}/v1/objects/{class_name}/{object_uuid}",
                json=query_obj,
                headers=self._api_headers(),
            ) as resp:
                resp.raise_for_status()

                LOGGER.info(
                    f"Updated object '{object_uuid}' of class '{class_name}' in Weaviate"
                )
                return
        except aiohttp.ClientError as err:
            LOGGER.warning(
                "Error communicating with Weaviate API: %s, request: %s", err, query_obj
            )
            raise WeaviateError("Unable to update object in Weaviate") from err

    async def seed_sample_data(self, class_name: str):
        """Seed some sample objects into our database."""
        data = []  # TODO

        for datum in data:
            await self.add_object(
                class_name=class_name,
                query=datum[0],
                content=datum[1],
                object_uuid=None,
            )
