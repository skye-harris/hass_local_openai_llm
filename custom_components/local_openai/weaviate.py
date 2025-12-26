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
            headers = {
                "Content-Type": "application/json",
            }

            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            async with self._aiohttp.post(
                f"{self._host}/v1/graphql", json=query_obj, headers=headers
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result.get("data", {}).get("Get", {}).get(class_name, [])
        except aiohttp.ClientError as err:
            LOGGER.warning("Error communicating with Weaviate API: %s", err)
            raise WeaviateError("Unable to query Weaviate") from err

    async def create_class(self, class_name: str):
        """Create our object class in Weaviate."""
        class_name = self.prepare_class_name(class_name)

        query_obj = {
            "class": class_name,
            "vectorizer": "text2vec-transformers",
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
            headers = {
                "Content-Type": "application/json",
            }

            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            async with self._aiohttp.post(
                f"{self._host}/v1/schema", json=query_obj, headers=headers
            ) as resp:
                resp.raise_for_status()
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
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            async with self._aiohttp.get(
                f"{self._host}/v1/schema/{class_name}", headers=headers
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
