import abc
import os
from typing import AsyncIterator

import httpx

import ragna
from ragna.core import Assistant, EnvVarRequirement, Requirement, Source

from bs4 import BeautifulSoup
import re
import requests
import base64
class ApiAssistant(Assistant):
    _API_KEY_ENV_VAR: str
    _API_BASE_URL: str
    @classmethod
    def requirements(cls) -> list[Requirement]:
        return [EnvVarRequirement(cls._API_KEY_ENV_VAR)]

    def icon(self) -> bytes:
        # Who even needs regex
        url = "https://" + ".".join([url for url in self._API_BASE_URL.split('/') if '.' in url][0].split('.')[1:])
        file = requests.get(url + '/favicon.ico')
        print(url)
        print(file)
        if file.status_code != requests.codes.ok:
            page = requests.get(url)
            print(url)
            soup = BeautifulSoup(page.text, features="lxml")

            # It's me, I need regex
            icons = soup.find_all('link', attrs={'rel': re.compile("^(shortcut icon|icon)$", re.I)})

            if 'https' not in icons[0].get('href'):
                icon_url = url + icons[0].get('href')
            else:
                icon_url = icons[0].get('href')

            file = requests.get(icon_url)
        return base64.b64encode(file.content)

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"User-Agent": f"{ragna.__version__}/{self}"},
            timeout=60,
        )
        self._api_key = os.environ[self._API_KEY_ENV_VAR]

    async def answer(
        self, prompt: str, sources: list[Source], *, max_new_tokens: int = 256
    ) -> AsyncIterator[str]:
        async for chunk in self._call_api(  # type: ignore[attr-defined, misc]
            prompt, sources, max_new_tokens=max_new_tokens
        ):
            yield chunk

    @abc.abstractmethod
    async def _call_api(
        self, prompt: str, sources: list[Source], *, max_new_tokens: int
    ) -> AsyncIterator[str]:
        ...
