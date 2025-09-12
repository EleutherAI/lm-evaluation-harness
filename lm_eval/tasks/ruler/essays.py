# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import asyncio
import glob
import os
from functools import cache
from typing import Dict

import html2text
import httpx
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm as async_tqdm


@cache
async def fetch_url(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


@cache
async def process_html_essay(
    client: httpx.AsyncClient, url: str, h: html2text.HTML2Text, temp_folder: str
) -> None:
    filename = url.split("/")[-1].replace(".html", ".txt")
    if os.path.exists(os.path.join(temp_folder, filename)):
        return None
    try:
        content = await fetch_url(client, url)
        soup = BeautifulSoup(content, "html.parser")
        specific_tag = soup.find("font")
        if specific_tag:
            parsed = h.handle(str(specific_tag))

            with open(
                os.path.join(temp_folder, filename), "w", encoding="utf-8"
            ) as file:
                file.write(parsed)
    except Exception as e:
        print(f"Failed to download {filename}: {str(e)}")


@cache
async def process_text_essay(
    client: httpx.AsyncClient, url: str, temp_folder: str
) -> None:
    filename = url.split("/")[-1]
    if os.path.exists(os.path.join(temp_folder, filename)):
        return None
    try:
        content = await fetch_url(client, url)
        with open(os.path.join(temp_folder, filename), "w", encoding="utf-8") as file:
            file.write(content)
    except Exception as e:
        print(f"Failed to download {filename}: {str(e)}")


@cache
async def get_essays() -> Dict[str, str]:
    temp_folder_repo = "essay_repo"
    temp_folder_html = "essay_html"
    os.makedirs(temp_folder_repo, exist_ok=True)
    os.makedirs(temp_folder_html, exist_ok=True)

    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_tables = True
    h.escape_all = True
    h.reference_links = False
    h.mark_code = False

    url_list = "https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Fetch URL list
        content = await fetch_url(client, url_list)
        urls = content.splitlines()

        # Separate HTML and text URLs
        html_urls = [url for url in urls if ".html" in url]
        text_urls = [url for url in urls if ".html" not in url]

        # Process HTML essays
        html_tasks = [
            process_html_essay(client, url, h, temp_folder_html) for url in html_urls
        ]
        await async_tqdm.gather(*html_tasks, desc="Downloading HTML essays")

        # Process text essays
        text_tasks = [
            process_text_essay(client, url, temp_folder_repo) for url in text_urls
        ]
        await async_tqdm.gather(*text_tasks, desc="Downloading text essays")

    # Collect results
    files_repo = sorted(glob.glob(os.path.join(temp_folder_repo, "*.txt")))
    files_html = sorted(glob.glob(os.path.join(temp_folder_html, "*.txt")))

    # Combine all texts
    text = ""
    for file in files_repo + files_html:
        with open(file, "r", encoding="utf-8") as f:
            text += f.read()

    return {"text": text}


@cache
def get_all_essays() -> Dict[str, str]:
    """Synchronous wrapper for get_essays()"""
    return asyncio.run(get_essays())
