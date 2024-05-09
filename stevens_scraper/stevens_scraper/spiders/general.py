import scrapy
import re
from bs4 import BeautifulSoup
import json


class AcademicCatalogSpider(scrapy.Spider):
    name = "general"
    allowed_domains = ["stevens.smartcatalogiq.com"]
    start_urls = []
    with open("department_urls.txt", "r") as fp:
        for line in fp:
            start_urls.append(line.strip())
    custom_settings = {
        "DEPTH_LIMIT": 0,
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 1.0,
    }

    def __init__(self):
        self.url_to_file_map = {}

    def parse(self, response):
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.select_one("div#main").text
        filename = response.url.replace("/", " ").replace(":", "_").strip()
        with open(
            f"C:/Users/smend/OneDrive/Desktop/Stevens/SEM2/Web Mining BIA 660-C/project/langchain_project/stevens_scraper/data/text/{filename}.txt",
            "w+",
        ) as fp:
            fp.write(text)
        with open(
            f"C:/Users/smend/OneDrive/Desktop/Stevens/SEM2/Web Mining BIA 660-C/project/langchain_project/stevens_scraper/data/html/{filename}.html",
            "w+",
        ) as f:
            f.write(response.text)
        self.url_to_file_map[response.url] = filename

    def close(self):
        with open(
            "C:/Users/smend/OneDrive/Desktop/Stevens/SEM2/Web Mining BIA 660-C/project/langchain_project/stevens_scraper/data/department_url_map.json",
            "w+",
        ) as file:
            json.dump(self.url_to_file_map, file, indent=4)
