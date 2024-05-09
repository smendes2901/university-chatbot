import scrapy
import re


class AcademicCatalogSpider(scrapy.Spider):
    name = "academic_catalog"
    allowed_domains = ["stevens.smartcatalogiq.com"]
    start_urls = [
        "https://stevens.smartcatalogiq.com/en/2023-2024/academic-catalog/",
    ]
    custom_settings = {
        "DEPTH_LIMIT": 5,
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 1.0,  # Delay between requests
    }

    def __init__(self):
        self.visited_urls = set()
        self.file = open("crawled_urls.txt", "a")

    def parse(self, response):
        if "text/html" in response.headers.get("Content-Type").decode("utf-8"):
            for href in response.css("a::attr(href)"):
                url = response.urljoin(href.extract())
                if url not in self.visited_urls and not self.is_file(url):
                    self.file.write(url + "\n")
                    self.visited_urls.add(url)
                    yield scrapy.Request(url, callback=self.parse)

    def is_file(self, url):
        return re.search(r"\.(pdf|docx|zip|rar)$", url, re.IGNORECASE) is not None

    def close(self):
        self.file.close()
