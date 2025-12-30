# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 09:46:40 2021

@author: TrungCT
Updated to work with modern IMDb layout (2024+)
"""

import re
import scrapy
from scrapy.exceptions import CloseSpider


class IMDBCrawler(scrapy.Spider):
    name = "full2MpaaCrawler"
    start_urls = [
        'https://www.imdb.com/search/title/?title_type=movie&release_date=2020-01-01,&sort=release_date,desc']
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    parseUrl = 'https://www.imdb.com'
    mpaaPath = '/parentalguide'

    def __init__(self, *args, **kwargs):
        super(IMDBCrawler, self).__init__(*args, **kwargs)
        self.limit = int(kwargs.get('limit', 10000))
        self.count = 0

    listOfCertificate = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'GP', 'M', 'M/PG', 'X']
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 0.2,  # Faster - 0.2 seconds
        'CONCURRENT_REQUESTS': 16,  # More concurrent requests
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'AUTOTHROTTLE_ENABLED': True,  # Auto-adjust speed based on server response
        'AUTOTHROTTLE_START_DELAY': 0.1,
        'AUTOTHROTTLE_MAX_DELAY': 1,
    }

    def parse(self, response):
        # Extract movie links
        movie_links = response.xpath('//a[contains(@href, "/title/tt")]/@href').getall()
        movie_links = list(set([link for link in movie_links if '/title/tt' in link and '/reviews' not in link]))
        
        for link in movie_links:
            if '/title/tt' in link:
                # Extract just the title ID part
                link_page = "/".join(link.split("/")[0:-1])
                yield scrapy.Request(self.parseUrl + link_page + self.mpaaPath, callback=self.parseMPAA, errback=self.errback)
        
        # Try to find next page
        next_page = response.xpath('//a[contains(@class, "next-page")]/@href').get()
        if next_page is None:
            next_page = response.xpath('//a[text()="Next"]/@href').get()
        
        if next_page is not None:
            current_start = int(re.search(r'start=(\d+)', response.url).group(1) if re.search(r'start=(\d+)', response.url) else 1)
            if current_start < 10000:
                yield response.follow(next_page, callback=self.parse)

    def parseMPAA(self, response):
        try:
            self.count += 1
            if self.count > self.limit:
                raise CloseSpider(f'Reached limit of {self.limit} movies')
            
            data = {}
            
            # Extract movie title from response header or text
            movie_title = response.xpath('//h1//text()').get('').strip()
            if not movie_title:
                # Alternative title extraction from page
                movie_title = response.xpath('//a[@href and contains(@href, "/title/")]/text()').get('').strip()
            data['Movie_Title'] = movie_title if movie_title else 'Unknown'
            
            # Extract movie ID from URL
            movie_id_match = re.search(r'/title/(tt\d+)', response.url)
            data['Movie_ID'] = int(movie_id_match.group(1).replace('tt', '')) if movie_id_match else 0
            
            if data['Movie_ID'] == 0:
                return
            
            # Try to extract certifications from the page
            list_of_certificate = []
            
            # Get all text content and search for certification patterns
            page_text = ' '.join(response.xpath('//text()').getall())
            
            # Simple approach: look for each certificate code in the page
            for cert in self.listOfCertificate:
                if f'United States: {cert}' in page_text or f'United States:{cert}' in page_text:
                    list_of_certificate.append(cert)
            
            data['ListOfCertificate'] = list(set(list_of_certificate))
            yield data
            
        except CloseSpider:
            raise
        except Exception as e:
            self.logger.warning(f"Error parsing MPAA: {response.url}")
    
    def errback(self, failure):
        self.logger.warning(f"Failed request: {failure.request.url}, Error: {failure.getErrorMessage()}")
