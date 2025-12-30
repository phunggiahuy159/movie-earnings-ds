# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 09:46:40 2021

@author: TrungCT
Updated to work with modern IMDb layout (2024+)
"""
import json
import re
import scrapy
from scrapy.exceptions import CloseSpider
from urllib.parse import unquote


class IMDBCrawler(scrapy.Spider):
    name = "full2ImdbCrawler"
    # Updated to crawl top-rated movies from all time - sorted by number of votes (most popular movies)
    # This will give us movies with complete data (released movies have all fields populated)
    start_urls = [
        'https://www.imdb.com/search/title/?title_type=feature&num_votes=1000,&sort=num_votes,desc']
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    parseUrl = 'https://www.imdb.com'
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 0.2,  # Faster - 0.2 seconds
        'CONCURRENT_REQUESTS': 16,  # More concurrent requests
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'AUTOTHROTTLE_ENABLED': True,  # Auto-adjust speed based on server response
        'AUTOTHROTTLE_START_DELAY': 0.1,
        'AUTOTHROTTLE_MAX_DELAY': 1,
    }
    
    def __init__(self, *args, **kwargs):
        super(IMDBCrawler, self).__init__(*args, **kwargs)
        self.limit = int(kwargs.get('limit', 10000))  # Default to 10000, can override with -a limit=100
        self.count = 0
        self.seen_urls = set()  # Track visited URLs to prevent duplicates

    def parse(self, response):
        # Extract movie links from search results using updated selectors
        # Try multiple selector patterns for robustness
        movie_links = response.xpath('//a[contains(@href, "/title/tt")]/@href').getall()
        
        # Filter to only title links (not episodes, reviews, fullcredits, etc)
        movie_links = [link for link in movie_links if '/title/tt' in link and any(x not in link for x in ['/reviews', '/fullcredits', '/episodes', '/news'])]
        movie_links = list(set(movie_links))  # Remove duplicates from same page
        
        for link in movie_links:
            # Normalize link to just /title/ttXXXXXXX/
            match = re.search(r'/title/(tt\d+)', link)
            if match:
                clean_link = f'/title/{match.group(1)}/'
                if clean_link not in self.seen_urls:
                    self.seen_urls.add(clean_link)
                    full_url = self.parseUrl + clean_link
                    yield scrapy.Request(full_url, callback=self.parseAMovie, errback=self.errback)
        
        # Try to find next page button
        next_page = response.xpath('//a[contains(@class, "next-page")]/@href').get()
        if next_page is None:
            next_page = response.xpath('//a[text()="Next"]/@href').get()
        
        if next_page is not None:
            current_start = int(re.search(r'start=(\d+)', response.url).group(1) if re.search(r'start=(\d+)', response.url) else 1)
            if current_start < 10000:
                yield response.follow(next_page, callback=self.parse)
        # from scrapy.shell import inspect_response
        # inspect_response(response, self)

    def parseAMovie(self, response):
        try:
            self.count += 1
            if self.count > self.limit:
                raise CloseSpider(f'Reached limit of {self.limit} movies')
            
            data = {}
            
            # Extract Movie ID from URL first
            movie_id_match = re.search(r'/title/(tt\d+)', response.url)
            if not movie_id_match:
                self.logger.warning(f"Could not extract Movie ID from {response.url}")
                return
            
            data['Movie_ID'] = int(movie_id_match.group(1).replace('tt', ''))
            
            # Try to extract from JSON-LD first (most reliable)
            json_ld = response.xpath('//script[@type="application/ld+json"]/text()').getall()
            movie_data = None
            
            self.logger.info(f"Found {len(json_ld)} JSON-LD scripts for {response.url}")
            
            for js in json_ld:
                try:
                    parsed = json.loads(js)
                    if isinstance(parsed, dict) and parsed.get('@type') == 'Movie':
                        movie_data = parsed
                        self.logger.info(f"Found Movie JSON-LD with duration={parsed.get('duration')}, rating={parsed.get('aggregateRating', {}).get('ratingValue') if isinstance(parsed.get('aggregateRating'), dict) else 'N/A'}")
                        break
                    elif isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and item.get('@type') == 'Movie':
                                movie_data = item
                                self.logger.info(f"Found Movie in JSON-LD list with duration={item.get('duration')}")
                                break
                except Exception as e:
                    self.logger.debug(f"Failed to parse JSON-LD: {e}")
                    continue
            
            if movie_data is None:
                self.logger.warning(f"No Movie JSON-LD found for {response.url}")
                movie_data = {}
            
            # Extract basic info
            movie_title = movie_data.get('name', '').strip()
            if not movie_title:
                # Fallback: try to get title from header
                movie_title = response.xpath('//span[@class="hero__primary-text"]/text()').get('').strip()
            if not movie_title:
                self.logger.warning(f"Could not extract Movie Title from {response.url}")
                return  # Skip if no title
            
            data['Movie_Title'] = movie_title
            
            # Budget (text, may contain currency symbol)
            budget_texts = response.xpath('//li[contains(., "Budget")]/following-sibling::li[1]//text()').getall()
            data['Budget'] = ''.join(budget_texts).strip() if budget_texts else ''
            
            # Cast - be more specific with selector
            cast_list = response.xpath('//a[@data-testid="title-cast-item__actor"]//text()').getall()
            data['Cast'] = [c.strip() for c in cast_list if c.strip()]
            
            # Directors and Writers - improved selectors
            directors = response.xpath('//li/a[contains(@href, "/name/nm")][ancestor::li//text()[contains(., "Director")]]/text()').getall()
            writers = response.xpath('//li/a[contains(@href, "/name/nm")][ancestor::li//text()[contains(., "Writer")]]/text()').getall()
            crew_combined = list(set([c.strip() for c in directors + writers if c.strip()]))
            data['Crew'] = crew_combined[:20]
            
            # Studios/Production companies - improved selectors
            studios = response.xpath('//a[contains(@href, "/company/co")]/text()').getall()
            if not studios:
                studios = response.xpath('//li[contains(text(), "Production company") or contains(text(), "Production")]/following-sibling::li[1]//a/text()').getall()
            if not studios:
                studios = response.xpath('//li[contains(., "Production")]/following-sibling::li//a[contains(@href, "/company/")]/text()').getall()
            # Filter out IMDbPro and clean
            data['Studios'] = [s.strip() for s in studios if s.strip() and 'IMDbPro' not in s and 'See production' not in s][:5]
            
            # Genre - get from JSON-LD if available, otherwise specific selectors
            genres = []
            if movie_data and 'genre' in movie_data:
                genre_data = movie_data['genre']
                genres = genre_data if isinstance(genre_data, list) else [genre_data] if genre_data else []
            
            # Fallback: scrape from page - look for genre links in the metadata area specifically
            if not genres:
                genres = response.xpath('//a[contains(@href, "/search/title?genres=")][not(ancestor::script)]/text()').getall()
            
            # Clean up: filter empty and script-injected content
            cleaned_genres = []
            for g in genres:
                g = str(g).strip() if g else ''
                # Skip if looks like script content or is too long
                if g and len(g) < 30 and not g.startswith('var ') and not g.startswith('window.'):
                    cleaned_genres.append(g)
            
            data['Genre'] = cleaned_genres
            
            # Keywords - filter junk entries
            keywords = []
            if movie_data and 'keywords' in movie_data:
                kw = movie_data['keywords']
                keywords = kw if isinstance(kw, list) else [kw] if kw else []
            junk_keywords = {'back to top', 'see more', 'also known as', 'more like this', 'explore more', 'credits'}
            data['Keywords'] = [k.strip() for k in keywords if k.strip() and k.strip().lower() not in junk_keywords][:15]
            
            # Languages - prefer JSON-LD, fallback to page scraping
            languages = []
            if movie_data and 'inLanguage' in movie_data:
                langs = movie_data['inLanguage']
                languages = langs if isinstance(langs, list) else [langs] if langs else []
            # Fallback: scrape from page
            if not languages:
                lang_links = response.xpath('//a[contains(@href, "primary_language=")]/text()').getall()
                if lang_links:
                    languages = lang_links
                else:
                    # Try data-testid selector
                    languages = response.xpath('//li[@data-testid="title-details-languages"]//a/text()').getall()
            data['Languages'] = [l.strip() for l in languages if l.strip() and len(l.strip()) < 25 and 'also' not in l.lower()][:5]
            
            # Countries - prefer JSON-LD, fallback to page
            countries = []
            if movie_data and 'countryOfOrigin' in movie_data:
                country_data = movie_data['countryOfOrigin']
                if isinstance(country_data, list):
                    countries = [c.get('name', '') if isinstance(c, dict) else str(c) for c in country_data if c]
                elif country_data:
                    countries = [country_data.get('name', '') if isinstance(country_data, dict) else str(country_data)]
            # Fallback: scrape from page
            if not countries:
                country_links = response.xpath('//a[contains(@href, "country_of_origin=")]/text()').getall()
                if country_links:
                    countries = country_links
                else:
                    countries = response.xpath('//li[@data-testid="title-details-origin"]//a/text()').getall()
            data['Countries'] = [c.strip() for c in countries if c and c.strip()][:3]
            
            # Filming locations - specific extraction with URL decoding
            locations = []
            all_loc_links = response.xpath('//a[contains(@href, "locations=")]/@href').getall()
            for link in all_loc_links[:10]:
                match = re.search(r'locations=([^&]+)', link)
                if match:
                    loc = unquote(match.group(1))  # Decode URL encoding (London%40%40%40 England -> London, England)
                    # Clean up @ symbols used as separators
                    loc = loc.replace('@', '').replace('  ', ' ').strip()
                    if loc and len(loc) < 50:
                        locations.append(loc)
            data['Filming_Location'] = list(set([loc.strip() for loc in locations if loc.strip()]))
            
            # Release date
            data['Release_Data'] = movie_data.get('datePublished', '')
            
            # Runtime - from JSON-LD first (reliable), fallback to XPath
            runtime = ''
            if movie_data and 'duration' in movie_data:
                # Duration is in ISO 8601 format like "PT148M" or "PT2H22M"
                runtime = str(movie_data['duration']).strip()
            
            if not runtime:
                # Fallback to page scraping - look for HTML element with runtime
                # Runtime typically appears as "duration" or "min" in metadata area
                runtime_text = response.xpath('//div[@data-testid="title-specs-item-duration"]//text()').get()
                if not runtime_text:
                    # Try alternate metadata selectors
                    runtime_spans = response.xpath('//span[contains(text(), "min")]/text()').getall()
                    # Filter for spans that look like "123 min" (short, numeric with "min")
                    for span in runtime_spans:
                        span = span.strip()
                        if span and 'min' in span.lower() and len(span) < 10:
                            runtime_text = span
                            break
                runtime = runtime_text.strip() if runtime_text else ''
            
            data['Runtime'] = runtime
            
            # Box office - improved selectors for worldwide gross (get the dollar amount, not the label)
            gross_text = ''
            # Try specific box office selectors - get the span with the money value
            gross_text = response.xpath('//li[@data-testid="title-boxoffice-cumulativeworldwidegross"]//span[@class="ipc-metadata-list-item__list-content-item"]/text()').get('')
            if not gross_text:
                # Try finding "Gross worldwide" label and get the value from the same list item
                gross_text = response.xpath('//li[.//span[text()="Gross worldwide"]]//span[@class="ipc-metadata-list-item__list-content-item"]/text()').get('')
            if not gross_text:
                # Try getting any span after "Gross worldwide" text
                gross_text = response.xpath('//span[text()="Gross worldwide"]/parent::*/following-sibling::*//text()').get('')
            if not gross_text:
                # Fallback: look for money format in box office section
                all_money = response.xpath('//section[@data-testid="BoxOffice"]//span[contains(text(), "$")]/text()').getall()
                if all_money:
                    # Get the largest amount (likely worldwide gross)
                    amounts = []
                    for m in all_money:
                        # Extract number from formats like "$1,234,567,890"
                        match = re.search(r'\$[\d,]+', m)
                        if match:
                            num_str = match.group().replace('$', '').replace(',', '')
                            try:
                                amounts.append((int(num_str), m))
                            except:
                                pass
                    if amounts:
                        gross_text = max(amounts, key=lambda x: x[0])[1]
            data['Gross_worldwide'] = gross_text.strip() if gross_text else ''
            
            # Rating - from JSON-LD first (most reliable), then fallback to page
            rating = ''
            if movie_data and 'aggregateRating' in movie_data:
                rating_obj = movie_data.get('aggregateRating', {})
                if isinstance(rating_obj, dict):
                    rating = str(rating_obj.get('ratingValue', '')).strip()
            
            if not rating:
                # Try specific IMDb rating selector - take the first digit value
                rating_texts = response.xpath('//div[@data-testid="hero-rating-bar__aggregate-rating__score"]//text()').getall()
                # Filter to first number-like entry (should be the rating itself, not "/" or "10")
                for text in rating_texts:
                    text = text.strip()
                    if text and '.' in text and len(text) < 5:  # Rating format: "8.5"
                        rating = text
                        break
            
            if not rating:
                # Last fallback: look for rating in ratingGroup spans
                rating_texts = response.xpath('//span[@data-testid="ratingGroup--imdb-rating"]//text()').getall()
                for text in rating_texts:
                    text = text.strip()
                    if text and '.' in text and len(text) < 5:
                        rating = text
                        break
            
            data['Rating'] = rating
            
            rating_count = ''
            if movie_data and 'aggregateRating' in movie_data:
                rating_count = movie_data.get('aggregateRating', {}).get('ratingCount', '')
            if not rating_count:
                rating_count = response.xpath('//div[@data-testid="hero-rating-bar__aggregate-rating__score"]/following-sibling::div//text()').get('')
            data['Rating_Count'] = str(rating_count).strip() if rating_count else ''
            
            # Extract MPAA Certification - improved extraction
            list_of_certificate = []
            # Try direct certification link
            cert_links = response.xpath('//a[contains(@href, "certificates=US:")]/text()').getall()
            if cert_links:
                for cert_text in cert_links:
                    for cert in ['G', 'PG-13', 'PG', 'R', 'NC-17', 'GP', 'M', 'M/PG', 'X', 'UNRATED']:
                        if cert in cert_text:
                            list_of_certificate.append(cert)
                            break
            # Fallback: check certificate section
            if not list_of_certificate:
                cert_section = response.xpath('//li[contains(., "Certificate")]/following-sibling::li[1]//text()').getall()
                if cert_section:
                    cert_text = ' '.join(cert_section)
                    for cert in ['G', 'PG-13', 'PG', 'R', 'NC-17', 'GP', 'M', 'M/PG', 'X', 'UNRATED']:
                        if f'United States:{cert}' in cert_text.replace(' ', '') or f'US:{cert}' in cert_text.replace(' ', ''):
                            list_of_certificate.append(cert)
                            break
            data['ListOfCertificate'] = list(set(list_of_certificate))
            
            yield data
        except CloseSpider:
            raise
        except Exception as e:
            import traceback
            self.logger.error(f"Error parsing movie: {response.url}, Error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
    
    def errback(self, failure):
        self.logger.warning(f"Failed request: {failure.request.url}, Error: {failure.getErrorMessage()}")
