# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import csv
import os
from itemadapter import ItemAdapter


class RealtimeCsvPipeline:
    """Pipeline that writes to CSV in real-time (flushes after each item)"""

    CSV_PATH = '/workspace/movie-earnings-ds/dataset/data.csv'
    FIELDS = ['Movie_ID', 'Movie_Title', 'Budget', 'Cast', 'Crew', 'Studios',
              'Genre', 'Keywords', 'Languages', 'Countries', 'Filming_Location',
              'Release_Data', 'Runtime', 'Gross_worldwide', 'Rating',
              'Rating_Count', 'ListOfCertificate']

    def open_spider(self, spider):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.CSV_PATH), exist_ok=True)
        # Open file and write header
        self.file = open(self.CSV_PATH, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS, extrasaction='ignore')
        self.writer.writeheader()
        self.file.flush()
        self.count = 0

    def close_spider(self, spider):
        self.file.close()
        spider.logger.info(f"CSV closed with {self.count} movies written to {self.CSV_PATH}")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        # Convert list fields to comma-separated strings
        if 'Cast' in adapter and isinstance(adapter['Cast'], list):
            adapter['Cast'] = ','.join(adapter['Cast'][:20])

        if 'Crew' in adapter and isinstance(adapter['Crew'], list):
            adapter['Crew'] = ','.join(adapter['Crew'][:10])

        if 'Studios' in adapter and isinstance(adapter['Studios'], list):
            adapter['Studios'] = ','.join(adapter['Studios'][:5])

        if 'Genre' in adapter and isinstance(adapter['Genre'], list):
            adapter['Genre'] = ','.join(adapter['Genre'])

        if 'Keywords' in adapter and isinstance(adapter['Keywords'], list):
            adapter['Keywords'] = ','.join(adapter['Keywords'][:15])

        if 'Languages' in adapter and isinstance(adapter['Languages'], list):
            adapter['Languages'] = ','.join(adapter['Languages'])

        if 'Countries' in adapter and isinstance(adapter['Countries'], list):
            adapter['Countries'] = ','.join(adapter['Countries'])

        if 'Filming_Location' in adapter and isinstance(adapter['Filming_Location'], list):
            adapter['Filming_Location'] = ','.join(adapter['Filming_Location'][:3])

        if 'ListOfCertificate' in adapter and isinstance(adapter['ListOfCertificate'], list):
            adapter['ListOfCertificate'] = ','.join(adapter['ListOfCertificate']) if adapter['ListOfCertificate'] else ''

        # Ensure empty strings for missing fields
        for field in self.FIELDS:
            if field not in adapter or adapter[field] is None:
                adapter[field] = ''

        # Write row and flush immediately for real-time updates
        self.writer.writerow(dict(adapter))
        self.file.flush()
        self.count += 1

        if self.count % 10 == 0:
            spider.logger.info(f"CSV Progress: {self.count} movies written")

        return item
