# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class ImdbcrawlerPipeline:
    """Pipeline to format scraped data to match data_joined.csv format"""
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        # Convert list fields to comma-separated strings
        # Cast: list -> "Actor1,Actor2,Actor3"
        if 'Cast' in adapter and isinstance(adapter['Cast'], list):
            adapter['Cast'] = ','.join(adapter['Cast'][:20])  # Limit to 20 cast members
        
        # Crew: list -> "Director1,Writer1,Writer2"
        if 'Crew' in adapter and isinstance(adapter['Crew'], list):
            adapter['Crew'] = ','.join(adapter['Crew'][:10])  # Limit to 10 crew members
        
        # Studios: list -> "Studio1,Studio2"
        if 'Studios' in adapter and isinstance(adapter['Studios'], list):
            adapter['Studios'] = ','.join(adapter['Studios'][:5])
        
        # Genre: list -> "Action,Adventure,Sci-Fi"
        if 'Genre' in adapter and isinstance(adapter['Genre'], list):
            adapter['Genre'] = ','.join(adapter['Genre'])
        
        # Keywords: list -> "keyword1,keyword2,keyword3"
        if 'Keywords' in adapter and isinstance(adapter['Keywords'], list):
            adapter['Keywords'] = ','.join(adapter['Keywords'][:15])
        
        # Languages: list -> "English,French"
        if 'Languages' in adapter and isinstance(adapter['Languages'], list):
            adapter['Languages'] = ','.join(adapter['Languages'])
        
        # Countries: list -> "United States,United Kingdom"
        if 'Countries' in adapter and isinstance(adapter['Countries'], list):
            adapter['Countries'] = ','.join(adapter['Countries'])
        
        # Filming_Location: list -> "Location1,Location2"
        if 'Filming_Location' in adapter and isinstance(adapter['Filming_Location'], list):
            adapter['Filming_Location'] = ','.join(adapter['Filming_Location'][:3])
        
        # ListOfCertificate: list -> "PG-13" (usually just one)
        if 'ListOfCertificate' in adapter and isinstance(adapter['ListOfCertificate'], list):
            adapter['ListOfCertificate'] = ','.join(adapter['ListOfCertificate']) if adapter['ListOfCertificate'] else ''
        
        # Ensure empty strings for missing fields
        for field in ['Movie_ID', 'Movie_Title', 'Budget', 'Cast', 'Crew', 'Studios', 
                      'Genre', 'Keywords', 'Languages', 'Countries', 'Filming_Location',
                      'Release_Data', 'Runtime', 'Gross_worldwide', 'Rating', 
                      'Rating_Count', 'ListOfCertificate']:
            if field not in adapter or adapter[field] is None:
                adapter[field] = ''
        
        return item
