# IMDb Movie Data Crawler

## Quick Start

```bash
cd /workspace/bor-prediction-analysis/src/crawler
bash run_full_crawl.sh [LIMIT]
```

## Examples

```bash
# Test with 10 movies (~30 seconds)
bash run_full_crawl.sh 10

# Small dataset (100 movies, ~1 minute)
bash run_full_crawl.sh 100

# Medium dataset (1000 movies, ~10 minutes)
bash run_full_crawl.sh 1000

# Large dataset (5000 movies, ~45 minutes)
bash run_full_crawl.sh 5000

# Maximum dataset (up to 10,000 movies, ~3-4 hours)
bash run_full_crawl.sh
```

## Output

**File**: `/workspace/bor-prediction-analysis/dataset/data_joined.csv`

**17 Fields**: Movie_ID, Movie_Title, Budget, Cast, Crew, Studios, Genre, Keywords, Languages, Countries, Filming_Location, Release_Data, Runtime, Gross_worldwide, Rating, Rating_Count, ListOfCertificate

## Notes

- Crawls most popular movies (ensures complete data including box office)
- Speed: ~2-3 movies per second
- Respects IMDb robots.txt with auto-throttling

