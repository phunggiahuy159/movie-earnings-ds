# IMDb Data Crawling Script

## Usage

```bash
cd /workspace/bor-prediction-analysis/src/crawler
bash run_full_crawl.sh [LIMIT]
```

## Examples

```bash
# Test with 10 movies
bash run_full_crawl.sh 10

# Scrape 100 movies
bash run_full_crawl.sh 100

# Scrape 1000 movies
bash run_full_crawl.sh 1000

# Scrape 5000 movies
bash run_full_crawl.sh 5000

# Full scrape (up to 10,000 movies)
bash run_full_crawl.sh
```

## Output

Data is saved to: `/workspace/bor-prediction-analysis/dataset/data_joined.csv`
