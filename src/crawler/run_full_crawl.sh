#!/bin/bash
# ==============================================================================
# IMDb Data Crawler - Main Execution Script
# ==============================================================================
# This script crawls IMDb to collect movie data with all required fields
# 
# Usage:
#   bash run_full_crawl.sh [limit]
#
# Examples:
#   bash run_full_crawl.sh 50        # Crawl 50 movies
#   bash run_full_crawl.sh 500       # Crawl 500 movies
#   bash run_full_crawl.sh 5000      # Crawl 5000 movies (large dataset)
#   bash run_full_crawl.sh           # Crawl up to 10,000 movies (default max)
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get limit from argument
LIMIT=${1:-""}

# Activate conda environment
echo -e "${BLUE}Activating conda environment 'movie'...${NC}"
source /workspace/miniconda3/bin/activate movie

# Change to crawler directory
cd /workspace/movie-earnings-ds/src/crawler

# Display banner
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                IMDb Movie Data Crawler                        ║"
echo "║                                                               ║"
echo "║  Extracts 17 fields from IMDb:                               ║"
echo "║  • Movie_ID, Movie_Title, Budget, Cast, Crew                 ║"
echo "║  • Studios, Genre, Keywords, Languages, Countries            ║"
echo "║  • Filming_Location, Release_Data, Runtime                   ║"
echo "║  • Gross_worldwide, Rating, Rating_Count, ListOfCertificate  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Estimate crawl info
if [ -z "$LIMIT" ]; then
    echo -e "${YELLOW}Mode: MAXIMUM CRAWL (up to 10,000 movies)${NC}"
    echo -e "${YELLOW}Estimated Time: 3-4 hours${NC}"
    echo -e "${YELLOW}Estimated Size: ~50-100 MB CSV file${NC}"
    LIMIT_ARG=""
else
    echo -e "${YELLOW}Mode: LIMITED CRAWL ($LIMIT movies)${NC}"
    # Calculate estimated time (~ 0.5 sec per movie)
    TIME_MINS=$((LIMIT / 120))
    if [ $TIME_MINS -lt 1 ]; then
        TIME_EST="< 1 minute"
    else
        TIME_EST="~$TIME_MINS minutes"
    fi
    echo -e "${YELLOW}Estimated Time: $TIME_EST${NC}"
    LIMIT_ARG="-a limit=$LIMIT"
fi

echo ""
echo -e "${BLUE}Starting crawl...${NC}"
echo "Output will be saved to: /workspace/movie-earnings-ds/dataset/data_joined.csv"
echo ""

# Clean up old output
rm -f /workspace/movie-earnings-ds/dataset/data.csv
rm -f /workspace/movie-earnings-ds/dataset/data_joined.csv

# Run the spider
scrapy crawl full2ImdbCrawler \
    -o /workspace/movie-earnings-ds/dataset/data.csv \
    $LIMIT_ARG

# Create backup/copy
if [ -f "/workspace/movie-earnings-ds/dataset/data.csv" ]; then
    cp /workspace/movie-earnings-ds/dataset/data.csv \
       /workspace/movie-earnings-ds/dataset/data_joined.csv
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ Crawl completed successfully!      ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
    echo ""
    echo "Output files:"
    echo "  • /workspace/movie-earnings-ds/dataset/data.csv"
    echo "  • /workspace/movie-earnings-ds/dataset/data_joined.csv"
    echo ""
    
    # Show statistics
    TOTAL_ROWS=$(($(wc -l < /workspace/movie-earnings-ds/dataset/data.csv) - 1))
    FILE_SIZE=$(du -h /workspace/movie-earnings-ds/dataset/data.csv | cut -f1)
    
    echo -e "${BLUE}Statistics:${NC}"
    echo "  • Total movies: $TOTAL_ROWS"
    echo "  • File size: $FILE_SIZE"
    echo "  • Fields: 17"
    echo ""
    
    echo -e "${BLUE}Sample data (first movie):${NC}"
    head -n 2 /workspace/movie-earnings-ds/dataset/data.csv | tail -n 1 | cut -c1-100
    echo "..."
    echo ""
    
else
    echo -e "${RED}✗ Error: Crawl failed - no output file created${NC}"
    exit 1
fi
