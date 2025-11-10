# Vibe Matcher

A simple recommendation system that matches fashion items based on their descriptions using TF-IDF vectorization and cosine similarity.

## Features

- **Local Processing**: No API keys or external services required
- **Fast Matching**: Uses efficient TF-IDF vectorization for quick similarity searches
- **Customizable**: Easy to add more products or modify the matching logic
- **Visual Feedback**: Generates latency plots for performance analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vibe-matcher.git
   cd vibe-matcher
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script with:
```bash
python vibe_matcher.py
```

### 1. Urban Street Style Search
```
Searching for: 'energetic urban street style'

Top matches:
                name similarity_score                                 tags
Urban Street Joggers            0.513 urban, streetwear, casual, energetic

Query completed in 3.18ms
```

### 2. Formal Wear Search
```
Searching for: 'elegant evening gown'

Top matches:
                 name similarity_score                                    tags
Designer Evening Gown            0.543 elegant, evening, formal, sophisticated

Query completed in 1.00ms
```

### 3. Winter Collection
```
Searching for: 'warm winter jacket'

Available winter items:
- Cozy Winter Sweater: Warm and soft oversized knit sweater
- Winter Parka Jacket: Heavy-duty winter parka with faux fur trim
```

## Performance

The system efficiently processes queries with an average response time under 5ms, as shown in the generated performance visualization:

![Query Latency Comparison](query_latencies.png)

## Customization

### Adding New Products
Edit the `_create_sample_data()` method in `vibe_matcher.py` to include your product catalog.

### Adjusting Search Sensitivity
Modify the `threshold` parameter in `find_similar_products()` to make matches more or less strict.


