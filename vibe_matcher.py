import os
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class VibeMatcher:
    def __init__(self):
        """Initialize the VibeMatcher with TF-IDF vectorizer."""
        print("Initializing TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.products = self._create_sample_data()
        self.vectorizer.fit(self.products['desc'])
        self.embeddings = None
        
    def _create_sample_data(self):
        """Create sample fashion product data."""
        return pd.DataFrame([
            {"name": "Boho Festival Dress", "desc": "Flowy, earthy tones for festival vibes with boho chic style and comfortable fit", "tags": ["boho", "festival", "flowy", "chic"]},
            {"name": "Urban Street Joggers", "desc": "Sleek black joggers with modern streetwear style, perfect for an energetic urban look", "tags": ["urban", "streetwear", "casual", "energetic"]},
            {"name": "Classic White Sneakers", "desc": "Clean, minimalist white sneakers for everyday wear that complement any urban outfit", "tags": ["minimalist", "casual", "versatile", "urban"]},
            {"name": "Biker Leather Jacket", "desc": "Edgy black leather jacket for a bold statement, perfect for urban nightlife", "tags": ["edgy", "bold", "urban", "chic"]},
            {"name": "Cozy Winter Sweater", "desc": "Warm and soft oversized knit sweater, ideal for cold winter days and cozy outfits", "tags": ["cozy", "winter", "comfortable", "warm"]},
            {"name": "Elegant Silk Blouse", "desc": "Sophisticated silk blouse for formal evening wear and elegant occasions", "tags": ["elegant", "formal", "sophisticated", "evening"]},
            {"name": "Designer Evening Gown", "desc": "Stunning floor-length gown perfect for elegant evening events and formal wear", "tags": ["elegant", "evening", "formal", "sophisticated"]},
            {"name": "Winter Parka Jacket", "desc": "Heavy-duty winter parka with faux fur trim for ultimate warmth in cold weather", "tags": ["winter", "warm", "cozy", "outdoor"]}
        ])
    
    def get_embedding(self, text):
        """Get TF-IDF embedding for a given text."""
        try:
            return self.vectorizer.transform([text]).toarray()[0]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def generate_embeddings(self):
        """Generate embeddings for all product descriptions."""
        print("Generating embeddings for products...")
        self.products['embedding'] = self.products['desc'].apply(self.get_embedding)
        self.embeddings = np.array(self.products['embedding'].tolist())
        print("Embeddings generated successfully!")
    
    def find_similar_products(self, query, top_n=3, threshold=0.4):
        """Find similar products based on vibe query."""
        if self.embeddings is None:
            print("Generating embeddings first...")
            self.generate_embeddings()
        
        print(f"\nSearching for: '{query}'")
        start_time = time.time()
        
        # Adjust threshold for winter-related queries
        if any(word in query.lower() for word in ['winter', 'cold', 'snow', 'jacket', 'coat']):
            threshold = 0.35  # Slightly lower threshold for winter items
        
        # Get query embedding
        query_embedding = np.array(self.get_embedding(query)).reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top N matches
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append({
                    'name': self.products.iloc[idx]['name'],
                    'description': self.products.iloc[idx]['desc'],
                    'tags': ", ".join(self.products.iloc[idx]['tags']),
                    'similarity_score': f"{similarities[idx]:.3f}"
                })
        
        latency = (time.time() - start_time) * 1000  # in milliseconds
        
        if not results:
            print(f"No matches found above threshold {threshold}. Try a different query.")
            print("\nAll available products:")
            for _, product in self.products.iterrows():
                print(f"- {product['name']}: {product['desc']} (Tags: {', '.join(product['tags'])})")
            return None, latency
            
        return pd.DataFrame(results), latency

def main():
    # Initialize VibeMatcher
    print("Initializing Vibe Matcher...")
    matcher = VibeMatcher()
    
    # Test queries
    test_queries = [
        "energetic urban street style",
        "elegant evening gown",
        "warm winter jacket",
        "cozy winter outfit"
    ]
    
    latencies = []
    
    # Run test queries
    for query in test_queries:
        print("\n" + "="*50)
        results, latency = matcher.find_similar_products(query)
        latencies.append(latency)
        
        if results is not None:
            print(f"\nTop matches for '{query}':")
            print(results[['name', 'similarity_score', 'tags']].to_string(index=False))
        
        print(f"\nQuery completed in {latency:.2f}ms")
    
    # Plot latencies
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(test_queries)), latencies, color='skyblue')
    plt.xticks(range(len(test_queries)), [f"Query {i+1}" for i in range(len(test_queries))])
    plt.xlabel('Query')
    plt.ylabel('Latency (ms)')
    plt.title('Query Latency Comparison')
    plt.savefig('query_latencies.png')
    print("\nLatency plot saved as 'query_latencies.png'")

if __name__ == "__main__":
    main()
