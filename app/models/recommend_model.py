from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
import json
import os

class Recommender:
    def __init__(self, embedding_dim=768, index_path="data/processed"):
        self.embeddings = HuggingFaceEmbeddings()
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.content_mapping = {}
        self.initialize_index()

    def initialize_index(self):
        """Initialize or load existing FAISS index"""
        index_file = os.path.join(self.index_path, "recommendations.index")
        mapping_file = os.path.join(self.index_path, "content_mapping.json")
        
        if os.path.exists(index_file) and os.path.exists(mapping_file):
            self.index = faiss.read_index(index_file)
            with open(mapping_file, 'r') as f:
                self.content_mapping = json.load(f)

    def add_items(self, items):
        """Add new items to the recommendation index"""
        if not items:
            return

        # Generate embeddings for items
        texts = [item.get('content', '') for item in items]
        embeddings = self.embeddings.embed_documents(texts)

        # Add to FAISS index
        item_ids = np.arange(len(items)) + len(self.content_mapping)
        self.index.add(np.array(embeddings))

        # Update mapping
        for id_, item in zip(item_ids, items):
            self.content_mapping[str(int(id_))] = item

        # Save updated index and mapping
        self.save_index()

    def recommend(self, query, k=5):
        """Get recommendations using FAISS similarity search"""
        if self.index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search similar items
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )

        # Get recommendations
        recommendations = []
        for idx, distance in zip(indices[0], distances[0]):
            if str(int(idx)) in self.content_mapping:
                item = self.content_mapping[str(int(idx))]
                recommendations.append({
                    'item': item,
                    'similarity_score': float(1 / (1 + distance))
                })

        return recommendations

    def save_index(self):
        """Save FAISS index and content mapping to disk"""
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
            
        faiss.write_index(self.index, os.path.join(self.index_path, "recommendations.index"))
        with open(os.path.join(self.index_path, "content_mapping.json"), 'w') as f:
            json.dump(self.content_mapping, f)
