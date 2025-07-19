"""
Dream Pipeline for QuickBooks Data Processing
Implements: CSV → Clean → Embed → Cache → LLM → Validate
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import os
from pathlib import Path
import pickle
from datetime import datetime
import logging

# AI/ML imports
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Validation imports
import pandera as pa
from pandera.typing import Series

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionEmbedder:
    """Handles embedding generation and similarity search for transactions."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.transactions = []
        
    def create_embeddings(self, transactions: List[Dict]) -> np.ndarray:
        """Create embeddings for transaction descriptions."""
        texts = [f"{t.get('vendor', '')} {t.get('memo', '')} {t.get('description', '')}".strip() 
                for t in transactions]
        return self.model.encode(texts, show_progress_bar=True)
    
    def build_index(self, transactions: List[Dict]):
        """Build FAISS index for similarity search."""
        embeddings = self.create_embeddings(transactions)
        dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        self.transactions = transactions
        
        logger.info(f"Built FAISS index with {len(transactions)} transactions")
    
    def find_similar(self, transaction: Dict, threshold: float = 0.8) -> Optional[Dict]:
        """Find similar transaction in cache."""
        if self.index is None:
            return None
            
        # Create embedding for new transaction
        text = f"{transaction.get('vendor', '')} {transaction.get('memo', '')} {transaction.get('description', '')}".strip()
        embedding = self.model.encode([text])
        
        # Search for similar transactions
        distances, indices = self.index.search(embedding.astype('float32'), k=5)
        
        # Check if any are above threshold
        for distance, idx in zip(distances[0], indices[0]):
            if distance > threshold and idx < len(self.transactions):
                similar_tx = self.transactions[idx]
                logger.info(f"Cache hit: Found similar transaction with similarity {distance:.3f}")
                return similar_tx
                
        return None


class TransactionCache:
    """Manages transaction cache with persistence."""
    
    def __init__(self, cache_file: str = "transaction_cache.pkl"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load existing cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get_cache_key(self, transaction: Dict) -> str:
        """Generate cache key for transaction."""
        # Create hash from vendor, amount, and date
        key_data = f"{transaction.get('vendor', '')}_{transaction.get('amount', 0)}_{transaction.get('date', '')}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, transaction: Dict) -> Optional[Dict]:
        """Get cached transaction."""
        key = self.get_cache_key(transaction)
        return self.cache.get(key)
    
    def set(self, transaction: Dict, category: str):
        """Cache transaction with category."""
        key = self.get_cache_key(transaction)
        self.cache[key] = {
            'transaction': transaction,
            'category': category,
            'cached_at': datetime.now().isoformat()
        }
        self._save_cache()
        logger.info(f"Cached transaction with category: {category}")


class GeminiCategorizer:
    """Handles category suggestions using Gemini AI."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        
    def suggest_category(self, transaction: Dict) -> str:
        """Suggest category for a transaction using Gemini."""
        try:
            # Create prompt for Gemini
            amount = transaction.get('amount', 0)
            if amount is None:
                amount = 0
            elif isinstance(amount, str):
                try:
                    amount = float(amount)
                except:
                    amount = 0
            
            prompt = f"""
            Categorize this QuickBooks transaction into one of these standard categories:
            - Office Supplies
            - Equipment & Technology  
            - Marketing & Advertising
            - Professional Services
            - Transportation
            - Employee Benefits
            - Miscellaneous
            
            Transaction details:
            - Vendor: {transaction.get('vendor', 'Unknown')}
            - Amount: ${amount:.2f}
            - Description: {transaction.get('description', '')}
            - Memo: {transaction.get('memo', '')}
            
            Return only the category name, nothing else.
            """
            
            response = self.model.generate_content(prompt)
            category = response.text.strip()
            
            logger.info(f"Gemini suggested category '{category}' for transaction")
            return category
            
        except Exception as e:
            logger.error(f"Failed to get Gemini suggestion: {e}")
            return "Miscellaneous"


class TransactionValidator:
    """Validates transaction data using Pandera schemas."""
    
    def __init__(self):
        self.schema = pa.DataFrameSchema({
            'date': pa.Column(pa.DateTime, nullable=True),
            'vendor': pa.Column(pa.String, nullable=True),
            'amount': pa.Column(pa.Float, nullable=True),
            'category': pa.Column(pa.String, nullable=True),
            'description': pa.Column(pa.String, nullable=True),
            'memo': pa.Column(pa.String, nullable=True)
        })
    
    def validate_transaction(self, transaction: Dict) -> bool:
        """Validate a single transaction."""
        try:
            # Convert to DataFrame for validation
            df = pd.DataFrame([transaction])
            self.schema.validate(df)
            return True
        except Exception as e:
            logger.error(f"Transaction validation failed: {e}")
            return False
    
    def validate_batch(self, transactions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Validate a batch of transactions, return valid and invalid."""
        valid = []
        invalid = []
        
        for tx in transactions:
            if self.validate_transaction(tx):
                valid.append(tx)
            else:
                invalid.append(tx)
                
        logger.info(f"Validation complete: {len(valid)} valid, {len(invalid)} invalid")
        return valid, invalid


class DreamPipeline:
    """Main pipeline implementing the dream architecture."""
    
    def __init__(self, gemini_api_key: str):
        self.embedder = TransactionEmbedder()
        self.cache = TransactionCache()
        self.categorizer = GeminiCategorizer(gemini_api_key)
        self.validator = TransactionValidator()
        
    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting transaction cleaning...")
        logger.info(f"Initial columns: {list(df.columns)}")
        logger.info(f"Initial row count: {len(df)}")

        # Try to map common alternative column names
        if 'vendor__name' in df.columns and 'vendor' not in df.columns:
            df['vendor'] = df['vendor__name']
        if 'Amount' in df.columns and 'amount' not in df.columns:
            df['amount'] = df['Amount']

        # Ensure required columns exist
        required_columns = ['vendor', 'amount', 'category', 'description', 'memo', 'date']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        # Remove exact duplicates
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {before - len(df)} exact duplicates")

        # Only drop rows if vendor and amount have any non-null data
        if df['vendor'].notna().any() and df['amount'].notna().any():
            before = len(df)
            df = df.dropna(subset=['vendor', 'amount'])
            logger.info(f"Removed {before - len(df)} rows with missing vendor/amount")
        else:
            logger.info("Skipping dropna: vendor/amount columns are empty")

        # Standardize vendor names
        df['vendor'] = df['vendor'].astype(str).str.strip().str.title()

        # Only filter by amount if it is numeric
        try:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            before = len(df)
            df = df[df['amount'] > 0]
            logger.info(f"Removed {before - len(df)} rows with non-positive amount")
        except Exception as e:
            logger.warning(f"Could not convert amount to numeric: {e}")

        logger.info(f"Cleaning complete: {len(df)} rows remaining")
        return df
    
    def process_transaction(self, transaction: Dict) -> Dict:
        """Process a single transaction through the pipeline."""
        # Step 1: Check cache first
        cached_result = self.cache.get(transaction)
        if cached_result:
            logger.info("Cache hit - using cached category")
            transaction['category'] = cached_result['category']
            return transaction
        
        # Step 2: Check for similar transactions using embeddings
        similar_tx = self.embedder.find_similar(transaction)
        if similar_tx:
            logger.info("Similar transaction found - using its category")
            transaction['category'] = similar_tx.get('category', 'Miscellaneous')
            # Cache this result
            self.cache.set(transaction, transaction['category'])
            return transaction
        
        # Step 3: Use Gemini AI for new transaction
        logger.info("No cache hit - using Gemini AI")
        category = self.categorizer.suggest_category(transaction)
        transaction['category'] = category
        
        # Cache the result
        self.cache.set(transaction, category)
        
        return transaction
    
    def process_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Process a batch of transactions."""
        logger.info(f"Processing batch of {len(transactions)} transactions")
        
        # Step 1: Clean the data
        df = pd.DataFrame(transactions)
        df = self.clean_transactions(df)
        transactions = df.to_dict('records')
        
        # Step 2: Build embedding index if we have existing transactions
        if len(transactions) > 10:  # Only build index for larger datasets
            self.embedder.build_index(transactions)
        
        # Step 3: Process each transaction
        processed = []
        for i, tx in enumerate(transactions):
            logger.info(f"Processing transaction {i+1}/{len(transactions)}")
            processed_tx = self.process_transaction(tx)
            processed.append(processed_tx)
        
        # Step 4: Validate results
        valid, invalid = self.validator.validate_batch(processed)
        
        if invalid:
            logger.warning(f"Found {len(invalid)} invalid transactions")
        
        return valid
    
    def run_pipeline(self, input_file: str, output_file: str):
        """Run the complete pipeline from CSV to final output."""
        logger.info("Starting Dream Pipeline...")
        
        # Load CSV
        df = pd.read_csv(input_file)
        transactions = df.to_dict('records')
        
        # Process through pipeline
        processed = self.process_batch(transactions)
        
        # Save results
        result_df = pd.DataFrame(processed)
        result_df.to_csv(output_file, index=False)
        
        logger.info(f"Pipeline complete! Results saved to {output_file}")
        logger.info(f"Processed {len(processed)} transactions")
        
        return result_df


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DreamPipeline(gemini_api_key="YOUR_GEMINI_API_KEY")
    
    # Run pipeline
    result = pipeline.run_pipeline(
        input_file="messy_transactions.csv",
        output_file="cleaned_transactions.csv"
    )
    
    print("Pipeline completed successfully!")
    print(f"Processed {len(result)} transactions") 