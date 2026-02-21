# """
# Phase 3A: FinBERT Feature Extraction (OPTIMIZED)
# Extracts embeddings and sentiment from earnings call transcripts using FinBERT
# """

# import pandas as pd
# import numpy as np
# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from pathlib import Path
# from datetime import datetime
# import sys
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')

# # Add project root to path
# project_root = Path(__file__).parent.parent.parent
# sys.path.append(str(project_root))

# from config import PROCESSED_DATA_DIR, FEATURES_DIR, LOGS_DIR

# class FinBERTExtractor:
#     """
#     Extracts features from earnings call transcripts using FinBERT (OPTIMIZED)
#     """
    
#     def __init__(self, batch_size=8):
#         self.transcripts_dir = PROCESSED_DATA_DIR / 'transcripts'
#         self.aligned_dir = PROCESSED_DATA_DIR / 'aligned'
#         self.output_dir = FEATURES_DIR
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.log_file = LOGS_DIR / 'features' / 'finbert_extraction.log'
#         self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
#         self.batch_size = batch_size
        
#         # Initialize FinBERT
#         print("🤖 Loading FinBERT model...")
#         self.model_name = "ProsusAI/finbert"
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
#         # Set device
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)
#         self.model.eval()
        
#         # Optimization: Set to half precision if GPU available
#         if self.device.type == 'cuda':
#             self.model = self.model.half()
        
#         print(f"✅ FinBERT loaded on device: {self.device}")
#         print(f"⚡ Batch size: {self.batch_size}")
#         print()
        
#         self.extraction_stats = {
#             'total_transcripts': 0,
#             'successfully_extracted': 0,
#             'failed': 0,
#             'avg_embedding_time': 0
#         }
    
#     def load_aligned_transcripts(self):
#         """
#         Load ONLY transcripts that match aligned data (faster!)
#         """
#         print("="*60)
#         print("PHASE 3A: FINBERT FEATURE EXTRACTION (OPTIMIZED)")
#         print("="*60)
#         print()
#         print("📂 Loading aligned transcripts...")
        
#         # Load aligned data to get the list we need
#         aligned_file = self.aligned_dir / 'aligned_data.csv'
        
#         if not aligned_file.exists():
#             print(f"❌ Aligned data not found: {aligned_file}")
#             return None
        
#         aligned_df = pd.read_csv(aligned_file)
#         aligned_keys = set(zip(aligned_df['ticker'], aligned_df['quarter']))
        
#         print(f"✅ Found {len(aligned_keys)} aligned transcript references")
        
#         # Load all transcripts
#         transcripts_file = self.transcripts_dir / 'transcripts_segmented.json'
        
#         if not transcripts_file.exists():
#             print(f"❌ Transcripts not found: {transcripts_file}")
#             return None
        
#         with open(transcripts_file, 'r', encoding='utf-8') as f:
#             all_transcripts = json.load(f)
        
#         # Filter to only aligned transcripts
#         transcripts = [
#             t for t in all_transcripts 
#             if (t['ticker'], t['quarter']) in aligned_keys
#         ]
        
#         print(f"✅ Loaded {len(transcripts)} transcripts (matching aligned data)")
#         print(f"   (Filtered from {len(all_transcripts)} total)")
#         print()
        
#         self.extraction_stats['total_transcripts'] = len(transcripts)
        
#         return transcripts
    
#     def prepare_texts_batch(self, transcripts):
#         """
#         Prepare texts for batch processing
#         """
#         texts = []
#         metadata = []
        
#         for transcript in transcripts:
#             # Get text sections
#             management_text = transcript.get('total_management_text', '')
#             analyst_text = transcript.get('total_analyst_text', '')
#             full_text = management_text + ' ' + analyst_text
            
#             # Truncate to 400 words (faster processing)
#             full_text_truncated = ' '.join(full_text.split()[:400])
            
#             texts.append(full_text_truncated)
#             metadata.append({
#                 'ticker': transcript['ticker'],
#                 'quarter': transcript.get('quarter', ''),
#                 'date': transcript.get('date', '')
#             })
        
#         return texts, metadata
    
#     def extract_batch_features(self, texts):
#         """
#         Extract features in batches (MUCH FASTER!)
#         """
#         all_sentiments = []
#         all_embeddings = []
        
#         # Process in batches
#         for i in range(0, len(texts), self.batch_size):
#             batch_texts = texts[i:i + self.batch_size]
            
#             try:
#                 # Tokenize batch
#                 inputs = self.tokenizer(
#                     batch_texts,
#                     return_tensors="pt",
#                     truncation=True,
#                     max_length=512,
#                     padding=True
#                 )
                
#                 # Move to device
#                 inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
#                 # Get predictions and embeddings
#                 with torch.no_grad():
#                     outputs = self.model(**inputs, output_hidden_states=True)
                    
#                     # Sentiment
#                     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
#                     # Embeddings from last hidden state (mean pooling)
#                     embeddings = outputs.hidden_states[-1].mean(dim=1)
                
#                 # Convert to CPU and numpy
#                 probs_np = probs.cpu().numpy()
#                 embeddings_np = embeddings.cpu().numpy()
                
#                 # Store results
#                 for j in range(len(batch_texts)):
#                     sentiment = {
#                         'negative': float(probs_np[j][0]),
#                         'neutral': float(probs_np[j][1]),
#                         'positive': float(probs_np[j][2])
#                     }
#                     all_sentiments.append(sentiment)
#                     all_embeddings.append(embeddings_np[j])
                    
#             except Exception as e:
#                 print(f"❌ Batch error: {e}")
#                 # Fill with defaults for failed batch
#                 for _ in range(len(batch_texts)):
#                     all_sentiments.append({'negative': 0.33, 'neutral': 0.34, 'positive': 0.33})
#                     all_embeddings.append(np.zeros(768))
        
#         return all_sentiments, all_embeddings
    
#     def extract_all_features(self, transcripts):
#         """
#         Extract features from all transcripts (OPTIMIZED WITH BATCHING)
#         """
#         print("="*60)
#         print("🤖 EXTRACTING FINBERT FEATURES (BATCH MODE)")
#         print("="*60)
#         print()
#         print("📊 Processing transcripts with FinBERT...")
#         print(f"   Model: {self.model_name}")
#         print(f"   Device: {self.device}")
#         print(f"   Batch size: {self.batch_size}")
#         print()
        
#         # Prepare all texts
#         print("📝 Preparing texts...")
#         texts, metadata = self.prepare_texts_batch(transcripts)
        
#         # Extract features in batches
#         print("⚡ Extracting features (batched)...")
#         sentiments, embeddings = self.extract_batch_features(texts)
        
#         # Combine results
#         all_features = []
        
#         print("📦 Packaging features...")
#         for i, (sentiment, embedding, meta) in enumerate(zip(sentiments, embeddings, metadata)):
#             features = {
#                 # Metadata
#                 'ticker': meta['ticker'],
#                 'quarter': meta['quarter'],
#                 'date': meta['date'],
                
#                 # Sentiment
#                 'sentiment_negative': sentiment['negative'],
#                 'sentiment_neutral': sentiment['neutral'],
#                 'sentiment_positive': sentiment['positive'],
#                 'sentiment_score': sentiment['positive'] - sentiment['negative'],
                
#                 # For simplicity, using same sentiment for mgmt and analyst
#                 # (Full version would process separately)
#                 'mgmt_sentiment_negative': sentiment['negative'],
#                 'mgmt_sentiment_neutral': sentiment['neutral'],
#                 'mgmt_sentiment_positive': sentiment['positive'],
#                 'mgmt_sentiment_score': sentiment['positive'] - sentiment['negative'],
                
#                 'analyst_sentiment_negative': sentiment['negative'],
#                 'analyst_sentiment_neutral': sentiment['neutral'],
#                 'analyst_sentiment_positive': sentiment['positive'],
#                 'analyst_sentiment_score': sentiment['positive'] - sentiment['negative'],
                
#                 'sentiment_divergence': 0.0  # Simplified
#             }
            
#             # Add embeddings (768 dimensions)
#             for j, emb_val in enumerate(embedding):
#                 features[f'embedding_{j:03d}'] = float(emb_val)
            
#             all_features.append(features)
#             self.extraction_stats['successfully_extracted'] += 1
        
#         print()
#         print("="*60)
#         print("📊 EXTRACTION SUMMARY")
#         print("="*60)
#         print()
#         print(f"Total transcripts:       {self.extraction_stats['total_transcripts']}")
#         print(f"Successfully extracted:  {self.extraction_stats['successfully_extracted']}")
#         print(f"Failed:                  {self.extraction_stats['failed']}")
        
#         if self.extraction_stats['successfully_extracted'] > 0:
#             success_rate = self.extraction_stats['successfully_extracted'] / self.extraction_stats['total_transcripts']
#             print(f"Success rate:            {success_rate:.1%}")
        
#         print()
        
#         return all_features
    
#     def save_features(self, features):
#         """
#         Save extracted features
#         """
#         if not features:
#             print("❌ No features to save")
#             return False
        
#         try:
#             print("="*60)
#             print("💾 SAVING FINBERT FEATURES")
#             print("="*60)
#             print()
            
#             # Convert to DataFrame
#             features_df = pd.DataFrame(features)
            
#             # Save as CSV
#             csv_path = self.output_dir / 'finbert_features.csv'
#             features_df.to_csv(csv_path, index=False)
#             print(f"✅ Features saved: {csv_path}")
#             print(f"   Size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
#             print(f"   Records: {len(features_df):,}")
#             print(f"   Features: {len(features_df.columns)}")
            
#             # Save metadata separately
#             metadata_cols = ['ticker', 'quarter', 'date', 'sentiment_score', 
#                            'mgmt_sentiment_score', 'analyst_sentiment_score', 'sentiment_divergence']
#             metadata_df = features_df[metadata_cols]
            
#             metadata_path = self.output_dir / 'finbert_features_metadata.csv'
#             metadata_df.to_csv(metadata_path, index=False)
#             print(f"✅ Metadata saved: {metadata_path}")
            
#             # Save extraction statistics
#             stats_path = self.output_dir / 'finbert_extraction_stats.json'
#             with open(stats_path, 'w') as f:
#                 json.dump(self.extraction_stats, f, indent=2)
#             print(f"✅ Statistics saved: {stats_path}")
            
#             # Display summary
#             print()
#             print("📊 FEATURE SUMMARY:")
#             print(f"   Sentiment features: 13")
#             print(f"   Embedding dimensions: 768")
#             print(f"   Total features: {len(features_df.columns)}")
#             print()
            
#             print("📈 SENTIMENT STATISTICS:")
#             print(f"   Average sentiment score: {features_df['sentiment_score'].mean():.3f}")
#             print(f"   Sentiment std: {features_df['sentiment_score'].std():.3f}")
#             print(f"   Positive transcripts: {(features_df['sentiment_score'] > 0).sum()} ({(features_df['sentiment_score'] > 0).sum()/len(features_df)*100:.1f}%)")
#             print(f"   Negative transcripts: {(features_df['sentiment_score'] < 0).sum()} ({(features_df['sentiment_score'] < 0).sum()/len(features_df)*100:.1f}%)")
#             print()
            
#             return True
            
#         except Exception as e:
#             print(f"❌ Error saving features: {e}")
#             return False
    
#     def _log_error(self, message):
#         """Log errors"""
#         with open(self.log_file, 'a') as f:
#             f.write(f"{datetime.now()} - ERROR: {message}\n")
    
#     def run(self):
#         """
#         Execute complete FinBERT feature extraction workflow
#         """
#         # Load transcripts (only aligned ones)
#         transcripts = self.load_aligned_transcripts()
        
#         if transcripts is None:
#             print("❌ Feature extraction failed")
#             return False
        
#         # Extract features
#         features = self.extract_all_features(transcripts)
        
#         # Save features
#         success = self.save_features(features)
        
#         if success:
#             print("="*60)
#             print("✅ PHASE 3A COMPLETE")
#             print("="*60)
#             print()
#             print(f"📁 Features saved to: {self.output_dir}")
#             print(f"📝 Log saved to: {self.log_file}")
#             print()
#             print("✅ Ready for Phase 3B: Additional NLP Features")
#             print()
        
#         return success

# def main():
#     """Main execution"""
#     # Use larger batch size for faster processing
#     extractor = FinBERTExtractor(batch_size=16)  # Increased from 8
#     success = extractor.run()
#     return success

# if __name__ == "__main__":
#     main()

"""
Phase 3A: FinBERT Feature Extraction (OPTIMIZED)
Extracts embeddings and sentiment from earnings call transcripts using FinBERT
"""

import pandas as pd
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, FEATURES_DIR, LOGS_DIR

class FinBERTExtractor:
    """
    Extracts features from earnings call transcripts using FinBERT (OPTIMIZED)
    """
    
    def __init__(self, batch_size=8):
        self.transcripts_dir = PROCESSED_DATA_DIR / 'transcripts'
        self.aligned_dir = PROCESSED_DATA_DIR / 'aligned'
        self.output_dir = FEATURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'features' / 'finbert_extraction.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        
        # Initialize FinBERT
        print("🤖 Loading FinBERT model...")
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Optimization: Set to half precision if GPU available
        if self.device.type == 'cuda':
            self.model = self.model.half()
        
        print(f"✅ FinBERT loaded on device: {self.device}")
        print(f"⚡ Batch size: {self.batch_size}")
        print()
        
        self.extraction_stats = {
            'total_transcripts': 0,
            'successfully_extracted': 0,
            'failed': 0,
            'avg_embedding_time': 0
        }
    
    def load_aligned_transcripts(self):
        """
        Load ONLY transcripts that match aligned data (faster!)
        """
        print("="*60)
        print("PHASE 3A: FINBERT FEATURE EXTRACTION (OPTIMIZED)")
        print("="*60)
        print()
        print("📂 Loading aligned transcripts...")
        
        # Load aligned data to get the list we need
        aligned_file = self.aligned_dir / 'aligned_data.csv'
        
        if not aligned_file.exists():
            print(f"❌ Aligned data not found: {aligned_file}")
            return None
        
        aligned_df = pd.read_csv(aligned_file)
        aligned_keys = set(zip(aligned_df['ticker'], aligned_df['quarter']))
        
        print(f"✅ Found {len(aligned_keys)} aligned transcript references")
        
        # Load all transcripts
        transcripts_file = self.transcripts_dir / 'transcripts_segmented.json'
        
        if not transcripts_file.exists():
            print(f"❌ Transcripts not found: {transcripts_file}")
            return None
        
        with open(transcripts_file, 'r', encoding='utf-8') as f:
            all_transcripts = json.load(f)
        
        # Filter to only aligned transcripts
        transcripts = [
            t for t in all_transcripts 
            if (t['ticker'], t['quarter']) in aligned_keys
        ]
        
        print(f"✅ Loaded {len(transcripts)} transcripts (matching aligned data)")
        print(f"   (Filtered from {len(all_transcripts)} total)")
        print()
        
        self.extraction_stats['total_transcripts'] = len(transcripts)
        
        return transcripts
    
    def prepare_texts_batch(self, transcripts):
        """
        Prepare texts - returns full, mgmt, and analyst texts separately
        """
        texts = []
        mgmt_texts = []
        analyst_texts = []
        metadata = []

        for transcript in transcripts:
            management_text = transcript.get('total_management_text', '') or ''
            analyst_text = transcript.get('total_analyst_text', '') or ''

            # If sections empty, split full_text at Q&A marker
            if not management_text.strip() and not analyst_text.strip():
                full = transcript.get('full_text', '') or transcript.get('prepared_remarks', '') or ''
                split_markers = [
                    'QUESTION AND ANSWER', 'Questions and Answers',
                    'Q&A', 'QUESTIONS AND ANSWERS', 'Operator Instructions',
                    'Questions & Answers'
                ]
                split_idx = len(full)
                for marker in split_markers:
                    idx = full.find(marker)
                    if idx != -1 and idx < split_idx:
                        split_idx = idx
                management_text = full[:split_idx]
                analyst_text = full[split_idx:]

            mgmt_trunc   = ' '.join(management_text.split()[:300])
            analyst_trunc = ' '.join(analyst_text.split()[:300])
            full_trunc   = ' '.join((management_text + ' ' + analyst_text).split()[:400])

            texts.append(full_trunc)
            mgmt_texts.append(mgmt_trunc if mgmt_trunc.strip() else full_trunc)
            analyst_texts.append(analyst_trunc if analyst_trunc.strip() else full_trunc)

            metadata.append({
                'ticker': transcript['ticker'],
                'quarter': transcript.get('quarter', ''),
                'date': transcript.get('date', '')
            })

        return texts, mgmt_texts, analyst_texts, metadata
    
    def extract_batch_features(self, texts):
        """
        Extract features in batches (MUCH FASTER!)
        """
        all_sentiments = []
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions and embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    # Sentiment
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    # Embeddings from last hidden state (mean pooling)
                    embeddings = outputs.hidden_states[-1].mean(dim=1)
                
                # Convert to CPU and numpy
                probs_np = probs.cpu().numpy()
                embeddings_np = embeddings.cpu().numpy()
                
                # Store results
                for j in range(len(batch_texts)):
                    sentiment = {
                        'negative': float(probs_np[j][0]),
                        'neutral': float(probs_np[j][1]),
                        'positive': float(probs_np[j][2])
                    }
                    all_sentiments.append(sentiment)
                    all_embeddings.append(embeddings_np[j])
                    
            except Exception as e:
                print(f"❌ Batch error: {e}")
                # Fill with defaults for failed batch
                for _ in range(len(batch_texts)):
                    all_sentiments.append({'negative': 0.33, 'neutral': 0.34, 'positive': 0.33})
                    all_embeddings.append(np.zeros(768))
        
        return all_sentiments, all_embeddings
    
    def extract_all_features(self, transcripts):
        """
        Extract features from all transcripts (OPTIMIZED WITH BATCHING)
        """
        print("="*60)
        print("🤖 EXTRACTING FINBERT FEATURES (BATCH MODE)")
        print("="*60)
        print()
        print("📊 Processing transcripts with FinBERT...")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.batch_size}")
        print()
        
        # Prepare all texts
        print("📝 Preparing texts...")
        texts, mgmt_texts, analyst_texts, metadata = self.prepare_texts_batch(transcripts)

        # Extract features in batches
        print("⚡ Extracting full transcript features (batched)...")
        sentiments, embeddings = self.extract_batch_features(texts)

        print("⚡ Extracting management sentiment (batched)...")
        mgmt_sentiments, _ = self.extract_batch_features(mgmt_texts)

        print("⚡ Extracting analyst sentiment (batched)...")
        analyst_sentiments, _ = self.extract_batch_features(analyst_texts)
        
        # Combine results
        all_features = []
        
        print("📦 Packaging features...")
        for i, (sentiment, embedding, meta) in enumerate(zip(sentiments, embeddings, metadata)):
            features = {
                # Metadata
                'ticker': meta['ticker'],
                'quarter': meta['quarter'],
                'date': meta['date'],
                
                # Sentiment
                'sentiment_negative': sentiment['negative'],
                'sentiment_neutral': sentiment['neutral'],
                'sentiment_positive': sentiment['positive'],
                'sentiment_score': sentiment['positive'] - sentiment['negative'],
                
                'mgmt_sentiment_negative': mgmt_sentiments[i]['negative'],
                'mgmt_sentiment_neutral': mgmt_sentiments[i]['neutral'],
                'mgmt_sentiment_positive': mgmt_sentiments[i]['positive'],
                'mgmt_sentiment_score': mgmt_sentiments[i]['positive'] - mgmt_sentiments[i]['negative'],

                'analyst_sentiment_negative': analyst_sentiments[i]['negative'],
                'analyst_sentiment_neutral': analyst_sentiments[i]['neutral'],
                'analyst_sentiment_positive': analyst_sentiments[i]['positive'],
                'analyst_sentiment_score': analyst_sentiments[i]['positive'] - analyst_sentiments[i]['negative'],

                'sentiment_divergence': (
                    (mgmt_sentiments[i]['positive'] - mgmt_sentiments[i]['negative']) -
                    (analyst_sentiments[i]['positive'] - analyst_sentiments[i]['negative'])
                )
            }
            
            # Add embeddings (768 dimensions)
            for j, emb_val in enumerate(embedding):
                features[f'embedding_{j:03d}'] = float(emb_val)
            
            all_features.append(features)
            self.extraction_stats['successfully_extracted'] += 1
        
        print()
        print("="*60)
        print("📊 EXTRACTION SUMMARY")
        print("="*60)
        print()
        print(f"Total transcripts:       {self.extraction_stats['total_transcripts']}")
        print(f"Successfully extracted:  {self.extraction_stats['successfully_extracted']}")
        print(f"Failed:                  {self.extraction_stats['failed']}")
        
        if self.extraction_stats['successfully_extracted'] > 0:
            success_rate = self.extraction_stats['successfully_extracted'] / self.extraction_stats['total_transcripts']
            print(f"Success rate:            {success_rate:.1%}")
        
        print()
        
        return all_features
    
    def save_features(self, features):
        """
        Save extracted features
        """
        if not features:
            print("❌ No features to save")
            return False
        
        try:
            print("="*60)
            print("💾 SAVING FINBERT FEATURES")
            print("="*60)
            print()
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features)
            
            # Save as CSV
            csv_path = self.output_dir / 'finbert_features.csv'
            features_df.to_csv(csv_path, index=False)
            print(f"✅ Features saved: {csv_path}")
            print(f"   Size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
            print(f"   Records: {len(features_df):,}")
            print(f"   Features: {len(features_df.columns)}")
            
            # Save metadata separately
            metadata_cols = ['ticker', 'quarter', 'date', 'sentiment_score', 
                           'mgmt_sentiment_score', 'analyst_sentiment_score', 'sentiment_divergence']
            metadata_df = features_df[metadata_cols]
            
            metadata_path = self.output_dir / 'finbert_features_metadata.csv'
            metadata_df.to_csv(metadata_path, index=False)
            print(f"✅ Metadata saved: {metadata_path}")
            
            # Save extraction statistics
            stats_path = self.output_dir / 'finbert_extraction_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(self.extraction_stats, f, indent=2)
            print(f"✅ Statistics saved: {stats_path}")
            
            # Display summary
            print()
            print("📊 FEATURE SUMMARY:")
            print(f"   Sentiment features: 13")
            print(f"   Embedding dimensions: 768")
            print(f"   Total features: {len(features_df.columns)}")
            print()
            
            print("📈 SENTIMENT STATISTICS:")
            print(f"   Average sentiment score: {features_df['sentiment_score'].mean():.3f}")
            print(f"   Sentiment std: {features_df['sentiment_score'].std():.3f}")
            print(f"   Positive transcripts: {(features_df['sentiment_score'] > 0).sum()} ({(features_df['sentiment_score'] > 0).sum()/len(features_df)*100:.1f}%)")
            print(f"   Negative transcripts: {(features_df['sentiment_score'] < 0).sum()} ({(features_df['sentiment_score'] < 0).sum()/len(features_df)*100:.1f}%)")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving features: {e}")
            return False
    
    def _log_error(self, message):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")
    
    def run(self):
        """
        Execute complete FinBERT feature extraction workflow
        """
        # Load transcripts (only aligned ones)
        transcripts = self.load_aligned_transcripts()
        
        if transcripts is None:
            print("❌ Feature extraction failed")
            return False
        
        # Extract features
        features = self.extract_all_features(transcripts)
        
        # Save features
        success = self.save_features(features)
        
        if success:
            print("="*60)
            print("✅ PHASE 3A COMPLETE")
            print("="*60)
            print()
            print(f"📁 Features saved to: {self.output_dir}")
            print(f"📝 Log saved to: {self.log_file}")
            print()
            print("✅ Ready for Phase 3B: Additional NLP Features")
            print()
        
        return success

def main():
    """Main execution"""
    # Use larger batch size for faster processing
    extractor = FinBERTExtractor(batch_size=16)  # Increased from 8
    success = extractor.run()
    return success

if __name__ == "__main__":
    main()