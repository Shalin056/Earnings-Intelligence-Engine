"""
Phase 2B: Speaker Segmentation
Separates Management vs Analyst utterances in earnings calls
"""

import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, LOGS_DIR

class SpeakerSegmenter:
    """
    Segments earnings call transcripts by speaker role
    Separates Management (executives) from Analysts (questioners)
    """
    
    def __init__(self):
        self.input_dir = PROCESSED_DATA_DIR / 'transcripts'
        self.output_dir = PROCESSED_DATA_DIR / 'transcripts'
        
        self.log_file = LOGS_DIR / 'preprocessing' / 'speaker_segmentation.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Patterns to identify speaker roles
        self.management_keywords = [
            'ceo', 'cfo', 'president', 'chief', 'executive', 'officer',
            'chairman', 'director', 'head of', 'vice president', 'vp',
            'treasurer', 'controller', 'management'
        ]
        
        self.analyst_keywords = [
            'analyst', 'research', 'investment', 'equity', 'credit',
            'securities', 'capital markets', 'managing director'
        ]
        
        self.segmentation_stats = {
            'total_processed': 0,
            'successfully_segmented': 0,
            'failed': 0,
            'avg_management_utterances': 0,
            'avg_analyst_utterances': 0,
            'avg_management_words': 0,
            'avg_analyst_words': 0
        }
    
    def identify_speaker_role(self, speaker_text):
        """
        Identify if speaker is Management or Analyst based on context
        
        Returns: 'management', 'analyst', or 'unknown'
        """
        if not speaker_text or not isinstance(speaker_text, str):
            return 'unknown'
        
        speaker_lower = speaker_text.lower()
        
        # Check for management keywords
        for keyword in self.management_keywords:
            if keyword in speaker_lower:
                return 'management'
        
        # Check for analyst keywords
        for keyword in self.analyst_keywords:
            if keyword in speaker_lower:
                return 'analyst'
        
        return 'unknown'
    
    def segment_prepared_remarks(self, prepared_remarks):
        """
        Segment prepared remarks section
        Typically all management speaking
        """
        if not prepared_remarks:
            return {
                'management_text': '',
                'analyst_text': '',
                'management_word_count': 0,
                'analyst_word_count': 0
            }
        
        # Prepared remarks are typically 100% management
        word_count = len(prepared_remarks.split())
        
        return {
            'management_text': prepared_remarks,
            'analyst_text': '',
            'management_word_count': word_count,
            'analyst_word_count': 0
        }
    
    def segment_qa_section(self, qa_section):
        """
        Segment Q&A section
        Alternates between Analysts (questions) and Management (answers)
        """
        if not qa_section:
            return {
                'management_text': '',
                'analyst_text': '',
                'management_word_count': 0,
                'analyst_word_count': 0,
                'management_utterances': 0,
                'analyst_utterances': 0
            }
        
        # Simple heuristic: Split by common Q&A patterns
        # In our sample data, we can identify by "Analyst X:" and "Management:" patterns
        
        management_parts = []
        analyst_parts = []
        
        # Split into paragraphs
        paragraphs = qa_section.split('\n\n')
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            # Check if this paragraph is a question (analyst) or answer (management)
            # Questions typically start with "Analyst", end with "?", or contain question words
            is_question = (
                para.strip().startswith('Analyst') or
                '?' in para or
                any(word in para.lower()[:100] for word in ['can you', 'could you', 'what', 'how', 'why', 'when', 'where'])
            )
            
            if is_question:
                analyst_parts.append(para)
            else:
                management_parts.append(para)
        
        # Join parts
        management_text = '\n\n'.join(management_parts)
        analyst_text = '\n\n'.join(analyst_parts)
        
        # Count words
        management_word_count = len(management_text.split())
        analyst_word_count = len(analyst_text.split())
        
        return {
            'management_text': management_text,
            'analyst_text': analyst_text,
            'management_word_count': management_word_count,
            'analyst_word_count': analyst_word_count,
            'management_utterances': len(management_parts),
            'analyst_utterances': len(analyst_parts)
        }
    
    def segment_transcript(self, transcript_dict):
        """
        Segment a complete transcript into management and analyst portions
        """
        try:
            # Get cleaned sections
            prepared_remarks = transcript_dict.get('prepared_remarks_cleaned', '')
            qa_section = transcript_dict.get('qa_section_cleaned', '')
            
            # Segment prepared remarks
            prep_segments = self.segment_prepared_remarks(prepared_remarks)
            
            # Segment Q&A
            qa_segments = self.segment_qa_section(qa_section)
            
            # Combine results
            segmented = {
                # Metadata
                'ticker': transcript_dict.get('ticker', ''),
                'company_name': transcript_dict.get('company_name', ''),
                'quarter': transcript_dict.get('quarter', ''),
                'date': transcript_dict.get('date', ''),
                'fiscal_year': transcript_dict.get('fiscal_year', 0),
                'fiscal_quarter': transcript_dict.get('fiscal_quarter', 0),
                
                # Prepared remarks segmentation
                'prepared_remarks_management_text': prep_segments['management_text'],
                'prepared_remarks_management_words': prep_segments['management_word_count'],
                
                # Q&A segmentation
                'qa_management_text': qa_segments['management_text'],
                'qa_analyst_text': qa_segments['analyst_text'],
                'qa_management_words': qa_segments['management_word_count'],
                'qa_analyst_words': qa_segments['analyst_word_count'],
                'qa_management_utterances': qa_segments.get('management_utterances', 0),
                'qa_analyst_utterances': qa_segments.get('analyst_utterances', 0),
                
                # Combined totals
                'total_management_text': prep_segments['management_text'] + '\n\n' + qa_segments['management_text'],
                'total_analyst_text': qa_segments['analyst_text'],
                'total_management_words': prep_segments['management_word_count'] + qa_segments['management_word_count'],
                'total_analyst_words': qa_segments['analyst_word_count'],
                
                # Ratios
                'management_analyst_word_ratio': 0,
                'segmentation_timestamp': datetime.now().isoformat()
            }
            
            # Calculate ratio
            total_words = segmented['total_management_words'] + segmented['total_analyst_words']
            if segmented['total_analyst_words'] > 0:
                segmented['management_analyst_word_ratio'] = segmented['total_management_words'] / segmented['total_analyst_words']
            elif total_words > 0:
                segmented['management_analyst_word_ratio'] = float('inf')  # Only management spoke
            
            return segmented
            
        except Exception as e:
            self._log_error(f"Error segmenting transcript {transcript_dict.get('ticker', 'UNKNOWN')}: {e}")
            return None
    
    def segment_all_transcripts(self):
        """
        Segment all cleaned transcripts
        """
        print("="*60)
        print("PHASE 2B: SPEAKER SEGMENTATION")
        print("="*60)
        print()
        
        # Load cleaned transcripts
        transcripts_file = self.input_dir / 'transcripts_cleaned.json'
        
        if not transcripts_file.exists():
            print(f"❌ Cleaned transcripts not found: {transcripts_file}")
            print("   Please run Phase 2A first.")
            return None
        
        print(f"📂 Loading cleaned transcripts from: {transcripts_file}")
        
        with open(transcripts_file, 'r', encoding='utf-8') as f:
            cleaned_transcripts = json.load(f)
        
        print(f"✅ Loaded {len(cleaned_transcripts)} cleaned transcripts")
        print()
        
        # Segment each transcript
        segmented_transcripts = []
        
        print("🗣️  Segmenting speakers...")
        for transcript in tqdm(cleaned_transcripts, desc="Segmenting"):
            segmented = self.segment_transcript(transcript)
            
            if segmented:
                segmented_transcripts.append(segmented)
                self.segmentation_stats['successfully_segmented'] += 1
                
                # Update running averages
                self.segmentation_stats['avg_management_words'] += segmented['total_management_words']
                self.segmentation_stats['avg_analyst_words'] += segmented['total_analyst_words']
            else:
                self.segmentation_stats['failed'] += 1
            
            self.segmentation_stats['total_processed'] += 1
        
        print()
        
        # Calculate final averages
        if segmented_transcripts:
            count = len(segmented_transcripts)
            self.segmentation_stats['avg_management_words'] /= count
            self.segmentation_stats['avg_analyst_words'] /= count
            
            # Count utterances
            total_mgmt_utterances = sum(t.get('qa_management_utterances', 0) for t in segmented_transcripts)
            total_analyst_utterances = sum(t.get('qa_analyst_utterances', 0) for t in segmented_transcripts)
            
            self.segmentation_stats['avg_management_utterances'] = total_mgmt_utterances / count
            self.segmentation_stats['avg_analyst_utterances'] = total_analyst_utterances / count
        
        # Display summary
        self._display_summary()
        
        return segmented_transcripts
    
    def save_segmented_transcripts(self, segmented_transcripts):
        """
        Save segmented transcripts
        """
        if not segmented_transcripts:
            print("❌ No segmented transcripts to save")
            return False
        
        try:
            print()
            print("="*60)
            print("💾 SAVING SEGMENTED TRANSCRIPTS")
            print("="*60)
            print()
            
            # Save as JSON (full data)
            json_path = self.output_dir / 'transcripts_segmented.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(segmented_transcripts, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON saved: {json_path}")
            print(f"   Size: {json_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save metadata/stats as CSV
            metadata_fields = [
                'ticker', 'company_name', 'quarter', 'date',
                'total_management_words', 'total_analyst_words',
                'management_analyst_word_ratio',
                'qa_management_utterances', 'qa_analyst_utterances'
            ]
            
            df = pd.DataFrame(segmented_transcripts)[metadata_fields]
            csv_path = self.output_dir / 'transcripts_segmented_metadata.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"✅ Metadata CSV saved: {csv_path}")
            
            # Save segmentation statistics
            stats_path = self.output_dir / 'segmentation_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(self.segmentation_stats, f, indent=2)
            
            print(f"✅ Statistics saved: {stats_path}")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving segmented transcripts: {e}")
            return False
    
    def _display_summary(self):
        """Display segmentation summary"""
        print("="*60)
        print("📊 SEGMENTATION SUMMARY")
        print("="*60)
        print()
        print(f"Total processed:         {self.segmentation_stats['total_processed']}")
        print(f"Successfully segmented:  {self.segmentation_stats['successfully_segmented']}")
        print(f"Failed:                  {self.segmentation_stats['failed']}")
        print()
        print(f"Avg management words:    {self.segmentation_stats['avg_management_words']:.0f}")
        print(f"Avg analyst words:       {self.segmentation_stats['avg_analyst_words']:.0f}")
        print(f"Avg management/analyst:  {self.segmentation_stats['avg_management_words'] / max(self.segmentation_stats['avg_analyst_words'], 1):.2f}x")
        print()
        print(f"Avg management utterances: {self.segmentation_stats['avg_management_utterances']:.1f}")
        print(f"Avg analyst utterances:    {self.segmentation_stats['avg_analyst_utterances']:.1f}")
        print()
    
    def _log_error(self, message):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")
    
    def run(self):
        """
        Execute complete speaker segmentation workflow
        """
        # Segment transcripts
        segmented_transcripts = self.segment_all_transcripts()
        
        if segmented_transcripts is None:
            print("❌ Segmentation failed")
            return False
        
        # Save segmented transcripts
        success = self.save_segmented_transcripts(segmented_transcripts)
        
        if success:
            print("="*60)
            print("✅ PHASE 2B COMPLETE")
            print("="*60)
            print()
            print(f"📁 Segmented data saved to: {self.output_dir}")
            print(f"📝 Log saved to: {self.log_file}")
            print()
            print("✅ Ready for Phase 2C: Financial Data Normalization")
            print()
        
        return success

def main():
    """Main execution"""
    segmenter = SpeakerSegmenter()
    success = segmenter.run()
    return success

if __name__ == "__main__":
    main()