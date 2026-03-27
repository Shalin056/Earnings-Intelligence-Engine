"""
Phase 2B: Speaker Segmentation
Segments transcripts by speaker role (Management vs Analysts)
ROBUST VERSION - Handles real transcript formats

FIXES APPLIED:
1. Improved Q&A section detection with more patterns
2. Better fallback when no Q&A section found
3. Enhanced analyst extraction from Q&A
4. Question-based extraction as last resort
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, LOGS_DIR


class SpeakerSegmenter:
    """
    Segments earnings call transcripts by speaker role
    ROBUST: Handles multiple real-world transcript formats
    """
    
    def __init__(self):
        self.input_dir = PROCESSED_DATA_DIR / 'transcripts'
        self.output_dir = PROCESSED_DATA_DIR / 'transcripts'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'preprocessing' / 'speaker_segmentation.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.segmentation_stats = {
            'total_processed': 0,
            'successfully_segmented': 0,
            'failed': 0,
            'avg_management_words': 0,
            'avg_analyst_words': 0,
            'avg_ratio': 0,
            'fallback_used': 0,  # Track how many used fallback segmentation
        }
        
        # ROBUST PATTERNS for real transcripts
        self.management_patterns = [
            # Executive titles
            r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–—]\s*(?:CEO|CFO|COO|CTO|President|Chairman|Chief)',
            r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+),\s*(?:CEO|CFO|COO|President|Chairman)',
            
            # Operator/Moderator
            r'(?:^|\n)(Operator):',
            r'(?:^|\n)(Moderator):',
            
            # Section headers
            r'Prepared Remarks?:?',
            r'Management Discussion:?',
            r'Opening Remarks?:?',
            r'Company Representatives?:?',
            
            # Generic management speaker
            r'(?:^|\n)(Management):',
            r'(?:^|\n)(Executive):',
        ]
        
        self.analyst_patterns = [
            # Analyst with firm
            r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–—]\s*(?:Analyst|Morgan Stanley|Goldman Sachs|JP Morgan|Bank of America|Citi|Barclays|UBS|Credit Suisse|Deutsche Bank|Wells Fargo|RBC|BMO)',
            r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+),\s*(?:Analyst)',
            
            # Q&A section markers
            r'Q&A|Question-and-Answer|Questions? and Answers?',
            r'Analyst Questions?:?',
            r'Question Session:?',
            
            # Generic analyst speaker
            r'(?:^|\n)(Analyst):',
            r'(?:^|\n)(Question):',
            
            # Specific question indicators
            r'(?:^|\n)Q:',
            r'(?:^|\n)Question\s+\d+:',
        ]
    
    def segment_by_sections(self, text):
        """
        Method 1: Segment by section headers (prepared remarks vs Q&A)
        
        IMPROVED VERSION:
        - More comprehensive Q&A markers
        - Fallback to speaker-based segmentation
        - Question extraction as last resort
        """
        management_text = ""
        analyst_text = ""
        
        # ── IMPROVED Q&A MARKERS ────────────────────────────────────
        # Real transcripts use many different formats
        qa_markers = [
            r'Questions?\s+and\s+Answers?',      # Standard
            r'Q\s*&\s*A\s+Session',               # Q&A Session
            r'Q\s*&\s*A\b',                       # Just Q&A
            r'Question\s+Session',                 # Question Session
            r'Question-and-Answer',                # Hyphenated
            r'Analyst\s+Questions?',               # Analyst Questions
            r'Questions?\s+from\s+Analysts?',      # Questions from Analysts
            r'(?:^|\n)Operator:\s*(?:Thank you|We will now|Our next question)',  # Operator transition
        ]
        
        qa_split = None
        for marker in qa_markers:
            matches = list(re.finditer(marker, text, re.IGNORECASE))
            if matches:
                qa_split = matches[0].start()
                break
        
        if qa_split:
            # Clear Q&A section found
            management_text = text[:qa_split]
            qa_section = text[qa_split:]
            analyst_text = self._extract_analyst_from_qa(qa_section)
        else:
            # ── FALLBACK: No clear Q&A section ─────────────────────
            # Many real transcripts don't have explicit section markers
            # Use speaker-based segmentation instead
            self.segmentation_stats['fallback_used'] += 1
            
            management_text, analyst_text = self.segment_by_speakers(text)
            
            # If STILL no analyst text found, extract questions
            if not analyst_text or len(analyst_text.split()) < 20:
                # Extract all question blocks (text containing ?)
                question_blocks = re.findall(
                    r'[^.!?]*\?[^.!?]*',  # Match sentences with ?
                    text
                )
                if question_blocks and len(question_blocks) >= 3:
                    # Only use questions if we found a reasonable number
                    analyst_text = ' '.join(question_blocks)
                    # Remove those blocks from management
                    for q in question_blocks:
                        management_text = management_text.replace(q, '', 1)
        
        return management_text, analyst_text
    
    def segment_by_speakers(self, text):
        """
        Method 2: Segment by speaker names/titles
        Used as fallback when no clear Q&A section exists
        """
        lines = text.split('\n')
        
        management_lines = []
        analyst_lines = []
        current_speaker = 'unknown'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line identifies a speaker
            is_mgmt = any(re.search(pat, line, re.IGNORECASE) 
                         for pat in self.management_patterns)
            is_analyst = any(re.search(pat, line, re.IGNORECASE) 
                           for pat in self.analyst_patterns)
            
            if is_mgmt and not is_analyst:
                current_speaker = 'management'
                management_lines.append(line)
            elif is_analyst and not is_mgmt:
                current_speaker = 'analyst'
                analyst_lines.append(line)
            else:
                # Continuation of previous speaker
                if current_speaker == 'management':
                    management_lines.append(line)
                elif current_speaker == 'analyst':
                    analyst_lines.append(line)
                else:
                    # Default to management if unclear
                    management_lines.append(line)
        
        management_text = ' '.join(management_lines)
        analyst_text = ' '.join(analyst_lines)
        
        return management_text, analyst_text
    
    def _extract_analyst_from_qa(self, qa_text):
        """
        Extract analyst portions from Q&A section
        
        IMPROVED VERSION:
        - Better question pattern matching
        - Multiple extraction strategies
        - Falls back to question sentences if needed
        """
        analyst_parts = []
        
        # ── IMPROVED QUESTION PATTERNS ─────────────────────────────
        question_patterns = [
            # Standard analyst intro with firm name
            r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–—]\s*(?:Analyst|Morgan Stanley|Goldman Sachs|JP\s*Morgan|Bank of America|Citi|Citigroup|Barclays|UBS|Credit Suisse|Deutsche Bank|Wells Fargo|RBC|BMO|Jefferies|Piper)',
            
            # Analyst without firm
            r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+),?\s*Analyst',
            
            # Question markers
            r'(?:^|\n)Q:',
            r'(?:^|\n)Question\s+\d+:',
            r'(?:^|\n)Analyst:',
            
            # Operator introducing analyst/question
            r'(?:^|\n)Operator:\s*(?:Our next question|The next question|Thank you)',
        ]
        
        # Try each pattern
        for pattern in question_patterns:
            parts = re.split(pattern, qa_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Take the parts AFTER the pattern (the actual questions)
                # Skip first part (before first match) and take every other part
                analyst_parts.extend(parts[1::2] if len(parts) > 2 else [parts[1]])
        
        # If no patterns matched, just look for question sentences
        if not analyst_parts:
            # Find sentences containing question marks
            questions = re.findall(
                r'[^.!?]*\?[^.!?]*',
                qa_text
            )
            if questions:
                analyst_parts = questions
        
        return ' '.join(analyst_parts)
    
    def segment_transcript(self, transcript_dict):
        """
        Segment a single transcript using multiple methods
        """
        try:
            full_text = transcript_dict.get('full_text_cleaned', '')
            
            if not full_text:
                return None
            
            # Try Method 1: Section-based (with improved fallback)
            management_text, analyst_text = self.segment_by_sections(full_text)
            
            # Calculate word counts
            mgmt_words = len(management_text.split())
            analyst_words = len(analyst_text.split())
            
            # Calculate statistics
            mgmt_utterances = len(re.findall(r'[.!?]+', management_text))
            analyst_utterances = len(re.findall(r'[.!?]+', analyst_text))
            
            ratio = mgmt_words / analyst_words if analyst_words > 0 else 0
            
            segmented = {
                **transcript_dict,
                'management_text': management_text,
                'analyst_text': analyst_text,
                'management_word_count': mgmt_words,
                'analyst_word_count': analyst_words,
                'management_utterances': mgmt_utterances,
                'analyst_utterances': analyst_utterances,
                'management_analyst_ratio': ratio,
                'segmentation_timestamp': datetime.now().isoformat()
            }
            
            return segmented
            
        except Exception as e:
            self._log_error(f"Error segmenting {transcript_dict.get('ticker', 'UNKNOWN')}: {e}")
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
        cleaned_file = self.input_dir / 'transcripts_cleaned.json'
        
        if not cleaned_file.exists():
            print(f"❌ Cleaned transcripts not found: {cleaned_file}")
            return None
        
        print(f"📂 Loading cleaned transcripts from: {cleaned_file}")
        
        with open(cleaned_file, 'r', encoding='utf-8') as f:
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
            else:
                self.segmentation_stats['failed'] += 1
            
            self.segmentation_stats['total_processed'] += 1
        
        print()
        
        # Calculate averages
        if segmented_transcripts:
            self.segmentation_stats['avg_management_words'] = np.mean([
                t['management_word_count'] for t in segmented_transcripts
            ])
            self.segmentation_stats['avg_analyst_words'] = np.mean([
                t['analyst_word_count'] for t in segmented_transcripts
            ])
            
            # Only calculate ratio for transcripts with analyst words
            ratios = [
                t['management_analyst_ratio'] for t in segmented_transcripts
                if t['management_analyst_ratio'] > 0
            ]
            if ratios:
                self.segmentation_stats['avg_ratio'] = np.mean(ratios)
        
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
            
            # Save as JSON
            json_path = self.output_dir / 'transcripts_segmented.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(segmented_transcripts, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON saved: {json_path}")
            print(f"   Size: {json_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save metadata CSV
            metadata_fields = [
                'ticker', 'company_name', 'quarter', 'date',
                'management_word_count', 'analyst_word_count',
                'management_analyst_ratio', 'segmentation_timestamp'
            ]
            
            df = pd.DataFrame(segmented_transcripts)[metadata_fields]
            csv_path = self.output_dir / 'transcripts_segmented_metadata.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"✅ Metadata CSV saved: {csv_path}")
            
            # Save statistics
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
        print(f"Avg management/analyst:  {self.segmentation_stats['avg_ratio']:.2f}x")
        print()
        
        # Report fallback usage
        fallback = self.segmentation_stats.get('fallback_used', 0)
        if fallback > 0:
            pct = fallback / self.segmentation_stats['total_processed'] * 100
            print(f"ℹ️  Fallback segmentation used: {fallback} transcripts ({pct:.1f}%)")
            print(f"   (No explicit Q&A section found, used speaker-based extraction)")
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
        
        # Save results
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
    import numpy as np  # Import here for stats
    segmenter = SpeakerSegmenter()
    success = segmenter.run()
    return success


if __name__ == "__main__":
    main()