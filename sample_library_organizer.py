import os
import shutil
from openai import OpenAI
from pathlib import Path
import librosa
import numpy as np
import pretty_midi
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AudioAnalysis:
    """Data class to hold analysis results"""
    path: Path
    category: str
    key: Optional[str] = None
    bpm: Optional[float] = None
    duration: Optional[float] = None

class SampleAnalyzer:
    """Helper class for analyzing audio characteristics"""
    
    @staticmethod
    def detect_key(audio_path):
        """
        Detect the musical key of an audio file using librosa.
        Returns the key as a string (e.g., 'C', 'Am')
        """
        try:
            y, sr = librosa.load(audio_path)
            
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Sum over time to get key profile
            chroma_sum = np.sum(chroma, axis=1)
            
            # Get the most prominent pitch class
            key_idx = np.argmax(chroma_sum)
            
            # Define pitch classes
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Detect if major or minor
            major_profile = librosa.feature.tonnetz(y=y, sr=sr)
            is_major = np.mean(major_profile[0]) > 0
            
            key = pitch_classes[key_idx]
            if not is_major:
                key += 'm'
                
            return key
        except (librosa.util.exceptions.ParameterError, ValueError, IOError) as e:
            logger.error(f"Error detecting key: {str(e)}")
            return None

    @staticmethod
    def detect_bpm(audio_path):
        """
        Detect the tempo (BPM) of an audio file.
        """
        try:
            y, sr = librosa.load(audio_path)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return round(tempo)
        except (librosa.util.exceptions.ParameterError, ValueError, IOError) as e:
            logger.error(f"Error detecting BPM: {str(e)}")
            return None

    @staticmethod
    def get_duration(audio_path):
        """
        Get the duration of the audio file in seconds.
        """
        try:
            y, sr = librosa.load(audio_path)
            return round(librosa.get_duration(y=y, sr=sr), 2)
        except (librosa.util.exceptions.ParameterError, ValueError, IOError) as e:
            logger.error(f"Error getting duration: {str(e)}")
            return None

class SampleOrganizer:
    def __init__(self, target_base_dir: str):
        """
        Initialize the sample organizer.
        
        Args:
            target_base_dir: Base directory where organized samples will be stored
        """
        self.llm = OpenAI()  # Will use API key from environment variable
        self.target_base_dir = Path(target_base_dir)
        self.analyzer = SampleAnalyzer()
        
        self.categories = {
            'kick': self.target_base_dir / 'kicks',
            'hihat': self.target_base_dir / 'hihats',
            'snare': self.target_base_dir / 'snares',
            'perc': self.target_base_dir / 'percussion',
            'melodic_loop': self.target_base_dir / 'melodic_loops',
            'percussion_loop': self.target_base_dir / 'percussion_loops',
            'fx': self.target_base_dir / 'fx',
            'other': self.target_base_dir / 'other'
        }
        
        self._create_category_folders()
        
    def _create_category_folders(self):
        """Create the category folders if they don't exist."""
        for folder in self.categories.values():
            folder.mkdir(parents=True, exist_ok=True)
    
    def _is_audio_file(self, file_path: Path) -> bool:
        """Check if the file is an audio file based on extension."""
        audio_extensions = {'.wav', '.mp3', '.aiff', '.flac', '.ogg'}
        return file_path.suffix.lower() in audio_extensions
    
    def _classify_sample(self, filename: str) -> str:
        """Use LLM to classify the sample based on its filename."""
        # Replace underscores with spaces for better LLM interpretation
        processed_filename = filename.replace('_', ' ')
        
        prompt = f"""Given the filename "{processed_filename}", classify it into one of these categories:
        - kick: For kick drums and bass drums, including BD
        - hihat: For hi-hats, cymbals, rides and similar high-frequency percussion, might be labeled Hat or HH or CH or OH
        - snare: For snare drums and similar sounds, inluding rim, clap
        - perc: For individual percussion hits
        - melodic_loop: For melodic patterns and musical loops, possibly including an instrument name or a key signature
        - percussion_loop: For rhythm-based loops
        - fx: For sound effects and transitions, sweep, fill, riser
        - other: For anything that doesn't fit above categories
        
        Respond with just the category name, nothing else."""
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4",  # Using a more reliable model
                messages=[
                    {"role": "system", "content": "You are a music sample classifier. Respond only with the exact category name, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Extract the text from the response
            category = response.choices[0].message.content.strip().lower()
            logger.info(f"Category detected: '{category}'")
            
            return category if category in self.categories else 'other'
        except Exception as e:
            logger.error(f"Error classifying sample: {str(e)}")
            return 'other'
    
    def _process_file(self, file_path: Path) -> AudioAnalysis:
        """Process a single audio file and return its analysis."""
        category = self._classify_sample(file_path.name)
        
        analysis = AudioAnalysis(
            path=file_path,
            category=category
        )
        
        # Only perform detailed analysis for certain categories
        if category in ['melodic_loop', 'percussion_loop']:
            analysis.key = self.analyzer.detect_key(file_path)
            analysis.bpm = self.analyzer.detect_bpm(file_path)
        
        analysis.duration = self.analyzer.get_duration(file_path)
        
        return analysis
    
    def _generate_new_filename(self, analysis: AudioAnalysis) -> Tuple[str, Dict]:
        """Generate a new filename with musical information."""
        base = analysis.path.stem
        suffix = analysis.path.suffix
        
        info_parts = []
        info_dict = {}
        
        if analysis.key and analysis.category in ['melodic_loop', 'percussion_loop']:
            info_parts.append(f"key-{analysis.key}")
            info_dict['key'] = analysis.key
            
        if analysis.bpm and analysis.category in ['melodic_loop', 'percussion_loop']:
            info_parts.append(f"bpm-{analysis.bpm}")
            info_dict['bpm'] = analysis.bpm
            
        if analysis.duration:
            info_parts.append(f"dur-{analysis.duration}s")
            info_dict['duration'] = analysis.duration
        
        if info_parts:
            new_filename = f"{base}__{'-'.join(info_parts)}{suffix}"
        else:
            new_filename = f"{base}{suffix}"
            
        return new_filename, info_dict
    
    def process_directory(self, source_dir: str, test_mode: bool = False):
        """
        Process audio files in the directory.
        
        Args:
            source_dir: Directory containing the audio files to organize
            test_mode: If True, only process 10 files and stop
        """
        source_dir = Path(source_dir)
        processed_count = 0
        
        logger.info("Starting sample organization...")
        if test_mode:
            logger.info("Running in TEST MODE - Will only process 10 files")
        
        for item in source_dir.rglob('*'):
            if item.is_file() and self._is_audio_file(item):
                try:
                    # Process the file
                    logger.info(f"\nProcessing: {item.name}")
                    analysis = self._process_file(item)
                    
                    # Generate new filename
                    new_filename, info = self._generate_new_filename(analysis)
                    dest_path = self.categories[analysis.category] / new_filename
                    
                    # Handle duplicate filenames
                    if dest_path.exists():
                        base = dest_path.stem
                        suffix = dest_path.suffix
                        counter = 1
                        while dest_path.exists():
                            dest_path = self.categories[analysis.category] / f"{base}_{counter}{suffix}"
                            counter += 1
                    
                    # Copy the file
                    shutil.copy2(item, dest_path)
                    
                    # Log information
                    logger.info(f"Category: {analysis.category}")
                    logger.info(f"New name: {dest_path.name}")
                    if info:
                        logger.info("Musical information detected:")
                        for k, v in info.items():
                            logger.info(f"  - {k}: {v}")
                    
                    processed_count += 1
                    if test_mode and processed_count >= 10:
                        logger.info("\nTest mode complete - processed 10 files")
                        logger.info("Check the results and run without test_mode to process all files")
                        return
                        
                except Exception as e:
                    logger.error(f"Error processing {item}: {str(e)}")
                    if test_mode:
                        logger.error("Error occurred during test mode - please check the error and try again")
                        return

def main():
    # Configuration
    SOURCE_DIR = "/path/to/your/samples"
    TARGET_DIR = "/path/to/organized/samples"
    
    # Initialize the organizer
    organizer = SampleOrganizer(target_base_dir=TARGET_DIR)
    
    # Run in test mode first
    logger.info("Running initial test...")
    organizer.process_directory(SOURCE_DIR, test_mode=True)
    
    # Ask for confirmation to continue with all files
    response = input("\nWould you like to process all remaining files? (y/n): ")
    if response.lower() == 'y':
        logger.info("\nProcessing all files...")
        organizer.process_directory(SOURCE_DIR, test_mode=False)
    else:
        logger.info("\nExiting - No additional files processed")

if __name__ == "__main__":
    main()