# Sample Library Organizer

An intelligent audio sample organizer that uses AI to categorize and organize your audio samples based on their content and musical characteristics.

## Features

- Automatically categorizes audio samples into meaningful categories (kicks, snares, hi-hats, percussion, melodic loops, etc.)
- Detects musical characteristics:
  - Key detection for melodic samples
  - BPM detection for loops
  - Duration calculation
- Organizes files into a clean folder structure
- Renames files with detected musical information
- Handles duplicates automatically
- Test mode available for trying out the organization on a small subset

## Requirements

- Python 3.8+
- Required Python packages:
  ```
  openai
  librosa
  numpy
  pretty_midi
  ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/sample-library-organizer.git
   cd sample-library-organizer
   ```

2. Install required packages:
   ```bash
   pip install openai librosa numpy pretty_midi
   ```

3. Set up your OpenAI API key:
   - Get an API key from OpenAI
   - Replace the API key in the script or set it as an environment variable

## Usage

1. Configure the script by modifying these variables in `main()`:
   ```python
   MODEL_PATH = "path/to/your/model"
   SOURCE_DIR = "path/to/your/samples"
   TARGET_DIR = "path/where/organized/samples/should/go"
   ```

2. Run the script:
   ```bash
   python sample_library_organizer.py
   ```

The script will:
1. Run in test mode first, processing 10 files
2. Ask for confirmation to process the remaining files
3. Organize all files into categorized folders
4. Add musical information to filenames where applicable

## Folder Structure

The script creates the following folder structure:
```
target_directory/
├── kicks/
├── snares/
├── hihats/
├── percussion/
├── melodic_loops/
├── percussion_loops/
├── fx/
└── other/
```

## File Naming Convention

Organized files follow this naming pattern:
```
original_name__key-Cm-bpm-128-dur-2.5s.wav
```
Where:
- `key-Cm`: Detected musical key (for melodic content)
- `bpm-128`: Detected tempo (for loops)
- `dur-2.5s`: Audio duration in seconds

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses OpenAI's GPT-4 for intelligent sample classification
- Built with librosa for audio analysis
- Inspired by the need for better sample organization in music production

## Note

This tool is designed for personal use in organizing audio sample libraries. Please ensure you have the rights to any audio files you process with this tool.