# object-detection


# VideoTools Setup Guide

## Setting up the Python Environment

1. **Create a new virtual environment**  
   Run the following command to create a virtual environment named `videotools_env`:
   ```bash
   python3 -m venv videotools_env

2. **Activate the virtual environment**
On Linux or macOS:
```bash
source videotools_env/bin/activate
```
On Windows:
```bash
.\videotools_env\Scripts\activate
```
Install required dependencies
Use the provided requirements.txt file to install all necessary libraries:

```bash
pip install -r requirements.txt
```

run 
```bash 
python VideoTools.py extract_frames "path/to/video.mp4" "path/to/output" --num_frames 50
```