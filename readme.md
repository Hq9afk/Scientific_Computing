# Installation

1. (https://www.python.org/downloads/)[Download python]

2. Create virtual environment

   ```bash
   python -m venv venv
   venv/Scripts/activate
   ```

3. Install packages

   ```bash
   pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128
   ```

4. Run

   ```bash
   python main.py
   ```
