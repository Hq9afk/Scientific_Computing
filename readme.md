# Installation

1. [Download python](https://www.python.org/downloads/)

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

5. Evaluate
   ```bash
   python predictions_and_errors.ppy
   ```