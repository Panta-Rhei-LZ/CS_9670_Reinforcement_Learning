# Project Workspace for CS 9670 Reinforcement Learning


## Setup Instructions

Before running the project, ensure you have Python 3.8 or higher installed. If you have a CUDA-enabled GPU, you can leverage it for faster training. Start by installing the required dependencies:

1. **Install PyTorch with CUDA Support**:
   Check your CUDA version and install the appropriate version of PyTorch by running:
   You can check https://pytorch.org/get-started/locally/ for more information.
   For example:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
   
2. **Install the remaining dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run**:  
   ```bash
   Run main.py to start
   ```
