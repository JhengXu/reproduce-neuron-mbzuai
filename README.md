# reproduce-neuron-mbzuai

## About the Base Project
This implementation is modified from the original work:  
**"Repetition Neurons: How Do Language Models Produce Repetitions?"**  
*Paper:* [arXiv:2410.13497](https://arxiv.org/abs/2410.13497)  
*Original Code:* [repetition_neuron](https://github.com/tatHi/repetition_neuron)

## Getting Started

### Prerequisites
- Python 3.10.8
- pip package manager
- Hugging Face account ([sign up](https://huggingface.co/join))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JhengXu/reproduce-neuron-mbzuai.git
   cd reproduce-neuron-mbzuai
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure paths**
   - Open `neuron.sh` in a text editor
   - Replace all instances of `yourpath` with your actual path
   - Example modification:
     ```bash
     # Change from:
     SAVE_PATH="yourpath/${MNAME}/${DNAME}"
     python yourpath/reproduce-neuron-mbzuai/neuron.py

     # To:
     SAVE_PATH="/path/to/your/directory/${MNAME}/${DNAME}"
     python /path/to/your/directory/neuron.py
     ```

4. **Set up Hugging Face Token**
   - Get your HF token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Open `neuron.py` and locate:
     ```python
     os.environ['HF_TOKEN'] = 'your HF_TOKEN'  # Replace with your actual token
     ```
   - Replace `'your HF_TOKEN'` with your actual token (keep the quotes):
     ```python
     os.environ['HF_TOKEN'] = 'hf_YourActualTokenHere'
     ```

5. **Make the script executable**
   ```bash
   chmod +x neuron.sh
   ```

### Usage
Run the configured script:
```bash
./neuron.sh
```
