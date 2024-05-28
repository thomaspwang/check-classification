### DiversaTech x Sofi Spring 2024

# Check Classification and Fraud Detection

## Initial Environment Setup

You only need perform steps 1 through  4 **once**. You can check if you've completed the environment by checking that a `.venv/` file exists in your local repository. 

<br> **Step 1** <br>
Make sure you have an updated Python version with
```
python --version  # should be 3.10 or higher
```

<br> **Step 2** <br>
Navigate to the root of this repository (`sofi-check-classification`) and create a new virtual environment with either
```
python -m venv venv
# or
python3 -m venv venv
```

<br> **Step 3** <br>
Activate the venv with
```
source venv/bin/activate
```
You should see a `venv` on the very left of the terminal text buffer like so:
![image](https://github.com/JermXT/sofi-check-classification/assets/82493352/c05a4041-b191-4baa-bd20-419e584e2d08)

<br> **Step 4** <br>
Install requirements with
```
pip install -r requirements.txt
```

<br> **Step 5** <br>
Even after installing `requirements.txt`, you may need need to manually install these:

```
pip install python-doctr
pip install torch torchvision torchaudio
```

Note: We've found that we needed to do this on EC2 instances (ml.g5.12xlarge). 
Note: `pip install torch torchvision torchaudio` assumes that your system has Cuda 12.1.

<br> **Step 5** <br>
TODO: Install LLaVA @jerli

## Folder Structure (TO-DO)

## TO-DO
- Configure environment variables automatically through `dotenv` or something instead of having redundant top-level variables such as `AWS_REGION` at the top of every file.
