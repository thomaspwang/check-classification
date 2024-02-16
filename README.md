# DiversaTech x Sofi Spring 2024<br>Check Classification and Fraud Detection

## Initial Environment Setup

You only need to do this **once**. You can check if you've completed the environment by checking that a `.venv/` file exists in your local repository. 

**Step 1** <br>
Make sure you have an updated Python version with
```
python --version  # should be 3.10 or higher
```

**Step 2** <br>
Navigate to the root of this repository (`sofi-check-classification`) and create a new virtual environment with either
```
python -m venv venv
# or
python3 -m venv venv
```

**Step 3** <br>
Activate the venv with
```
source venv/bin/activate
```
You should see a `venv` on the very left of the terminal text buffer like so:
![image](https://github.com/JermXT/sofi-check-classification/assets/82493352/c05a4041-b191-4baa-bd20-419e584e2d08)

**Step 4** <br>
Install requirements with
```
pip install -r requirements.txt
```

**Step 5** <br>
Install our project as a package in editable mode with
```
pip install -e .
```
This allows our root-level scripts to be importable as packages in our testing suite.
