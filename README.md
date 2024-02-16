### DiversaTech x Sofi Spring 2024

# Check Classification and Fraud Detection

## Initial Environment Setup

You only need to do this **once**. You can check if you've completed the environment by checking that a `.venv/` file exists in your local repository. 

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
Install our project as a package in editable mode with
```
pip install -e .
```
This allows our root-level scripts to be importable as packages in our testing suite.

<br><br>
## Installing New Dependencies
If you want to use a package that hasn't already been installed, activate your virtual environment (if it hasn't been already) and run
```
pip install <whatever package you want>
```
If the package will be used across our team, let Tommy and Jeremy know and then add that package to our `requirements.txt` by
```
pip freeze > requirements.txt
```
