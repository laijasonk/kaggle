# Instructions to set up the workspace

## Install Pycharm

## Install python
Download python from:
https://www.python.org/downloads/release/python-363/

Launch the python install executable and choose a manual installation 

Install in for example C:\Python36-64 
Installing with a simple path like C:\ is recommended

## Set up virtual environment
Create a folder to store your virtual environment for example in C:\Users\JOUS\venv

Clone the repo in a new folder for example Documents\10009

Open pycharm

Create a virtual environment from this python distribution:
- In Pycharm create a new project
- Click on the little wheel at the end of the line on Interpreter
- Choose 'Create a VirtualEnv'
- It opens a window 
- Inside the window: choose Base interpreter (the python excutable) for example C:Python36-64\Tools\Python.exe
- Inside the window: choose Location (where your virtual environment folder is) for example C:\Users\JOUS\venv
- Inside the window: choose the name of your virtual environment for example 10009
- Click Ok
- Location in the New project window is where you choose to work for example Documents/10009
- Push create 

Your workspace and virtual environment is now all set up

## Install the packages

You should now be inside Pycharm in your project environment

```
pip install "./packages/numpy-1.14.1+mkl-cp36-cp36m-win_amd64.whl"
pip install "./packages/scipy-1.0.0-cp36-cp36m-win_amd64.whl"
pip install "./packages/scikit_image-0.13.1-cp36-cp36m-win_amd64.whl
pip install "./packages/scikit_learn-0.19.1-cp36-cp36m-win_amd64.whl
pip install matplotlib
pip install tensorflow
```

To navigate in the terminal you can use cd to change directory, cd .. and dir to list what is the folder
