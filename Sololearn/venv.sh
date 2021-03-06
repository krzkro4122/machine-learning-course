# Step 1: Update your repositories
sudo apt-get update
# Step 2: Install pip for Python 3
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
sudo apt install python3-pip
# Step 3: Use pip to install virtualenv
sudo pip3 install virtualenv 
# Step 4: Launch your Python 3 virtual environment, here the name of my virtual environment will be mlEnv
virtualenv -p python3 ~/.virtualenv/mlEnv
# Step 5: Activate your new Python 3 environment. There are two ways to do this
. /home/chris/.virtualenv/mlEnv/bin/activate # or source env3/bin/activate which does exactly the same thing
# you can make sure you are now working with Python 3
python -- version
# this command will show you what is going on: the python executable you are using is now located inside your virtualenv repository
which python 
# Step 6: code your stuff
# Step 7: done? leave the virtual environment
# deactivate
