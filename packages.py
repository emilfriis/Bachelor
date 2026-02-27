# A list of packages that are used in the project including explanations
# Update this file when new packages are added or removed
# Makes for cleaner code in the notebooks


# -------------------
# Data, plotting etc. 
# -------------------
import pandas as pd                 # for data manipulation and analysis    
import numpy as np                  # for numerical operations
import matplotlib.pyplot as plt     # for plotting
import yfinance as yf               # for downloading financial data

# -------------------
# Estimation methods
# -------------------
import statsmodels.tsa.ar_model import AutoReg # autoregressive models

# -------------------
# GitHub, file handling etc.
# -------------------
import sys                              # provides access to Pythons search path
import os                               # gives Python access to file system 
sys.path.append(os.path.abspath(".."))  # search for modules in parent directory