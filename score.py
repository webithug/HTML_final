import pandas as pd
import glob
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy import optimize as op
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import json
import math
import csv

def error_funct(predict, real, total):
    return 3*abs( (predict-real)/total )*()
