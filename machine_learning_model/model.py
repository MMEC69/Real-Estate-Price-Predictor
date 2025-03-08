import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

# Functions for the work
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# to convert range values to a mean
def convert_sqft_to_num(x):
    tokens = x.split("-")
    if(len(tokens) == 2):
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

# reading the data set
df1 = pd.read_csv("./dataset/Bengaluru_House_Data.csv")
print(df1.head())
print(df1.shape)

print(df1.groupby("area_type")["area_type"].agg("count"))

df2 = df1.drop(["area_type", "society", "balcony", "availability"], axis="columns")
print(df2.head())

# data cleaning -> handle the values
print(df2.isnull().sum())
print("\n")

# dropping the null rows instead of filling with medians (data set is large)
df3 = df2.dropna()
print(df3.isnull().sum())
print("\n")

print(df3.shape)

print(df3["size"].unique())

# creating new cols
df3["bhk"] = df3["size"].apply(lambda x: int(x.split(" ")[0]))

print(df3.head())

print(df3["bhk"].unique())

# for the bhk that has 43 bhk seems like an error given that its total_sqft is small
# need to clean up that later
print(df3[df3.bhk > 20])
print("\n")

print(df3.total_sqft.unique())

print(df3[~df3["total_sqft"].apply(is_float)].head(10))

# testing the function
print("\n")
print(convert_sqft_to_num("2166"))
print(convert_sqft_to_num("2100 - 2850"))
print(convert_sqft_to_num("34.46Sq. Meter"))

# data processing stages
df4 = df3.copy()
df4["total_sqft"] = df4["total_sqft"].apply(convert_sqft_to_num)
print(df4.head(10))

print(df4.loc[30])
print(df4.loc[410])

# feature and dimentional reduction
