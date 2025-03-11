import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pickle
import json

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

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams["figure.figsize"] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color = "blue", label = "2 BHK", s = 50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker="+", color= "green", label = "3 BHK", s = 50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    plt.show()

def plotter_bar_chart(df):
    matplotlib.rcParams["figure.figsize"] = (20,10)
    plt.hist(df.price_per_sqft, rwidth=0.8)
    plt.xlabel("Price Per Square Feet")
    plt.ylabel("Count")
    plt.show()

def plotter_hist_bathrooms(df):
    plt.hist(df.bath, rwidth=0.8)
    plt.xlabel("Number of bathrooms")
    plt.ylabel("Count")
    plt.show()

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby("location"):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean" : np.mean(bhk_df.price_per_sqft),
                "std" : np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk-1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats["mean"])].index.values)
    return df.drop(exclude_indices, axis = "index")

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        "linear_regression" : {
            "model": LinearRegression(),
            "params": {
                # "normalize": [True, False]
            }
        },
        "lasso": {
            "model": Lasso(),
            "params": {
                "alpha": [1,2],
                "selection": ["random", "cyclic"]
            }
        },
        "decision_tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "criterion": ["mse", "friedman_mse"],
                "splitter": ["best", "random"]
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config["model"], config["params"], cv = cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            "model": algo_name,
            "best_score": gs.best_score_,
            "best_params": gs.best_params_
        })
    return pd.DataFrame(scores, columns=["model", "best_score", "best_params"])

def predict_price(location, sqft, bath, bhk, X):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]

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
df5 = df4.copy()
df5["price_per_sqft"] = df5["price"]*100000/df5["total_sqft"]
print(df5.head())

# Since there is large number of unique locations it leads to dimensional curse
print(df5.location.unique())
print(len(df5.location.unique()))

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby("location")["location"].agg("count").sort_values(ascending=False)
print(location_stats)

# any location that has less than 10 as other locations
print(len(location_stats[location_stats<=10]))

location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10)

print(len(df5.location.unique()))
df5.location = df5.location.apply(lambda x: "other" if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))

print(df5.head())

# Outlier detection and removal
# Outlier is extreme data points but it's necessary to remove them for reduce extreme variation
print(df5[df5.total_sqft/df5.bhk<300].head())
print(df5.shape)

df6 = df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.shape)

print(df6.price_per_sqft.describe())

df7 = remove_pps_outliers(df6)
print(df7.shape)

# plot to see anomalies within bhk and price_per_sqft
# plot_scatter_chart(df7, "Rajaji Nagar")

df8 = remove_bhk_outliers(df7)
print(df8.shape)
# plot_scatter_chart(df8, "Hebbal")

# plotter_bar_chart(df8)

print(df8.bath.unique())

print(df8[df8.bath>10])

# plotter_hist_bathrooms(df8)
print("\n")

print(df8[df8.bath>df8.bhk+2])
print("\n")

df9 = df8[df8.bath<df8.bhk+2]
print(df9.shape)
print("\n")

# Data frame is looking pretty clean now
# For ml training unnecessary features will be dropped
df10 = df9.drop(["size", "price_per_sqft"], axis="columns")
print(df10.head(3))
print("\n")

dummies = pd.get_dummies(df10.location)
print(dummies.head(3))

df11 = pd.concat([df10, dummies.drop("other", axis="columns")], axis="columns")
print(df11.head(3))

df12 = df11.drop("location", axis="columns")
print(df12.head(2))
print(df12.shape)

# X should contain only independent variable
x = df12.drop("price", axis="columns")
print(x.head())

y = df12.price
print(y.head())


# data set is divided to training set and testing set
# training set is used for model training
# test set is for evaluation of model perfromance
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
print(lr_clf.score(x_test, y_test))

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

print(cross_val_score(LinearRegression(), x, y, cv=cv))

print(find_best_model_using_gridsearchcv(x,y))

# Now the prediction
# print(x.columns)
print(predict_price("1st Phase JP Nagar", 1000, 2, 2, x))
print(predict_price("1st Phase JP Nagar", 1000, 3, 3, x))

print(predict_price("Indira Nagar", 1000, 2, 2, x))
print(predict_price("Indira Nagar", 1000, 3, 3, x))

# exporting the model into a pickle file
with open("bangalore_home_prices_model.pickle", "wb") as f:
    pickle.dump(lr_clf, f)

# exporting column information into a json file
columns = {
    "data_columns" : [col.lower() for col in x.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))

