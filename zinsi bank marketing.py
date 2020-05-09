import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('bank_marketing.csv')

print(df.head(10))
print("_____________________________________________________________-")
print(df.shape)
print("_____________________________________________________________-")
print(df.columns)
print("_____________________________________________________________-")
print(df.dtypes)

# display all columns
pd.set_option("display.max.columns", None)

# rounds to two decimal pointe
pd.set_option("display.precision", 2)

print("_____________________________________________________________-")
print(df.head(10))

# .info() quick discription of the data
# numbrt of rows / number of non_nullvalues / attribute type
print("_____________________________________________________________-")
print(df.info())

# summry for the numerical attributes
print("_____________________________________________________________-")
print(df.describe())

# columns <--> rows
print("_____________________________________________________________-")
print(df.describe().transpose())

# describe including objects
print("_____________________________________________________________-")
print(df.describe(include = np.object))

# .value_counts( to find how many categories exist
# and how many districts belong to each category
print("_____________________________________________________________-")
print(df["job"].value_counts())
print("_____________________________________________________________-")
print(df["marital"].value_counts())
print("_____________________________________________________________-")
print(df["education"].value_counts())
print("_____________________________________________________________-")
print(df["default"].value_counts())
print("_____________________________________________________________-")
print(df["housing"].value_counts())
print("_____________________________________________________________-")
print(df["loan"].value_counts())
print("_____________________________________________________________-")
print(df["contact"].value_counts())
print("_____________________________________________________________-")
print(df["month"].value_counts())
print("_____________________________________________________________-")
print(df["poutcome"].value_counts())
print("_____________________________________________________________-")
print(df["y"].value_counts())

# plot histogram for each neumerical attribute
#print("_____________________________________________________________-")
#df.hist(bins = 50, figsize = (20, 15))
#plt.show()

# create a test test (avoid data snooping)
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print("_____________________________________________________________-")
print(train_set.shape)
print(test_set.shape)

# create a copy toplay with it without harming the original
bank = train_set.copy()

# visualize the relation beteen two attributes
# for example in this case "age" and "balance"
#print("_____________________________________________________________-")
#bank.plot(kind="scatter", x="age", y="balance")

# Looking for Corralation :
#corr_matrix = bank.corr()

# note: just for numerical attributs
#from sklearn.preprocessing import LabelEncoder

#bank['y'] = LabelEncoder().fit_transform(bank['y'] )

# now see how much each attribute correlates with the 'y'
# note: just for numerical attributs
#print("_____________________________________________________________-")
#print("How much each attribute correlates with the 'y'")
#print(corr_matrix['y'].sort_values(ascending=False))

# Prepare The Data For Machine Learning Algorith:
bank = train_set.drop(["y"], axis=1)
labels = train_set["y"].copy()

from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder().fit_transform(labels)
labels = pd.DataFrame(labels)

# Handeling Text and Categorical Fearuers:
df_cat = ['job', 'marital', 'education', 'default', 
          'housing', 'loan', 'contact','month',
          'poutcome']

df_num = ['age', 'balance', 'day', 'duration', 'campaign',
          'pdays', 'previous']

b = pd.get_dummies(bank["job"], drop_first = True)
c = pd.get_dummies(bank["marital"], drop_first = True)
d = pd.get_dummies(bank["education"], drop_first = True)
e = pd.get_dummies(bank["default"], drop_first = True)
f = pd.get_dummies(bank["housing"], drop_first = True)
g = pd.get_dummies(bank["loan"], drop_first = True)
h = pd.get_dummies(bank["contact"], drop_first = True)
i = pd.get_dummies(bank["month"], drop_first = True)
j = pd.get_dummies(bank["poutcome"], drop_first = True)

bank_drop = bank.drop(['job', 'marital', 'education', 'default',
              'housing', 'loan', 'contact', 'month',
              'poutcome'] , axis = 1)

bank_cat = pd.concat([bank_drop, b, c, d, e, f, g, h, i, j], axis = 1)
print("_____________________________________________________________-")
print(bank_cat.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(bank_cat)

# returns a np array
bank_cat_scaled = scaler.transform(bank_cat)

bank_cat_scaled = pd.DataFrame(bank_cat_scaled)

bank_train = bank_cat_scaled.copy()
bank_labels = labels.copy()

bank_test  = test_set.copy()

# Select and Train a Model:

# train set is bank_train
# train set labrls is bank_labels
# test set is bank_test

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(bank_train, bank_labels)

# Done!
# now, try it out with instances from the training set

some_data = bank_train.iloc[:5]

some_data = scaler.transform(some_data)

# in case of the pipeline is readu we should use:
# some data_prepared = full_pipeline.transform(some_data) 
some_labels = bank_labels.iloc[:5]

print("_____________________________________________________________-")
print("log_reg predections:")
print("Predicفions: ", log_reg.predict(some_data))
print("Labels: ", list(some_labels))


# measure the RMSE on the whole training set

from sklearn.metrics import mean_squared_error

bank_predictions_log = log_reg.predict(bank_train)
log_mse = mean_squared_error(bank_labels, bank_predictions_log)
log_rmse = np.sqrt(log_mse)
print("_____________________________________________________________-")
print("log_reg RMSE:")
print(log_rmse)


from sklearn.model_selection import cross_val_score
log_scores = cross_val_score(log_reg, bank_train, bank_labels,
                         scoring ="neg_mean_squared_error", cv =10)
log_rmse_scores = np.sqrt(-log_scores)

print("_____________________________________________________________-")
print(";og_reg cross_val_scores:")
print("Scores: ", log_rmse_scores)
print("Mean: ", log_rmse_scores.mean())
print("Standard deviation: ", log_rmse_scores.std())

# now lets try SVM

from sklearn.svm import SVC

svr = SVC(kernel = 'linear')
svr_reg = svr.fit(bank_train, bank_labels)

print("_____________________________________________________________-")
print("svr_reg predections:")
print("Predictions: ", svr_reg.predict(some_data))
print("Labels: ", list(some_labels))

bank_predictions_svr = svr_reg.predict(bank_train)
svr_mse = mean_squared_error(bank_labels, bank_predictions_svr)
svr_rmse = np.sqrt(svr_mse)

print("_____________________________________________________________-")
print("svr_reg RMSE:")
print(svr_rmse)

svr_scores = cross_val_score(svr_reg, bank_train, bank_labels,
                         scoring ="neg_mean_squared_error", cv =10)

svr_rmse_scores = np.sqrt(-svr_scores)

print("_____________________________________________________________-")
print("svr_reg cross_val_scores:")
print("Scores: ", svr_rmse_scores)
print("Mean: ", svr_rmse_scores.mean())
print("Standard deviation: ", svr_rmse_scores.std())













