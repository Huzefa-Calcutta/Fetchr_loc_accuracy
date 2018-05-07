import numpy as np
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
from patsylearn import PatsyTransformer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from sklearn.pipeline import Pipeline
import geohash as gh
import pickle

#from geopy.distance import geodesic
#from geopy.geocoders import Nominatim,GoogleV3
#from geopy.exc import GeocoderTimedOut

data_1 = pd.read_csv("anon_dataset_10_2016.csv")
data_2 = pd.read_csv("anon_dataset_11_2016.csv")
data_3 = pd.read_csv("anon_dataset_12_2016.csv")
data_4 = pd.read_csv("anon_dataset_01_2017.csv")
print((data_4.iloc[0,5]))

data = pd.concat([data_1, data_2, data_3, data_4])
print(data.shape)

data["commitment_date"] = pd.to_datetime(data["commitment_date"])
data["commitment_date_weekday"] = data["commitment_date"].dt.weekday_name

data[["delivery_lat","delivery_lang"]] = data["delivery_coordinates"].str.split(";",expand = True)
data[["Schedule_lat","Schedule_lang"]] = (data.scheduled_coordinates.str.split(";",expand = True)[[0,1]])

data["Schedule_lat"] = data["Schedule_lat"].str.split(".",expand = True)[0]+"."+data["Schedule_lat"].str.split(".",expand = True)[1]
data['Schedule_lang'] =data['Schedule_lang'].str.replace(" ","")
data["Schedule_lang"] = data["Schedule_lang"].str.split(".",expand = True)[0]+"."+data["Schedule_lang"].str.split(".",expand = True)[1]

data.isnull().sum()
data.dropna(subset = ["delivery_lat","delivery_lang","Schedule_lat","Schedule_lang"],how = 'any' ,inplace = True)

data.delivery_lat = pd.to_numeric(data.delivery_lat)
data.delivery_lang = pd.to_numeric(data.delivery_lang)
data["Schedule_lat"] = pd.to_numeric(data["Schedule_lat"])
data['Schedule_lang'] = pd.to_numeric(data['Schedule_lang'])

def location_type(lat,lang):
    return gh.encode(lat,lang,precision=4)
    '''
    try:
        return geolocator.geocode(geolocator.reverse(loc1).address, timeout=None).raw['type']
    except GeocoderTimedOut:
       return None
    '''

data["Schedule_Deliver_diff"] = 2*np.arcsin(np.sqrt(
                                            np.absolute(np.sin(np.radians(data["Schedule_lat"]-data["delivery_lat"])*0.5)**2 +
                                       np.cos(data["Schedule_lat"])*np.cos(data["delivery_lat"])*
                                                    np.sin(np.radians(data["Schedule_lang"]-data["delivery_lang"])*0.5)*
                                                    np.sin(np.radians(data["Schedule_lang"]-data["delivery_lang"])*0.5))))*6371

data["Schedule_loc_type"] = np.vectorize(location_type)(data["Schedule_lat"],data["Schedule_lang"])

pd.unique(data["Schedule_loc_type"])
data_loc_type_count = data.groupby(by = 'Schedule_loc_type').size().reset_index(name="counts")

data_loc_type_count[data_loc_type_count['counts'] < 100].shape
loc_type_less_freq = list(data_loc_type_count[data_loc_type_count['counts'] < 100]['Schedule_loc_type'])

# All the loc_type having count of less than 100 are grouped together
data.loc[data["Schedule_loc_type"].isin(loc_type_less_freq),"Schedule_loc_type"] = "Other"

data["accurate"] = data["Schedule_Deliver_diff"] < 0.7

data.accurate.sum()


loc_type = seaborn.countplot("Schedule_loc_type",hue = "accurate" ,data = data)
loc_type.set_xticklabels(labels=pd.unique(data["Schedule_loc_type"]),rotation = 90)
plt.clf()

Schedule_channel = seaborn.countplot("schedule_channel",hue="accurate", data = data)
plt.clf()

delivery_slot_from = seaborn.countplot("time_slot_from", hue="accurate", data=data, order=sorted(pd.unique(data["time_slot_from"])))
plt.clf()

delivery_slot_to = seaborn.countplot("time_slot_to", hue="accurate", data = data, order=sorted(pd.unique(data["time_slot_to"])))
plt.clf()

delivery_slot = seaborn.countplot("delivery_slot", hue="accurate", data = data, order=sorted(pd.unique(data["delivery_slot"])))
delivery_slot.set_xticklabels(labels=sorted(pd.unique(data["delivery_slot"])), rotation = 90)
plt.clf()

day_list = ["Sunday","Monday", "Tuesday", "Wednesday","Thursday", "Friday", "Saturday"]
delivery_comm_date = seaborn.countplot("commitment_date_weekday", hue="accurate", data = data, order=day_list)

# Schedule delivery time during the entire working hours i.e. between 8 am to 6 pm
data["Is_delivery_working_hours"] = (data["time_slot_from"].str[:2].astype(int) >= 9) & (data["time_slot_to"].str[:2].astype(int) <= 18)
# Whether committed delivery day is  friday (weekend) or not
data["Is_committed_delivery_friday"] = (data["commitment_date_weekday"]=="Friday").astype(int)

# dataframe of predictor variables
feature_df = data[["Schedule_loc_type", "schedule_channel" , "Is_committed_delivery_friday","Is_delivery_working_hours"]]
y_df = data["accurate"]


interaction_transformer = PatsyTransformer("C(Schedule_loc_type) + C(schedule_channel)+ Is_committed_delivery_friday + Is_delivery_working_hours + C(Schedule_loc_type):C(schedule_channel)")
non_interaction_transformer = PatsyTransformer("C(Schedule_loc_type) + C(schedule_channel)+ Is_committed_delivery_friday + Is_delivery_working_hours")

naive_bayes_clf = Pipeline([("encoding",non_interaction_transformer),("nb_clf",GaussianNB())]) # Because we assume conditional independence among diffrent predictor variables in Naive Bayes, interaction terms are not necessary
logistic_clf = Pipeline([('int_feature',interaction_transformer),('log_clf', LogisticRegression(C = 10.0,fit_intercept=True,random_state=100))])
supportvector_clf = Pipeline([('int_feature',interaction_transformer),('svm_clf', svm.SVC(kernel='linear',C=1, gamma=0.1,random_state = 100))])
randomforest_clf = Pipeline([('int_feature',interaction_transformer),('rf_clf', RandomForestClassifier(n_estimators = 500,max_features = 0.7,max_depth = 5, random_state=100,oob_score=True))])

# no parameter tuning required for Naive Bayes
naive_bayes_clf = naive_bayes_clf.fit(feature_df,data["accurate"])


#Defining parameter grid space for logistic, SVM and Random forest
parameters_logistic = {'log_clf__C' : (1,10,100,1000,10000),'log_clf__penalty': ('l2','l1')}
parameters_svm = {'svm_clf__C':(1,10,100,1000,10000),'svm_clf__gamma':(0.0001,0.001,0.01,0.1),'svm_clf__kernel':('linear','rbf','poly')}
parameters_rf = {'rf_clf__max_depth': (2,4,6,8,10,12,16,20),'rf_clf__max_features': (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)}

#Defining Grid search
opt_logistic = GridSearchCV(logistic_clf,parameters_logistic,cv=KFold(n_splits=5,random_state=100))
opt_svm = GridSearchCV(supportvector_clf,parameters_svm,cv=KFold(n_splits=5,random_state=100))# n_jobs = -1 if the system is linux or MacOS which enables parallel processing
opt_rf = GridSearchCV(randomforest_clf,parameters_rf,cv=KFold(n_splits=5,random_state=100))# n_jobs = -1 if the system is linux or MacOS which enables parallel processing

#Logistic model training and parameter tuning starts
opt_logistic = opt_logistic.fit(feature_df,data["accurate"])


#SVM model training and parameter tuning starts
opt_svm = opt_svm.fit(feature_df,data["accurate"])


#Random forest model training and parameter tuning starts
opt_rf = opt_rf.fit(feature_df,data["accurate"])

#Top score for each of the method

print(opt_logistic.best_score_)
print(opt_svm.best_score_)
print(opt_rf.best_score_)


# Preparing the test dataset
data_test = pd.read_csv("test_dataset_02_2017.csv")
data_test["commitment_date"] = pd.to_datetime(data_test["commitment_date"])
data_test["commitment_date_weekday"] = data_test["commitment_date"].dt.weekday_name

data_test[["Schedule_lat","Schedule_lang"]] = (data_test.scheduled_coordinates.str.split(";",expand = True)[[0,1]])
data_test["Schedule_lat"] = pd.to_numeric(data_test["Schedule_lat"],errors='ignore')
data_test['Schedule_lang'] = pd.to_numeric(data_test['Schedule_lang'],errors='ignore')

data_test = data_test[data_test["Schedule_lat"] != None]
data_test = data_test[data_test["Schedule_lang"] != None]


data_test["Is_delivery_working_hours"] = (data_test["time_slot_from"].str[:2].astype(int) >= 8) & (data_test["time_slot_to"].str[:2].astype(int) > 15) & (data["time_slot_to"].str[:2].astype(int) <= 18).astype(int)
data_test["Is_delivery_1st_half"] = (data_test["time_slot_from"].str[:2].astype(int) >= 8) & (data_test["time_slot_to"].str[:2].astype(int) <=14).astype(int)
data_test["Is_delivery_2nd_half"] = (data_test["time_slot_from"].str[:2].astype(int) > 14) & (data_test["time_slot_to"].str[:2].astype(int) <=18).astype(int)

# Whether committed delivery day is  friday (weekend) or not
data_test["Is_committed_delivery_friday"] = (data_test["commitment_date_weekday"]=="Friday").astype(int)

# dataframe of predictor variables
feature_test_df = data_test[["Schedule_loc_type", "schedule_channel" , "Is_committed_delivery_friday","Is_delivery_working_hours","Is_delivery_1st_half","Is_delivery_2nd_half"]]

#predicting on test set
predicted_class_nb = pd.Series(naive_bayes_clf.predict(feature_test_df),index = feature_test_df.index,name = 'NB_predict')
predicted_class_svm = pd.Series(opt_logistic.predict(feature_test_df),index = feature_test_df.index,name = 'Logistic_predict')
predicted_class_svm = pd.Series(opt_svm.predict(feature_test_df),index = feature_test_df.index,name = 'SVM_predict')
predicted_class_rf = pd.Series(opt_rf.predict(feature_test_df),index = feature_test_df.index,name= 'Random_forest_predict')

result = pd.concat([feature_test_df,predicted_class_nb,predicted_class_svm,predicted_class_rf],axis=1)
result.to_csv("test_results.csv")
