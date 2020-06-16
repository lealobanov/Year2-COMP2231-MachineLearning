#OPERATION INSTRUCTIONS
#_______________________

#To run, navigate to the directory containing 'dataset' folder and classifier.py. 
#In the terminal, type python3 classifier.py
    #If your machine defaults to Python 3, type python classifier.py
#Follow the on-screen instructions: type 1 or 2 and press enter, to specify the target feature 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

print("1 - PASS/DISTINCTION - FAIL/WITHDRAWN")
print("2 - PASS/DISTINCTION - FAIL")
print('')
variant = ''
while variant not in ["1","2"]:
    variant = input("Please select the target feature for the machine learning algorithms: ")
print("Working... please wait")
print('')
#Read in CSV files
#-----------------
#Read in studentRegistration.csv, studentInfo.csv, courses.csv
#Merge datasets into 1 CSV file along unique identifier [id_student, code_module, code_presentation]
studentRegistration = pd.read_csv("dataset/studentRegistration.csv")
studentInfo = pd.read_csv("dataset/studentInfo.csv")
courses = pd.read_csv("dataset/courses.csv")
studentRegInfo = pd.merge(studentRegistration, studentInfo, on=['id_student', 'code_module', 'code_presentation'], how='outer')
studentRegInfoCourses = pd.merge(studentRegInfo , courses, on=['code_module', 'code_presentation'], how='outer')

#Read in assessment data and merge on id_assessment column
assessments = pd.read_csv("dataset/assessments.csv")
assessments.drop('weight', axis=1, inplace=True)
studentAssessment = pd.read_csv("dataset/studentAssessment.csv")
studentAssessment.drop('is_banked', axis=1, inplace=True)
studentAssessmentData = pd.merge(studentAssessment, assessments, on='id_assessment')

#Constructing new features
#Compute avg TMA, CMA scores
#---------------------------
#Compute avg TMA score values per unique ['id_student', 'code_module', 'code_presentation'] grouping
TMAcalc = studentAssessmentData[studentAssessmentData['assessment_type'] == 'TMA'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    avg_TMA=('score', 'mean')
)
TMAmerge = pd.merge(studentRegInfoCourses, TMAcalc, on=['id_student', 'code_module', 'code_presentation'], how="left")

#Compute avg CMA score values per unique ['id_student', 'code_module', 'code_presentation'] grouping
CMAcalc = studentAssessmentData[studentAssessmentData['assessment_type'] == 'CMA'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    avg_CMA=('score', 'mean')
)
CMAmerge = pd.merge(TMAmerge, CMAcalc, on=['id_student', 'code_module', 'code_presentation'], how="left")

#Compute avg exam score values per unique ['id_student', 'code_module', 'code_presentation'] grouping
Examcalc = studentAssessmentData[studentAssessmentData['assessment_type'] == 'Exam'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    avg_Exam=('score', 'mean')
)
Exammerge = pd.merge(CMAmerge, Examcalc, on=['id_student', 'code_module', 'code_presentation'], how="left")

#Read in VLE data
vle = pd.read_csv("dataset/vle.csv")
#Delete columns for week-from, week-to parameters; majority missing values
vle.drop('week_from', axis=1, inplace=True)
vle.drop('week_to', axis=1, inplace=True)
studentVle = pd.read_csv("dataset/studentVle.csv")
studentVleData = pd.merge(studentVle, vle, on= ['id_site', 'code_module', 'code_presentation'], how='left')

#Constructing new features
#Compute total sum clicks, avg clicks, and total visits across all VLE categories
totalVLEclicks = studentVleData.groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_VLE=('sum_click', 'sum'),
    avg_VLE=('sum_click', 'mean'),
    total_VLE_visits = ('sum_click','count')
)
VLEtotal_avg = pd.merge(Exammerge , totalVLEclicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#Compute sum clicks, avg clicks, and total visits PER VLE CATEGORY
#-----------------------------------------------------------------
#forumng VLE category
forumng_clicks = studentVleData[studentVleData['activity_type'] == 'forumng'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_forumng = ('sum_click', 'sum'),
    avg_forumng=('sum_click', 'mean'),
    total_forumng_visits = ('sum_click','count')
)
forumng = pd.merge(VLEtotal_avg, forumng_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")


#oucontent VLE category
oucontent_clicks = studentVleData[studentVleData['activity_type'] == 'oucontent'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_oucontent = ('sum_click', 'sum'),
    avg_oucontent=('sum_click', 'mean'),
    total_oucontent_visits = ('sum_click','count')
)
oucontent = pd.merge(forumng, oucontent_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#subpage VLE category
subpage_clicks = studentVleData[studentVleData['activity_type'] == 'subpage'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_subpage = ('sum_click', 'sum'),
    avg_subpage=('sum_click', 'mean'),
    total_subpage_visits = ('sum_click','count')
    
)
subpage = pd.merge(oucontent, subpage_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#homepage VLE category
homepage_clicks = studentVleData[studentVleData['activity_type'] == 'homepage'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_homepage = ('sum_click', 'sum'),
    avg_homepage=('sum_click', 'mean'),
    total_homepage_visits = ('sum_click','count')
)
homepage = pd.merge(subpage, homepage_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#quiz VLE category
quiz_clicks = studentVleData[studentVleData['activity_type'] == 'quiz'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_quiz = ('sum_click', 'sum'),
    avg_quiz=('sum_click', 'mean'),
    total_quiz_visits = ('sum_click','count')
)
quiz = pd.merge(homepage, quiz_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#resource VLE category
resource_clicks = studentVleData[studentVleData['activity_type'] == 'resource'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_resource = ('sum_click', 'sum'),
    avg_resource=('sum_click', 'mean'),
    total_resource_visits = ('sum_click','count')
)
resource = pd.merge(quiz, resource_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#url VLE category
url_clicks = studentVleData[studentVleData['activity_type'] == 'url'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_url = ('sum_click', 'sum'),
    avg_url=('sum_click', 'mean'),
    total_url_visits = ('sum_click','count') 
)
url = pd.merge(resource, url_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#ouwiki VLE category
ouwiki_clicks = studentVleData[studentVleData['activity_type'] == 'ouwiki'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_ouwiki = ('sum_click', 'sum'),
    avg_ouwiki=('sum_click', 'mean'),
    total_ouwiki_visits = ('sum_click','count')
)
ouwiki = pd.merge(url, ouwiki_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#oucollaborate VLE category
oucollaborate_clicks = studentVleData[studentVleData['activity_type'] == 'oucollaborate'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_oucollaborate = ('sum_click', 'sum'),
    avg_oucollaborate=('sum_click', 'mean'),
    total_oucollaborate_visits = ('sum_click','count')
)
oucollaborate = pd.merge(ouwiki, oucollaborate_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#externalquiz VLE category
externalquiz_clicks = studentVleData[studentVleData['activity_type'] == 'externalquiz'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_externalquiz = ('sum_click', 'sum'),
    avg_externalquiz=('sum_click', 'mean'),
    total_externalquiz_visits = ('sum_click','count')
)
externalquiz = pd.merge(oucollaborate, externalquiz_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")


#page VLE category
page_clicks = studentVleData[studentVleData['activity_type'] == 'page'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_page = ('sum_click', 'sum'),
    avg_page=('sum_click', 'mean'),
    total_page_visits = ('sum_click','count')
)
page = pd.merge(externalquiz, page_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#questionnaire VLE category
questionnaire_clicks = studentVleData[studentVleData['activity_type'] == 'questionnaire'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_questionnaire = ('sum_click', 'sum'),
    avg_questionnaire=('sum_click', 'mean'),
    total_questionnaire_visits = ('sum_click','count')
)
questionnaire = pd.merge(page, questionnaire_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#ouelluminate VLE category
ouelluminate_clicks = studentVleData[studentVleData['activity_type'] == 'ouelluminate'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_ouelluminate = ('sum_click', 'sum'),
    avg_ouelluminate=('sum_click', 'mean'),
    total_ouelluminate_visits = ('sum_click','count')
)
ouelluminate = pd.merge(questionnaire, ouelluminate_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#glossary VLE category
glossary_clicks = studentVleData[studentVleData['activity_type'] == 'glossary'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_glossary = ('sum_click', 'sum'),
    avg_glossary=('sum_click', 'mean'),
    total_glossary_visits = ('sum_click','count')
)
glossary = pd.merge(ouelluminate, glossary_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#dataplus VLE category
dataplus_clicks = studentVleData[studentVleData['activity_type'] == 'dataplus'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_dataplus = ('sum_click', 'sum'),
    avg_dataplus=('sum_click', 'mean'),
    total_dataplus_visits = ('sum_click','count')
)
dataplus = pd.merge(glossary, dataplus_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#dualpane VLE category
dualpane_clicks = studentVleData[studentVleData['activity_type'] == 'dualpane'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_dualpane = ('sum_click', 'sum'),
    avg_dualpane=('sum_click', 'mean'),
    total_dualpane_visits = ('sum_click','count')
)
dualpane = pd.merge(dataplus, dualpane_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#repeatactivity VLE category
repeatactivity_clicks = studentVleData[studentVleData['activity_type'] == 'repeatactivity'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_repeatactivity = ('sum_click', 'sum'),
    avg_repeatactivity=('sum_click', 'mean'),
    total_repeatactivity_visits = ('sum_click','count')
)
repeatactivity = pd.merge(dualpane, repeatactivity_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#htmlactivity VLE category
htmlactivity_clicks = studentVleData[studentVleData['activity_type'] == 'htmlactivity'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_htmlactivity = ('sum_click', 'sum'),
    avg_htmlactivity=('sum_click', 'mean'),
    total_htmlactivity_visits = ('sum_click','count')
)
htmlactivity = pd.merge(repeatactivity, htmlactivity_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#sharedsubpage VLE category
sharedsubpage_clicks = studentVleData[studentVleData['activity_type'] == 'sharedsubpage'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_sharedsubpage = ('sum_click', 'sum'),
    avg_sharedsubpage=('sum_click', 'mean'),
    total_sharedsubpage_visits = ('sum_click','count')
)
sharedsubpage = pd.merge(htmlactivity, sharedsubpage_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#folder VLE category
folder_clicks = studentVleData[studentVleData['activity_type'] == 'folder'].groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_folder = ('sum_click', 'sum'),
    avg_folder=('sum_click', 'mean'),
    total_folder_visits = ('sum_click','count')
)
data = pd.merge(sharedsubpage, folder_clicks, on=['id_student', 'code_module', 'code_presentation'], how = "left")

#Feature engineering
#--------------------
#Binary assignment to 'gender' feature; male = 1, female = 0
le_gender = LabelEncoder()
data['gender'] = le_gender.fit_transform(data.gender)

#One-hot encode 'region' feature
data['region'] = pd.Categorical(data['region'])
one_hot_region = pd.get_dummies(data['region'], prefix = 'region')
data = pd.concat([data, one_hot_region ], axis=1)

#One-hot encode 'highest_education' feature
data['highest_education'] = pd.Categorical(data['highest_education'])
one_hot_highestedu = pd.get_dummies(data['highest_education'], prefix = 'edu')
data = pd.concat([data, one_hot_highestedu ], axis=1)

#Compute mean of band
def split_mean(x):
    split_list = x.split('-')
    mean = (float(split_list[0])+float(split_list[1]))/2
    return mean

#IMD_band - replace with mean of band
#Remove '%' sign
data['imd_band'] = data['imd_band'].str.replace('%','')
#Eliminate rows missing imd_band value from dataset
data['imd_band'].replace('', np.nan, inplace=True)
data.dropna(subset=['imd_band'], inplace=True)
#Replace with mean of band range
data['imd_band_mean'] = data['imd_band'].apply(lambda x: split_mean(x))

#Age_band - replace with mean of band range
#Modify '55<=' to 55-90 (maintaining interval band size of 35)
data['age_band'] =  data['age_band'].str.replace('<=','-90')
data['age_band_mean'] = data['age_band'].apply(lambda x: split_mean(x))

#Binary assignment to 'disability' feature; N = 0, Y = 1
le_disability = LabelEncoder()
data['disability'] = le_disability.fit_transform(data.disability)

#Resolve missing data values
#-------------------------------------------

#Resolve missing TMA, CMA, Exam score values
#Missing TMA, CMA scores replaced with mean of column
data['avg_TMA'].replace('', np.nan, inplace=True)
data['avg_CMA'].replace('', np.nan, inplace=True)
data.fillna(data.mean()['avg_TMA':'avg_CMA'], inplace=True)

#Missing Exam scores replaced with 0
data['avg_Exam'].replace('', np.nan, inplace=True)
data['avg_Exam'].fillna(0, inplace=True)

#Records with missing registration dates removed
data['date_registration'].replace('', np.nan, inplace=True)
data.dropna(subset=['date_registration'], inplace=True)

#Resolve missing VLE values
#df.isnull().sum(axis = 0) - check number of null values per column

#Records with missing total/avg VLE clicks and visits across all categories are removed
data.dropna(subset=['total_VLE'], inplace=True)
data.dropna(subset=['avg_VLE'], inplace=True)
data.dropna(subset=['total_VLE_visits'], inplace=True)

#Dropped VLE categories where more than 50% of data values were null; in remaining categories, records with missing values were replaced by the column mean

#Forumnng VLE category - 11.2% null
data['total_forumng'].fillna(data["total_forumng"].mean(), inplace=True)
data['avg_forumng'].fillna(data["avg_forumng"].mean(), inplace=True)
data['total_forumng_visits'].fillna(data["total_forumng_visits"].mean(), inplace=True)

#Oucontent VLE category - 8.1% null
data['total_oucontent'].fillna(data["total_oucontent"].mean(), inplace=True)
data['avg_oucontent'].fillna(data["avg_oucontent"].mean(), inplace=True)
data['total_oucontent_visits'].fillna(data["total_oucontent_visits"].mean(), inplace=True)

#Subpage VLE category - 3.7% null
data['total_subpage'].fillna(data["total_subpage"].mean(), inplace=True)
data['avg_subpage'].fillna(data["avg_subpage"].mean(), inplace=True)
data['total_subpage_visits'].fillna(data["total_subpage_visits"].mean(), inplace=True)

#Homepage VLE category - 0.1% null
data['total_homepage'].fillna(data["total_homepage"].mean(), inplace=True)
data['avg_homepage'].fillna(data["avg_homepage"].mean(), inplace=True)
data['total_homepage_visits'].fillna(data["total_homepage_visits"].mean(), inplace=True)

#Quiz VLE category - 31.7% null
data['total_quiz'].fillna(data["total_quiz"].mean(), inplace=True)
data['avg_quiz'].fillna(data["avg_quiz"].mean(), inplace=True)
data['total_quiz_visits'].fillna(data["total_quiz_visits"].mean(), inplace=True)

#Resource VLE category - 5.7% null
data['total_resource'].fillna(data["total_resource"].mean(), inplace=True)
data['avg_resource'].fillna(data["avg_resource"].mean(), inplace=True)
data['total_resource_visits'].fillna(data["total_resource_visits"].mean(), inplace=True)

#Url VLE category - 18.7% null
data['total_url'].fillna(data["total_url"].mean(), inplace=True)
data['avg_url'].fillna(data["avg_url"].mean(), inplace=True)
data['total_url_visits'].fillna(data["total_url_visits"].mean(), inplace=True)

#Ouwiki VLE category - 64.3% null - dropped
data = data.drop('total_ouwiki', 1)
data = data.drop('avg_ouwiki', 1)
data = data.drop('total_ouwiki_visits', 1)

#Oucollaborate VLE category - 63.4% null - dropped
data = data.drop('total_oucollaborate', 1)
data = data.drop('avg_oucollaborate', 1)
data = data.drop('total_oucollaborate_visits', 1)

#Externalquiz VLE category - 83.6% null - dropped
data = data.drop('total_externalquiz', 1)
data = data.drop('avg_externalquiz', 1)
data = data.drop('total_externalquiz_visits', 1)

#Page VLE category - 70.9% null - dropped
data = data.drop('total_page', 1)
data = data.drop('avg_page', 1)
data = data.drop('total_page_visits', 1)

#Questionnaire VLE category - 84.9% null - dropped
data = data.drop('total_questionnaire', 1)
data = data.drop('avg_questionnaire', 1)
data = data.drop('total_questionnaire_visits', 1)

#Ouelluminate VLE category - 91.6% null - dropped
data = data.drop('total_ouelluminate', 1)
data = data.drop('avg_ouelluminate', 1)
data = data.drop('total_ouelluminate_visits', 1)

#Glossary VLE category - 79.3% null - dropped
data = data.drop('total_glossary', 1)
data = data.drop('avg_glossary', 1)
data = data.drop('total_glossary_visits', 1)

#Dataplus VLE category - 91% null - dropped
data = data.drop('total_dataplus', 1)
data = data.drop('avg_dataplus', 1)
data = data.drop('total_dataplus_visits', 1)

#Dualpane VLE category - 87.7% null - dropped
data = data.drop('total_dualpane', 1)
data = data.drop('avg_dualpane', 1)
data = data.drop('total_dualpane_visits', 1)

#Repeatactivity VLE category - 99.9% null - dropped
data = data.drop('total_repeatactivity', 1)
data = data.drop('avg_repeatactivity', 1)
data = data.drop('total_repeatactivity_visits', 1)

#Htmlactivity VLE category - 93.5% null - dropped
data = data.drop('total_htmlactivity', 1)
data = data.drop('avg_htmlactivity', 1)
data = data.drop('total_htmlactivity_visits', 1)

#Sharedsubpage VLE category - 99.6% null - dropped
data = data.drop('total_sharedsubpage', 1)
data = data.drop('avg_sharedsubpage', 1)
data = data.drop('total_sharedsubpage_visits', 1)

#Folder VLE category - 93.6% null - dropped
data = data.drop('total_folder', 1)
data = data.drop('avg_folder', 1)
data = data.drop('total_folder_visits', 1)

#Drop engineered columns
data = data.drop('date_unregistration', 1)
data = data.drop('region', 1)
data = data.drop('highest_education', 1)
data = data.drop('imd_band', 1)
data = data.drop('age_band', 1)
data = data.drop('code_module', 1)
data = data.drop('code_presentation', 1)
data = data.drop('id_student', 1)

#Binary assignment to 'final_result' feature
if variant == "1":
    # VARIANT 1- Map pass/distinction to 1, fail/withdraw to 0
    data['final_result'] = pd.Categorical(data['final_result'])
    parse_results = {"final_result":     {"Distinction": 1, "Pass": 1, "Fail":0, "Withdrawn":0}}
    data.replace(parse_results, inplace=True)
elif variant == "2": 
    # VARIANT 2- Map pass/distinction to 1, fail to 0; drop withdrawn labels
    data.drop(data.loc[data['final_result']=="Withdrawn"].index, inplace=True)
    data['final_result'] = pd.Categorical(data['final_result'])
    parse_results = {"final_result":     {"Distinction": 1, "Pass": 1, "Fail":0}}
    data.replace(parse_results, inplace=True)  

#----------------------------------------------
#Derive test, training, and validation sets
#X - features
#y - labels


target_attribute = data['final_result']
data = data.drop(columns = ['final_result'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(data, target_attribute, test_size=0.2, random_state=0)

#Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#-------------------------------------------------------------
#Configure machine learning model 1 - RANDOM FOREST CLASSIFIER
print('')
print("MODEL 1 - RANDOM FOREST CLASSIFIER")
print("________________________________")
print('')

#Create the model
model1 = RandomForestClassifier(n_estimators=33, 
                            min_samples_split=5,
                            min_samples_leaf=1,
                            max_depth = 31,
                            bootstrap = True,
                            max_features = 'sqrt',
                            random_state = 42)


#Tune N_estimators (number of trees in random forest) hyperparameter and visualize ROC curve
def RF_tune_nestimators():
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    train_results = []
    test_results = []
    for estimator in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1 = plt.plot(n_estimators, train_results, 'b')
    line2 = plt.plot(n_estimators, test_results, 'r')
    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.show()
    return
#RF_tune_nestimators()

#Tune max_depth (depth of each tree in the forest) hyperparameter and visualize ROC curve
def RF_tune_maxdepth():
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
        rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()
    return
#RF_tune_maxdepth()

#Tune min_samples_split (minimum number of samples required to split an internal node) hyperparameter and visualize ROC curve
def RF_tune_minsamplessplit():
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
        rf = RandomForestClassifier(min_samples_split=min_samples_split)
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
    line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Min samples split')
    plt.show()
    return
#RF_tune_minsamplessplit()

#Tune min_samples_leaf (minimum number of samples required to be at a leaf node) hyperparameter and visualize ROC curve
def RF_tune_minsamplesleaf():
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_leaf in min_samples_leafs:
        rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
    line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Min samples leaf')
    plt.show()
    return
#RF_tune_minsamplesleaf()


#Hyper-parameter tuning Random Forest model with k-fold cross validation and visualize ROC curve
def rf_hyperparams_grid():
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'bootstrap': [True],
        'max_depth': [4, 8, 10, 12, 16, 20,40],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split':[2, 5, 10],
        'n_estimators': [20, 33, 50, 100, 200]
    }
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                            cv = 5, n_jobs = -1, verbose = 2)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_, grid_search.best_score_)
    return
#rf_hyperparams_grid()

# Fit on training data
model1.fit(X_train, y_train)   

#Cross validation to check for bias and possible over fitting
from sklearn.model_selection import cross_val_score
print("Model 1 5-Fold Cross Validation Scores: " , cross_val_score(model1, X_train, y_train, cv=5))

#Generate class predictions
rf_predictions = model1.predict(X_test)
from sklearn.metrics import accuracy_score
print("Model 1 Accuracy: " , accuracy_score(y_test, rf_predictions))
print('')
from sklearn import metrics
print("Model 1 Classification Report: " )
print('')
print(metrics.classification_report(rf_predictions, y_test))
print('')
from sklearn.metrics import confusion_matrix
print('Model 1 Confusion Matrix : \n' + str(confusion_matrix(y_test,rf_predictions)))
print('')
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, rf_predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("Model 1 AUC: ",roc_auc)
print('')

#Visualize model ROC curve plotting false positive rate against true positive rate
def model1ROCvisual():
    plt.plot(false_positive_rate, true_positive_rate,'r-',label = 'Random Forest')
    plt.plot([0,1],[0,1],'k-',label='random')
    plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return
#model1ROCvisual()


#-------------------------------------------------------------
#Configure machine learning model 2 - LOGISTIC REGRESSION
print("MODEL 2 - LOGISTIC REGRESSION")
print("______________________________")
print('')

from sklearn.linear_model import LogisticRegression

#Create the model
model2 = LogisticRegression(max_iter = 10000, C=5, penalty='l2')

#Tune C hyperparameter (regularization parameter)
def logreg_C_tune():
    Cs = np.linspace(0.1, 250, 100, endpoint=True)
    train_results = []
    test_results = []
    for C in Cs:
        rf = LogisticRegression(C=C)
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(Cs, train_results, 'b', label='Train AUC')
    line2, = plt.plot(Cs, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('C')
    plt.show()
    return
#logreg_C_tune()


#Hyper-parameter tuning logistic regression using parameter grid
def logreg_hyperparams_grid():
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C':np.logspace(-4,4,20)
    }
    clf = LogisticRegression()
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_, grid_search.best_score_)
    return 
#logreg_hyperparams_grid()


#Fit the model
model2.fit(X_train, y_train)

#Cross validation to check for bias and possible over fitting
print("Model 2 5-Fold Cross Validation Scores: " , cross_val_score(model2, X_train, y_train, cv=5))

#Generate class predictions
logreg_predictions = model2.predict(X_test)
print('')
print("Model 2 Accuracy: " , accuracy_score(y_test, logreg_predictions))
print('')
print("Model 2 Classification Report: ")
print(metrics.classification_report(logreg_predictions, y_test))
print('')
print('Model 2 Confusion Matrix : \n' + str(confusion_matrix(y_test,logreg_predictions)))
print('')
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, logreg_predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("Model 2 AUC: ",roc_auc)
print('')

#Visualize model ROC curve plotting false positive rate against true positive rate
def model2ROCvisual():
    plt.plot(false_positive_rate, true_positive_rate,'r-',label = 'Logistic Regression')
    plt.plot([0,1],[0,1],'k-',label='random')
    plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return 
#model2ROCvisual()

#Export updated pandas dataframe to CSV for future use
#data.to_csv('cleaned_data.csv')
