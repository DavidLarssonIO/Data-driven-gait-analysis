import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np

# --------------------------------------------------------------------------------------------------    
# Function that will evaluate the selected model and return performance metrics for analysis
# A generic function is created to evaluate the model performance. 
# This is going to be used as a benchmark to evaluate how much the performance of the model has improved
# based on hyperparameter tuning.

def evaluate(model, X_valid, y_valid, title):
    
    # Build the k-fold cross-validator
    kfold = KFold(n_splits=5)
    all_y_pred = cross_val_predict(model, X_valid, y_valid, cv=kfold)
    
    results = np.ndarray(shape=(11))
    CM_arr = metrics.confusion_matrix(y_valid, all_y_pred)

    plot_confusion_matrix(CM_arr, title)
    
    results[0] = metrics.accuracy_score(y_valid, all_y_pred) #Accuracy
    results[1] = metrics.balanced_accuracy_score(y_valid, all_y_pred) #Balanced Accuracy
    results[2] = metrics.precision_score(y_valid, all_y_pred,average='macro')   #Precision
    results[3] = metrics.recall_score(y_valid, all_y_pred,average='macro')   #Sensitivity/Recall/True positive rate (TPR)
    results[4] = metrics.f1_score(y_valid, all_y_pred, average='macro')
    results[5] = metrics.matthews_corrcoef(y_valid, all_y_pred)
    results[6] = metrics.mean_absolute_error(y_valid, all_y_pred)
    results[7] = metrics.mean_squared_error(y_valid, all_y_pred)
    results[8] = np.sqrt(metrics.mean_squared_error(y_valid, all_y_pred))
    results[9] = model.score(X_valid,y_valid)

    print(f'Performance Metrics for {title} :')
    print('----------------------------------------------------------')
    print('Accuracy Score:', results[0])
    print('Balanced Accuracy Score:', results[1])
    print('Precision: (Macro Avg)', results[2])
    print('Recall: (Macro Avg)', results[3])
    print('F-1 Score (Macro Avg):', results[4])
    print('MCC Score:', results[5])
    print('Mean Absolute Error:', results[6])
    print('Mean Squared Error:', results[7])
    print('Root Mean Squared Error:', results[8])
    print('Classifier Score:', results[9])

    print('\nClassification Report: \n', metrics.classification_report(y_valid, all_y_pred))
    
    return results, all_y_pred, CM_arr




# --------------------------------------------------------------------------------------------------    
# Function to plot the confusion matrix for the evaluated models

def plot_confusion_matrix(CM_arr, title):
    file_prefix = f'Confusion Matrix : {title}'
    temp_labels = ['Gear 0', 'Gear 2', 'Gear 3', 'Gear 4']

    x = np.true_divide(CM_arr, CM_arr.sum(axis=1, keepdims=True))

    plt.figure()
    sns.heatmap(x, annot=True,fmt='0.2%', xticklabels=temp_labels, yticklabels=temp_labels, cmap='YlGnBu')
    plt.title(file_prefix)
    plt.xlabel("Predicted Gear")
    plt.ylabel("True Gear")
    #plt.gcf()
    #plt.savefig('{}_Confusion Matrix'.format(file_prefix), bbox_to_anchor='tight')
    plt.show()
    #plt.close()


# --------------------------------------------------------------------------------------------------    
# Creating a correlation matrix to visualize the correlation between variables
# This will help us perform some feature selection based on filtering by correlation scores.

def plot_correlation_heatmap(df):

    corr = df.corr()

    plt.figure(figsize = (8,8))

    # Visualizing the correlation matrix
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap="YlGnBu",
        square=True,
        linewidths=.5,
        annot=True
    )
    plt.title('Training Data Set Correlation Matrix')

    #plt.gcf()
    #plt.savefig('Training Data Set Correlation Matrix', bbox_to_anchor='tight')

    plt.show()
    
    
# --------------------------------------------------------------------------------------------------    
# Creating a function that standardizes and scales the dataset
# Import libraries for different scalers if standard scaler is not preferred

def scale_dataset(df):
    
    data = df

    sc = StandardScaler()

    data = sc.fit_transform(data)
    data = pd.DataFrame(data)
    
    data.columns=df.columns

    return data

# --------------------------------------------------------------------------------------------------    

# Creating a function to plot all the features of a random forest classifier sorted by 
# feature importance. Takes in the model as an argument and plots the features in 
# descending order of their importance as calculated by the 'feature importance' feature of an RFC

def plot_feature_importance(classifier_obj, X_train, y_train):
    
    sel = SelectFromModel(classifier_obj)
    sel.fit(X_train, y_train)

    # Save the Feature Importances sorted by their score
    imp_score = np.array(sorted(zip(map(lambda x: round(x, 4), classifier_obj.feature_importances_), pd.DataFrame(X_train).columns.values), reverse=True))

    imp_features = int((np.count_nonzero(sel.get_support())))
    #imp_features = 3

    fig = plt.figure(figsize=(15, 5))

    plt.plot(list(imp_score[:,1]),list(imp_score[:,0].astype(float)), label='Feature Importance')
    plt.axvline(imp_features, color='r', linestyle='--', label='Maximum Selected Features = {} for Analysis'.format(imp_features))
    plt.xlabel("No. of Features")
    plt.ylabel("Feature Importance Score")
    plt.title("Dataset Features based on Importance Score")
    plt.legend(loc='best')
    plt.grid(True,linewidth=0.25)
    #plt.gcf()
    #plt.savefig('RFC Feature Selection Plot', bbox_to_anchor='tight')
    plt.show()
    #plt.close()
