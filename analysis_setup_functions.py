import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn import metrics
from IPython.display import HTML




# --------------------------------------------------------------------------------------------------    
# Function to generate the trianing and validation datasets based on the number of skiers specified
# for the split. Skiers are randomly picked from the list of available 10 skiers. Everytime this
# function is called, a random set of skiers are split into these sets.

def split_train_test_data(df_info, skier_list, validation_skiers, to_train=False, special_case=99):

    print('**********************************************************************************************')
    
    val_skiers_list = []
    train_skiers_list = []
    
    if to_train:

        print('Randomly Splitting Data into Training and Validation Sets...')

        X = df_info[df_info.columns.difference(['Skier','Pole', 'Other pole time', 'Gear','Peak time'])]
        y = (df_info[['Gear']])
        y_plot_data = y

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y)
        y_train = y_train.values.ravel()
        y_valid = y_valid.values.ravel()

        print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
        
    else:
        print('Randomly Splitting Skiers into Training and Validation Sets...')
        val_skiers_list = rnd.sample(list(skier_list), validation_skiers)
        
        if special_case == 0:
            print('ADDITIONAL EXPERIMENT!...')
            print('Selecting Shortest Skiers for Validation...')
            print('Shortest Skiers are also all women, have shortest pole length and lowest mass...')
            val_skiers_list = [10,7,4]

        if special_case == 1:
            print('ADDITIONAL EXPERIMENT!...')
            print('Selecting Tallest Skiers for Validation...')
            print('Tallest Skiers have longest pole length and high mass...')
            val_skiers_list = [1,2,3]

        if special_case == 2:
            print('ADDITIONAL EXPERIMENT!...')
            print('Selecting Skiers with Lowest Body Mass Index (BMI) for Validation...')
            val_skiers_list = [2,8,9,10]

        if special_case == 3:
            print('ADDITIONAL EXPERIMENT!...')
            print('Selecting Skiers with Highest Body Mass Index (BMI) for Validation...')
            val_skiers_list = [1,3,4,7]

            
        train_skiers_list = list(set(skier_list) - set(val_skiers_list))

        print(f'Training Skiers : {train_skiers_list}')
        print(f'Validation Skiers : {val_skiers_list}')

        X = df_info[df_info.columns.difference(['Pole', 'Other pole time', 'Gear','Peak time'])]
        y = (df_info[['Gear','Skier']])

        # Storing validation data separately which will later be used to plot comparison of prediction
        y_plot_data = df_info.loc[(df_info['Skier'].isin(val_skiers_list))]

        # Creating training and validation sets based on random skiers stored above
        X_train = X.loc[(X['Skier'].isin(train_skiers_list))]
        X_valid = X.loc[(X['Skier'].isin(val_skiers_list))]
        y_train = y.loc[(y['Skier'].isin(train_skiers_list))]
        y_valid = y.loc[(y['Skier'].isin(val_skiers_list))]

        X_train = X_train[X_train.columns.difference(['Skier'])]
        X_valid = X_valid[X_valid.columns.difference(['Skier'])]
        y_train = y_train[y_train.columns.difference(['Skier'])]
        y_train = y_train.values.ravel()
        y_valid = y_valid[y_valid.columns.difference(['Skier'])]
        y_valid = y_valid.values.ravel()

        #print('Dataframe Sizes:')
        #print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
        
    return X_train, X_valid, y_train, y_valid, y_plot_data, train_skiers_list, val_skiers_list



# --------------------------------------------------------------------------------------------------    
# Function that will evaluate the selected model and return performance metrics for analysis
# A generic function is created to evaluate the model performance. 
# This is going to be used as a benchmark to evaluate how much the performance of the model has improved
# based on hyperparameter tuning.

def evaluate(model,X_train, y_train, X_valid, y_valid, title):
    
    # Fit the model with training data
    model.fit(X_train, y_train)

    print('Evaluating Data with 5-fold CV...\n')
    # Build the k-fold cross-validator
    kfold = KFold(n_splits=5)
    all_y_pred = cross_val_predict(model, X_valid, y_valid, cv=kfold)
    
    results = np.ndarray(shape=(11))
    CM_arr = metrics.confusion_matrix(y_valid, all_y_pred)

    # Create a dataframe that will store results in one table
    results_df = pd.DataFrame(columns=list(['Accuracy',
                                            'Balanced Accuracy',
                                            'Precision (Macro Avg)',
                                            'Recall (Macro Avg)',
                                            'F1 (Macro Avg)',
                                            'MCC',
                                            'Classifier Score']))
    
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

    results = np.around(results, decimals=3)
    
    print(f'Performance Metrics for {title} :')
    print('----------------------------------------------------------')
    #print('Accuracy Score:', results[0])
    #print('Balanced Accuracy Score:', results[1])
    #print('MCC Score:', results[5])
    #print('F-1 Score (Macro Avg):', results[4])
    #print('Classifier Score:', results[9])
    #print('Precision: (Macro Avg)', results[2])
    #print('Recall: (Macro Avg)', results[3])
    #print('Mean Absolute Error:', results[6])
    #print('Mean Squared Error:', results[7])
    #print('Root Mean Squared Error:', results[8])
    
    temp_all_metrics =  ["" for x in range(7)]
    temp_all_metrics[0:6] = results[0:6]
    temp_all_metrics[6] = results[9]

    # Print Results variable in a cleaner compact horizontal manner
    results_df.loc[len(results_df)] = temp_all_metrics
    display( HTML( results_df.to_html().replace("\\n","<br>") ) )

    
    class_report = metrics.classification_report(y_valid, all_y_pred)
    print('\nClassification Report: \n', class_report)
    
    plot_confusion_matrix(CM_arr, title)

    return results, all_y_pred, CM_arr, class_report


# --------------------------------------------------------------------------------------------------    
# A function to perform hyperparameter tuning for the random forest classifier model

def tune_random_forest(X_train, y_train):
    
    print('Performing Hyperparameter Tuning...\n')
    #-----------------------------------------------------------------------------------------
    # SETTING UP PARAMETERS FOR HYPERPARAMETER TUNING
    #-----------------------------------------------------------------------------------------
    # Specifiying the range of Number of trees in the random forest
    n_estimators = [int(x) for x in np.linspace(start = 6, stop = 120, num = 6)]

    # Specifying the range of Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
    max_depth.append(None)

    # Specifying the range of Minimum number of samples required to split a node
    min_samples_split = [1, 2, 4, 6]

    # Specifying the range of Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 6]

    # Specifying the Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Specifying the Class weight
    class_weight = ['balanced', 'balanced_subsample']

    # Now, we create the random grid that would store all the values as specified above
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                  'class_weight' : class_weight
                  }


    #-----------------------------------------------------------------------------------------
    # PERFORMING HYPERPARAMETER TUNING
    #-----------------------------------------------------------------------------------------
    # Now, using the random grid created above, we start to search for the best hyperparameters

    # First we create the base model that we want to tune by specifying no parameters
    rf = RandomForestClassifier()

    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                  n_iter = 100, scoring='balanced_accuracy', 
                                  cv = 3,
                                   #verbose=2,
                                   # random_state=42,
                                   n_jobs=-1,
                                  return_train_score=True)

    # Fit the random search model
    rf_random.fit(X_train, y_train);

    print('Best Parameters after Randomized Search CV : ')
    print(rf_random.best_params_)
    print('\n')

    #-----------------------------------------------------------------------------------------
    # EVALUATE MODEL WITH CHOSEN PARAMETERS
    #-----------------------------------------------------------------------------------------
    # Build the Random Forest Classifier model based on best set of parameters chosen above
    rfc = rf_random.best_estimator_
    
    # Fit the model with training data
    rfc.fit(X_train, y_train)

    return rfc, rf_random.best_params_


# --------------------------------------------------------------------------------------------------    
# Function to perform hyperparameter tuning for a Multiclass Logistic Regression Problem

def tune_logistic_regression(X_train, y_train):

    print('Performing Hyperparameter Tuning...\n')
    #-----------------------------------------------------------------------------------------
    # SETTING UP PARAMETERS FOR HYPERPARAMETER TUNING
    #-----------------------------------------------------------------------------------------
    # Create regularization penalty space
    penalty = ['l1', 'l2', 'elasticnet']

    # Specify if Dual or primal formulation
    dual=[True,False]

    # Create regularization hyperparameter space
    C = np.logspace(0, 2.5, 15)

    # define grid for different class weights
    balance=['balanced']

    #Specify the solver
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    
    # Specifying the range of maximum oterations to enable convergence
    max_iter = np.linspace(100, 10000, 100).astype(int)

    param_grid = dict(dual=dual, class_weight=balance, C=C, solver=solver, penalty=penalty, max_iter=max_iter)


    #-----------------------------------------------------------------------------------------
    # PERFORMING HYPERPARAMETER TUNING
    #-----------------------------------------------------------------------------------------
    # Now, using the random grid created above, we start to search for the best hyperparameters

    # First we create the base model that we want to tune by specifying no parameters
    lr = LogisticRegression()

    lr_random = RandomizedSearchCV(estimator=lr, param_distributions=param_grid,
                                   n_jobs=-1, scoring='balanced_accuracy', cv = 3,
                                  return_train_score=True)

    #lr_random_result = lr_random.fit(X_train, y_train)
    lr_random.fit(X_train, y_train)


    print('Best Parameters after Randomized Search CV : ')
    print(lr_random.best_params_)
    print('\n')

    #-----------------------------------------------------------------------------------------
    # EVALUATE MODEL WITH CHOSEN PARAMETERS
    #-----------------------------------------------------------------------------------------
    # Build the Random Forest Classifier model based on best set of parameters chosen above
    lr = lr_random.best_estimator_

    # Fit the model with training data
    lr.fit(X_train, y_train)

    return lr, lr_random.best_params_
    
    
    
# --------------------------------------------------------------------------------------------------    
# Function that wil perform a majority filtering on the list of predicted labels based on the width 
#  the window width. It calculates the label occurring most frequently in the current window and
# sets the current value to that label.
# This function can be useful to smooth the predicted labels based on surrounding values.

def majority_filter_traditional(seq, width):
    offset = width // 2
    seq = [0] * offset + seq
    result = []
    for i in range(len(seq) - offset):
        a = seq[i:i+width]
        result.append(max(set(a), key=a.count))
    return result
 

    
# --------------------------------------------------------------------------------------------------    
# Function that uses the above majority filtering technique to refine the predicted labels and
# calculate the performance metrics again to determine if there has been an improvement in the 
# model due to this method.
    
def refine_predicted_labels( y_valid, all_y_pred, title):
    
    all_y_pred = majority_filter_traditional(list(all_y_pred), 15)  

    results = np.ndarray(shape=(11))
    CM_arr = metrics.confusion_matrix(y_valid, all_y_pred)

    # Create a dataframe that will store results in one table
    results_df = pd.DataFrame(columns=list(['Accuracy',
                                            'Balanced Accuracy',
                                            'Precision (Macro Avg)',
                                            'Recall (Macro Avg)',
                                            'F1 (Macro Avg)',
                                            'MCC']))
    
    results[0] = metrics.accuracy_score(y_valid, all_y_pred) #Accuracy
    results[1] = metrics.balanced_accuracy_score(y_valid, all_y_pred) #Balanced Accuracy
    results[2] = metrics.precision_score(y_valid, all_y_pred,average='macro')   #Precision
    results[3] = metrics.recall_score(y_valid, all_y_pred,average='macro')   #Sensitivity/Recall/True positive rate (TPR)
    results[4] = metrics.f1_score(y_valid, all_y_pred, average='macro')
    results[5] = metrics.matthews_corrcoef(y_valid, all_y_pred)
    results[6] = metrics.mean_absolute_error(y_valid, all_y_pred)
    results[7] = metrics.mean_squared_error(y_valid, all_y_pred)
    results[8] = np.sqrt(metrics.mean_squared_error(y_valid, all_y_pred))

    results = np.around(results, decimals=3)
    
    print(f'Performance Metrics for {title} :')
    print('----------------------------------------------------------')
    #print('Accuracy Score:', results[0])
    #print('Balanced Accuracy Score:', results[1])
    #print('MCC Score:', results[5])
    #print('F-1 Score (Macro Avg):', results[4])
    #print('Classifier Score:', results[9])
    #print('Precision: (Macro Avg)', results[2])
    #print('Recall: (Macro Avg)', results[3])
    #print('Mean Absolute Error:', results[6])
    #print('Mean Squared Error:', results[7])
    #print('Root Mean Squared Error:', results[8])
    
    temp_all_metrics =  ["" for x in range(6)]
    temp_all_metrics[0:6] = results[0:6]

    # Print Results variable in a cleaner compact horizontal manner
    results_df.loc[len(results_df)] = temp_all_metrics
    display( HTML( results_df.to_html().replace("\\n","<br>") ) )

    
    class_report = metrics.classification_report(y_valid, all_y_pred)
    print('\nClassification Report: \n', class_report)
    
    plot_confusion_matrix(CM_arr, title)

    return results, all_y_pred, CM_arr, class_report

    

# --------------------------------------------------------------------------------------------------    
# Function to plot the original gear profile versus the predicted gears to understand 
# which of the data points from the original data were inccorectly classified
# Viewing this plot helps to understand accuracy of the model results
# Set individual_skier=True if you want the plots to be returned individually for all
# skiers of the validation dataset

def plot_predicted_gear_comparison(y_plot_data,all_y_pred,individual_skier=False):

    plot_df = y_plot_data
    plot_df = plot_df.assign(Predicted_Gear=all_y_pred)
    skier_list = plot_df['Skier'].unique()
    
    if individual_skier:

        for i in range(0,len(skier_list)):
            temp_df = plot_df.loc[(plot_df['Skier'] == skier_list[i])]
            fig, axs = plt.subplots(figsize=(15,4))
            axs.set_facecolor((248/255, 248/255, 248/255))
            #plt.title(f'Skier {skier_list[i]} : Original vs Predicted Gears Comparison')
            plt.xlabel('Observations')
            plt.plot(range(0,temp_df.shape[0]), temp_df['Gear'],alpha=0.75, color='green', label='True Gear Profile')
            plt.scatter(range(0,temp_df.shape[0]), temp_df['Predicted_Gear'],alpha=0.5, c=temp_df.Predicted_Gear, cmap='plasma', label='Predicted Gears')
            axs.set_yticks([-1,0,2,3,4])
            #axs.set_yticklabels(['','Gear 0', 'Gear 2', 'Gear 3', 'Gear 4'])
            axs.set_yticklabels(['','Gear 0 \n (Double \n Poling)', 'Gear 2', 'Gear 3', 'Gear 4'])
            plt.legend(loc='lower left')
            plt.show()
        
    else:    

        temp_df = plot_df

        fig, axs = plt.subplots(figsize=(17,5))
        axs.set_facecolor((248/255, 248/255, 248/255))
        #plt.title(f'All Validation Skiers Combined : Original vs Predicted Gears Comparison')
        plt.xlabel('Observations')
        plt.plot(range(0,temp_df.shape[0]), temp_df['Gear'],alpha=0.75, color='green', label='True Gear Profile')
        plt.scatter(range(0,temp_df.shape[0]), temp_df['Predicted_Gear'],alpha=0.5, c=temp_df.Predicted_Gear, cmap='plasma', label='Predicted Gears')
        axs.set_yticks([-1,0,2,3,4])
        #axs.set_yticklabels(['','Gear 0', 'Gear 2', 'Gear 3', 'Gear 4'])
        axs.set_yticklabels(['','Gear 0 \n (Double \n Poling)', 'Gear 2', 'Gear 3', 'Gear 4'])
        plt.legend(loc='lower left')
        plt.show()
    

    
    
# --------------------------------------------------------------------------------------------------    
# Function to plot the confusion matrix for the evaluated models

def plot_confusion_matrix(CM_arr, title):
    #file_prefix = f'Confusion Matrix : Accuracy of {title}'
    #temp_labels = ['Gear 0', 'Gear 2', 'Gear 3', 'Gear 4']
    temp_labels = ['Gear 0 \n(Double \n Poling)', 'Gear 2', 'Gear 3', 'Gear 4']


    # Calculation to convert predicted numbers to accuracy
    x = np.true_divide(CM_arr, CM_arr.sum(axis=1, keepdims=True))

    #plt.figure()
    figure = plt.subplots(figsize=(5.25,4.5))
    
    sns.heatmap(x, annot=True,fmt='0.2%', xticklabels=temp_labels, yticklabels=temp_labels, cmap='YlGnBu',cbar=False)
    #plt.title(file_prefix)
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
    #plt.axvline(imp_features, color='r', linestyle='--', label='Maximum Selected Features = {} for Analysis'.format(imp_features))
    plt.xlabel("No. of Features")
    plt.ylabel("Feature Importance Score")
    plt.title("Dataset Features based on Importance Score")
    plt.legend(loc='best')
    plt.grid(True,linewidth=0.25)
    #plt.gcf()
    #plt.savefig('RFC Feature Selection Plot', bbox_to_anchor='tight')
    plt.show()
    #plt.close()
    
    print('\nFeature Importance Scores:\n')
    print(imp_score)
