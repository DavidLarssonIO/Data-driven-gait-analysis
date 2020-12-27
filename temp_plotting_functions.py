# --------------------------------------------------------------------------------------------------    
# Function to plot the original gear profile versus the predicted gears to understand 
# which of the data points from the original data were inccorectly classified
# Viewing this plot helps to understand accuracy of the model results

# y_valid     ---> True Gear Labels
# all_y_pred  ---> Predicted Gear Labels

def temp_plot_predicted_gear_comparison(y_valid,all_y_pred):

    fig, axs = plt.subplots(figsize=(17,5))
    axs.set_facecolor((248/255, 248/255, 248/255))
    
    # Dont need to include the below title command, so that the professor can provide any caption to the figure as theyd like
    #plt.title(f'All Validation Skiers Combined : Original vs Predicted Gears Comparison')
    plt.xlabel('Observations')
    
    '''
    This plots the original labels of the test set:
    X Axis -> No. of observations. Change the range X value as per your code
    Y Axis -> True Gear Labels
    '''
    plt.plot(range(0,len(y_valid)), y_valid,alpha=0.75, color='green', label='True Gear Profile')
    
    '''
    This plots the predicted labels of the test set as a scatter plot:
    X Axis -> No. of observations. Change the range X value as per your code
    Y Axis -> Predicted Gear Labels
    '''
    plt.scatter(range(0,len(y_valid)), all_y_pred,alpha=0.5, c=all_y_pred, cmap='plasma', label='Predicted Gears')

    
    axs.set_yticks([-1,0,2,3,4])
    axs.set_yticklabels(['','Gear 0 \n (Double \n Poling)', 'Gear 2', 'Gear 3', 'Gear 4'])
    plt.legend(loc='lower left')
    #plt.gcf()
    #plt.savefig('{}_Your-File-Name'.format('temp'), bbox_to_anchor='tight')
    plt.show()


    
    
# --------------------------------------------------------------------------------------------------    
# Function to plot the confusion matrix for the evaluated models
# This uses a confusion matrix calculated as below as an argument to generate the plot.
# CM_arr = metrics.confusion_matrix(y_valid, all_y_pred)

# y_valid     ---> True Gear Labels
# all_y_pred  ---> Predicted Gear Labels

def temp_plot_confusion_matrix(CM_arr):

    #file_prefix = f'Confusion Matrix'
    
    temp_labels = ['Gear 0 \n(Double \n Poling)', 'Gear 2', 'Gear 3', 'Gear 4']

    # Calculation to convert predicted numbers to accuracy
    x = np.true_divide(CM_arr, CM_arr.sum(axis=1, keepdims=True))
    
    figure = plt.subplots(figsize=(5.25,4.5))
    #plt.figure()
    sns.heatmap(x, annot=True,fmt='0.2%', xticklabels=temp_labels, yticklabels=temp_labels, cmap='YlGnBu',cbar=False)
    
    # Dont need to include the below title command, so that the professor can provide any caption to the figure as theyd like
    #plt.title(file_prefix)
    plt.xlabel("Predicted Gear")
    plt.ylabel("True Gear")
    #plt.gcf()
    #plt.savefig('{}_Your-File-Name'.format(file_prefix), bbox_to_anchor='tight')
    plt.show()
    #plt.close()
