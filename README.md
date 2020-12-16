# Data Driven Gait Analysis
## Classification of Cross-Country Skiing Techniques Using Supervised Learning
### TRA105 Digitalization in Sports | Chalmers University of Technology
#### Group Members:
 - David Larsson [larsdav@student.chalmers.se]
 - Savya Sachi Gupta [savya@student.chalmers.se]
---

Power meters are used to improve athlete performance and in cross-country skiing, power is dependent on technique (gear) used. In this project, we build a random forest classifier for predicting which gears a skier uses based on data from [Skisens](https://skisens.com) force measuring pole handles. 

## In this Package:
Below are the files used in our project with brief descriptions of what functions they serve. You may click on the links provided to directly navigate to these files. 

1. [functions/**analysis_setup_functions.py**](https://github.com/DavidLarssonIO/Data-driven-gait-analysis/blob/master/functions/analysis_setup_functions.py) : File containing all functions to perform classification, such as hyperparameter tuning, training, testing and visualization
2. [functions/**dataframe_functions.py**](https://github.com/DavidLarssonIO/Data-driven-gait-analysis/blob/master/functions/dataframe_functions.py) : File containing all data processing functions for raw data, such as cleaning, creating calculated fields, calibrating etc.
3. [main-data-classification](https://github.com/DavidLarssonIO/Data-driven-gait-analysis/blob/master/main-data-classification.ipynb) : Main notebook where we load data and run our classification models. This notebook imports all functions from the above two files and uses them. You may re-run this notebook for new data, further instruction of how to use the notebook are provided within the notebook.
4. [report-notebook/main-report-notebook.md](https://github.com/DavidLarssonIO/Data-driven-gait-analysis/blob/master/report-notebook/main-report-notebook.md) : This is the markdown/report version of the Jupyter notebook with our results, discussions and observations for future reference. 

*Note: It is important to download all files from this package, ensure all libraries used in this project are installed on your system and check the file paths mentioned in the notebook are pointing to your data correctly for smooth execution.*
***For detailed explanation and functioning of these operations, refer to these files and their comments***

