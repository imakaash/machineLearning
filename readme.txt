Notes on the provided data:
Did you note something about the data? Does the data influence your choice of ML model.
The dataset contains combination of both categorical and numerical features. Some of the important discovery which I made was following:
The year attribute has values as low as 0 which is certainly wrong. Also, the kilometers attribute for few records have strangely very large values which is not possible for a cars lifetime.
Also, in few categorical variables, the value is same but represented in different cases, such as 'SUV' and 'suv' which represent the same meaning ideally.

With the raw data the scores turned out to be real bad. So I used outlier imputation (IQR based and Percentile trimming on lower and outer bounds) as well as few feature engineering methods on selected attributes.

Used ML model:
Which ML model did you use? Did you use some special hyperparameter settings?
I used Ridge model for the final output file generation as it performed better than the other models I tried with. The reason for using Ridge and Lasso models was to use it capability for feature selection and since this dataset had lots of features using them proved to improve the score to some extent.
I played around with the hyperparameter alpha and found out the default setting to be giving better results. Couldn't try other methods due to the large size of data and memory restrictions of the hardware.

Validation:
Did you do some validation of the trained model? If yes, how would you rate the performance of your model?
I validated my results using the validation/test set created out of the training dataset to evaluate the performance of the model. The best model gave an score (r2) of 0.20346167390860792 on training and 0.2071252384960891 on test set with MAE being 3697.4669448756217 for the test set predictions.
These are certainly not good scores and has a huge scope of further improvement.