# bugbustingdiabetes
## Check out the app ---> click [here](google.com)
### Inspiration for [streamlit](https://streamlit.io/) app layout from [playground](https://github.com/ahmedbesbes/playground/tree/d0617ea8f5f9782583104b6987b813c5163b0d38) by Ahmed Besbes

```
Type 2 diabetes (T2D) mellitus is a metabolic disorder that occurs in 
humans. It is a chronic illness caused by the body’s lack of insulin 
production or insulin resistance. Insulin is a hormone produced by the 
pancreatic beta cells, which regulates the amount of glucose in the 
bloodstream by promoting glucose absorption into liver, fat, and skeletal 
muscle cells1. The aim of this project is to develop models to predict the 
risk of diabetes. This study used the Pima Diabetes dataset. As part of 
data exploration, we examined the types of columns and the missing values 
for both the features and target column in the dataset. A flaw in this 
dataset was that it was biased towards certain groups who were predisposed 
to T2D. If a person already falls into that group, this data set provides 
excellent predictability of their risk of developing diabetes. 
Additionally, the more pregnancies a woman had, the more reliable the 
dataset was for predicting her chances of developing T2D.

We created a machine learning model and a sequential model that could
predict the outcome of a patient having diabetes. With the machine
learning model, we then ran a KNN and SVM on the data and calculated the
accuracy of the model. The initial model had a 67% accuracy of KNN and 73%
accuracy for SVM. We wanted to increase the accuracy of the model, so we
looked at the correlation of the features to the outcome of diabetes. We
found that the top four correlated features were (in order): Glucose, BMI,
Age, and Pregnancies. We extracted these four features from the data and
improved the accuracy of KNN to 73% and SVM to 77%. With the sequential
model, we created a sequential model with 2 layers and sigmoid for the
activation. We trained the model and ran test data through the model. The
model gave a 67% accuracy with the test data. Running the model again
using the top 4 correlated features increases the accuracy to 75%. We plan
to further assess how we can increase the accuracy of these models by
tuning them.

References
qbal A, Novodvorsky P, Heller SR. Recent Updates on Type 1 Diabetes
Mellitus Management for Clinicians. Diabetes Metab J. 2018 Feb;42(1):3-18
Erratum in: Diabetes Metab J. 2018 Apr;42(2):177 

```