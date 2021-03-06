model_infos = {
    "Bug Buster's Neural Network": """
        Bug Buster's Neural Network:
        - This model uses Keras Functional API
        - Created by the Bug Busters
    """,
    "Bug Buster's Testing Network": """
        Bug Buster's Testing Network:
        - This is a Sequential model
        - This model is an attempt to prevent our model from overtraining
        - The validation accuracy stops increasing in the Bug Buster's Neural Network and the Simple Neural Network
        - A dropout layer was added to the input layer and each node has a 20% chance of dropping
    """,
    "Simple Neural Network": """
        Simple Neural Network by Krishna Reddy Maryada:
        - This model uses Keras Functional API
        - Model taken from Kaggle
    """,
    "KNN": """
        K Neighbors Classifier (machine learning model):
        - The K-nearest neighbors algorithm is a non-parametric classification method
        and is used for classification and regression.
        - We chose 2 for the value of n_neighbors when we create the K Neighbors Classifier.
    """,
    "SVM": """
        Support Vector Machine (machine learning model):
        - Support Vector Machines are machine learning models that analyze data for classification
        and regression analysis.
        - The SVM takes the set of training values from the diabetes dataset with 2 classifications
        (0 for no diabetes, 1 for diabetes) and assigns new examples to one category or the other.
        SVM maps training examples to points in space so it can maximize the width of the gap between
        the two categories. 
    """,
}

model_urls = {
    "Bug Buster's Neural Network": "https://colab.research.google.com/drive/1Lizd4yhYgtrL0XHH_Ck_6GTwC7RY-oX3",
    "Bug Buster's Testing Network": "https://drive.google.com/file/d/1xLtFilJ9xXLv0Cc12wZ_2OVHFNBrcDlM/view?usp=sharing",
    "Simple Neural Network": "https://www.kaggle.com/kredy10/simple-neural-network-for-diabetes-prediction",
    "KNN": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
    "SVM": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
}