# mlops_vnit2
This is the Repository for MLOPs course
print('Hello World')
Add new File
import pandas as pd
import mlflow
import mlflow.sklearn
## enable autologging
# mlflow.sklearn.autolog()

mlflow.set_tracking_uri(uri="http://127.0.0.1:5004")

## enable autologging
# mlflow.sklearn.autolog()

mlflow.set_tracking_uri(uri="http://127.0.0.1:5004")

## create a new MLflow Experiment
mlflow.set_experiment("Manufacturing_Dept")
df = pd.read_csv("titanic.csv")

df = pd.read_csv("titanic.csv")
df = df.fillna(0)
df.info()
df.head(3)
df["gender_enc"]=df["Sex"].astype('category').cat.codes
df["embark_enc"]=df["Embarked"].astype('category').cat.codes
X = df[["Pclass","Age","gender_enc","embark_enc","Fare","SibSp","Parch"]]
Y = df["Survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
with mlflow.start_run():
    # step:1 initialise the model class
    model = DecisionTreeClassifier(criterion="entropy",max_depth=5)
    mlflow.log_params({'criterion':'entropy','max_depth':5})
    #step:2 train the model over training data
    model.fit(X_train,y_train)
    mlflow.log_params({'train_size':X_train.shape[0]})
    #step:3 predict this over test_set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)*100
    mlflow.log_metric("accuracy",acc)
    mlflow.set_tag("Training info","Basic Decsion Tree model on titanic dataset")
