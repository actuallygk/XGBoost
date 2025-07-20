import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=42)
d_train = xgb.DMatrix(x_train, label = y_train)
d_test = xgb.DMatrix(x_test,label=y_test)
params = {
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': 3,
    'max_depth': 4,
    'eta': 0.3,
    'seed': 42
}
epochs = 50
model = xgb.train(params,d_train,epochs)
y_pred = model.predict(d_test)
accuracy=accuracy_score(y_pred,y_test)
print(accuracy)