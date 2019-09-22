from common.classification import zero_one_accuracy
from sklearn.model_selection import train_test_split
from common.data_handler.load_mnist import load_mnist
from machine_learning_models.knn import knn
x,y = load_mnist(500)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf_base = knn(k=5)

clf_base.fit(x_train,y_train)
clf_base.predict(x_test)
print(y_test)
print(zero_one_accuracy(y_test,clf_base.predict(x_test)))