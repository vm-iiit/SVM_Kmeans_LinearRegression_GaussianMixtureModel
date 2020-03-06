from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


from q3 import Airfoil as ar
model3 = ar()
model3.train('./Datasets/Question-3/airfoil.csv') # Path to the train.csv will be provided
prediction3 = model3.predict('./Datasets/Question-3/airfoil (test).csv') # Path to the test.csv will be provided
prediction3 should be Python 1-D List
'''WE WILL CHECK THE MEAN SQUARED ERROR OF PREDICTIONS WITH THE GROUND TRUTH VALUES'''


from q4 import Weather as wr
model4 = wr()
model4.train('./Datasets/Question-4/train.csv') # Path to the train.csv will be provided 
prediction4 = model4.predict('./Datasets/Question-4/test.csv') # Path to the test.csv will be provided
prediction4 should be Python 1-D List
# '''WE WILL CHECK THE MEAN SQUARED ERROR OF PREDICTIONS WITH THE GROUND TRUTH VALUES'''


from q5 import AuthorClassifier as ac
auth_classifier = ac()
auth_classifier.train('./Datasets/Question-5/Train.csv') # Path to the train.csv will be provided
predictions = auth_classifier.predict('./Datasets/Question-5/Test.csv') # Path to the test.csv will be provided

'''WE WILL CHECK THE PREDICTIONS WITH THE GROUND TRUTH LABELS'''


from q6 import Cluster as cl
cluster_algo = cl()
predictions = cluster_algo.cluster('./Datasets/Question-6/') 

'''SCORE BASED ON THE ACCURACY OF CLUSTER LABELS'''