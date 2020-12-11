import pandas as pd
import preprocessing
import learning as learn
import testing as test

def main() :
  csv_data = pd.read_csv('https://firebasestorage.googleapis.com/v0/b/fir-crud-36cbe.appspot.com/o/Iris.csv?alt=media&token=71bdac3f-96e5-4aae-9b60-78025c1d3330')
  csv_data = preprocessing.normalization(csv_data)
  dfTraining, dfTesting = preprocessing.splitData(csv_data)
  biases = (0.0001, 0.0001)
  weights = learn.learning(dfTraining, 0.5, biases)
  test.testing(dfTesting, weights, biases)

main()