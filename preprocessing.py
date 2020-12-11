import pandas as pd

def splitData(csv_data) :
  setosa_index = 1
  versicolor_index = 50
  virginica_index = 100
  dfTraining = csv_data.iloc[0:1,:]
  i = 1
  while i < 60 :
    if (i + 1) % 3 == 0 :
      dfTraining = dfTraining.append(csv_data.iloc[virginica_index:virginica_index+1,:])
      virginica_index += 1
    elif (i + 1) % 2 == 0 :
      dfTraining = dfTraining.append(csv_data.iloc[versicolor_index:versicolor_index+1,:])
      versicolor_index += 1
    else :
      dfTraining = dfTraining.append(csv_data.iloc[setosa_index:setosa_index+1,:])
      setosa_index += 1
    i += 1

  dfTesting = csv_data.iloc[30:50,:]
  dfTesting = dfTesting.append(csv_data.iloc[80:100,:])
  dfTesting = dfTesting.append(csv_data.iloc[130:150,:])
  
  return (dfTraining, dfTesting)

def searchMinMax(csv_data) :
  column = csv_data['SepalLengthCm']
  maxSepalLengthCm = column.max()
  minSepalLengthCm = column.min()

  column = csv_data['SepalWidthCm']
  maxSepalWidthCm = column.max()
  minSepalWidthCm = column.min()

  column = csv_data['PetalLengthCm']
  maxPetalLengthCm = column.max()
  minPetalLengthCm = column.min()

  column = csv_data['PetalWidthCm']
  maxPetalWidthCm = column.max()
  minPetalWidthCm = column.min()

  return (maxSepalLengthCm, minSepalLengthCm, maxSepalWidthCm, 
                 minSepalWidthCm, maxPetalLengthCm, minPetalLengthCm, 
                 maxPetalWidthCm, minPetalWidthCm)

def normalization(csv_data) :
  maxSepalLengthCm, minSepalLengthCm, maxSepalWidthCm, minSepalWidthCm, maxPetalLengthCm, minPetalLengthCm, maxPetalWidthCm, minPetalWidthCm = searchMinMax(csv_data)
  #############################
  listNewSepalLength = []
  listNewSepalWidth = []
  listNewPetalLength = []
  listNewPetalWidth = []
  for i in range(0, len(csv_data)) :
    listNewSepalLength.append((csv_data.iloc[i]['SepalLengthCm'] - minSepalLengthCm) / (maxSepalLengthCm - minSepalLengthCm))
    listNewSepalWidth.append((csv_data.iloc[i]['SepalWidthCm'] - minSepalWidthCm) / (maxSepalWidthCm - minSepalWidthCm))
    listNewPetalLength.append((csv_data.iloc[i]['PetalLengthCm'] - minPetalLengthCm) / (maxPetalLengthCm - minPetalLengthCm))
    listNewPetalWidth.append((csv_data.iloc[i]['PetalWidthCm'] - minPetalWidthCm) / (maxPetalWidthCm - minPetalWidthCm))
  newDf = pd.DataFrame({'SepalLengthCm' : listNewSepalLength,
                        'SepalWidthCm' : listNewSepalWidth,
                        'PetalLengthCm' : listNewPetalLength,
                        'PetalWidthCm' : listNewPetalLength})
  csv_data.update(newDf)
  return csv_data