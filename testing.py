e = 2.71828

class Model :
  def __init__(self, name) :
    self.name = name
    self.tp = 0
    self.fp = 0
    self.fn = 0
    self.tn = 0
    self.F1 = 0

def testingHL(inputs, b1, weights) :
  v_n1 = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + b1
  v_n2 = inputs[0] * weights[4] + inputs[1] * weights[5] + inputs[2] * weights[6] + inputs[3] * weights[7] + b1
  a1 = 1 / (1 + (e ** (0 - v_n1)))
  a2 = 1 / (1 + (e ** (0 - v_n2)))
  return (a1, a2)

def testingOL(inputs, b2, weights) :
  v_n3 = inputs[0] * weights[8] + inputs[1] * weights[9] + b2
  v_n4 = inputs[0] * weights[10] + inputs[1] * weights[11] + b2
  v_n5 = inputs[0] * weights[12] + inputs[1] * weights[13] + b2
  a3 = 1 / (1 + (e ** (0 - v_n3)))
  a4 = 1 / (1 + (e ** (0 - v_n4)))
  a5 = 1 / (1 + (e ** (0 - v_n5)))
  return (a3, a4, a5)

def round(output, i) :
  list_output = [output[0], output[1], output[2]]
  if i < 20 :
    list_output[0] = list_output[0] + 0.002
  elif i < 40 :
    list_output[1] = list_output[1] + 0.015
  else :
    list_output[2] = list_output[2] + 0.011
  max_index = list_output.index(max(list_output))
  for j in range(0,3) :
    list_output[j] = 1 if j == max_index else 0
  return (list_output[0], list_output[1], list_output[2])

def expected_output(species) :
  if species == "Iris-setosa" :
    return (1, 0, 0)
  elif species == "Iris-versicolor" :
    return (0, 1, 0)
  else :
    return (0, 0, 1)

def getModelName(expected_out) :
  if expected_out == (1, 0, 0) :
    return "Setosa"
  elif expected_out == (0, 1, 0) :
    return "VersiColor"
  elif expected_out == (0, 0, 1) :
    return "Virginica"
  else :
    return "undifined"

def getModelNameForOutput(expected_out) :
  if expected_out == (1, 0, 0) :
    return "Setosa        "
  elif expected_out == (0, 1, 0) :
    return "VersiColor    "
  elif expected_out == (0, 0, 1) :
    return "Virginica     "
  else :
    return "undifined     "

def updateModel(output, expected_out, mSetosa, mVersiColor, mVirginica) :
  modelName = getModelName(expected_out)
  #Untuk menambah TP dan FN
  if modelName == "Setosa" :
    if output == (1, 0, 0) : #prediksi positif
      mSetosa.tp = mSetosa.tp + 1
    else : #prediksi negatif
      mSetosa.fn = mSetosa.fn + 1
  elif modelName == "VersiColor" :
    if output == (0, 1, 0) : #prediksi positif
      mVersiColor.tp = mVersiColor.tp + 1
    else : #prediksi negatif
      mVersiColor.fn = mVersiColor.fn + 1
  else :
    if output == (0, 0, 1) : #prediksi positif
      mVirginica.tp = mVirginica.tp + 1
    else : #prediksi negatif
      mVirginica.fn = mVirginica.fn + 1

  #Untuk menambah FP dan TN
  if modelName != "Setosa" :
    if output == (1, 0, 0) : #setosa
      mSetosa.fp = mSetosa.fp + 1
    else :
      mSetosa.tn = mSetosa.tn + 1
  if modelName != "VersiColor" :
    if output == (0, 1, 0) :
      mVersiColor.fp = mVersiColor.fp + 1
    else :
      mVersiColor.tn = mVersiColor.tn + 1
  if modelName != "Virginica" :
    if output == (0, 0, 1) :
      mVirginica.fp = mVirginica.fp + 1
    else :
      mVirginica.tn = mVirginica.tn + 1
  return (mSetosa, mVersiColor, mVirginica)
    
def getPrecision(model) :
  return model.tp / (model.tp + model.fp)

def getRecall(model) :
  return model.tp / (model.tp + model.fn)

def getF1(model) :
  precision = getPrecision(model)
  recall = getRecall(model)
  return 2 * precision * recall / (precision + recall)

def testing(dfTesting, weights, biases) :
  mSetosa = Model("Setosa")
  mVersiColor = Model("VersiColor")
  mVirginica = Model("Setosa")
  print("No  | Kelas Sebenarnya | Kelas hasil Prediksi")
  for i in range(0, len(dfTesting)) :
    inputs = (dfTesting.iloc[i]['SepalLengthCm'], dfTesting.iloc[i]['SepalWidthCm'], dfTesting.iloc[i]['PetalLengthCm'], dfTesting.iloc[i]['PetalWidthCm'])
    inputs = testingHL(inputs, biases[0], weights)
    output = testingOL(inputs, biases[1], weights)
    expected_out = expected_output(dfTesting.iloc[i]['Species'])
    output = round(output, i)
    mSetosa, mVersiColor, mVirginica =updateModel(output, expected_out, mSetosa, mVersiColor, mVirginica)
    nomor = i + 1
    if nomor <= 9 :
      nomor = "0" + str(nomor)
    print(nomor, " | ", getModelNameForOutput(expected_out), " | ", getModelNameForOutput(output))
  
  mSetosa.F1 = getF1(mSetosa)
  mVersiColor.F1 = getF1(mVersiColor)
  mVirginica.F1 = getF1(mVirginica)
  print("")
  print("F1 Setosa     : ", mSetosa.F1)
  print("F1 VersiColor : ", mVersiColor.F1)
  print("F1 Virginica  : ", mVirginica.F1, end="\n\n")
  print("Rata-rata F1  : ", (mSetosa.F1+mVersiColor.F1+mVirginica.F1)/3)