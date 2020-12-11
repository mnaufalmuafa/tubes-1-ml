e = 2.71828

def backwardHL(inputs, tuple_A, a_expectations, learning_rate, weights) :
  w1_error = ((tuple_A[2] - a_expectations[0]) * tuple_A[2] * (1- tuple_A[2]) * weights[8] + (tuple_A[3] - a_expectations[1]) * tuple_A[3] * (1 - tuple_A[3]) * weights[10] + (tuple_A[4] - a_expectations[2]) * tuple_A[4] * (1 - tuple_A[4]) * weights[12]) + tuple_A[0] * (1 - tuple_A[0]) * inputs[0]
  w2_error = ((tuple_A[2] - a_expectations[0]) * tuple_A[2] * (1- tuple_A[2]) * weights[8] + (tuple_A[3] - a_expectations[1]) * tuple_A[3] * (1 - tuple_A[3]) * weights[10] + (tuple_A[4] - a_expectations[2]) * tuple_A[4] * (1 - tuple_A[4]) * weights[12]) + tuple_A[0] * (1 - tuple_A[0]) * inputs[1]
  w3_error = ((tuple_A[2] - a_expectations[0]) * tuple_A[2] * (1- tuple_A[2]) * weights[8] + (tuple_A[3] - a_expectations[1]) * tuple_A[3] * (1 - tuple_A[3]) * weights[10] + (tuple_A[4] - a_expectations[2]) * tuple_A[4] * (1 - tuple_A[4]) * weights[12]) + tuple_A[0] * (1 - tuple_A[0]) * inputs[2]
  w4_error = ((tuple_A[2] - a_expectations[0]) * tuple_A[2] * (1- tuple_A[2]) * weights[8] + (tuple_A[3] - a_expectations[1]) * tuple_A[3] * (1 - tuple_A[3]) * weights[10] + (tuple_A[4] - a_expectations[2]) * tuple_A[4] * (1 - tuple_A[4]) * weights[12]) + tuple_A[0] * (1 - tuple_A[0]) * inputs[3]

  w5_error = ((tuple_A[2] - a_expectations[0]) * tuple_A[2] * (1- tuple_A[2]) * weights[9] + (tuple_A[3] - a_expectations[1]) * tuple_A[3] * (1 - tuple_A[3]) * weights[11] + (tuple_A[4] - a_expectations[2]) * tuple_A[4] * (1 - tuple_A[4]) * weights[13]) + tuple_A[1] * (1 - tuple_A[1]) * inputs[0]
  w6_error = ((tuple_A[2] - a_expectations[0]) * tuple_A[2] * (1- tuple_A[2]) * weights[9] + (tuple_A[3] - a_expectations[1]) * tuple_A[3] * (1 - tuple_A[3]) * weights[11] + (tuple_A[4] - a_expectations[2]) * tuple_A[4] * (1 - tuple_A[4]) * weights[13]) + tuple_A[1] * (1 - tuple_A[1]) * inputs[0]
  w7_error = ((tuple_A[2] - a_expectations[0]) * tuple_A[2] * (1- tuple_A[2]) * weights[9] + (tuple_A[3] - a_expectations[1]) * tuple_A[3] * (1 - tuple_A[3]) * weights[11] + (tuple_A[4] - a_expectations[2]) * tuple_A[4] * (1 - tuple_A[4]) * weights[13]) + tuple_A[1] * (1 - tuple_A[1]) * inputs[0]
  w8_error = ((tuple_A[2] - a_expectations[0]) * tuple_A[2] * (1- tuple_A[2]) * weights[9] + (tuple_A[3] - a_expectations[1]) * tuple_A[3] * (1 - tuple_A[3]) * weights[11] + (tuple_A[4] - a_expectations[2]) * tuple_A[4] * (1 - tuple_A[4]) * weights[13]) + tuple_A[1] * (1 - tuple_A[1]) * inputs[0]

  weights[0] = weights[0] - learning_rate * w1_error
  weights[1] = weights[1] - learning_rate * w2_error
  weights[2] = weights[2] - learning_rate * w3_error
  weights[3] = weights[3] - learning_rate * w4_error
  weights[4] = weights[4] - learning_rate * w5_error
  weights[5] = weights[5] - learning_rate * w6_error
  weights[6] = weights[6] - learning_rate * w7_error
  weights[7] = weights[7] - learning_rate * w8_error

  return weights

def backwardOL(tuple_A, a_expectations, learning_rate, weights) :
  w9_error = (tuple_A[2] - a_expectations[0]) + tuple_A[2] * (1 - tuple_A[2]) + tuple_A[0]
  w10_error = (tuple_A[2] - a_expectations[0]) + tuple_A[2] * (1 - tuple_A[2]) + tuple_A[1]

  w11_error = (tuple_A[3] - a_expectations[1]) + tuple_A[3] * (1 - tuple_A[3]) + tuple_A[0]
  w12_error = (tuple_A[3] - a_expectations[1]) + tuple_A[3] * (1 - tuple_A[3]) + tuple_A[1]

  w13_error = (tuple_A[4] - a_expectations[2]) + tuple_A[4] * (1 - tuple_A[4]) + tuple_A[0]
  w14_error = (tuple_A[4] - a_expectations[2]) + tuple_A[4] * (1 - tuple_A[4]) + tuple_A[1]

  weights[8] = weights[8] - learning_rate * w9_error
  weights[9] = weights[9] - learning_rate * w10_error
  weights[10] = weights[10] - learning_rate * w11_error
  weights[11] = weights[11] - learning_rate * w12_error
  weights[12] = weights[12] - learning_rate * w13_error
  weights[13] = weights[13] - learning_rate * w14_error

  return weights

def backwardPropagation(inputs, tuple_V, tuple_A, tuple_error, total_error, a_expectations, learning_rate, weights) :
  weights = backwardOL(tuple_A, a_expectations, learning_rate, weights)
  weights = backwardHL(inputs, tuple_A, a_expectations, learning_rate, weights)
  return weights

def forwardHL(inputs, b1, weights) :
  v_n1 = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + b1
  v_n2 = inputs[0] * weights[4] + inputs[1] * weights[5] + inputs[2] * weights[6] + inputs[3] * weights[7] + b1
  a1 = 1 / (1 + (e ** (0 - v_n1)))
  a2 = 1 / (1 + (e ** (0 - v_n2)))
  return ((v_n1, v_n2) ,(a1, a2))

def forwardOL(inputs, b2, weights) :
  v_n3 = inputs[0] * weights[8] + inputs[1] * weights[9] + b2
  v_n4 = inputs[0] * weights[10] + inputs[1] * weights[11] + b2
  v_n5 = inputs[0] * weights[12] + inputs[1] * weights[13] + b2
  a3 = 1 / (1 + (e ** (0 - v_n3)))
  a4 = 1 / (1 + (e ** (0 - v_n4)))
  a5 = 1 / (1 + (e ** (0 - v_n5)))
  return ((v_n3, v_n4, v_n5), (a3, a4, a5))

def forwardPropagation(inputs, biases, a_expectations, weights) :
  tuple_V_HL, tuple_A_HL = forwardHL(inputs, biases[0], weights)
  tuple_V_OL, tuple_A_OL = forwardOL(tuple_A_HL, biases[1], weights)
  e1 = 0.5 * (a_expectations[0] - tuple_A_OL[0]) * (a_expectations[0] - tuple_A_OL[0])
  e2 = 0.5 * (a_expectations[1] - tuple_A_OL[1]) * (a_expectations[1] - tuple_A_OL[1])
  e3 = 0.5 * (a_expectations[2] - tuple_A_OL[2]) * (a_expectations[2] - tuple_A_OL[2])

  tuple_V = tuple_V_HL + tuple_V_OL
  tuple_A = tuple_A_HL + tuple_A_OL
  tuple_error = (e1, e2, e3)
  total_error = e1 + e2 + e3

  return (tuple_V, tuple_A, tuple_error, total_error)

def expected_a(species) :
  if species == "Iris-setosa" :
    return (1, 0, 0)
  elif species == "Iris-versicolor" :
    return (0, 1, 0)
  else :
    return (0, 0, 1)

def learning(dfTraining, learning_rate, biases) :
  weights = [0.8, 0.7, 0.4, 0.8, 0.1, 0.4, 0.9, 0.4, 0.9, 0.2, 0.8, 0.5, 0.3, 0.7]
  i = 0
  while i < 1 : # 500
    for j in range(0, len(dfTraining)) : #len(dfTraining)
      SepalLengthCm = dfTraining.iloc[j]['SepalLengthCm']
      SepalWidthCm = dfTraining.iloc[j]['SepalWidthCm']
      PetalLengthCm = dfTraining.iloc[j]['PetalLengthCm']
      PetalWidthCm = dfTraining.iloc[j]['PetalWidthCm']

      a_expectations = expected_a(dfTraining.iloc[j]['Species'])
      inputs = (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)

      tuple_V, tuple_A, tuple_error, total_error = forwardPropagation(inputs, biases, a_expectations, weights)
      weights = backwardPropagation(inputs, tuple_V, tuple_A, tuple_error, total_error, a_expectations, learning_rate, weights)
      i += 1
  return weights