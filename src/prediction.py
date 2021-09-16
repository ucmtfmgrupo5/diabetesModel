import pickle
import os
import dill
import pandas as pd
import json

class Prediction:

    model = None

    def __init__(self):
        currrentPath = os.path.dirname(os.path.abspath(__file__))
        parentPath = os.path.abspath(os.path.join(currrentPath, os.pardir))
        filename = parentPath+'/modelDiabetesModel/modelDiabetesModel'
        print(filename)
        infile = open(filename,'rb')
        self.model = pickle.load(infile)
        infile.close()


    def makePrediction(self, inputDF):

        #Transform input DF into proper format
        inputModel = inputDF.values

        #Predict values from input
        predictions = self.model.predict(inputModel)


        #Transform output prediction into dataframe
        predictionDF = pd.DataFrame(predictions, columns=['prediction'])

        #Build output response object 
        Body = []
        for index, row in inputDF.iterrows():
            pregnant = row['pregnant']
            glucose = row['glucose']
            pressure = row['pressure']
            triceps = row['triceps']
            mass = row['mass']
            pedigree = row['pedigree']
            age = row['age']

            prediction = predictionDF.iloc[index]['prediction']

            InputVals = {}

            InputVals['pregnant'] = pregnant
            InputVals['glucose'] = glucose
            InputVals['pressure'] = pressure
            InputVals['triceps'] = triceps
            InputVals['mass'] = mass
            InputVals['pedigree'] = pedigree
            InputVals['age'] = age
            PredictionBody = {}

            PredictionBody['input'] = InputVals
            PredictionBody['prediction'] = prediction

            Body.append(PredictionBody)

            output = json.dumps(Body)
        
        return output


if __name__ == "__main__":
    predictionObject = Prediction()
    currrentPath = os.path.dirname(os.path.abspath(__file__))   
    parentPath = os.path.abspath(os.path.join(currrentPath, os.pardir))
    filenameDill = parentPath+'/predictionDiabetesModel/prediction'
    with open(filenameDill, "wb") as f:
     dill.dump(predictionObject, f)
