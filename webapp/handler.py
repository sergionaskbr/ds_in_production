import os
import pandas as pd
import pickle
from flask             import Flask, request, Response
from pathlib           import Path
from rossmann.Rossmann import Rossmann

# loading model
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / 'model_rossmann.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# inicialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json(silent=True)
    
    if test_json: # there is data
        if isinstance(test_json, dict):  # unique example (se o json tiver uma unica linha)
            test_raw = pd.DataFrame(test_json, index=[0])         
            
        else:  # multiple example (se receber varios jsons concatenados)
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # instantiate rossmann class
        pipeline = Rossmann()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_predict(model, test_raw, df3)
        
        return df_response
        
    else:
        return Response('{}', status=200, mimetype='application/json')

# running locally
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
