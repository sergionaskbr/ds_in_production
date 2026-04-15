import pandas as pd
import pickle
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann

# loading model
model = pickle.load(open('/mnt/d/repos/ds_prod/models/model_rossmann.pkl', 'rb')) 
# no caso do modelo e preciso colocar o caminho inteiro
# quando a API comecar ela vai carregar o modelo em memoria

# inicialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance(test_json, dict):  # unique example (se o json tiver uma unica linha)
            test_raw = pd.DataFrame(test_json, index=[0])         
            
        else:  # multiple example (se receber varios jsons concatenados)
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # Instantiate Rossmann Class
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

if __name__ == '__main__':
    app.run('0.0.0.0')
