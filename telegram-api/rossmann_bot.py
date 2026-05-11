#imports
import json
import os 
import pandas as pd
import requests 

from flask import Flask, request, Response


# constants
token = os.environ.get('TELEGRAM_BOT_TOKEN')

# # bot info
# https://api.telegram.org/bot8406535365:AAFqP0Bp-TyspQ_1sQ7bMihsDDHzOswnfSk/getMe

# # get updates
# https://api.telegram.org/bot8406535365:AAFqP0Bp-TyspQ_1sQ7bMihsDDHzOswnfSk/getUpdates

# # webhook 
# https://api.telegram.org/bot8406535365:AAFqP0Bp-TyspQ_1sQ7bMihsDDHzOswnfSk/setWebhook?url=https://rossmann-telegram-bot-d8n7.onrender.com

# # send message
# https://api.telegram.org/bot8406535365:AAFqP0Bp-TyspQ_1sQ7bMihsDDHzOswnfSk/sendMessage?chat_id=895677125&text=hi bot 


def send_message(chat_id, text):
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    
    payload = {
        'chat_id': chat_id,
        'text': text
    }

    r = requests.post(url, json=payload)
    
    print(f'Status Code {r.status_code}')
    
    return None


def load_dataset(store_id):
    # loading test dataset
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_path)

    test_path = os.path.join(project_root, 'data', 'test.csv')
    store_path = os.path.join(project_root, 'data', 'store.csv')

    df10 = pd.read_csv(test_path)
    df_store_raw = pd.read_csv(store_path)

    # merge test dataset + store
    df_test = pd.merge(df10, df_store_raw, how='left', on='Store')

    # choose store for prediction 
    df_test = df_test[df_test['Store'] == store_id] 

    if not df_test.empty:
        # remove closed days
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()] 
        df_test = df_test.drop('Id', axis=1)

        # convert dataframe to json
        data = json.dumps(df_test.to_dict(orient='records'))

    else:
        data = 'error'

    return data


def predict(data):

    # API Call
    url = 'https://rossmann-api-d4bw.onrender.com/rossmann/predict' 
    header = {'Content-type': 'application/json'}
    data = data 

    r = requests.post(url, data=data, headers=header)
    print('Status Code{}'.format(r.status_code)) 

    d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())

    return d1


def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/', '')

    try:
        store_id = int(store_id)

    except ValueError:
        store_id = 'error'

    return chat_id, store_id


# inicialize API
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])  
def index():
    if request.method == 'POST':
        message = request.get_json()

        chat_id, store_id = parse_message(message)

        if store_id != 'error':
            # loading data
            data = load_dataset(store_id)

            if data != 'error':
                # prediction
                d1 = predict(data)

                # calculation
                d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()
                value = d2['prediction'].values[0]
                value_eur = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                # send message    
                msg = f"Store Number {d2['store'].values[0]} will sell €{value_eur} in the next 6 weeks"
                
                send_message(chat_id, msg)
                return Response('OK', status=200)

            else:
                send_message(chat_id, 'Store not available')
                return Response('OK', status=200)
        
        else: 
            send_message(chat_id, 'Store ID was not found')
            return Response('OK', status=200)


    else:
        return '<h1> Rossmann Telegram BOT </h1>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# # showing prediction 
# d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()

# for i in range(len(d2)):
#     value = d2.loc[i, 'prediction']
#     value_eur = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
#     print(f"Store Number {d2.loc[i, 'store']} will sell €{value_eur} in the next 6 weeks")

