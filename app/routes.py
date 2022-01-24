from app import app
from flask import render_template
import pickle, numpy as np
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

@app.route('/')
@app.route('/index')
def index():
    modeles = {'titre': 'Regression Lineaire'}
    parametres = [
        {"par1":"10", "par2":"uniform"},
        {"par1":"20", "par2":"lineaire"}
    ]
    return render_template('index.html', titre_page='Accueil', mod=modeles, page_no=1, param=parametres)

from app.forms import ModelePredictionForm

from flask import  flash, redirect

@app.route('/form_input', methods=['GET', 'POST'])
def form_input():
    form = ModelePredictionForm()
    if form.validate_on_submit():
        #formuler le data
        sample_data = [form.Action.data]
        clean_data = [str(i) for i in sample_data]
        data = []
        data.extend(clean_data[0:3])
        # Reshape the Data
        ex1 = np.array(data).reshape(1, -1)
        #lire le modele
        rl_model = pickle.load(open('modele_LR.pkl', 'rb'))
        resultat_prediction = rl_model.predict(ex1)
        flash('Action: {} donne Y={}'.format(
            form.Action.data, resultat_prediction))
        return redirect('/index')
    return render_template('form_input.html', title='Modele de Prediction', form=form)