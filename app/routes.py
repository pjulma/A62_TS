from app import app
from flask import render_template, request
import pickle, numpy as np
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA,  ARIMAResults, ARIMAResultsWrapper


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
from app.forms import ActionPredictionForm
import collect as collect
import eda as eda


from flask import  flash, redirect
"""
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
"""
@app.route('/inputPredAction', methods=['GET', 'POST'])
def inputPredAction():
    form = ActionPredictionForm()
    actionList=['AAPL', 'AMZN', 'WMT', 'NFLX','MAR','AAL']
    print("Before IF: ")
    #if form.validate_on_submit():
    #thisAction=[form.action]

    ####trying to get selected value with ... form.action
    """
    sample_data = [form.action]
    clean_data = [float(i) for i in sample_data]
    data = []
    data.extend(clean_data[0:4])
    print("PRINT DATA: " + str(data))
    """

    ####trying to get selected value with ... flash
    flash("getting FLASH action: {}".format(form.action))
    
    #trying to get selected value with ... request.form.get
    select = request.form.get('actionSelect')
    print("PRINT select: " + str(select))

    #Starting collect phase
    myCollect = collect
    myCollect.startCollect()

    #Starting EDA phase
    myEda = eda
    myEda.startEDA()

    #Prediction phase
    rl_model = ARIMAResults.load('modele_LR.pkl')
    #rl_model = pickle.load(open('model2.pkl', 'rb'))
    #loaded = ARIMAResults.load('model2.pkl')
    #resultat_prediction = loaded.predict(select)
    resultat_prediction = rl_model.predict(select)
    print("resultat_prediction:" + resultat_prediction)
    
    
    return render_template("inputPredAction.html",actionList=actionList, form=form)

