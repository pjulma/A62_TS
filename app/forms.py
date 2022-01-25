from flask_wtf import FlaskForm
from wtforms import StringField,   SubmitField
from wtforms.validators import DataRequired

class ModelePredictionForm(FlaskForm):
    tv = StringField('TV', validators=[DataRequired()])
    radio = StringField('Radio', validators=[DataRequired()])
    journaux = StringField('Journaux', validators=[DataRequired()])

    submit = SubmitField('Prédire')

class ActionPredictionForm(FlaskForm):
    action = StringField('Action', validators=[DataRequired()])
    submit = SubmitField('Prédire')
