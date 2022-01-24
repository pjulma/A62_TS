from flask_wtf import FlaskForm
from wtforms import StringField,   SubmitField
from wtforms.validators import DataRequired

class ModelePredictionForm(FlaskForm):
    Action = StringField('Action', validators=[DataRequired()])
    submit = SubmitField('Prédire Action')