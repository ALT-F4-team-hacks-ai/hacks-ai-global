import datetime
import sqlalchemy
from sqlalchemy import orm
from .db_session import SqlAlchemyBase
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField
from wtforms import BooleanField, SubmitField
from wtforms.validators import DataRequired


class ClassesForm(FlaskForm):
    title = StringField('Заголовок', validators=[DataRequired()])
    is_private = BooleanField("Приватная лекция")


class Classes(SqlAlchemyBase):
    __tablename__ = 'classes'
    categories = orm.relationship("Category",
                                  secondary="association",
                                  backref="classes")
    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    title = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    content = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    terms = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    a = str(datetime.datetime.now())
    a = a[:19]
    created_date = sqlalchemy.Column(sqlalchemy.String,
                                     default=a)
    is_private = sqlalchemy.Column(sqlalchemy.Boolean, default=True)
    user_id = sqlalchemy.Column(sqlalchemy.Integer,
                                sqlalchemy.ForeignKey("users.id"))
    inprocess = sqlalchemy.Column(sqlalchemy.Boolean, default=True)
    audio = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    user = orm.relationship('User')
