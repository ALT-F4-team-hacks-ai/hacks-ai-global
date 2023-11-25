import datetime
import sqlalchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from .db_session import SqlAlchemyBase
from sqlalchemy import orm

class User(SqlAlchemyBase, UserMixin):
    __tablename__ = 'users'
    a = """"\u041f\u043e\u0431\u0435\u0434\u0430": "\u0443\u0441\u043f\u0435\u0445 \u0432 \u0441\u0440\u0430\u0436\u0435\u043d\u0438\u0438, \u0441\u043e\u0440\u0435\u0432\u043d\u043e\u0432\u0430\u043d\u0438\u0438, \u0437\u0430\u043a\u043e\u043d\u0447\u0438\u0432\u0448\u0438\u0439\u0441\u044f \u043f\u043e\u0440\u0430\u0436\u0435\u043d\u0438\u0435\u043c \u0441\u043e\u043f\u0435\u0440\u043d\u0438\u043a\u0430. "}"""
    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    name = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    email = sqlalchemy.Column(sqlalchemy.String,
                              index=True, unique=True, nullable=True)
    glossary = sqlalchemy.Column(sqlalchemy.JSON,
                              index=True, nullable=True, default=a)
    hashed_password = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    created_date = sqlalchemy.Column(sqlalchemy.DateTime,
                                     default=datetime.datetime.now)
    classes = orm.relationship("Classes", back_populates='user')
    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.hashed_password, password)

    def __repr__(self):
        return f"<User> {self.id} {self.name} {self.email}"