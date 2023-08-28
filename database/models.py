from sqlalchemy import ForeignKey
from app import db

class User(db.Model):
    """用户表"""
    __tablename__ = 'User'
    uid = db.Column(db.Integer, primary_key=True)  # 默认设置自增长
    username = db.Column(db.String(128), unique=True)
    password = db.Column(db.String(128))
    avatar = db.Column(db.String(1024))
    phoneNumber = db.Column(db.String(128))


class Browse(db.Model):
    __tablename__ = 'Browse'
    uid = db.Column(db.Integer, ForeignKey('User.uid'), primary_key=True)
    time = db.Column(db.DATETIME)
    text = db.Column(db.String(128))
    ifQuestion = db.Column(db.String(128))


class Collect(db.Model):
    __tablename__ = 'Collect'
    uid = db.Column(db.Integer, ForeignKey(User.uid), primary_key=True)
    time = db.Column(db.DATETIME)
    text = db.Column(db.String(128))
    ifQuestion = db.Column(db.String(128))