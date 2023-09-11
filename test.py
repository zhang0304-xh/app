# # App.py
# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_migrate import Migrate
# from flask_login import (
#     UserMixin,
#     login_user,
#     LoginManager,
#     current_user,
#     logout_user,
#     login_required,
# )
#
# login_manager = LoginManager()
# login_manager.session_protection = "strong"
# login_manager.login_view = "login"
# login_manager.login_message_category = "info"
# db = SQLAlchemy()
# migrate = Migrate()
# bcrypt = Bcrypt()
#
#
# def create_app():
#     App = Flask(__name__)
#
#
# App.secret_key = 'secret-key'
# App.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
# App.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# login_manager.init_app(App)
# db.init_app(App)
# migrate.init_app(App, db)
# bcrypt.init_app(App)
# # 注册路由
# from auth import auth as auth_blueprint
# from main import main as main_blueprint
#
# App.register_blueprint(auth_blueprint)
# App.register_blueprint(main_blueprint)
# return App
# # models.py
# from App import db, login_manager, bcrypt
# from flask_login import UserMixin
#
#
# class User(db.Model, UserMixin):
#     """用户表"""
#
#
# __tablename__ = 'User'
# uid = db.Column(db.Integer, primary_key=True)  # 默认设置自增长
# username = db.Column(db.String(128), unique=True)
# password = db.Column(db.String(128))
# avatar = db.Column(db.String(1024))
# phoneNumber = db.Column(db.String(128))
#
#
# # 定义get_id方法，返回用户id
# def get_id(self):
#     return self.uid
#
#
# # auth.py
# from flask import Blueprint, render_template, request, redirect, url_for, flash
# from models import User
# from App import db, bcrypt, login_manager
# from flask_login import login_user, logout_user, login_required
#
# auth = Blueprint('auth', __name__)
#
#
# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))
#
#
# @auth.route('/register')
# def register():
#     return render_template('register.html')
#
#
# @auth.route('/register_check', methods=['POST', 'GET'])
# def register_check():
#     if request.method == 'POST':
#         uname = request.form['username']
#
#
# upass = request.form['password']
# uava = request.files['avatar']
# upnumber = request.form['phoneNumber']
# # 检查用户名是否已存在
# user = User.query.filter_by(username=uname).first()
# if user:
#     flash('用户名已存在，请重新输入', 'error')
# return redirect(url_for('auth.register'))
# # 对密码进行哈希加密
# hashed_password = bcrypt.generate_password_hash(upass).decode('utf-8')
# u1 = User(username=uname, password=hashed_password, avatar=uava, phoneNumber=upnumber)
# db.session.add(u1)
# db.session.commit()
# flash('注册成功，请登录', 'success')
# return redirect(url_for('auth.login'))
#
#
# @auth.route('/login')
# def login():
#     return render_template('login.html')
#
#
# @auth.route('/login_check', methods=['POST', 'GET'])
# def login_check():
#     if request.method == 'POST':
#         uname = request.form['username']
#
#
# upass = request.form['password']
# # 查询数据库中是否有该用户
# user = User.query.filter_by(username=uname).first()
# if not user:
#     flash('用户名不存在，请重新输入', 'error')
# return redirect(url_for('auth.login'))
# # 检查密码是否正确
# if not bcrypt.check_password_hash(user.password, upass):
#     flash('密码错误，请重新输入', 'error')
# return redirect(url_for('auth.login'))
# # 登录用户并重定向到个人中心页面
# login_user(user)
# flash('登录成功', 'success')
# return redirect(url_for('main.profile'))
#
#
# @auth.route('/logout')
# @login_required
# def logout():
#
#
# # 注销用户并重定向到首页
# logout_user()
# flash('注销成功', 'success')
# return redirect(url_for('main.index'))
# # views.py
# from flask import Blueprint, render_template
# from flask_login import login_required, current_user
#
# main = Blueprint('main', __name__)
#
#
# @main.route('/')
# def index():
#     return render_template('index.html')
#
#
# @main.route('/profile')
# @login_required
# def profile():
#
#
# # 使用current_user变量来显示用户信息
# return render_template('profile.html', user=current_user)
