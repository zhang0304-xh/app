# 初始化文件
from flask import Flask
from .views import api_v1
from .extensions import init_exts

def create_app():
    app = Flask(__name__)
    # 注册蓝图
    app.register_blueprint(blueprint=api_v1)

    # 配置数据库
    db_uri='mysql+pymysql://root:123456@127.0.0.1:3306/APP?charset=utf8mb4'
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 初始化插件
    init_exts(app)

    return app