from flask import Flask
from extension import db,cors
from config import Config
from resources import main

def create_app():
    # 初始化flask
    app = Flask(__name__)
    # 从对象设置配置信息
    app.config.from_object(Config)
    # 第三方扩展初始化
    db.init_app(app)
    #cors.init(app)
    # 注册蓝图
    app.register_blueprint(main)
    return app

