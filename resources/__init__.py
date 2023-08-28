from flask import Blueprint
from flask_restful import Api
from resources.views import *

# 定义蓝图，main为蓝图名字
main = Blueprint('main', __name__)
# 实例化api
api = Api(main)

# 设置路由
api.add_resource(MyResource, '/my', defaults={"uid": None}, method=['GET', ])
api.add_resource(LoginResource, '/my/login', method=['POST', ])
api.add_resource(SuccessLoginResource, '/my/success_login', method=['GET', ])
api.add_resource(FailLoginResource, '/my/fail_login', method=['GET', ])
api.add_resource(RegisterResource, '/my/register', method=['POST', 'GET'])
api.add_resource(SuccessRegisterResource, '/my/success_register', method=['GET', ])
api.add_resource(CollectedCardResource, '/my/collected_card', defaults={"uid": None}, method=['POST', 'GET'])
api.add_resource(CollectedQuestionResource, '/my/collected_question', defaults={"uid": None}, method=['POST', 'GET'])
api.add_resource(LookedCardResource, '/my/looked_card', defaults={"uid": None}, method=['POST', 'GET'])
api.add_resource(LookedQuestionResource, '/my/looked_question', defaults={"uid": None}, method=['POST', 'GET'])
api.add_resource(HelpResource, '/my/help', defaults={"uid": None}, method=['GET', ])
