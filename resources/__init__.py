from flask import Blueprint
from flask_restful import Api
from resources.views import *

# 定义蓝图，main为蓝图名字
main = Blueprint('main', __name__)
# 实例化api
api = Api(main)

# 设置路由
api.add_resource(MyResource, '/my')
api.add_resource(LoginResource, '/my/login')
api.add_resource(SuccessLoginResource, '/my/success_login')
api.add_resource(FailLoginResource, '/my/fail_login')
api.add_resource(RegisterResource, '/my/register')
api.add_resource(SuccessRegisterResource, '/my/success_register')
api.add_resource(CollectedCardResource, '/my/collected_card')
api.add_resource(CollectedQuestionResource, '/my/collected_question')
api.add_resource(LookedCardResource, '/my/looked_card')
api.add_resource(LookedQuestionResource, '/my/looked_question')
api.add_resource(HelpResource, '/my/help')
