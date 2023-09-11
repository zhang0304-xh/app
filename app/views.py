# 路由 + 视图函数
from flask import Blueprint
from .models import *
import json
from py2neo import Graph

from flask import render_template, \
    request, abort, redirect, url_for, session, make_response
from sqlalchemy import and_

api_v1 =Blueprint('my',__name__)

@api_v1.route('/')
def my_index():
    pass

@api_v1.route('/my')  # 个人中心页面
def index():
    # message.user = None
    return render_template('my.html')


# 用户登录页面
@api_v1.route('/my/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        # 获取数据库所有信息，存下来，把用户输入与数据库比对
        user = User.query.all()
        for i in user:
            if request.form['username'] == i.username and request.form['password'] == i.password:
                response = make_response('success_login')  # 登录成功，返回给前端的值
                session['islogin'] = 'true'  # 是否已经登录的标识
                session['uid'] = i.uid
                session['username'] = i.username  # 根据自己的需求，在session里存储一些值
                session['avatar'] = i.avatar
                session['phoneNumber'] = i.phoneNumber
                # response.set_cookie('username', i.username,
                #                     max_age=30 * 24 * 3600) #max_age，cookie的存活时间，这里表示一个月
                # response.set_cookie('avatar', i.avatar)
                # response.set_cookie('phoneNumber', i.phoneNumber, max_age=30 * 24 * 3600)
                #
                # return redirect(url_for('success_login'))
                # user2 = {
                #     'uid': i.uid,
                #     'username': i.username,
                #     'password': i.password,
                #     'avatar': i.avatar,
                #     'phoneNumber': i.phoneNumber
                # }
                # g.uid = user2['uid']
                # g.username = user2['username']
                # g.password = user2['password']
                # g.avatar = user2['avatar']
                # g.phoneNumber = user2['phoneNumber']
                # print(g)
                # print(g.username)
                return redirect(url_for('success_login'))  # 跳到success函数所指向的url
        return redirect(url_for('fail_login'))
    # elif request.method == "GET":#没有登录，就返回个人中心页面
    #     return redirect(url_for(index))


@api_v1.route('/my/success_login')
def success_login():
    if 'username' in session:
        username1 = session['username']
        uid1 = session['uid']
        avatar1 = session['avatar']
        phoneNumber1 = session['phoneNumber']
    return render_template('success_login.html', username=username1, uid=uid1, avatar=avatar1, phoneNumber=phoneNumber1)


@api_v1.route('/my/fail_login')  # 登录失败
def fail_login():
    return render_template('unlogin.html')


# #用户退出
# @App.route('/logout')
# def logout():
#     # 清空session和Cookie，页面跳转
#     session.clear()
#
#     response = make_response('注销并进行重定向', 302)  # 302状态码表示重定向
#
#     # 两种重定向的方法
#     # response.headers['Location'] = '/'
#     response.headers['Location'] = url_for('index.home')  # 另一种方式，括号里的值写请求为“/” 的 - Blueprint名.方法名
#
#     # 两种清除cookie的方式
#     response.delete_cookie('username')
#     response.set_cookie('password', '', max_age=0)
#
#     return response



# 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']


@api_v1.route('/my/register')
def register():
    return render_template('register.html')


# 用户注册
@api_v1.route('/my/register_check', methods=['POST', 'GET'])
def register_check():
    if request.method == 'POST':
        uname = request.form['username']
        upass = request.form['password']
        uava = request.files['avatar']
        uava = uava.enconde('base64', 'strict')
        upnumber = request.form['phoneNumber']
        u1 = User(username=uname, password=upass, avatar=uava, phoneNumber=upnumber)
        db.session.add(u1)
        db.session.commit()
        return redirect(url_for('success_register'))
        # 跳到success函数所指向的url

    elif request.method == "GET":  # 没有登录，就返回个人中心页面
        return redirect(url_for('index'))


@api_v1.route('/my/success_register')
def success_register():
    return render_template('success_register.html')


# 查看收藏的知识卡片
@api_v1.route('/my/collected_card', methods=['POST', 'GET'])
def collected_card():  # 需要前端传来该用户的id
    uidd = request.args.get('type1')
    if(uidd != None):
        collected_card1 = Collect.query.filter(and_(Collect.uid == uidd, Collect.ifQuestion == 1))
        for i in collected_card1:
            uid1 = i.uid
            text1 = i.text
            time1 = i.time
        return render_template('print_collect_card.html', uid=uid1,text=text1,time=time1)
    return abort(304)


# 查看收藏的问题
@api_v1.route('/my/collected_question', methods=['POST', 'GET'])
def collected_question():  # 需要前端传来该用户的id
    uidd = request.args.get('type1')
    if (uidd != None):
        collected_question1 = Collect.query.filter(and_(Collect.uid == uidd, Collect.ifQuestion == '0'))
        print(collected_question1)
        for i in collected_question1:

            text1 = i.text
            time1 = i.time
        return render_template('print_collect_question.html', uid=uidd,text=text1,time=time1)
    return abort(404)

# 查看搜索过的知识卡片
@api_v1.route('/my/looked_card', methods=['POST', 'GET'])
def looked_card():  # 需要前端传来该用户的id
    uidd = request.args.get('type1')
    if (uidd != None):
        looked_card1 = Browse.query.filter(and_(Browse.uid == uidd, Browse.ifQuestion == 1))
        for i in looked_card1:
            uid1 = i.uid
            text1 = i.text
            time1 = i.time
        return render_template('print_look_card.html', uid=uid1,text=text1,time=time1)
    return abort(404)

# 查看搜索过的问题
@api_v1.route('/my/looked_question', methods=['POST', 'GET'])
def looked_question():  # 需要前端传来该用户的id
    uidd = request.args.get('type1')
    if (uidd != None):
        looked_question1 = Browse.query.filter(and_(Browse.uid == uidd, Browse.ifQuestion == 0))
        for i in looked_question1:
            uid1 = i.uid
            text1 = i.text
            time1 = i.time
        return render_template('print_look_question.html', uid=uid1,text=text1,time=time1)
    return(404)

# 帮助反馈(返回数据库中管理员的联系方式）
# @App.route('/my/help', message=['POST','GET'])
# def help():
#     if request.method == 'POST':
#         pass

# 创建存图片的类
# class ImageFile(db.Model):
#     __tablename__ = 'ImageFile'
#     id = db.Column(db.Integer, primary_key=True)
#     image_name = db.Column(db.String(30), index=True)
#     image = db.Column(db.LargeBinary(length=2048))

def Get_UserData():
    user = User.query.all()  # 返回列表
    return user


# if __name__ == '__main__':
#     app.run(debug=True)
#     u = User(username='1', password='1', phoneNumber='111')
#     db.session.add(u)
#     db.session.commit()
#     # db.drop_all() #清除所有表
#     # db.create_all()  # 创建所有的表
#     # #创建对象，插入数据
#     # role1 = Role(name='admin')
#     # # session记录到对象任务中
#     # db.session.add(role1)
#     # # 提交任务
#     # db.session
