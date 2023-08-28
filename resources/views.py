from flask import jsonify, request, render_template, redirect, url_for, session
from flask_restful import Resource,abort
from sqlalchemy import and_
import json

from database.models import *


class MyResource(Resource):
    def get(self, uid):
        if not uid:
            return {'message': '未登录'}#用message记录用户登录状态,若未登录 不会传用户id
        user = User.query.filter(User.id == uid)#查找对应的用户信息
        # 生成json格式数据
        infoJson = {}
        dataJson = json.loads(json.dumps(infoJson))
        for i in user:
            dataJson['uid'] = i.uid
            dataJson['message'] = '已登录'
            dataJson['username'] = i.username
            dataJson['password'] = i.password
            dataJson['avatar'] = i.avatar
            dataJson['phoneNumber'] = i.phoneNumber
            ansJson = json.dumps(dataJson, ensure_ascii=False)
        return ansJson
        # return jsonify(message="个人中心页面")

class LoginResource(Resource):
    def post(self):
        # 获取数据库所有信息，存下来，把用户输入与数据库比对
        user = User.query.all()
        for i in user:
            if request.form['username'] == i.username and request.form['password'] == i.password:
                session['islogin'] = 'true'
                session['uid'] = i.uid
                session['username'] = i.username
                session['avatar'] = i.avatar
                session['phoneNumber'] = i.phoneNumber
                return jsonify(message="success_login")
        return jsonify(message="fail_login")

class SuccessLoginResource(Resource):
    def get(self):
        if 'username' in session:
            username1 = session['username']
            uid1 = session['uid']
            avatar1 = session['avatar']
            phoneNumber1 = session['phoneNumber']
        return jsonify(username=username1, uid=uid1, avatar=avatar1, phoneNumber=phoneNumber1)

class FailLoginResource(Resource):
    def get(self):
        return jsonify(message="unlogin")

class RegisterResource(Resource):#注册
    def get(self):
        return render_template('register.html')

    def post(self):
        uname = request.form['username']
        upass = request.form['password']
        uava = request.files['avatar']
        uava = uava.enconde('base64', 'strict')
        upnumber = request.form['phoneNumber']
        u1 = User(username=uname, password=upass, avatar=uava, phoneNumber=upnumber)
        db.session.add(u1)
        db.session.commit()
        #生成json格式数据
        infoJson = {}
        dataJson = json.loads(json.dumps(infoJson))
        dataJson['username'] = uname
        dataJson['password'] = upass
        dataJson['avatar'] = uava
        dataJson['phoneNumber'] = upnumber
        ansJson = json.dumps(dataJson, ensure_ascii=False)

        return ansJson
        # return redirect(url_for('success_register'))

class SuccessRegisterResource(Resource):
    def get(self):
        return render_template('success_register.html')

class CollectedCardResource(Resource):
    def get(self, uid):
        if uid:
            collected_card1 = Collect.query.filter(and_(Collect.uid == uidd, Collect.ifQuestion == 1))
            infoJson = {}
            dataJson = json.loads(json.dumps(infoJson))
            # for i in collected_card1:
            #     uid1 = i.uid
            #     text1 = i.text
            #     time1 = i.time
            for i in collected_card1:
                dataJson['uid'] = i.uid
                dataJson['text'] = i.text
                dataJson['time'] = i.time
            ansJson = json.dumps(dataJson, ensure_ascii=False)
            return ansJson
            # return render_template('print_collect_card.html', uid=uid1, text=text1, time=time1)
        abort(404)

class CollectedQuestionResource(Resource):
    def get(self, uid):
        if uid:
            collected_question1 = Collect.query.filter(and_(Collect.uid == uidd, Collect.ifQuestion == '0'))
            infoJson = {}
            dataJson = json.loads(json.dumps(infoJson))
            for i in collected_question1:
                dataJson['text'] = i.text
                dataJson['time'] = i.time
            ansJson = json.dumps(dataJson, ensure_ascii=False)
            return ansJson
        abort(404)

class LookedCardResource(Resource):
    def get(self, uid):
        if uid:
            looked_card1 = Browse.query.filter(and_(Browse.uid == uidd, Browse.ifQuestion == 1))
            infoJson = {}
            dataJson = json.loads(json.dumps(infoJson))
            for i in looked_card1:
                dataJson['uid'] = i.uid
                dataJson['text'] = i.text
                dataJson['time'] = i.time
            ansJson = json.dumps(dataJson, ensure_ascii=False)
            return ansJson
        abort(404)

class LookedQuestionResource(Resource):
    def get(self, uid):
        if uid:
            looked_question1 = Browse.query.filter(and_(Browse.uid == uidd, Browse.ifQuestion == 0))
            infoJson = {}
            dataJson = json.loads(json.dumps(infoJson))
            for i in looked_question1:
                dataJson['uid'] = i.uid
                dataJson['text'] = i.text
                dataJson['time'] = i.time
            ansJson = json.dumps(dataJson, ensure_ascii=False)
            return ansJson
        abort(404)

class HelpResource(Resource):
    def get(self):
        # 返回数据库中管理员的联系方式
        return jsonify(contact='admin@example.com')