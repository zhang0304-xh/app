from flask import jsonify, request, render_template, redirect, url_for, session
from flask_restful import Resource,abort
from sqlalchemy import and_

from database.models import *


class MyResource(Resource):
    def get(self):
        return jsonify(message="个人中心页面")

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

class RegisterResource(Resource):
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
        return redirect(url_for('success_register'))

class SuccessRegisterResource(Resource):
    def get(self):
        return render_template('success_register.html')

class CollectedCardResource(Resource):
    def get(self):
        uidd = request.args.get('type1')
        if uidd is not None:
            collected_card1 = Collect.query.filter(and_(Collect.uid == uidd, Collect.ifQuestion == 1))
            for i in collected_card1:
                uid1 = i.uid
                text1 = i.text
                time1 = i.time
            return render_template('print_collect_card.html', uid=uid1, text=text1, time=time1)
        abort(404)

class CollectedQuestionResource(Resource):
    def get(self):
        uidd = request.args.get('type1')
        if uidd is not None:
            collected_question1 = Collect.query.filter(and_(Collect.uid == uidd, Collect.ifQuestion == '0'))
            print(collected_question1)
            for i in collected_question1:
                text1 = i.text
                time1 = i.time
            return render_template('print_collect_question.html', uid=uidd, text=text1, time=time1)
        abort(404)

class LookedCardResource(Resource):
    def get(self):
        uidd = request.args.get('type1')
        if uidd is not None:
            looked_card1 = Browse.query.filter(and_(Browse.uid == uidd, Browse.ifQuestion == 1))
            for i in looked_card1:
                uid1 = i.uid
                text1 = i.text
                time1 = i.time
            return render_template('print_look_card.html', uid=uid1, text=text1, time=time1)
        abort(404)

class LookedQuestionResource(Resource):
    def get(self):
        uidd = request.args.get('type1')
        if uidd is not None:
            looked_question1 = Browse.query.filter(and_(Browse.uid == uidd, Browse.ifQuestion == 0))
            for i in looked_question1:
                uid1 = i.uid
                text1 = i.text
                time1 = i.time
            return render_template('print_look_question.html', uid=uid1, text=text1, time=time1)
        abort(404)

class HelpResource(Resource):
    def get(self):
        # 返回数据库中管理员的联系方式
        return jsonify(contact='admin@example.com')