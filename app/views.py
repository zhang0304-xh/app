# 路由 + 视图函数
import base64
import json

from flask import Blueprint, jsonify
from .models import *
import model
from flask import render_template, \
    request, abort, redirect, url_for, session, make_response
from sqlalchemy import and_
import json

from py2neo import Graph

api_v1 = Blueprint('my', __name__)


@api_v1.route('/my')  # 个人中心页面
def index():
    # message.user = None
    return render_template('my.html')


# 修改用户信息
@api_v1.route('/my/modify')
def register():
    return render_template('modifyMessage.html')


@api_v1.route('/my/modify_implement', methods=['POST', 'GET'])
def modify2():
    if request.method == 'POST':
        # uid2 = request.form['uid']#用户id
        uid2 = "9"
        # User.query.filter(User.uid == uid2)

        # for i in user:
        #     print(user.username)
        uname = request.form['username']  # 用户名
        if (uname != None):
            User.query.filter(User.uid == uid2).update({"username": uname})
            db.session.commit()
        upass = request.form['password']  # 密码
        if (upass != None):
            User.query.filter(User.uid == uid2).update({"password": upass})
            db.session.commit()
            # upass2 = request.form['password2']
            # if(request.form['avatar'] != None):
            # uava = request.files['avatar']#头像
            # uava = uava.enconde('base64', 'strict')
            # user.update({"avatar":uava})
            pass
        uemail = request.form['uemail']  # 密码
        if (uemail != None):
            User.query.filter(User.uid == uid2).update({"email": uemail})
            db.session.commit()
        upnumber = request.form['phoneNumber']  # 手机号t
        if (upnumber != None):
            User.query.filter(User.uid == uid2).update({"phoneNumber": upnumber})
            db.session.commit()

        return render_template('success_modify.html')
    else:
        return render_template('fail_modify.html')


# 用户反馈信息
@api_v1.route('/my/feedback', methods=['POST', 'GET'])
def modify():
    if request.method == 'POST':
        try:  # 捕获数据库异常，防止用户重复反馈报错
            uid2 = request.form['uid']  # 用户id
            umessage = request.form['message']  # 用户反馈的信息
            m1 = User(uid=uid2, message=umessage)  # 用户信息对象
           # db.session.add(u1)
            db.session.commit()
        except:
            info1 = "该问题您已经反馈过了，请勿重复操作"
            return render_template('fail_feedback.html', info=info1)
        return render_template('success_feedback.html')


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
                return redirect(url_for('my.success_login'))  # 跳到success函数所指向的url
        return redirect(url_for('my.fail_login'))
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


'''
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
        # uava = request.files['avatar'].read()
        uava = ""
        # print(uava)

        # data = uava.read()  # 读取文件内容
        # uava = uava.encode('base64', 'strict')
        # avatar_data = request.files['avatar'].read()  # 读取文件数据为字节数组
        # avatar_str = base64.b64encode(uava.endoce('utf-8')) # 将字节数组编码为字符串
        upnumber = request.form['phoneNumber']
        u1 = User(username=uname, password=upass, avatar=uava, phoneNumber=upnumber)
        db.session.add(u1)
        db.session.commit()
        return render_template('print_userMessage.html',name=uname)
        # 跳到success函数所指向的url

    elif request.method == "GET":  # 没有登录，就返回个人中心页面
        return redirect(url_for('index'))


# @api_v1.route('/my/print_userMessage')
# def success_register():
#     return render_template('print_userMessage.html')


# 查看收藏的知识卡片
@api_v1.route('/my/collected_card', methods=['POST', 'GET'])
def collected_card():  # 需要前端传来该用户的id
    uidd = request.args.get('type1')
    if(uidd != None):
        collected_card1 = Collect.query.filter(and_(Collect.uid == uidd, Collect.ifQuestion == "1")).all()
        # print(collected_card1)
        # print(type(collected_card1))
        # list1 = None
        # for i in collected_card1:
        #     uid1 = i.uid
        #     text1 = i.text
        #     time1 = i.time
        #     print(text1)


        return render_template('print_collect_card.html', collected_card=collected_card1)
    return abort(304)


# 查看收藏的问题
@api_v1.route('/my/collected_question', methods=['POST', 'GET'])
def collected_question():  # 需要前端传来该用户的id
    uidd = request.args.get('type1')
    if (uidd != None):
        collected_question1 = Collect.query.filter(and_(Collect.uid == uidd, Collect.ifQuestion == '0')).all()
        # print(collected_question1)
        # for i in collected_question1:
        #
        #     text1 = i.text
        #     time1 = i.time
        return render_template('print_collect_question.html', collected_question=collected_question1)
    return abort(404)

# 查看搜索过的知识卡片
@api_v1.route('/my/looked_card', methods=['POST', 'GET'])
def looked_card():  # 需要前端传来该用户的id
    uidd = request.args.get('type1')
    if (uidd != None):
        looked_card1 = Browse.query.filter(and_(Browse.uid == uidd, Browse.ifQuestion == 1)).all()
        # for i in looked_card1:
        #     uid1 = i.uid
        #     text1 = i.text
        #     time1 = i.time
        return render_template('print_look_card.html', looked_card=looked_card1)
    return abort(404)

# 查看搜索过的问题
@api_v1.route('/my/looked_question', methods=['POST', 'GET'])
def looked_question():  # 需要前端传来该用户的id
    uidd = request.args.get('type1')
    if (uidd != None):
        looked_question1 = Browse.query.filter(and_(Browse.uid == uidd, Browse.ifQuestion == 0)).all()
        # for i in looked_question1:
        #     uid1 = i.uid
        #     text1 = i.text
        #     time1 = i.time
        return render_template('print_look_question.html', looked_question=looked_question1)
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
'''
api_v2 = Blueprint('hii', __name__)


@api_v2.route('/')
def my_index():
    return render_template('main_kg7.html')


@api_v2.route('/node1_data')
def get_node_data():
    # chart_data = {
    #     'links': [
    #         {"source": "青菜", "value": 3, "target": "青菜菌核病"},
    #         {"source": "青菜", "value": 3, "target": "青菜绵腐病"},
    #         {"source": "青菜", "value": 3, "target": "青菜炭疽病"},
    #         # 添加更多 links 对象...
    #     ],
    #     'nodes': [
    #         {"group": 0, "class": "作物", "size": 20, "id": "萝卜"},
    #         {"group": 0, "class": "作物", "size": 20, "id": "丝瓜"},
    #         {"group": 0, "class": "作物", "size": 20, "id": "西瓜"},
    #         # 添加更多 nodes 对象...
    #     ]
    # }
    # return jsonify(chart_data)
    # 假设后端返回的JSON数据为data
    with open('app/starwar_alldata.json', 'r') as file:
        data_node = json.load(file)
    return jsonify(data_node)

@api_v2.route('/node_data')
def get_carddata():
    #假设后端返回的JSON数据为data

    graph = Graph("http://localhost:7474", auth=("neo4j", "12345678"))

    # get links这部分要增多的话可以简化

    sas = f'MATCH (n:`作物`) RETURN  collect(n.name) as cast LIMIT 50'
    c_corn = graph.run(sas).data()
    corn = c_corn[0].get('cast')
    print(corn)
    #corn = corn + corn_data
    #corn = ["萝卜", "丝瓜", "西瓜", "豌豆", "草莓", "青菜"]

    data = []
    data2 = []
    for i in range(0,25):
        sas = f'MATCH path=(m:`作物`)<-[r]-(d:`病害`)   WHERE m.name = "{corn[i]}"   RETURN m.name as source,r.weight as value,d.name as target LIMIT 40'
        c_data = graph.run(sas).data()
        data = data +c_data


        #print(data)
    for i in range(len(data)):
        data[i]['value'] = 3
        data[i]['target'] = data[i]['target'] + "3"
    # print("data:")
    # print(data)


    for i in range(0,25):
        sas = f'MATCH path=(m:`作物`)<-[r]-(d:`虫害`)   WHERE m.name = "{corn[i]}"   RETURN m.name as source,r.weight as value,d.name as target LIMIT 40'
        c_data2 = graph.run(sas).data()
        data2 = data2 +c_data2
        # print(data2)
        for i in range(len(data2)):
            data2[i]['value'] = 3

        # print("data2:")
        # print(data2)

    all_links_data = data + data2
    #print("all_links")
    #print(all_links_data)
    # links_dict = {"links":all_links_data}
    # print(links_dict)
    # all_data1=json.dumps(links_dict, ensure_ascii=False)
    # print(all_data1)

    # 作物note
    zw_list = []
    for i in range(0, 25):
        keys = ["group", "class", "size", "id"]
        values = [0, "作物", 20, f"{corn[i]}"]
        zw_dict = dict(zip(keys, values))
        zw_list.append(zw_dict)
    # print("作物list")
    # print(zw_list)

    # 虫害note
    ch_list = []
    for i in range(0, 25):
        sas = f'MATCH path=(m:`作物`)<-[r]-(p:`虫害`) WHERE m.name ="{corn[i]}" RETURN  collect(p.name ) as cast'
        ch_note = graph.run(sas).data()
        # print(bh_note)

        ch_note_list = ch_note[0].get("cast")
        # print(bh_note_list)
        for n in ch_note_list:
            keys = ["group", "class", "size", "id"]
            values = [1, "虫害", 5, f"{n}"]
            ch_dict = dict(zip(keys, values))
            # print(ch_dict)
            ch_list.append(ch_dict)
    #   print(corn[i])
    # print("虫害list")
    # print(ch_list)

    # 病害note
    bh_list = []
    for i in range(0, 25):
        sas = f'MATCH path=(m:`作物`)<-[r]-(p:`病害`) WHERE m.name ="{corn[i]}" RETURN  collect(p.name ) as cast'
        bh_note = graph.run(sas).data()
        # print(bh_note)

        bh_note_list = bh_note[0].get("cast")
        #print(bh_note_list)
        for n in bh_note_list:
            keys = ["group", "class", "size", "id"]
            values = [2, "病害", 8, f"{n}3"]
            bh_dict = dict(zip(keys, values))
            # print(ch_dict)
            bh_list.append(bh_dict)
    #   print(corn[i])
    # print("病害list")
        #print(bh_list)



    # 数据拼接
    all_notes_data = zw_list + ch_list + bh_list


    all_links_data2 = []
    for item in all_links_data:
        all_links_data2.append(frozenset(item.items()))
    all_links_data2 = [dict(x) for x in set(tuple(x) for x in all_links_data2)]

    all_notes_data2 = []
    for item in all_notes_data:
        all_notes_data2.append(frozenset(item.items()))
    all_notes_data2 = [dict(x) for x in set(tuple(x) for x in all_notes_data2)]

    f_dict = {"links": all_links_data2, "nodes": all_notes_data2}
    # print(f_dict)


    return f_dict


# 每日推荐（随机取一个病害节点，返回其病害名，症状，危害的作物名，学名
@api_v2.route('/dayrecommended', methods=['POST', 'GET'])
def get_showdata():
    graph = Graph("http://localhost:7474", auth=("neo4j", "12345678"))
    rsas = f'MATCH path=(m:`作物`)<-[r]-(p:`病害`)  RETURN  p.name as cast order by rand() limit 1'
    r_note = graph.run(rsas).data()
    rr_note = r_note[0].get("cast")

    sas = f'MATCH (n:`病害`)-[r:`症状`]->(m:`症状`) WHERE n.name="{rr_note}"  return n.name as 病害 , m.name as 症状'
    zz_note = graph.run(sas).data()

    sas2 = f'MATCH (n:`病害`)-[r:`危害作物`]->(m:`作物`) WHERE n.name="{rr_note}"  return m.name as 危害作物'
    zw_note = graph.run(sas2).data()

    sas3 = f'MATCH (n:`病害`)-[r:`学名`]->(m:`学名`) WHERE n.name="{rr_note}"  return m.name as 学名'
    xm_note = graph.run(sas3).data()

    # sas4 = f'MATCH (n:`病害`)-[r:`简介`]->(m:`简介`) WHERE n.name="{rr_note}" return m.name as 简介'
    # jj_note = graph.run(sas4).data()

    dict_note1 = zz_note[0] | zw_note[0] | xm_note[0]  # | jj_note[0]
    list_note1 = [key for key in dict_note1]
    list_note2 = [value for value in dict_note1.values()]
    list_note3 = list_note1 + list_note2
    #这里前端能不能写如果报错取不出数据（没有传给数据或者传出数据为空）则重新拉取一次数据（重新调用，刷新）
    #print(list_note3)
    # print(dict_note1)
    #print(type(list_note3))
    # dict_note2 = list(dict_note1.items())
    # print(dict_note2)
    # dict_note3 = list(dict_note2)
    # print(dict_note3)
    return list_note3  # jsonify(f_dict)


# formatted_json(render_template('sohw_day.html'))




# 问答
@api_v2.route('/api/tryChat', methods=['POST', 'GET'])
def getdata():
    a = request.form.get('mydata')
    word = str(a)
    sent = model.predict(word)
    answ = sent[0].get('content')
    #chatWord = json.dumps(answ, ensure_ascii=False)
    # #print(answ)
    return answ#chatWord
    '''
    # #sent = [{'病害': '玉米大斑病'}, '防治方法']
    # sent1 = sent[0]
    # for key, value in sent1.items():
    #     sent1_keys = key
    #     sent1_value = value
    # #print(sent1_keys, sent1_value)
    # sent2 = sent[1]
    # #print(sent2)
    # asas = f'MATCH (n:`{sent1_keys}`) - [r:`{sent2}`] -> (m) WHERE n.name="{sent1_value}" RETURN m.name as content'
    # answ = graph.run(asas).data()'''










# 表单接收,请求获取网页结果给后端
@api_v2.route('/resourceshome', methods=['POST', 'GET'])
def get_sent():
    sent = request.form.get('sent')
    return sent
