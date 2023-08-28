from flask import Flask
from common import create_app

app = create_app()


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello'


if __name__ == '__main__':
    app.run(host='0.0.0.0')
