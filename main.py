import os
import sqlite3

from flask import Flask, render_template, redirect, request, abort, flash, url_for
from werkzeug.utils import secure_filename
from pipeline import TextProcessor
from data import db_session
from data.users import User
from data.classes import Classes, ClassesForm
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from forms.login import LoginForm
from forms.user import RegisterForm
import requests
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = '4827c890c1b84580a2efd2fb7257aa8d'
login_manager = LoginManager()
login_manager.init_app(app)
UPLOAD_FOLDER = 'static/il/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'aiff'}



def main():
    db_session.global_init('db/base.db')
    app.run()


@app.route("/")
def visit():
    return render_template("visit.html")


@login_manager.user_loader
def load_classes(user_id):
    db_sess = db_session.create_session()
    return db_sess.query(Classes).get(id)

@login_manager.user_loader
def load_user(user_id):
    db_sess = db_session.create_session()
    return db_sess.query(User).get(user_id)


@app.route("/login")
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect('/feed')
    form = LoginForm()
    if form.validate_on_submit():
        db_sess = db_session.create_session()
        user = db_sess.query(User).filter(User.email == form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            return redirect("/createnewclass")
        return render_template('login.html',
                               message="Неправильный логин или пароль",
                               form=form)
    return render_template('login.html', title='Авторизация', form=form)



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect("/")


@app.route("/register", methods=['GET', 'POST'])
def reqister():
    form = RegisterForm()
    if form.validate_on_submit():
        if form.password.data != form.password_again.data:
            return render_template('registration.html', title='Регистрация',
                                   form=form,
                                   message="Пароли не совпадают")
        db_sess = db_session.create_session()
        if db_sess.query(User).filter(User.email == form.email.data).first():
            return render_template('registration.html', title='Регистрация',
                                   form=form,
                                   message="Такой пользователь уже есть")
        user = User(
            name=form.name.data,
            email=form.email.data,
        )
        user.set_password(form.password.data)
        db_sess.add(user)
        db_sess.commit()
        return redirect('/login')
    return render_template('registration.html', title='Регистрация', form=form)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/createnewclass', methods=['GET', 'POST'])
@login_required
def add_classes():
    form = ClassesForm()
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Не могу прочитать файл')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Нет выбранного файла')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print(1)
            filename = secure_filename(file.filename)
            connect = sqlite3.connect('db/base.db')
            cursor = connect.cursor()
            cursor.execute("SELECT id FROM classes ORDER BY id DESC LIMIT 1")
            a = str(cursor.fetchall())
            a = int(a[a.index('(') + 1:a.index(')') - 1])
            connect.close()
            db_sess = db_session.create_session()
            classes = Classes()
            d = str(a + 1) + filename[filename.index('.'):]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], d))
            classes.title = form.title.data
            classes.content = TextProcesso.process_text(d)[0]
            classes.terms = TextProcesso.process_text(d)[1]
            classes.inprocess = 0
            classes.audio = d
            classes.is_private = form.is_private.data
            current_user.classes.append(classes)
            db_sess.merge(current_user)
            db_sess.commit()
            return redirect(f'/classes/{a + 1}')
    return render_template('createnewclass.html', title='Новый конспект', form=form)


@app.route('/classes/<int:id>', methods=['GET', 'POST'])
@login_required
def viewing_classes(id):
    if request.method == "GET":
        form = ClassesForm()
        db_sess = db_session.create_session()
        classes = db_sess.query(Classes).filter(Classes.id == id).first()
        terms = classes.terms
        if classes.is_private == 0:
            if classes.inprocess == 1:
                return render_template('notready.html')
            else:
                if classes.user_id == current_user.id:
                    form.is_private.data = classes.is_private
                    classes.is_private = form.is_private.data
                    db_sess.commit()
                    return render_template('classes.html',
                                           text=classes.content,
                                           terms=terms,
                                           audio=classes.audio,
                                           form=form,
                                           title='Лекция')
                else:
                    return render_template('classes.html',
                                           text=classes.content,
                                           terms=terms,
                                           audio=classes.audio,
                                           title='Лекция')
        else:
            if classes.user_id == current_user.id:
                if classes.inprocess == 1:
                    return render_template('notready.html')
                else:
                    form.is_private.data = classes.is_private
                    classes.is_private = form.is_private.data
                    db_sess.commit()
                    print('комит2')
                    return render_template('classes.html',
                                           text=classes.content,
                                           terms=terms,
                                           audio=classes.audio,
                                           form=form,
                                           title='Лекция')
            else:
                return render_template('private.html')
@app.route('/profile/<int:id>', methods=['GET'])
def viewing_profile(id):
    db_sess = db_session.create_session()
    user = db_sess.query(User).filter(User.id == id).first()
    classes = db_sess.query(Classes).filter(Classes.user_id == id)
    return render_template('profile.html',
                           classes=classes,
                           user=user,
                           title=user.name
                           )


if __name__ == '__main__':
    main()
    app.run()
