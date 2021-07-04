from flask import Flask, render_template, request
from summary import * 

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def index():
    text = ""
    result = ""

    if request.method == 'POST':
        acc = request.form.get('vec')
        procent = request.form.get('proc')
        proc = int(procent)


        if acc == '1':
            text = request.form['nameText']
            result = algorithm(text, 'word2vec_Oftalmologie_400_1_10_final.txt', 400, proc)
            return render_template('index.html',result = result, text = text)

        elif acc == '2':
            text = request.form['nameText']
            result = algorithm(text, 'corola.300.20.txt', 300, proc)
            return render_template('index.html',result = result, text = text)
        else:
            text = request.form['nameText']
            result = 'Nici un rezultat'
            return render_template('index.html',result = result, text = text)
    else:
        return render_template('index.html',result = result, text = text)

if __name__ == "__main__":
    app.run(debug = True)
