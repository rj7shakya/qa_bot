from flask import Flask, render_template, request

from utils.predict import load_and_predict

app = Flask(__name__)

@app.route("/")
def qa():
    return render_template("qa.html")
  
@app.route("/chat",methods=['GET','POST'])
def chat():
  if request.method == 'POST':
    response = load_and_predict(request.form['question'])
    return render_template("chat.html", res = response, ques = request.form['question'])
  
  return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)
