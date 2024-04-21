from flask import Flask, render_template,request
import torch
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model1 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
ckpt_path = "flipkart_qa_v1.ckpt"
model1.load_state_dict(torch.load(ckpt_path))


def predict(question):
  inputs = tokenizer.encode("answer the question:" + question, return_tensors="pt")
  output_ids = model1.generate(inputs, max_length=1024, num_return_sequences=1)
  output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  return output

app = Flask(__name__)

@app.route("/")
def qa():
    return render_template("qa.html")
  
@app.route("/chat",methods=['GET','POST'])
def chat():
  if request.method == 'POST':
    res = predict(request.form['question'])
    return render_template("chat.html",res=res,ques=request.form['question'])
    
  else:
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)
