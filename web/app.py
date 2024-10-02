from flask import Flask, request, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

tokenizer = T5Tokenizer.from_pretrained('model/t5-small')
model = T5ForConditionalGeneration.from_pretrained('model/t5-small')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        inputs = tokenizer.encode("translate English to English: " + user_input, return_tensors="pt")
        outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
