from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('model')
model = T5ForConditionalGeneration.from_pretrained('model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    inputs = tokenizer.encode("translate English to English: " + user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return render_template('index.html', user_input=user_input, response=response)

if __name__ == "__main__":
    app.run(debug=True)
