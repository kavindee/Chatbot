from flask import Flask, request, jsonify
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load the model and tokenizer once at startup
peft_model_id = "GymBoy2/lora-flan-t5-large-chat2"
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id).to("cpu")
model.eval()


def inference(input_sent):
    input_ids = tokenizer(input_sent, return_tensors="pt", truncation=True, max_length=256).input_ids.to("cpu")
    outputs = model.generate(input_ids=input_ids, top_p=0.9, max_length=256)
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]


@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get user input from the request JSON
        data = request.json
        user_input = data.get("instruction", "")

        if not user_input:
            return jsonify({"error": "No instruction provided"}), 400

        # Format the input and perform inference
        formatted_input = "instruction: " + user_input + ". output: "
        response = inference(formatted_input)

        # Return the result as JSON
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
