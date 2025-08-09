from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook_receiver():
    data = request.json
    print(f"Received webhook data: {data}")
    # You can add your custom logic here to process the data
    return jsonify({"message": "Webhook received!"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)



from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/v1/hackrx/run', methods=['POST', 'GET']) # Allow POST and GET for flexibility
def hackrx_webhook():
    if request.method == 'POST':
        data = request.json
        print(f"Received POST data at /api/v1/hackrx/run: {data}")
        # Process the data here as needed
        return jsonify({"message": "HackRX webhook received successfully!"}), 200
    elif request.method == 'GET':
        # You might want to return a simple confirmation for GET requests
        return jsonify({"message": "HackRX webhook endpoint is active!"}), 200
    else:
        return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    # Use a port that's likely to be open, or specify one.
    # For local testing, 5000 is common.
    # For deployment, you'd use a more robust server like Gunicorn.
    app.run(debug=True, port=5000)