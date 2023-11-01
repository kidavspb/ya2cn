from flask import Flask, request, jsonify, render_template, url_for
from main import analyze_comment

app = Flask(__name__)

positive_comments = 0
negative_comments = 0


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/get_sentiment_ratio')
def get_sentiment_ratio():
    total = positive_comments + negative_comments
    total = 1 if total == 0 else total
    return jsonify({
        'positive_percentage': (positive_comments / total) * 100,
        'negative_percentage': (negative_comments / total) * 100
    })


@app.route('/reset_counters', methods=['POST'])
def reset_counters():
    global positive_comments, negative_comments
    positive_comments = 0
    negative_comments = 0
    return jsonify({"status": "success"}), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    comment = data['comment']
    result = analyze_comment(comment)
    if result == 'Positive':
        global positive_comments
        positive_comments += 1
    else:
        global negative_comments
        negative_comments += 1
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(debug=True)
