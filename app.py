import flask
import models
import time

app = flask.Flask(__name__)
intent_bert = models.IntentBert(weights="model.h5", debug=True)


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    try:
        # do stuff
        query = flask.request.args.get("query")
        pred = intent_bert.predict(query)
        data["result"] = pred
        data["success"] = True
    except Exception as e:
        print(" * [e] Error:", query, "->", e)
        data["error"] = str(e)
    return flask.jsonify(data)

# warm-up

TEST_QUERY = "How do I turn off my computer?"
intent_bert.predict(TEST_QUERY)

BENCHMARK = True

if BENCHMARK:
    print("Starting benchmark:")

    start_time = time.time()

    for i in range(1000):
        _query = str(i) + " " + TEST_QUERY
        intent_bert.predict(_query)

    end_time = time.time()

    print("Total time taken:", round(end_time - start_time, 1))
    print("Query per second:", 1000//(end_time - start_time))
    print("Latency (ms):", (end_time - start_time)/1000 * 1e3)

# if file was executed by itself, start the server process

if __name__ == "__main__":
    print(" * [i] Starting Flask server")
    app.run(host='0.0.0.0', port=5000)
    