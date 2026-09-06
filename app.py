from flask import Flask, render_template ,request
import LinearRegressionGrades , LinearConsumo
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/concepts")
def concepts():
    return render_template("concepts.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/types")
def types():
    return render_template("types.html")


@app.route("/algorithms")
def algorithms():
    return render_template("algorithms.html")


@app.route("/examples")
def examples():
    return render_template("examples.html")


@app.route("/trends")
def trends():
    return render_template("trends.html")


@app.route("/LinearRegression/", methods = ["GET","POST"])
def LRregessionGrades():
    calculateGradeResult = None
    if request.method == "POST":
        hours= float(request.form["hours"])
        calculateGradeResult = LinearRegressionGrades.calculateGrade(hours)
    return render_template("temoLinearRegression.html", result = calculateGradeResult) 

@app.route("/LinearConsumo/", methods = ["GET","POST"])
def RegressionConsumo():
    calculateConsumo = None
    if request.method == "POST":
        habitantes = float(request.form["habitantes"])
        calculateConsumo = LinearConsumo.calculateConsumo(habitantes)
    return render_template("LinearConsumo.html", result = calculateConsumo) 

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

