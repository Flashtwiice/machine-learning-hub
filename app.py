from flask import Flask, render_template
import LinearRegressionGrades
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


@app.route("/LinearRegression/")
def LRregessionGrades():
    calculateGradeResult = LinearRegressionGrades.calculateGrade(5)
    return str(calculateGradeResult)

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
