from flask import Flask, render_template, request, abort
import LinearRegressionGrades, LinearConsumo , clusteringE
from use_cases import USE_CASES

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


@app.route("/Cluster/")
def Cluster():
    info = clusteringE.implementClustering()
    return info["Result"]



@app.route("/use-cases/<int:case_id>")
def use_case(case_id):
    case = USE_CASES.get(case_id)
    if case is None:
        abort(404)
    return render_template("use_case.html", case=case, case_id=case_id)


@app.route("/linear-regression/concepts")
def linear_regression_concepts():
    return render_template("linear_regression_concepts.html")


@app.route("/linear-regression/application", methods=["GET", "POST"])
def linear_regression_application():
    prediction = None
    error = None

    if request.method == "POST":
        raw_value = request.form.get("occupants", "").strip()

        if not raw_value:
            error = "Please enter a value for the number of occupants."
        else:
            try:
                occupants = float(raw_value)
                if occupants <= 0:
                    error = "The number of occupants must be a positive number."
                else:
                    prediction = round(float(LinearConsumo.calculateConsumo(occupants)), 2)
            except ValueError:
                error = "Please enter a valid numeric value."

    return render_template(
        "linear_regression_application.html",
        prediction=prediction,
        error=error,
        record_count=LinearConsumo.RECORD_COUNT,
        chart=LinearConsumo.generate_plot(),
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

