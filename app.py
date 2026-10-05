from flask import Flask, render_template, request, abort
import LinearRegressionGrades, LinearConsumo
from use_cases import USE_CASES
import logistic_regression_diabetes as logistic_regression
import random_forest_purchase as random_forest
import kmeans_manual
import kmeans_energy
from reinforcement import train, GRID, START, GOAL, ACTION_NAMES
import rl_gridworld
import rl_content

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


@app.route("/logistic-regression/concepts")
def logistic_regression_concepts():
    return render_template("logistic_regression_concepts.html")


@app.route("/logistic-regression/application", methods=["GET", "POST"])
def logistic_regression_application():
    result = None
    error = None

    if request.method == "POST":
        raw_value = request.form.get("glucose", "").strip()

        if not raw_value:
            error = "Please enter a value for the glucose level."
        else:
            try:
                glucose = float(raw_value)
                if glucose <= 0:
                    error = "The glucose level must be a positive number."
                else:
                    result = logistic_regression.predict_class(glucose)
            except ValueError:
                error = "Please enter a valid numeric value."

    return render_template(
        "logistic_regression_application.html",
        result=result,
        error=error,
        record_count=logistic_regression.RECORD_COUNT,
        train_count=logistic_regression.TRAIN_COUNT,
        test_count=logistic_regression.TEST_COUNT,
        chart=logistic_regression.generate_plot(),
    )


@app.route("/logistic-regression/evaluation-metrics")
def logistic_regression_metrics():
    return render_template(
        "logistic_regression_metrics.html",
        metrics=logistic_regression.get_metrics(),
        test_count=logistic_regression.TEST_COUNT,
    )


@app.route("/random-forest/concepts")
def random_forest_concepts():
    return render_template("random_forest_concepts.html")


@app.route("/random-forest/application", methods=["GET", "POST"])
def random_forest_application():
    result = None
    error = None

    if request.method == "POST":
        raw_age = request.form.get("age", "").strip()
        raw_income = request.form.get("income", "").strip()
        raw_time = request.form.get("time_on_site", "").strip()

        if not raw_age or not raw_income or not raw_time:
            error = "Please fill in age, income and time on site."
        else:
            try:
                age = float(raw_age)
                income = float(raw_income)
                time_on_site = float(raw_time)
                if age <= 0 or income <= 0 or time_on_site <= 0:
                    error = "All values must be positive numbers."
                else:
                    result = random_forest.predict_class(age, income, time_on_site)
            except ValueError:
                error = "Please enter valid numeric values."

    return render_template(
        "random_forest_application.html",
        result=result,
        error=error,
        record_count=random_forest.RECORD_COUNT,
        train_count=random_forest.TRAIN_COUNT,
        test_count=random_forest.TEST_COUNT,
        chart=random_forest.generate_plot(),
    )


@app.route("/random-forest/evaluation-metrics")
def random_forest_metrics():
    return render_template(
        "random_forest_metrics.html",
        metrics=random_forest.get_metrics(),
        lr_metrics=logistic_regression.get_metrics(),
        test_count=random_forest.TEST_COUNT,
    )


@app.route("/unsupervised/concepts")
def unsupervised_concepts():
    return render_template("unsupervised_concepts.html")


@app.route("/unsupervised/manual-exercise")
def unsupervised_manual():
    return render_template(
        "unsupervised_manual.html",
        record_count=kmeans_manual.RECORD_COUNT,
        describe=kmeans_manual.DESCRIBE,
        initial_centroids=kmeans_manual.INITIAL_CENTROID_ROWS,
        initial_plot=kmeans_manual.INITIAL_PLOT,
        iterations=kmeans_manual.ITERATION_RESULTS,
    )


@app.route("/unsupervised/clustering-application")
def unsupervised_application():
    return render_template(
        "unsupervised_application.html",
        raw_count=kmeans_energy.RAW_COUNT,
        record_count=kmeans_energy.RECORD_COUNT,
        preprocessing=kmeans_energy.PREPROCESSING,
        config=kmeans_energy.KMEANS_CONFIG,
        n_iter=kmeans_energy.N_ITER,
        k_comparison=kmeans_energy.K_COMPARISON,
        summary=kmeans_energy.CLUSTER_SUMMARY,
        silhouette=kmeans_energy.SILHOUETTE,
        silhouette_label=kmeans_energy.interpret_silhouette(kmeans_energy.SILHOUETTE),
        rows=kmeans_energy.TABLE_ROWS,
        chart=kmeans_energy.generate_plot(),
    )


@app.route('/reinforcement', methods=['GET', 'POST'])
def reinforcement():
    result = None
    if request.method == 'POST':
        result = train(episodes=1000)
 
    # Pass the result and grid settings to the template.
    return render_template(
        'reinforcement.html',
        result=result,
        grid=GRID,
        start=START,
        goal=GOAL,
        actions=ACTION_NAMES,
    )


def _rl_context(**extra):
    context = dict(
        layout=rl_gridworld.LAYOUT,
        legend=rl_gridworld.LEGEND,
        counts=rl_gridworld.CELL_COUNTS,
        total_cells=rl_gridworld.ROWS * rl_gridworld.COLUMNS,
        reward_table=rl_gridworld.REWARD_TABLE,
        parameters=rl_gridworld.PARAMETERS,
        params=rl_content.P,
        actions=rl_gridworld.ACTION_NAMES,
    )
    context.update(extra)
    return context


@app.route("/reinforcement-learning/concepts")
def rl_concepts():
    return render_template(
        "rl_concepts.html",
        **_rl_context(glossary=rl_content.GLOSSARY, epsilons=rl_gridworld.epsilon_schedule()),
    )


@app.route("/reinforcement-learning/application", methods=["GET", "POST"])
def rl_application():
    result = rl_gridworld.train() if request.method == "POST" else None
    return render_template("rl_application.html", **_rl_context(result=result))


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

