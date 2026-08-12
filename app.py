import base64
import io
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for server use
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model.pkl"
CLUSTER_DATA_PATH = "cluster_data.pkl"

print("Loading model and cluster data...")
kmeans = pickle.load(open(MODEL_PATH, "rb"))
cluster_data = pickle.load(open(CLUSTER_DATA_PATH, "rb"))
print("Model and cluster data loaded successfully.")

X_TRAIN = cluster_data["X"]          # shape (n_customers, 2)
Y_TRAIN = cluster_data["y"]          # cluster label per training customer
FEATURE_NAMES = cluster_data["feature_names"]

N_CLUSTERS = kmeans.n_clusters
CENTROIDS = kmeans.cluster_centers_

OVERALL_MEAN_INCOME = X_TRAIN[:, 0].mean()
OVERALL_MEAN_SCORE = X_TRAIN[:, 1].mean()

CLUSTER_COLORS = [
    "#ff6b6b", "#06d6a0", "#00b4d8", "#ffd166", "#8e44ad",
    "#ff85a1", "#43aa8b", "#f77f00", "#577590", "#e63946"
]


def describe_cluster(cluster_index):
    """
    Build a human-readable label for a cluster based on where its
    centroid sits relative to the overall mean income/spending score.
    This avoids hardcoding fixed segment names, since KMeans cluster
    ordering/labels are arbitrary and depend on random_state.
    """
    centroid_income, centroid_score = CENTROIDS[cluster_index]

    income_level = "High" if centroid_income >= OVERALL_MEAN_INCOME else "Low"
    spending_level = "High" if centroid_score >= OVERALL_MEAN_SCORE else "Low"

    label = f"{income_level} Income, {spending_level} Spending"

    descriptions = {
        ("High", "High"): "Premium customers — high income and high spending. Prime targets for loyalty programs and premium offers.",
        ("High", "Low"): "Cautious high earners — high income but low spending. Potential targets for targeted promotions to increase engagement.",
        ("Low", "High"): "Value seekers — limited income but high spending. Respond well to discounts and value-driven marketing.",
        ("Low", "Low"): "Budget-conscious customers — low income and low spending. Lower priority for premium campaigns.",
    }

    description = descriptions.get((income_level, spending_level),
                                    "Customers with moderate income and spending patterns.")

    return label, description


def generate_cluster_plot(new_point, predicted_cluster):
    fig, ax = plt.subplots(figsize=(7, 6))

    for cluster_id in range(N_CLUSTERS):
        mask = Y_TRAIN == cluster_id
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        ax.scatter(
            X_TRAIN[mask, 0], X_TRAIN[mask, 1],
            s=45, alpha=0.65, color=color,
            label=f"Cluster {cluster_id}"
        )

    ax.scatter(
        CENTROIDS[:, 0], CENTROIDS[:, 1],
        s=220, c="black", marker="X", label="Centroids"
    )

    ax.scatter(
        [new_point[0]], [new_point[1]],
        s=320, c="gold", marker="*",
        edgecolors="black", linewidths=1.5,
        label="New Customer", zorder=5
    )

    ax.set_xlabel(FEATURE_NAMES[0])
    ax.set_ylabel(FEATURE_NAMES[1])
    ax.set_title(f"Customer Segments (New Customer → Cluster {predicted_cluster})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", n_clusters=N_CLUSTERS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "No form data received."}), 400

        try:
            annual_income = float(data.get("Annual_Income", ""))
            spending_score = float(data.get("Spending_Score", ""))
        except (ValueError, TypeError):
            return jsonify({"error": "Please enter valid numeric values for income and spending score."}), 400

        if not (0 <= spending_score <= 100):
            return jsonify({"error": "Spending Score must be between 0 and 100."}), 400

        if annual_income < 0:
            return jsonify({"error": "Annual Income cannot be negative."}), 400

        new_point = np.array([[annual_income, spending_score]])
        predicted_cluster = int(kmeans.predict(new_point)[0])

        label, description = describe_cluster(predicted_cluster)
        plot_image = generate_cluster_plot(new_point[0], predicted_cluster)

        return jsonify({
            "cluster": predicted_cluster,
            "label": label,
            "description": description,
            "plot_image": plot_image
        }), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Route not found."}), 404


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": "Internal server error. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True)
