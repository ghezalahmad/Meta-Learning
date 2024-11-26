import os
import pandas as pd
import numpy as np
import torch
from flask import Flask, render_template, request, send_file
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

class MAMLModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(MAMLModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)

def compute_novelty(sample, training_data):
    distances = np.linalg.norm(training_data - sample, axis=1)
    normalized_distances = distances / np.linalg.norm(training_data.max(axis=0) - training_data.min(axis=0))
    return 1 - normalized_distances.mean()

def compute_uncertainty(predictions):
    return np.std(predictions, axis=1)

def compute_utility(predictions, weights=None):
    """
    Example utility function: weighted ratio of targets.
    Adjust as necessary for your use case.
    """
    if weights is None:
        weights = [1.0] * predictions.shape[1]
    return (predictions[:, 0] * weights[0]) / (predictions[:, 1] * weights[1])

def multi_objective_loss(predictions, targets, weights=None):
    num_targets = targets.shape[1]
    if weights is None:
        weights = [1.0] * num_targets
    loss = 0.0
    for i in range(num_targets):
        mse = torch.mean((predictions[:, i] - targets[:, i])**2)
        loss += weights[i] * mse
    return loss

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    data = pd.read_csv(file_path)
    columns = data.columns.tolist()
    return render_template("configure.html", file_path=file.filename, columns=columns)

def compute_utility(predictions, weights=None):
    """
    Compute utility scores.
    - For two targets: Compute a weighted ratio.
    - For one target: Use the value directly as the utility.
    """
    num_targets = predictions.shape[1]
    if num_targets == 1:
        return predictions[:, 0]  # Single target, utility is just the predicted value

    if weights is None:
        weights = [1.0] * num_targets  # Default equal weights

    # For two or more targets
    return (predictions[:, 0] * weights[0]) / (predictions[:, 1] * weights[1])


@app.route("/configure", methods=["POST"])
def configure():
    try:
        # Form inputs
        file_path = os.path.join(UPLOAD_FOLDER, request.form["file_path"])
        input_columns = request.form.getlist("input_columns")
        target_columns = request.form.getlist("target_columns")

        # Load data
        data = pd.read_csv(file_path)

        # Separate known targets for training and unknown targets for prediction
        known_targets = ~data[target_columns[0]].isna()  # Assume one target column
        inputs_train = data.loc[known_targets, input_columns]
        targets_train = data.loc[known_targets, target_columns]
        inputs_infer = data.loc[~known_targets, input_columns]
        idx_samples = data.loc[~known_targets, "Idx_Sample"]  # Ensure `Idx_sample` is included

        # Scale data
        scaler_inputs = StandardScaler()
        scaler_targets = StandardScaler()

        inputs_train_scaled = scaler_inputs.fit_transform(inputs_train)
        inputs_infer_scaled = scaler_inputs.transform(inputs_infer)
        targets_train_scaled = scaler_targets.fit_transform(targets_train)

        # Meta-learning model training
        meta_model = MAMLModel(len(input_columns), len(target_columns))
        maml_training(meta_model, inputs_train_scaled, targets_train_scaled)

        # Predictions for discovery points
        inputs_infer_tensor = torch.tensor(inputs_infer_scaled, dtype=torch.float32)
        with torch.no_grad():
            predictions_scaled = meta_model(inputs_infer_tensor).numpy()
        predictions = scaler_targets.inverse_transform(predictions_scaled)

        # Calculate utility, novelty, and uncertainty
        novelty_scores = [compute_novelty(sample, inputs_train_scaled) for sample in inputs_infer_scaled]
        uncertainty_scores = np.std(predictions_scaled, axis=1)  # Simulated uncertainty as std deviation
        utility_scores = predictions[:, 0] / (1 + uncertainty_scores)  # Example utility formula

        # Compile results
        result_df = pd.DataFrame({
            "Idx_Sample": idx_samples.astype(int).values,
            "Utility": utility_scores,
            "Novelty": novelty_scores,
            "Uncertainty": uncertainty_scores,
            **{col: predictions[:, i] for i, col in enumerate(target_columns)},
            **inputs_infer.reset_index(drop=True).to_dict(orient="list"),
        })

        result_file_path = os.path.join(RESULT_FOLDER, "processed_results.csv")
        result_df.to_csv(result_file_path, index=False)

        # Generate Plotly scatter plot
        scatter_fig = go.Figure()

        scatter_fig.add_trace(go.Scatter(
            x=inputs_infer[input_columns[0]],
            y=predictions[:, 0],
            mode='markers',
            marker=dict(
                size=10,
                color=utility_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Utility")
            ),
            text=[
                f"Idx_Sample: {idx}<br>Utility: {utility:.2f}<br>Novelty: {novelty:.2f}<br>Uncertainty: {uncertainty:.2f}"
                for idx, utility, novelty, uncertainty in zip(idx_samples, utility_scores, novelty_scores, uncertainty_scores)
            ],
            hoverinfo='text'
        ))

        scatter_fig.update_layout(
            title="Discovery Scatter Plot",
            xaxis_title=input_columns[0],
            yaxis_title=target_columns[0],
            template="plotly_white"
        )

        scatter_plot = json.dumps(scatter_fig, cls=PlotlyJSONEncoder)

        # Render results
        table_columns = result_df.columns.tolist()
        table_data = result_df.values.tolist()

        return render_template(
            "results.html",
            scatter_plot=scatter_plot,
            table_columns=table_columns,
            table_data=table_data,
        )
    except Exception as e:
        return f"Error: {str(e)}", 500





@app.route('/download', methods=['GET'])
def download():
    result_file_path = os.path.join(RESULT_FOLDER, "processed_results.csv")
    if not os.path.exists(result_file_path):
        return "No results to download!", 404
    return send_file(result_file_path, as_attachment=True)

def maml_training(meta_model, inputs_scaled, targets_scaled, num_tasks=4, num_epochs=50, inner_lr=0.005, outer_lr=0.0005):
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=outer_lr)
    clustering = AgglomerativeClustering(n_clusters=num_tasks)
    task_labels = clustering.fit_predict(inputs_scaled)
    tasks = {i: {"inputs": [], "targets": []} for i in range(num_tasks)}

    for i, label in enumerate(task_labels):
        tasks[label]["inputs"].append(inputs_scaled[i])
        tasks[label]["targets"].append(targets_scaled[i])

    for task_id in tasks:
        tasks[task_id]["inputs"] = torch.tensor(np.array(tasks[task_id]["inputs"]), dtype=torch.float32)
        tasks[task_id]["targets"] = torch.tensor(np.array(tasks[task_id]["targets"]), dtype=torch.float32)

    num_targets = targets_scaled.shape[1]
    weights = [1.0] * num_targets

    for epoch in range(num_epochs):
        meta_loss = 0.0
        for task_id, task in tasks.items():
            task_model = MAMLModel(len(inputs_scaled[0]), num_targets)
            task_model.load_state_dict(meta_model.state_dict())
            task_optimizer = torch.optim.SGD(task_model.parameters(), lr=inner_lr)

            support_inputs = task["inputs"][: int(0.8 * len(task["inputs"]))]
            query_inputs = task["inputs"][int(0.8 * len(task["inputs"])):]
            support_targets = task["targets"][: int(0.8 * len(task["targets"]))]
            query_targets = task["targets"][int(0.8 * len(task["targets"])):]

            for _ in range(5):
                preds = task_model(support_inputs)
                loss = multi_objective_loss(preds, support_targets, weights)
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()

            query_preds = task_model(query_inputs)
            meta_loss += multi_objective_loss(query_preds, query_targets, weights)

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Meta Loss: {meta_loss.item():.4f}")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
