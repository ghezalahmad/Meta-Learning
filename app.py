import os
import pandas as pd
import numpy as np
import torch
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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


def multi_objective_loss(predictions, targets, weight_strength=0.5, weight_co2=0.5):
    mse_strength = torch.mean((predictions[:, 0] - targets[:, 0])**2)
    mse_co2 = torch.mean((predictions[:, 1] - targets[:, 1])**2)
    return weight_strength * mse_strength + weight_co2 * mse_co2


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

    # Load dataset to fetch column names
    data = pd.read_csv(file_path)
    columns = data.columns.tolist()

    return render_template("configure.html", file_path=file.filename, columns=columns)


@app.route('/configure', methods=['POST'])
def configure():
    file_path = request.form['file_path']
    input_columns = request.form.getlist('input_columns')
    target_columns = request.form.getlist('target_columns')
    optimization = request.form['optimization']
    weight = float(request.form.get('weight', 1.0))
    threshold = request.form.get('threshold', None)
    curiosity = float(request.form.get('curiosity', 0.0))
    explore_exploit = request.form['explore_exploit']

    # Load dataset and pass configuration to the Meta-Learning pipeline
    data = pd.read_csv(file_path)
    # Apply the Meta-Learning logic here

    # For demonstration purposes, return the received configurations
    return render_template(
        'results.html',  # Ensure this template is created
        input_columns=input_columns,
        target_columns=target_columns,
        optimization=optimization,
        weight=weight,
        threshold=threshold,
        curiosity=curiosity,
        explore_exploit=explore_exploit
    )


def maml_training(meta_model, inputs_scaled, targets_scaled, num_tasks=4, num_epochs=50, inner_lr=0.005, outer_lr=0.0005):
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=outer_lr)
    clustering = AgglomerativeClustering(n_clusters=num_tasks)
    task_labels = clustering.fit_predict(inputs_scaled)

    tasks = {i: {"inputs": [], "targets": []} for i in range(num_tasks)}
    for i, label in enumerate(task_labels):
        tasks[label]["inputs"].append(inputs_scaled[i])
        tasks[label]["targets"].append(targets_scaled[i])

    for task_id in tasks:
        tasks[task_id]["inputs"] = torch.tensor(tasks[task_id]["inputs"], dtype=torch.float32)
        tasks[task_id]["targets"] = torch.tensor(tasks[task_id]["targets"], dtype=torch.float32)

    for epoch in range(num_epochs):
        meta_loss = 0.0
        for task_id, task in tasks.items():
            task_model = MAMLModel(len(inputs_scaled[0]), 2)
            task_model.load_state_dict(meta_model.state_dict())
            task_optimizer = torch.optim.SGD(task_model.parameters(), lr=inner_lr)

            support_inputs = task["inputs"][: int(0.8 * len(task["inputs"]))]
            query_inputs = task["inputs"][int(0.8 * len(task["inputs"])):]
            support_targets = task["targets"][: int(0.8 * len(task["targets"]))]
            query_targets = task["targets"][int(0.8 * len(task["targets"])):]

            for _ in range(5):
                preds = task_model(support_inputs)
                loss = multi_objective_loss(preds, support_targets, weight_strength=0.5, weight_co2=0.5)
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()

            query_preds = task_model(query_inputs)
            meta_loss += multi_objective_loss(query_preds, query_targets)

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Meta Loss: {meta_loss.item():.4f}")


if __name__ == "__main__":
    app.run(debug=True)
