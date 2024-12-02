import os
import pandas as pd
import numpy as np
import torch
import streamlit as st
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import plotly.express as px
import torch.optim as optim  # Import PyTorch's optimizer module
from skopt import gp_minimize
from skopt.space import Real
import json
import plotly.graph_objects as go


# Set up directories
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Define MAML model
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
    
def meta_train(meta_model, data, input_columns, target_columns, epochs, inner_lr, outer_lr, num_tasks=5):
    """
    Meta-train the MAML model using simulated tasks.

    Args:
        meta_model (MAMLModel): The MAML model to train.
        data (pd.DataFrame): The dataset containing input and target columns.
        input_columns (list): Columns used as input features.
        target_columns (list): Columns used as target properties.
        epochs (int): Number of meta-training epochs.
        inner_lr (float): Learning rate for the inner loop.
        outer_lr (float): Learning rate for the outer loop.
        num_tasks (int): Number of tasks to simulate.

    Returns:
        MAMLModel: The trained meta-model.
    """
    optimizer = optim.Adam(meta_model.parameters(), lr=outer_lr)
    loss_function = torch.nn.MSELoss()

    for epoch in range(epochs):
        meta_loss = 0.0

        for task in range(num_tasks):
            # Simulate a task by sampling a subset of the data
            task_data = data.sample(frac=0.2)  # Use 20% of the data for this task
            inputs = torch.tensor(task_data[input_columns].values, dtype=torch.float32)
            targets = torch.tensor(task_data[target_columns].values, dtype=torch.float32)

            # Split into support set (inner loop) and query set (outer loop)
            num_support = int(len(inputs) * 0.8)
            support_inputs, query_inputs = inputs[:num_support], inputs[num_support:]
            support_targets, query_targets = targets[:num_support], targets[num_support:]

            # Inner loop: Task-specific adaptation
            task_model = MAMLModel(len(input_columns), len(target_columns), hidden_size=hidden_size)
            task_model.load_state_dict(meta_model.state_dict())  # Clone the meta-model
            task_optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)

            for _ in range(5):  # Inner loop iterations
                task_predictions = task_model(support_inputs)
                task_loss = loss_function(task_predictions, support_targets)
                task_optimizer.zero_grad()
                task_loss.backward()
                task_optimizer.step()

            # Outer loop: Meta-optimization
            query_predictions = task_model(query_inputs)
            query_loss = loss_function(query_predictions, query_targets)
            meta_loss += query_loss

        # Update meta-model parameters
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Meta-Loss: {meta_loss.item():.4f}")

    return meta_model


# Utility function
def calculate_utility(predictions, uncertainties, apriori, curiosity, weights, max_or_min, thresholds=None):
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    weights = np.array(weights).reshape(1, -1)
    max_or_min = np.array(max_or_min)

    # Normalize predictions
    prediction_std = predictions.std(axis=0, keepdims=True).clip(min=1e-6)
    prediction_mean = predictions.mean(axis=0, keepdims=True)
    normalized_predictions = (predictions - prediction_mean) / prediction_std

    # Adjust for min/max optimization
    for i, mode in enumerate(max_or_min):
        if mode == "min":
            normalized_predictions[:, i] *= -1

    # Apply weights to predictions
    weighted_predictions = normalized_predictions * weights[:, :predictions.shape[1]]

    # Normalize uncertainties
    uncertainty_std = uncertainties.std(axis=0, keepdims=True).clip(min=1e-6)
    normalized_uncertainties = uncertainties / uncertainty_std
    weighted_uncertainties = normalized_uncertainties * weights[:, :uncertainties.shape[1]]

    # Handle apriori constraints
    if apriori is not None and apriori.shape[1] > 0:
        apriori = np.array(apriori)
        apriori_std = apriori.std(axis=0, keepdims=True).clip(min=1e-6)
        apriori_mean = apriori.mean(axis=0, keepdims=True)
        normalized_apriori = (apriori - apriori_mean) / apriori_std

        # Align apriori dimensions with predictions
        if apriori.shape[1] != predictions.shape[1]:
            apriori = np.resize(apriori, (apriori.shape[0], predictions.shape[1]))

        # Apply thresholds
        if thresholds is not None:
            thresholds = np.array(thresholds).reshape(1, -1)
            for i, (thresh, mode) in enumerate(zip(thresholds[0], max_or_min)):
                if i < apriori.shape[1] and thresh is not None:  # Ensure index is valid
                    if mode == "min":
                        normalized_apriori[:, i] = np.where(
                            apriori[:, i] > thresh, 0, normalized_apriori[:, i]
                        )
                    elif mode == "max":
                        normalized_apriori[:, i] = np.where(
                            apriori[:, i] < thresh, 0, normalized_apriori[:, i]
                        )

        weighted_apriori = normalized_apriori * weights[:, :apriori.shape[1]]
        apriori_utility = weighted_apriori.sum(axis=1)
    else:
        apriori_utility = np.zeros(predictions.shape[0])

    # Combine all utility components
    utility = (
        weighted_predictions.sum(axis=1)
        + (curiosity * 10) * weighted_uncertainties.sum(axis=1)  # Amplify uncertainty impact
        + apriori_utility
    )
    return utility



# Novelty calculation
def calculate_novelty(features, labeled_features):
    if labeled_features.shape[0] == 0:
        return np.zeros(features.shape[0])
    distances = distance_matrix(features, labeled_features)
    min_distances = distances.min(axis=1)
    max_distance = min_distances.max()
    novelty = min_distances / (max_distance + 1e-6)
    return novelty

# Scatter plot
def plot_scatter_matrix(result_df, target_columns, utility_scores):
    scatter_data = result_df[target_columns + ["Utility"]].copy()
    scatter_data["Utility"] = utility_scores

    fig = px.scatter_matrix(
        scatter_data,
        dimensions=target_columns,
        color="Utility",
        color_continuous_scale="Viridis",
        title="Scatter Matrix of Target Properties",
        labels={col: col for col in target_columns},
    )
    fig.update_traces(diagonal_visible=False)
    return fig


def restore_session(session_data):
    """
    Restore session variables from a session file.
    Args:
        session_data (dict): Parsed JSON data from the session file.
    Returns:
        dict: Restored session variables.
    """
    restored_session = {
        "input_columns": session_data.get("input_columns", []),
        "target_columns": session_data.get("target_columns", []),
        "apriori_columns": session_data.get("apriori_columns", []),
        "weights_targets": session_data.get("weights_targets", []),
        "weights_apriori": session_data.get("weights_apriori", []),
        "thresholds_targets": session_data.get("thresholds_targets", []),
        "thresholds_apriori": session_data.get("thresholds_apriori", []),
        "curiosity": session_data.get("curiosity", 0.0),
        "results": pd.DataFrame(session_data.get("results", [])) if "results" in session_data else pd.DataFrame(),
    }
    return restored_session

def create_tsne_plot(data, features, utility_col="Utility", perplexity=20, learning_rate=200):
    """
    Create a t-SNE plot for the dataset.

    Args:
        data (pd.DataFrame): The dataset with features and utility.
        features (list): The list of feature column names.
        utility_col (str): Column name representing utility scores.
        perplexity (int): Perplexity parameter for t-SNE.
        learning_rate (int): Learning rate for t-SNE optimization.

    Returns:
        plotly.graph_objects.Figure: A scatter plot in t-SNE space.
    """
    # Validate input data
    if len(features) == 0:
        raise ValueError("No features selected for t-SNE.")

    if utility_col not in data.columns:
        raise ValueError(f"The column '{utility_col}' is not in the dataset.")

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(data) - 1),
        n_iter=350,
        random_state=42,
        init="pca",
        learning_rate=learning_rate,
    )

    # Fit t-SNE on the selected feature columns
    tsne_result = tsne.fit_transform(data[features])

    # Create a dataframe with t-SNE results
    tsne_result_df = pd.DataFrame({
        "t-SNE-1": tsne_result[:, 0],
        "t-SNE-2": tsne_result[:, 1],
        utility_col: data[utility_col].values,
    })

    # Generate scatter plot
    fig = px.scatter(
        tsne_result_df,
        x="t-SNE-1",
        y="t-SNE-2",
        color=utility_col,
        title="t-SNE Visualization of Data",
        labels={"t-SNE-1": "t-SNE Dimension 1", "t-SNE-2": "t-SNE Dimension 2"},
        color_continuous_scale="Viridis",
    )

    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=800, legend_title_text="Utility")
    return fig




# Streamlit layout
st.set_page_config(page_title="MAML Dashboard", layout="wide")
st.title("MAML Dashboard")

# Sidebar Configuration




# Model Configuration Section
# Model Configuration Section
with st.sidebar.expander("Model Configuration", expanded=True):  # Expanded by default
    # Hidden Size
    hidden_size = st.slider(
        "Hidden Size (MAML):", 
        min_value=64, 
        max_value=256, 
        step=16, 
        value=hidden_size if 'hidden_size' in locals() else 128,  # Use session data if available
        help="The number of neurons in the hidden layers of the MAML model. Larger sizes capture more complex patterns but increase training time."
    )

    # Learning Rate
    learning_rate = st.slider(
        "Learning Rate:", 
        min_value=0.001, 
        max_value=0.1, 
        step=0.001, 
        value=learning_rate if 'learning_rate' in locals() else 0.01,  # Use session data if available
        help="The step size for updating model weights during optimization. Higher values accelerate training but may overshoot optimal solutions."
    )

    # Curiosity
    curiosity = st.slider(
        "Curiosity (Explore vs Exploit):", 
        min_value=-2.0, 
        max_value=2.0, 
        value=curiosity if 'curiosity' in locals() else 0.0,  # Use session data if available
        step=0.1, 
        help="Balances exploration and exploitation. Negative values focus on high-confidence predictions, while positive values prioritize exploring uncertain regions."
    )


# Learning Rate Scheduler Section
with st.sidebar.expander("Learning Rate Scheduler", expanded=False):  # Collapsed by default
    scheduler_type = st.selectbox(
        "Scheduler Type:", 
        ["None", "CosineAnnealing", "ReduceLROnPlateau"], 
        help="Choose the learning rate scheduler type:\n- None: Keeps the learning rate constant.\n- CosineAnnealing: Gradually reduces the learning rate in a cosine curve.\n- ReduceLROnPlateau: Lowers the learning rate when the loss plateaus."
    )

    scheduler_params = {}
    if scheduler_type == "CosineAnnealing":
        scheduler_params["T_max"] = st.slider(
            "T_max (CosineAnnealing):", 
            min_value=10, 
            max_value=100, 
            step=10, 
            value=50, 
            help="The number of iterations over which the learning rate decreases in a cosine curve."
        )
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler_params["factor"] = st.slider(
            "Factor (ReduceLROnPlateau):", 
            min_value=0.1, 
            max_value=0.9, 
            step=0.1, 
            value=0.5, 
            help="The factor by which the learning rate is reduced when the loss plateaus."
        )
        scheduler_params["patience"] = st.slider(
            "Patience (ReduceLROnPlateau):", 
            min_value=1, 
            max_value=10, 
            step=1, 
            value=5, 
            help="The number of epochs to wait before reducing the learning rate after the loss plateaus."
        )

# Meta-Training Configuration Section
with st.sidebar.expander("Meta-Training Configuration", expanded=False):  # Collapsed by default
    meta_epochs = st.slider(
        "Meta-Training Epochs:", 
        min_value=10, 
        max_value=100, 
        step=10, 
        value=50, 
        help="The number of iterations over all simulated tasks during meta-training. Higher values improve adaptability but increase training time."
    )

    inner_lr = st.slider(
        "Inner Loop Learning Rate:", 
        min_value=0.001, 
        max_value=0.1, 
        step=0.001, 
        value=0.01, 
        help="The learning rate for task-specific adaptation during the inner loop. Controls how quickly the model adapts to a single task."
    )

    outer_lr = st.slider(
        "Outer Loop Learning Rate:", 
        min_value=0.001, 
        max_value=0.1, 
        step=0.001, 
        value=0.01, 
        help="The learning rate for updating meta-parameters in the outer loop. A lower value ensures stability, while a higher value speeds up training."
    )

    num_tasks = st.slider(
        "Number of Tasks:", 
        min_value=2, 
        max_value=10, 
        step=1, 
        value=5, 
        help="The number of tasks (subsets of the dataset) simulated during each epoch of meta-training. More tasks improve generalization but increase computation."
    )

# Sidebar: Restore Session
# Sidebar: Restore Session
st.sidebar.header("Restore Session")
uploaded_session = st.sidebar.file_uploader(
    "Upload Session File (JSON):",
    type=["json"],
    help="Upload a previously saved session file to restore your configuration."
)

if uploaded_session:
    try:
        # Load and parse the uploaded JSON session file
        session_data = json.load(uploaded_session)
        restored_session = restore_session(session_data)
        
        # Apply restored session values
        input_columns = restored_session["input_columns"]
        target_columns = restored_session["target_columns"]
        apriori_columns = restored_session["apriori_columns"]
        weights_targets = restored_session["weights_targets"]
        weights_apriori = restored_session["weights_apriori"]
        thresholds_targets = restored_session["thresholds_targets"]
        thresholds_apriori = restored_session["thresholds_apriori"]
        curiosity = restored_session["curiosity"]
        result_df = restored_session["results"]

        st.sidebar.success("Session restored successfully!")

        # Display restored dataset and results (optional)
        if not result_df.empty:
            st.write("### Restored Results Table")
            st.dataframe(result_df, use_container_width=True)

    except Exception as e:
        st.sidebar.error(f"Failed to restore session: {str(e)}")


# File upload
# Initialize input, target, and apriori columns globally
input_columns = []
target_columns = []
apriori_columns = []

# File upload
uploaded_file = st.file_uploader("Upload Dataset (CSV format):", type=["csv"])
if uploaded_file:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    data = pd.read_csv(file_path)
    st.success("Dataset uploaded successfully!")
    st.dataframe(data)

    # Feature selection
    st.header("Select Features")
    input_columns = st.multiselect(
        "Input Features:",
        options=data.columns.tolist(),
        default=input_columns  # Initialized as empty list above
    )

    remaining_columns = [col for col in data.columns if col not in input_columns]
    target_columns = st.multiselect(
        "Target Properties:",
        options=remaining_columns,
        default=target_columns  # Initialized as empty list above
    )

    remaining_columns_aprior = [col for col in remaining_columns if col not in target_columns]
    apriori_columns = st.multiselect(
        "A Priori Properties:",
        options=remaining_columns_aprior,
        default=apriori_columns  # Initialized as empty list above
    )



    # Target settings
    st.header("Target Settings")
    max_or_min_targets = []
    weights_targets = []
    thresholds_targets = []
    for col in target_columns:
        with st.expander(f"Target: {col}"):
            optimize_for = st.radio(f"Optimize {col} for:", ["Maximize", "Minimize"], index=0)
            weight = st.number_input(f"Weight for {col}:", value=1.0, step=0.1)
            threshold = st.text_input(f"Threshold (optional) for {col}:", value="")
            max_or_min_targets.append("max" if optimize_for == "Maximize" else "min")
            weights_targets.append(weight)
            thresholds_targets.append(float(threshold) if threshold else None)

    
    # Apriori Settings
    st.header("Apriori Settings")
    max_or_min_apriori = []
    weights_apriori = []
    thresholds_apriori = []

    for col in apriori_columns:
        with st.expander(f"A Priori: {col}"):
            optimize_for = st.radio(f"Optimize {col} for:", ["Maximize", "Minimize"], index=0)
            weight = st.number_input(f"Weight for {col}:", value=1.0, step=0.1)
            threshold = st.text_input(f"Threshold (optional) for {col}:", value="")
            max_or_min_apriori.append("max" if optimize_for == "Maximize" else "min")
            weights_apriori.append(weight)
            thresholds_apriori.append(float(threshold) if threshold else None)


    # Experiment execution
    if st.button("Run Experiment"):
        if not input_columns or not target_columns:
            st.error("Please select at least one input feature and one target property.")
        else:
            try:
                # Prepare data
                known_targets = ~data[target_columns[0]].isna()
                inputs_train = data.loc[known_targets, input_columns]
                targets_train = data.loc[known_targets, target_columns]
                inputs_infer = data.loc[~known_targets, input_columns]

                # Handle Idx_Sample (assign sequential IDs if not present)
                if "Idx_Sample" in data.columns:
                    idx_samples = data.loc[~known_targets, "Idx_Sample"].reset_index(drop=True)
                else:
                    idx_samples = pd.Series(range(1, len(inputs_infer) + 1), name="Idx_Sample")

                # Scale input data
                scaler_inputs = StandardScaler()
                inputs_train_scaled = scaler_inputs.fit_transform(inputs_train)
                inputs_infer_scaled = scaler_inputs.transform(inputs_infer)

                # Scale target data
                scaler_targets = StandardScaler()
                targets_train_scaled = scaler_targets.fit_transform(targets_train)

                # Scale a priori data (if selected)
                if apriori_columns:
                    apriori_data = data[apriori_columns]
                    scaler_apriori = StandardScaler()
                    apriori_scaled = scaler_apriori.fit_transform(apriori_data.loc[known_targets])
                    apriori_infer_scaled = scaler_apriori.transform(apriori_data.loc[~known_targets])
                else:
                    apriori_infer_scaled = np.zeros((inputs_infer.shape[0], 1))  # Default to zeros if no a priori data

                # Meta-learning predictions and training
                meta_model = MAMLModel(len(input_columns), len(target_columns), hidden_size=hidden_size)
                # Perform meta-training
                st.write("### Meta-Training Phase")
                meta_model = meta_train(
                    meta_model=meta_model,
                    data=data,
                    input_columns=input_columns,
                    target_columns=target_columns,
                    epochs=50,  # Meta-training epochs
                    inner_lr=0.01,  # Learning rate for inner loop
                    outer_lr=learning_rate,  # Outer loop learning rate (from sidebar)
                    num_tasks=5  # Number of simulated tasks
                )
                st.write("Meta-training completed!")

                # Initialize the optimizer for the MAML model
                optimizer = optim.Adam(meta_model.parameters(), lr=learning_rate)

                # Function to initialize the learning rate scheduler based on user selection
                def initialize_scheduler(optimizer, scheduler_type, **kwargs):
                    """
                    Initialize the learning rate scheduler based on the user selection.

                    Args:
                        optimizer (torch.optim.Optimizer): Optimizer for the model.
                        scheduler_type (str): The type of scheduler selected by the user.
                        kwargs: Additional parameters for the scheduler.

                    Returns:
                        Scheduler object or None.
                    """
                    if scheduler_type == "CosineAnnealing":
                        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
                    elif scheduler_type == "ReduceLROnPlateau":
                        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
                    else:
                        return None

                # Initialize the scheduler with user-selected parameters
                scheduler = initialize_scheduler(optimizer, scheduler_type, **scheduler_params)

                # Convert training data to PyTorch tensors
                inputs_train_tensor = torch.tensor(inputs_train_scaled, dtype=torch.float32)
                targets_train_tensor = torch.tensor(targets_train_scaled, dtype=torch.float32)

                # Training loop
                epochs = 50  # Number of training epochs
                loss_function = torch.nn.MSELoss()  # Define the loss function (can be adjusted)

                for epoch in range(epochs):  # Loop over the epochs
                    meta_model.train()  # Ensure the model is in training mode

                    # Forward pass: Predict outputs for training inputs
                    predictions_train = meta_model(inputs_train_tensor)

                    # Compute the loss between predictions and actual targets
                    loss = loss_function(predictions_train, targets_train_tensor)

                    # Backward pass and optimization
                    optimizer.zero_grad()  # Clear gradients from the previous step
                    loss.backward()        # Compute gradients via backpropagation
                    optimizer.step()       # Update model parameters using gradients

                    # Update the scheduler if applicable
                    if scheduler_type == "ReduceLROnPlateau":
                        scheduler.step(loss)  # Adjust learning rate based on the loss
                    elif scheduler_type == "CosineAnnealing":
                        scheduler.step()      # Adjust learning rate based on the schedule

                    # Log progress every 10 epochs
                    #if (epoch + 1) % 10 == 0:
                    #    st.write(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

                # After training, move to inference
                meta_model.eval()  # Set the model to evaluation mode
                inputs_infer_tensor = torch.tensor(inputs_infer_scaled, dtype=torch.float32)
                with torch.no_grad():
                    predictions_scaled = meta_model(inputs_infer_tensor).numpy()
                predictions = scaler_targets.inverse_transform(predictions_scaled)

                # Ensure predictions are always 2D (reshape for single target property)
                # Calculate uncertainty
                if len(target_columns) == 1:
                    # Perturbation-based uncertainty
                    num_perturbations = 20  # Number of perturbations
                    noise_scale = 0.1  # Adjust noise scale for exploration
                    perturbed_predictions = []

                    for _ in range(num_perturbations):
                        # Add noise to the input tensor
                        perturbed_input = inputs_infer_tensor + torch.normal(0, noise_scale, size=inputs_infer_tensor.shape)
                        perturbed_prediction = meta_model(perturbed_input).detach().numpy()
                        perturbed_predictions.append(perturbed_prediction)

                    # Stack all perturbed predictions and compute the variance
                    perturbed_predictions = np.stack(perturbed_predictions, axis=0)  # Shape: (num_perturbations, num_samples, target_dim)
                    uncertainty_scores = perturbed_predictions.std(axis=0).mean(axis=1, keepdims=True)  # Variance across perturbations
                else:
                    # For multiple target properties, compute uncertainty for each row
                    uncertainty_scores = np.std(predictions_scaled, axis=1, keepdims=True)




                # Novelty calculation
                novelty_scores = calculate_novelty(inputs_infer_scaled, inputs_train_scaled)

                # Utility calculation
                if apriori_infer_scaled.ndim == 1:
                    apriori_infer_scaled = apriori_infer_scaled.reshape(-1, 1)

                utility_scores = calculate_utility(
                    predictions,
                    uncertainty_scores,
                    apriori_infer_scaled,
                    curiosity=curiosity,
                    weights=weights_targets + (weights_apriori if len(apriori_columns) > 0 else []),  # Combine weights
                    max_or_min=max_or_min_targets + (max_or_min_apriori if len(apriori_columns) > 0 else []),  # Combine min/max
                    thresholds=thresholds_targets + (thresholds_apriori if len(apriori_columns) > 0 else []),  # Combine thresholds
                )


                # Ensure all arrays are of the same length
                num_samples = len(inputs_infer)
                idx_samples = idx_samples[:num_samples]  # Adjust length if necessary
                predictions = predictions[:num_samples]
                utility_scores = utility_scores[:num_samples]
                novelty_scores = novelty_scores[:num_samples]
                uncertainty_scores = uncertainty_scores[:num_samples]

                # Create result dataframe (exclude training samples)
                # global result_df
                result_df = pd.DataFrame({
                    "Idx_Sample": idx_samples,
                    "Utility": utility_scores,
                    "Novelty": novelty_scores,
                    "Uncertainty": uncertainty_scores.flatten(),
                    **{col: predictions[:, i] for i, col in enumerate(target_columns)},
                    **inputs_infer.reset_index(drop=True).to_dict(orient="list"),
                }).sort_values(by="Utility", ascending=False).reset_index(drop=True)

                # After training or inference
                if uploaded_file and "result_df" in globals() and not result_df.empty:
                    st.header("Generate t-SNE Plot")
                    if st.button("Generate t-SNE Plot"):
                        try:
                            # Validate the required columns
                            if "Utility" not in result_df.columns or len(input_columns) == 0:
                                st.error("Please ensure the dataset has utility scores and input features.")
                            else:
                                # Generate the t-SNE plot
                                tsne_plot = create_tsne_plot(
                                    data=result_df,
                                    features=input_columns,
                                    utility_col="Utility",
                                )
                                st.plotly_chart(tsne_plot)
                        except Exception as e:
                            st.error(f"An error occurred while generating the t-SNE plot: {str(e)}")


                
                
                # Display results
                st.write("### Results Table")
                st.dataframe(result_df, use_container_width=True)

                # Add a download button for predictions
                st.write("### Download Predictions")
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                # Scatter plot
                if len(target_columns) > 1:
                    scatter_fig = plot_scatter_matrix(result_df, target_columns, utility_scores)
                    st.write("### Utility in Output Space (Scatter Matrix)")
                    st.plotly_chart(scatter_fig)
                else:
                    st.write("### Utility vs Target Property")
                    single_scatter_fig = px.scatter(
                        result_df,
                        x=target_columns[0],  # Single target column
                        y="Utility",          # Utility score
                        title=f"Utility vs {target_columns[0]}",
                        labels={target_columns[0]: target_columns[0], "Utility": "Utility"},
                        color="Utility",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(single_scatter_fig)
                # Radar Chart
                st.write("### Radar Chart: Performance Overview")
                categories = target_columns + ["Utility", "Novelty", "Uncertainty"]
                values = [predictions[:, i].mean() for i in range(len(target_columns))] + [
                    utility_scores.mean(), novelty_scores.mean(), uncertainty_scores.mean()
                ]
                radar_fig = go.Figure()
                radar_fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Metrics'))
                radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
                st.plotly_chart(radar_fig)

                 # Export Session Data
                session_data = {
                    "input_columns": input_columns,
                    "target_columns": target_columns,
                    "apriori_columns": apriori_columns,
                    "weights_targets": weights_targets,
                    "weights_apriori": weights_apriori,
                    "thresholds_targets": thresholds_targets,
                    "thresholds_apriori": thresholds_apriori,
                    "curiosity": curiosity,
                
                    "results": result_df.to_dict(orient="records")
                }
                session_json = json.dumps(session_data, indent=4)
                st.download_button(
                    label="Download Session as JSON",
                    data=session_json,
                    file_name="session.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")



