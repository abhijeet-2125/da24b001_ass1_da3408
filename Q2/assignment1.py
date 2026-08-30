import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("mnist-mlp-classifier")
print("Tracking URI:", mlflow.get_tracking_uri())

#oading data
print("Fetching MNIST (this can take a minute the first time)...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

X, _, y, _ = train_test_split(X, y, train_size=10000, stratify=y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def train_and_log(hidden_layer_sizes, learning_rate_init, batch_size, run_name):
    with mlflow.start_run(run_name=run_name):
        #parameters
        mlflow.log_param("hidden_layer_sizes", str(hidden_layer_sizes))
        mlflow.log_param("learning_rate_init", learning_rate_init)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("predictor", "MLPClassifier")
        mlflow.log_param("dataset", "MNIST")

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=40,
            early_stopping=False,
            random_state=42,
        )
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_test, test_preds)
        val_f1 = f1_score(y_test, test_preds, average="macro")
        train_loss = model.loss_

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_f1_macro", val_f1)
        mlflow.log_metric("train_loss", train_loss)

        mlflow.set_tag("team", "data-science")
        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=[
        "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"
    ]
)

        run_id = mlflow.active_run().info.run_id
        print(
            f"Logged run {run_id} | {run_name} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} train_loss={train_loss:.4f}"
        )
        return run_id

configs = [
    {"hidden_layer_sizes": (64,),        "learning_rate_init": 0.001, "batch_size": 32},
    {"hidden_layer_sizes": (64,),        "learning_rate_init": 0.01,  "batch_size": 32},
    {"hidden_layer_sizes": (128,),       "learning_rate_init": 0.001, "batch_size": 32},
    {"hidden_layer_sizes": (128,),       "learning_rate_init": 0.01,  "batch_size": 32},
    {"hidden_layer_sizes": (128, 64),    "learning_rate_init": 0.001, "batch_size": 64},
    {"hidden_layer_sizes": (128, 64),    "learning_rate_init": 0.01,  "batch_size": 64},
]

run_ids = []
for i, cfg in enumerate(configs, start=1):
    run_name = f"mlp-{cfg['hidden_layer_sizes']}-lr{cfg['learning_rate_init']}-bs{cfg['batch_size']}"
    rid = train_and_log(**cfg, run_name=run_name)
    run_ids.append(rid)

print("\nAll run IDs:", run_ids)

runs_df = mlflow.search_runs(
    experiment_names=["mnist-mlp-classifier"],
    order_by=["metrics.val_accuracy DESC"],
)
display_cols = [c for c in runs_df.columns if c in (
    "run_id", "tags.mlflow.runName", "params.hidden_layer_sizes",
    "params.learning_rate_init", "params.batch_size",
    "metrics.val_accuracy", "metrics.train_loss",
)]
print(runs_df[display_cols].head(10).to_string(index=False))

best_run = runs_df.iloc[0]
print(f"\nBest run: {best_run['run_id']}  (val_accuracy={best_run['metrics.val_accuracy']:.4f})")

