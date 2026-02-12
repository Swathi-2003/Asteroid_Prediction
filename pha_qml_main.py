'''# pha_qml_main.py
import numpy as np
import matplotlib.pyplot as plt
from preprocess import X_train, X_test, y_train, y_test
from qml_model import run_vqc, run_qsvc
from sklearn.svm import SVC

print("✅ OneHotEncoder patch applied (for Qiskit compatibility)\n")

print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
print(f"Unique classes: {np.unique(y_train)}")
print("Running Quantum Models...\n")

vqc_acc = run_vqc(X_train, X_test, y_train, y_test)
qsvc_acc = run_qsvc(X_train, X_test, y_train, y_test)

# Classical SVM
clf = SVC(kernel="rbf")
clf.fit(X_train[:, :2], y_train)
svm_acc = clf.score(X_test[:, :2], y_test)
print(f"✅ Classical SVM Accuracy: {svm_acc:.3f}")

# Comparison graph
models = ["Classical SVM", "QSVC", "VQC"]
scores = [svm_acc, qsvc_acc, vqc_acc]
plt.bar(models, scores, color=["gray", "skyblue", "orange"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

print("\n✅ All models finished successfully!")

# pha_qml_main.py
from preprocess import preprocess_data
from qml_model import run_vqc, run_qsvc
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---- Preprocess ----
X_train, X_test, y_train, y_test = preprocess_data()

print("\nRunning Quantum Models...")

# ---- Run Quantum Models ----
vqc_acc = run_vqc(X_train, X_test, y_train, y_test)
qsvc_acc = run_qsvc(X_train, X_test, y_train, y_test)

# ---- Classical SVM ----
print("\n🚀 Running Classical SVM for comparison")
clf = SVC(kernel="rbf", C=1)
clf.fit(X_train[:, :2], y_train)
y_pred = clf.predict(X_test[:, :2])
svm_acc = accuracy_score(y_test, y_pred)
print(f"✅ Classical SVM Accuracy: {svm_acc:.3f}")

# ---- Comparison Graph ----
plt.figure(figsize=(6, 4))
plt.bar(["VQC", "QSVC", "Classical SVM"], [vqc_acc, qsvc_acc, svm_acc], color=["#6fa8dc", "#93c47d", "#f6b26b"])
plt.ylabel("Accuracy")
plt.title("Model Comparison: Quantum vs Classical")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("comparison_accuracy.png")
plt.show()

print("\n✅ All models completed. Comparison chart saved as 'comparison_accuracy.png'.")
'''
'''# pha_qml_main.py
import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_and_select
from classical_models import run_classical_models
from qml_model import run_vqc, run_pqsvc

# ---- Config ----
K_FEATURES = 4
USE_CLOUD = False   # set True to run QML on BlueQubit (make sure token env var is set)

# 1) Preprocess and select top-4 features (9 methods voting)
print("🔁 Preprocessing and feature selection (9-method voting)...")
X_train, X_test, y_train, y_test, selected_features = preprocess_and_select(k_features=K_FEATURES, verbose=True)

# 2) Run classical models (4 algorithms)
print("\n💻 Running 4 Classical ML models (DecisionTree, KNN, AdaBoost, Ridge)...")
classical_results, classical_metrics = run_classical_models(X_train, X_test, y_train, y_test)

# 3) Run QML models (VQC and PQSVC) using the selected 4 features
print("\n⚛️ Running Quantum ML models (VQC and PQSVC)...")
# Pass USE_CLOUD to qml functions by setting in qml_model.py OR use default variable there.
vqc_acc = run_vqc(X_train, X_test, y_train, y_test, qubits=K_FEATURES, reps=3, maxiter=40, use_cloud=USE_CLOUD)
pqsvc_acc = run_pqsvc(X_train, X_test, y_train, y_test, qubits=K_FEATURES, reps=1, num_steps=100, C=100.0, use_cloud=USE_CLOUD)

# 4) Combine results
results = classical_results.copy()
results["VQC (Quantum)"] = vqc_acc
results["PQSVC (Quantum)"] = pqsvc_acc

print("\n📊 Final accuracy comparison (all models):")
for name, val in results.items():
    print(f"{name:25s}: {val:.4f}")

# 5) Plot comparison
labels = list(results.keys())
scores = [results[k] for k in labels]

plt.figure(figsize=(10,6))
bars = plt.bar(labels, scores, color=["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f"])
plt.ylim(0,1)
plt.title("Accuracy: Classical (4) vs Quantum (VQC,PQSVC)")
plt.xticks(rotation=20, ha='right')
for i,b in enumerate(bars):
    plt.text(b.get_x() + b.get_width()/2, scores[i] + 0.01, f"{scores[i]:.3f}", ha='center')
plt.tight_layout()
plt.show()
'''

# pha_qml_main.py
import matplotlib.pyplot as plt
from preprocess import preprocess_and_select
from classical_models import run_classical_models
from qml_model import run_vqc, run_pqsvc

# ---- Config ----
K_FEATURES = 4
USE_CLOUD = False

print("🔁 Preprocessing and feature selection...")
X_train, X_test, y_train, y_test, selected = preprocess_and_select(k_features=K_FEATURES, verbose=True)

# Classical models
print("\n💻 Running Classical ML models...")
classical_results, _ = run_classical_models(X_train, X_test, y_train, y_test)

# Quantum models
print("\n⚛️ Running Quantum ML models...")
vqc_metrics = run_vqc(X_train, X_test, y_train, y_test, qubits=K_FEATURES, reps=3, maxiter=40, use_cloud=USE_CLOUD)
pqsvc_metrics, pqsvc_cm = run_pqsvc(X_train, X_test, y_train, y_test, qubits=K_FEATURES, reps=1, use_cloud=USE_CLOUD)

# Summary display
print("\n\n📊 --- FINAL METRICS SUMMARY ---")
print("VQC:")
for name, vals in vqc_metrics.items():
    acc, prec, rec, f1 = vals
    print(f"{name:15s} | Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")

print("\nPQSVC (for each C):")
for C, vals in pqsvc_metrics.items():
    acc, prec, rec, f1 = vals
    print(f"C={C:<6} | Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
    print(pqsvc_cm[C])

# Bar graph comparison (just best quantum vs avg classical)
labels = list(classical_results.keys()) + ["VQC (RealAmp)", "VQC (EffSU2)", "PQSVC (best C)"]
scores = list(classical_results.values()) + [
    max(vqc_metrics["RealAmplitudes"][0], 0),
    max(vqc_metrics["EfficientSU2"][0], 0),
    max(v[0] for v in pqsvc_metrics.values())
]

plt.figure(figsize=(10,6))
bars = plt.bar(labels, scores, color=["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#ffd92f","#a6d854","#e5c494"])
plt.ylim(0,1)
plt.title("Accuracy Comparison: Classical vs Quantum Models")
plt.xticks(rotation=20, ha='right')
for i,b in enumerate(bars):
    plt.text(b.get_x()+b.get_width()/2, scores[i]+0.01, f"{scores[i]:.3f}", ha='center')
plt.tight_layout()
plt.show()


