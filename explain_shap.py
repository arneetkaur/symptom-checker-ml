import numpy as np
import shap
import matplotlib.pyplot as plt
from pipeline import model, symptoms, X_train

explainer = shap.LinearExplainer(model, X_train)

def explain(selected, top_n=8, save_as="shap_example.png"):
    x = np.zeros((1, len(symptoms)))
    for s in selected:
        x[0, symptoms.index(s)] = 1

    pred_idx = int(np.argmax(model.predict_proba(x)[0]))
    disease = model.classes_[pred_idx]

    sv = np.array(explainer.shap_values(x))
    # SHAP versions differ in output layout; handle both
    if sv.shape[0] == len(model.classes_):
        vals = sv[pred_idx, 0, :]
    else:
        vals = sv[0, :, pred_idx]

    order = np.argsort(np.abs(vals))[::-1][:top_n]
    names = [symptoms[i] for i in order][::-1]
    contrib = [vals[i] for i in order][::-1]

    plt.figure(figsize=(8, 5))
    colors = ["tab:red" if c > 0 else "tab:blue" for c in contrib]
    plt.barh(names, contrib, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"SHAP: top features for predicted disease '{disease}'")
    plt.xlabel("Contribution to model output")
    plt.tight_layout()
    plt.savefig(save_as, dpi=150)
    plt.show()

    print("Predicted:", disease)
    for n, c in zip(names[::-1], contrib[::-1]):
        print(f"  {n}: {c:+.3f}")

if __name__ == "__main__":
    explain(["itching", "skin_rash", "nodal_skin_eruptions"])