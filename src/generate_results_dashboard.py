from pathlib import Path
import json
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import timm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results_dashboard"
HTML_PATH = OUTPUT_DIR / "model-results-dashboard.html"

BASE_DATA_DIR = PROJECT_ROOT / "data" / "cars_body_type"
OLD_DATA_DIR = PROJECT_ROOT / "collected_crops" / "split"
REFINED_DATA_DIR = PROJECT_ROOT / "collected_crops" / "refined_split"

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 0

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_dataset(dataset_dir: Path):
    ds = ImageFolder(dataset_dir, transform=get_eval_transform(), allow_empty=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return ds, loader


def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint["model_name"]

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, checkpoint


def infer(model, loader):
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


def compute_per_class_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    rows = []

    for i, class_name in enumerate(class_names):
        tn, fp, fn, tp = mcm[i].ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = report[class_name]["f1-score"]
        support = int(report[class_name]["support"])

        rows.append({
            "class_name": class_name,
            "support": support,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "precision": precision,
            "precision_percent": precision * 100.0,
            "recall": recall,
            "recall_percent": recall * 100.0,
            "specificity": specificity,
            "specificity_percent": specificity * 100.0,
            "f1_score": f1,
            "f1_score_percent": f1 * 100.0,
        })

    return pd.DataFrame(rows), report


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_confusion_matrix_images(cm, class_names, title_prefix):
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title(f"{title_prefix} - Confusion Matrix")
    ax1.set_ylabel("True label")
    ax1.set_xlabel("Predicted label")
    img_abs = fig_to_base64(fig1)

    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    out = np.zeros_like(cm_norm, dtype=np.float64)
    np.divide(cm_norm, row_sums, out=out, where=row_sums != 0)
    cm_norm_df = pd.DataFrame(out, index=class_names, columns=class_names)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm_df, annot=True, fmt=".2f", cmap="Blues", ax=ax2)
    ax2.set_title(f"{title_prefix} - Normalized Confusion Matrix")
    ax2.set_ylabel("True label")
    ax2.set_xlabel("Predicted label")
    img_norm = fig_to_base64(fig2)

    return img_abs, img_norm


def evaluate_case(case_name: str, checkpoint_path: Path, dataset_dir: Path, group_key: str, display_name: str):
    if not checkpoint_path.exists() or not dataset_dir.exists():
        return None

    model, class_names, checkpoint = load_model(checkpoint_path)
    ds, loader = load_dataset(dataset_dir)

    if ds.classes != class_names:
        return {
            "case_name": case_name,
            "display_name": display_name,
            "group_key": group_key,
            "checkpoint_name": checkpoint_path.name,
            "dataset_label": group_key,
            "status": "class_mismatch",
            "dataset_classes": ds.classes,
            "checkpoint_classes": class_names,
        }

    y_true, y_pred = infer(model, loader)

    labels = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_df, report = compute_per_class_metrics(y_true, y_pred, class_names)
    cm_abs, cm_norm = make_confusion_matrix_images(cm, class_names, display_name)

    case_dir = RESULTS_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    per_class_df.to_csv(case_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"metric": "accuracy", "value": acc, "value_percent": acc * 100.0},
        {"metric": "macro_f1", "value": macro_f1, "value_percent": macro_f1 * 100.0},
        {"metric": "weighted_f1", "value": weighted_f1, "value_percent": weighted_f1 * 100.0},
    ]).to_csv(case_dir / "summary_table.csv", index=False, encoding="utf-8-sig")

    with open(case_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {
        "case_name": case_name,
        "display_name": display_name,
        "group_key": group_key,
        "checkpoint_name": checkpoint_path.name,
        "dataset_label": group_key,
        "num_samples": len(ds),
        "accuracy": acc,
        "accuracy_percent": acc * 100.0,
        "macro_f1": macro_f1,
        "macro_f1_percent": macro_f1 * 100.0,
        "weighted_f1": weighted_f1,
        "weighted_f1_percent": weighted_f1 * 100.0,
        "best_valid_f1_from_checkpoint": checkpoint.get("best_valid_f1", None),
        "per_class_metrics": per_class_df.round(4).to_dict(orient="records"),
        "confusion_matrix_abs_b64": cm_abs,
        "confusion_matrix_norm_b64": cm_norm,
        "status": "ok",
    }


def build_html(results):
    payload = json.dumps(results, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model results dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg:#f7f6f2;
      --surface:#fbfbf9;
      --surface-2:#f3f0ec;
      --text:#28251d;
      --muted:#72706b;
      --border:#d4d1ca;
      --primary:#01696f;
      --primary-2:#0c4e54;
      --shadow:0 10px 30px rgba(0,0,0,.08);
      --radius:18px;
    }}

    [data-theme="dark"] {{
      --bg:#171614;
      --surface:#1c1b19;
      --surface-2:#22211f;
      --text:#e6e1d8;
      --muted:#a7a29a;
      --border:#393836;
      --primary:#4f98a3;
      --primary-2:#227f8b;
      --shadow:0 10px 30px rgba(0,0,0,.35);
    }}

    * {{ box-sizing:border-box; }}

    body {{
      margin:0;
      font-family:Inter,system-ui,sans-serif;
      background:var(--bg);
      color:var(--text);
    }}

    .app {{
      max-width:1400px;
      margin:0 auto;
      padding:24px;
    }}

    .topbar {{
      display:flex;
      justify-content:space-between;
      gap:16px;
      align-items:center;
      margin-bottom:24px;
    }}

    .brand {{
      display:flex;
      align-items:center;
      gap:12px;
    }}

    .logo {{
      width:42px;
      height:42px;
      border-radius:12px;
      background:linear-gradient(135deg,var(--primary),var(--primary-2));
      display:grid;
      place-items:center;
      color:white;
      font-weight:800;
      box-shadow:var(--shadow);
    }}

    .title h1 {{
      margin:0;
      font-size:clamp(1.6rem,2vw,2.3rem);
    }}

    .title p {{
      margin:4px 0 0;
      color:var(--muted);
    }}

    .toolbar {{
      display:flex;
      gap:12px;
      align-items:center;
      flex-wrap:wrap;
    }}

    button {{
      border:0;
      cursor:pointer;
    }}

    .ghost, .tab {{
      background:var(--surface);
      color:var(--text);
      border:1px solid var(--border);
      border-radius:999px;
      padding:10px 14px;
      font-weight:600;
    }}

    .tabs {{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      margin-bottom:20px;
    }}

    .tab.active {{
      background:var(--primary);
      color:#fff;
      border-color:var(--primary);
    }}

    .panel {{
      display:none;
    }}

    .panel.active {{
      display:block;
    }}

    .grid {{
      display:grid;
      grid-template-columns:repeat(12,1fr);
      gap:16px;
    }}

    .card {{
      background:var(--surface);
      border:1px solid var(--border);
      border-radius:var(--radius);
      padding:18px;
      box-shadow:var(--shadow);
    }}

    .kpi {{
      grid-column:span 3;
    }}

    .kpi .label {{
      color:var(--muted);
      font-size:.9rem;
    }}

    .kpi .value {{
      margin-top:8px;
      font-size:clamp(1.6rem,3vw,2.4rem);
      font-weight:800;
    }}

    .kpi .sub {{
      margin-top:6px;
      color:var(--muted);
      font-size:.92rem;
    }}

    .wide {{
      grid-column:span 12;
    }}

    .half {{
      grid-column:span 6;
    }}

    .section-head {{
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:12px;
      margin-bottom:14px;
    }}

    .section-head h2, .section-head h3 {{
      margin:0;
    }}

    .section-head p {{
      margin:4px 0 0;
      color:var(--muted);
    }}

    .metric-table, .class-table {{
      width:100%;
      border-collapse:collapse;
      font-size:.95rem;
    }}

    .metric-table th, .metric-table td, .class-table th, .class-table td {{
      padding:10px 12px;
      border-bottom:1px solid var(--border);
      text-align:left;
      vertical-align:top;
    }}

    .metric-table th, .class-table th {{
      color:var(--muted);
      font-weight:700;
    }}

    .badge {{
      display:inline-flex;
      align-items:center;
      gap:6px;
      padding:6px 10px;
      border-radius:999px;
      background:rgba(1,105,111,.12);
      color:var(--primary);
      font-weight:700;
      font-size:.82rem;
    }}

    .img-wrap img {{
      width:100%;
      height:auto;
      border-radius:14px;
      border:1px solid var(--border);
      background:#fff;
    }}

    .small {{
      color:var(--muted);
      font-size:.9rem;
    }}

    .error {{
      border-color:#b24c72;
      background:rgba(178,76,114,.08);
    }}

    .footer-note {{
      margin-top:18px;
      color:var(--muted);
    }}

    @media (max-width:980px) {{
      .kpi, .half {{
        grid-column:span 12;
      }}

      .app {{
        padding:16px;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <div class="logo" aria-hidden="true">MR</div>
        <div class="title">
          <h1>Dashboard wyników modeli</h1>
          <p>Trzy zakładki: test źródłowy, walidacja adaptacji i końcowy test docelowy.</p>
        </div>
      </div>
      <div class="toolbar">
        <button class="ghost" data-theme-toggle aria-label="Przełącz motyw">🌙</button>
      </div>
    </header>

    <nav class="tabs" aria-label="Widoki wyników">
      <button class="tab active" data-tab="source_test">1. Test źródłowy</button>
      <button class="tab" data-tab="adapt_valid">2. Adaptacja stage 2</button>
      <button class="tab" data-tab="target_test">3. Końcowy test docelowy</button>
    </nav>

    <main id="panels"></main>
  </div>

<script>
const results = {payload};

const tabInfo = {{
  source_test: {{
    title: "Test źródłowy",
    desc: "Porównanie modeli na oryginalnym zbiorze testowym."
  }},
  adapt_valid: {{
    title: "Adaptacja stage 2",
    desc: "Ocena na zbiorze walidacyjnym używanym przy adaptacji do domeny docelowej."
  }},
  target_test: {{
    title: "Końcowy test docelowy",
    desc: "Najważniejsza ocena końcowa na niezależnym zbiorze docelowym."
  }}
}};

function pct(v) {{
  return typeof v === "number" ? `${{v.toFixed(2)}}%` : "—";
}}

function num(v) {{
  return typeof v === "number" ? v.toFixed(4) : "—";
}}

function makeMetricRows(okItems) {{
  return okItems.map(item => `
    <tr>
      <td><strong>${{item.display_name}}</strong><div class="small">${{item.checkpoint_name}}</div></td>
      <td>${{item.num_samples ?? "—"}}</td>
      <td>${{pct(item.accuracy_percent)}}</td>
      <td>${{pct(item.macro_f1_percent)}}</td>
      <td>${{pct(item.weighted_f1_percent)}}</td>
      <td>${{item.best_valid_f1_from_checkpoint !== null && item.best_valid_f1_from_checkpoint !== undefined ? num(item.best_valid_f1_from_checkpoint) : "—"}}</td>
    </tr>
  `).join("");
}}

function makeClassTable(item) {{
  if (!item.per_class_metrics) return '<p class="small">Brak tabeli klas.</p>';

  const rows = item.per_class_metrics.map(r => `
    <tr>
      <td><strong>${{r.class_name}}</strong></td>
      <td>${{r.support}}</td>
      <td>${{pct(r.precision_percent)}}</td>
      <td>${{pct(r.recall_percent)}}</td>
      <td>${{pct(r.specificity_percent)}}</td>
      <td>${{pct(r.f1_score_percent)}}</td>
      <td>${{r.TP}}</td>
      <td>${{r.FP}}</td>
      <td>${{r.FN}}</td>
    </tr>
  `).join("");

  return `
    <div class="card wide">
      <div class="section-head">
        <div>
          <h3>Metryki per klasa — ${{item.display_name}}</h3>
          <p>Precision, recall, specificity i F1 dla każdej klasy.</p>
        </div>
        <span class="badge">Per class</span>
      </div>
      <div style="overflow:auto;">
        <table class="class-table">
          <thead>
            <tr>
              <th>Klasa</th>
              <th>Support</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>Specificity</th>
              <th>F1</th>
              <th>TP</th>
              <th>FP</th>
              <th>FN</th>
            </tr>
          </thead>
          <tbody>${{rows}}</tbody>
        </table>
      </div>
    </div>
  `;
}}

function panelHtml(groupKey, items) {{
  const okItems = items.filter(x => x.status === "ok");
  const badItems = items.filter(x => x.status !== "ok");
  const best = [...okItems].sort((a, b) => (b.macro_f1 ?? -1) - (a.macro_f1 ?? -1))[0];

  const panelCards = okItems.map(item => `
    <section class="grid" style="margin-top:16px;">
      <div class="card kpi">
        <div class="label">Accuracy</div>
        <div class="value">${{pct(item.accuracy_percent)}}</div>
        <div class="sub">${{item.display_name}}</div>
      </div>

      <div class="card kpi">
        <div class="label">Macro F1</div>
        <div class="value">${{pct(item.macro_f1_percent)}}</div>
        <div class="sub">Kluczowa metryka wieloklasowa</div>
      </div>

      <div class="card kpi">
        <div class="label">Weighted F1</div>
        <div class="value">${{pct(item.weighted_f1_percent)}}</div>
        <div class="sub">Ważone wsparciem klas</div>
      </div>

      <div class="card kpi">
        <div class="label">Liczba próbek</div>
        <div class="value">${{item.num_samples ?? "—"}}</div>
      </div>

      <div class="card half img-wrap">
        <div class="section-head">
          <div>
            <h3>Confusion matrix</h3>
            <p>${{item.display_name}}</p>
          </div>
          <span class="badge">Absolute</span>
        </div>
        <img alt="Confusion matrix" src="data:image/png;base64,${{item.confusion_matrix_abs_b64}}">
      </div>

      <div class="card half img-wrap">
        <div class="section-head">
          <div>
            <h3>Normalized matrix</h3>
            <p>${{item.display_name}}</p>
          </div>
          <span class="badge">Normalized</span>
        </div>
        <img alt="Normalized confusion matrix" src="data:image/png;base64,${{item.confusion_matrix_norm_b64}}">
      </div>

      ${{makeClassTable(item)}}
    </section>
  `).join("");

  const badHtml = badItems.map(item => `
    <div class="card error wide">
      <div class="section-head">
        <div>
          <h3>${{item.display_name}}</h3>
          <p>Nie udało się policzyć wyników dla tego przypadku.</p>
        </div>
      </div>
      <p class="small">Status: ${{item.status}}</p>
      <p class="small">Checkpoint: ${{item.checkpoint_name}}</p>
      <p class="small">Zbiór: ${{item.dataset_label}}</p>
    </div>
  `).join("");

  return `
    <section class="panel ${{groupKey === "source_test" ? "active" : ""}}" data-panel="${{groupKey}}">
      <div class="card wide">
        <div class="section-head">
          <div>
            <h2>${{tabInfo[groupKey].title}}</h2>
            <p>${{tabInfo[groupKey].desc}}</p>
          </div>
          ${{best ? `<span class="badge">Najlepszy Macro F1: ${{best.display_name}} (${{pct(best.macro_f1_percent)}})</span>` : ""}}
        </div>

        <div style="overflow:auto;">
          <table class="metric-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Próbki</th>
                <th>Accuracy</th>
                <th>Macro F1</th>
                <th>Weighted F1</th>
                <th>Best valid F1</th>
              </tr>
            </thead>
            <tbody>${{makeMetricRows(okItems)}}</tbody>
          </table>
        </div>
      </div>

      ${{badHtml}}
      ${{panelCards}}
    </section>
  `;
}}

const groups = ["source_test", "adapt_valid", "target_test"];
const panels = document.getElementById("panels");
panels.innerHTML = groups.map(g => panelHtml(g, results.filter(r => r.group_key === g))).join("");

document.querySelectorAll(".tab").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.querySelector(`[data-panel="${{btn.dataset.tab}}"]`)?.classList.add("active");
  }});
}});

(function() {{
  const t = document.querySelector("[data-theme-toggle]");
  const r = document.documentElement;
  let d = matchMedia("(prefers-color-scheme:dark)").matches ? "dark" : "light";
  r.setAttribute("data-theme", d);
  t.textContent = d === "dark" ? "☀️" : "🌙";

  t.addEventListener("click", () => {{
    d = d === "dark" ? "light" : "dark";
    r.setAttribute("data-theme", d);
    t.textContent = d === "dark" ? "☀️" : "🌙";
  }});
}})();
</script>
</body>
</html>
"""

    HTML_PATH.write_text(html, encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "case_name": "stage2_on_original_test",
            "display_name": "Stage 2",
            "checkpoint_path": CHECKPOINT_DIR / "best_stage2.pth",
            "dataset_dir": BASE_DATA_DIR / "test",
            "group_key": "source_test",
        },
        {
            "case_name": "stage2_adapted_on_original_test",
            "display_name": "Stage 2 adapted",
            "checkpoint_path": CHECKPOINT_DIR / "best_stage2_adapted.pth",
            "dataset_dir": BASE_DATA_DIR / "test",
            "group_key": "source_test",
        },
        {
            "case_name": "stage3_on_original_test",
            "display_name": "Stage 3",
            "checkpoint_path": CHECKPOINT_DIR / "best_stage3.pth",
            "dataset_dir": BASE_DATA_DIR / "test",
            "group_key": "source_test",
        },
        {
            "case_name": "stage2_on_stage2_adaptation_valid",
            "display_name": "Stage 2",
            "checkpoint_path": CHECKPOINT_DIR / "best_stage2.pth",
            "dataset_dir": OLD_DATA_DIR / "valid",
            "group_key": "adapt_valid",
        },
        {
            "case_name": "stage2_adapted_on_stage2_adaptation_valid",
            "display_name": "Stage 2 adapted",
            "checkpoint_path": CHECKPOINT_DIR / "best_stage2_adapted.pth",
            "dataset_dir": OLD_DATA_DIR / "valid",
            "group_key": "adapt_valid",
        },
        {
            "case_name": "stage3_on_stage2_adaptation_valid",
            "display_name": "Stage 3",
            "checkpoint_path": CHECKPOINT_DIR / "best_stage3.pth",
            "dataset_dir": OLD_DATA_DIR / "valid",
            "group_key": "adapt_valid",
        },
        {
            "case_name": "stage2_on_refined_test",
            "display_name": "Stage 2",
            "checkpoint_path": CHECKPOINT_DIR / "best_stage2.pth",
            "dataset_dir": REFINED_DATA_DIR / "test",
            "group_key": "target_test",
        },
        {
            "case_name": "stage2_adapted_on_refined_test",
            "display_name": "Stage 2 adapted",
            "checkpoint_path": CHECKPOINT_DIR / "best_stage2_adapted.pth",
            "dataset_dir": REFINED_DATA_DIR / "test",
            "group_key": "target_test",
        },
        {
            "case_name": "stage3_on_refined_test",
            "display_name": "Stage 3",
            "checkpoint_path": CHECKPOINT_DIR / "best_stage3.pth",
            "dataset_dir": REFINED_DATA_DIR / "test",
            "group_key": "target_test",
        },
    ]

    results = []

    for case in cases:
        result = evaluate_case(
            case_name=case["case_name"],
            checkpoint_path=case["checkpoint_path"],
            dataset_dir=case["dataset_dir"],
            group_key=case["group_key"],
            display_name=case["display_name"],
        )
        if result is not None:
            results.append(result)

    pd.DataFrame([
        {k: v for k, v in r.items() if k not in [
            "per_class_metrics",
            "confusion_matrix_abs_b64",
            "confusion_matrix_norm_b64"
        ]}
        for r in results
    ]).to_csv(RESULTS_DIR / "all_results_summary.csv", index=False, encoding="utf-8-sig")

    with open(RESULTS_DIR / "dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    build_html(results)

    print(f"Dashboard saved to: {HTML_PATH}")
    print(f"Data saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()