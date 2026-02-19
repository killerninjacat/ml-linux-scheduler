import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# GLOBAL STYLE (Research Paper Style)
# =====================================================

plt.style.use("ggplot")

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

DATA_DIR = Path("data/raw")

# =====================================================
# Automatically locate newest files
# =====================================================

def latest_file(prefix):
    files = sorted(DATA_DIR.glob(f"{prefix}_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No {prefix} files found")
    return files[-1]


state_file = latest_file("state")
rapl_file = latest_file("rapl")

print(f"Using state file: {state_file}")
print(f"Using rapl file : {rapl_file}")

df = pd.read_json(state_file, lines=True)
rapl = pd.read_json(rapl_file, lines=True)

# =====================================================
# GRAPH 1 — Label Distribution
# =====================================================

plt.figure()

counts = df["decision"].value_counts().sort_index()

bars = plt.bar(
    ["No Migration", "Migration"],
    counts,
    color=["#4C72B0", "#DD8452"]
)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height,
             f"{int(height)}",
             ha='center',
             va='bottom')

plt.title("Dataset Label Distribution")
plt.ylabel("Number of Samples")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# GRAPH 2 — CPU Load Distribution
# =====================================================

plt.figure()

plt.hist(
    df["src_load"],
    bins=40,
    color="#55A868",
    edgecolor="black",
    alpha=0.85
)

plt.title("Source CPU Load Distribution")
plt.xlabel("CPU Load (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# =====================================================
# GRAPH 3 — Migration Timeline
# =====================================================

plt.figure()

df["time"] = (df["timestamp"] - df["timestamp"].min()) / 1e9
migrations = df[df["decision"] == 1]

plt.scatter(
    migrations["time"],
    [1]*len(migrations),
    s=10,
    alpha=0.6,
    color="#C44E52"
)

plt.title("Migration Events Over Time")
plt.xlabel("Time (seconds)")
plt.yticks([])
plt.tight_layout()
plt.show()

# =====================================================
# GRAPH 4 — Energy Consumption Trend
# =====================================================

plt.figure()

rapl["time"] = (rapl["timestamp"] - rapl["timestamp"].min()) / 1e9

plt.plot(
    rapl["time"],
    rapl["total_uj"] / 1e6,
    linewidth=2.5,
    color="#8172B3"
)

plt.title("Energy Consumption Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Energy (Joules)")
plt.tight_layout()
plt.show()

# =====================================================
# GRAPH 5 — NUMA Migration Ratio
# =====================================================

plt.figure()

labels = ["Same NUMA Node", "Cross NUMA Node"]
values = df["cross_node"].value_counts().sort_index()

plt.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#64B5CD", "#F28E2B"],
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)

plt.title("NUMA Migration Distribution")
plt.tight_layout()
plt.show()
