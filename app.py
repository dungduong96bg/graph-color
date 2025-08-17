import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import random
import pandas as pd
import time

# =========================
# Cấu hình app
# =========================
st.set_page_config(page_title="Server/BTS Maintenance Scheduling", layout="wide")
st.title("Mô phỏng lập lịch bảo trì Server/BTS và so sánh thời gian thuật toán")

# =========================
# Load icon
# =========================
bts_img = Image.open("bts.png")
server_img = Image.open("server.png")

# =========================
# Sidebar config
# =========================
num_nodes = st.sidebar.number_input("Số server/BTS", min_value=20, max_value=50, value=30)
prob_edge = st.sidebar.number_input("Xác suất xung đột (0–1)", min_value=0.0, max_value=1.0, value=0.3)
seed = st.sidebar.number_input("Random seed", value=42)
st.sidebar.markdown("**Chạy:**")
run_button = st.sidebar.button("Tìm lịch tối ưu và so sánh thời gian")


# =========================
# Sinh đồ thị ngẫu nhiên
# =========================
@st.cache_data
def generate_graph(n, p, seed=None):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    nodes_type = {i: random.choice(["BTS", "Server"]) for i in range(n)}
    nx.set_node_attributes(G, nodes_type, "type")
    return G


# =========================
# Thuật toán tô màu
# =========================
def greedy_coloring(G):
    color = {}
    for v in G.nodes():
        used = {color[u] for u in G.neighbors(v) if u in color}
        c = 1
        while c in used:
            c += 1
        color[v] = c
    return color


def dsatur_coloring(G):
    color = {}
    uncolored = set(G.nodes())
    degrees = dict(G.degree())
    while uncolored:
        max_sat = max([len({color[u] for u in G.neighbors(v) if u in color}) for v in uncolored])
        candidates = [v for v in uncolored if len({color[u] for u in G.neighbors(v) if u in color}) == max_sat]
        v = max(candidates, key=lambda x: degrees[x]) if len(candidates) > 1 else candidates[0]
        used = {color[u] for u in G.neighbors(v) if u in color}
        c = 1
        while c in used:
            c += 1
        color[v] = c
        uncolored.remove(v)
    return color


def backtracking_coloring(G):
    n = len(G.nodes())
    nodes = list(G.nodes())
    color = {}

    def assign(v_index, k):
        if v_index >= n:
            return True
        v = nodes[v_index]
        for c in range(1, k + 1):
            if all(color.get(u, 0) != c for u in G.neighbors(v)):
                color[v] = c
                if assign(v_index + 1, k):
                    return True
                color[v] = 0
        return False

    for k in range(1, n + 1):
        color.clear()
        if assign(0, k):
            break
    return color


# =========================
# Hàm đo thời gian (ms)
# =========================
def measure_time(func, G):
    start = time.time()
    result = func(G)
    end = time.time()
    return result, (end - start) * 1000  # milliseconds


# =========================
# Chạy app khi nhấn nút
# =========================
if run_button:
    G = generate_graph(num_nodes, prob_edge, seed)

    algos = {
        "Greedy": greedy_coloring,
        "DSATUR": dsatur_coloring,
        "Backtracking": backtracking_coloring
    }

    times = {}
    results = {}

    for name, func in algos.items():
        result, t = measure_time(func, G)
        results[name] = result
        times[name] = t

    # =========================
    # Vẽ đồ thị network
    # =========================
    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    scale = 0.15 / (num_nodes ** 0.5)  # scale icon theo số node

    for node in G.nodes():
        x, y = pos[node]
        img = bts_img if G.nodes[node]["type"] == "BTS" else server_img
        ax.imshow(img, extent=(x - scale, x + scale, y - scale, y + scale), zorder=2)

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=max(1, 3 - num_nodes / 10), zorder=1)

    ax.axis('off')
    st.subheader("Đồ thị server/BTS")
    st.pyplot(plt)

    # =========================
    # Bảng lịch bảo trì Greedy
    # =========================
    st.subheader("Kết quả lịch bảo trì (Greedy)")
    df = pd.DataFrame({
        "Server/BTS": [f"{node} ({G.nodes[node]['type']})" for node in G.nodes()],
        "Slot bảo trì": [results["Greedy"][node] for node in G.nodes()]
    }).sort_values("Slot bảo trì")
    st.dataframe(df)

    # =========================
    # So sánh thời gian chạy (line plot)
    # =========================
    df_time = pd.DataFrame({
        "Thuật toán": list(times.keys()),
        "Thời gian (ms)": list(times.values())
    }).sort_values("Thời gian (ms)")

    st.subheader("So sánh thời gian chạy các thuật toán (ms)")
    st.dataframe(df_time)

    colors = ["skyblue", "orange", "green"]
    plt.figure(figsize=(6, 4))
    for i, algo in enumerate(df_time["Thuật toán"]):
        plt.plot(algo, df_time["Thời gian (ms)"][i], 'o', color=colors[i], markersize=10, label=algo)
    plt.plot(df_time["Thuật toán"], df_time["Thời gian (ms)"], color='gray', linestyle='--', linewidth=1)
    plt.ylabel("Thời gian (ms)")
    plt.title("So sánh thời gian chạy thuật toán tô màu")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
