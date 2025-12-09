import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# print(os.path.dirname(os.path.dirname(__file__)))

from evaluate2 import rl_results, baseline_results, ranking_lists


# --- 1. Define Policies/Functions (For completeness) ---
def always_increase(obs, step): return 4
def always_decrease(obs, step): return 0
def always_hold(obs, step): return 2
def aimd_like(obs, step):
    loss_rate = obs[3]; queue_norm = obs[1]
    return 0 if loss_rate > 0.01 or queue_norm > 0.8 else 3
def queue_based(obs, step):
    queue_norm = obs[1]
    if queue_norm < 0.3: return 3
    elif queue_norm > 0.7: return 1
    else: return 2
def gradual_increase(obs, step):
    queue_norm = obs[1]; loss_rate = obs[3]
    if loss_rate > 0.05: return 0
    elif queue_norm > 0.9: return 1
    elif queue_norm > 0.6: return 2
    else: return 3

# --- 2. Define Model/Environment Data ---
environments = {
    'Easy': 'CongestionControl-Easy-v0',
    'Medium': 'CongestionControl-Medium-v0',
    'Hard': 'CongestionControl-Hard-v0',
}
baselines_list = ['Always Increase', 'Always Decrease', 'Always Hold', 'AIMD-like', 'Queue-based', 'Gradual Increase']

# --- 3. HARDCODED RESULTS (Performance Matrix Data) ---
# rl_results_hardcoded = {
#     'Easy Model': {'Easy': {'mean': 1184, 'std': 105}, 'Medium': {'mean': 1079, 'std': 127}, 'Hard': {'mean': 815, 'std': 154}},
#     'Medium Model (Curriculum)': {'Easy': {'mean': 1092, 'std': 123}, 'Medium': {'mean': 1095, 'std': 111}, 'Hard': {'mean': 751, 'std': 149}},
#     'Hard Model (Curriculum)': {'Easy': {'mean': 1042, 'std': 139}, 'Medium': {'mean': 1077, 'std': 115}, 'Hard': {'mean': 921, 'std': 133}},
# }
# rl_results_hardcoded = rl_results
rl_models = list(rl_results.keys())


# baseline_results_hardcoded = {
#     'Always Increase': {'Easy': {'mean': -13677, 'std': 1374}, 'Medium': {'mean': -7885, 'std': 854}, 'Hard': {'mean': -3618, 'std': 411}},
#     'Always Decrease': {'Easy': {'mean': 163, 'std': 21}, 'Medium': {'mean': 148, 'std': 25}, 'Hard': {'mean': 129, 'std': 21}},
#     'Always Hold': {'Easy': {'mean': 1007, 'std': 122}, 'Medium': {'mean': 878, 'std': 114}, 'Hard': {'mean': 721, 'std': 101}},
#     'AIMD-like': {'Easy': {'mean': 598, 'std': 85}, 'Medium': {'mean': 401, 'std': 89}, 'Hard': {'mean': 78, 'std': 67}},
#     'Queue-based': {'Easy': {'mean': -4288, 'std': 451}, 'Medium': {'mean': -3805, 'std': 378}, 'Hard': {'mean': -3470, 'std': 382}},
#     'Gradual Increase': {'Easy': {'mean': 989, 'std': 95}, 'Medium': {'mean': 745, 'std': 99}, 'Hard': {'mean': 567, 'std': 88}},
# }
# baseline_results = baseline_results

# --- 4. HARDCODED RANKING LISTS ---
# _ranking_lists = {
#     'Easy': [
#         ('Easy Model', 1173.5, '🏆'), ('Gradual Increase', 936.7, '🥈'), ('Medium Model (Curriculum)', 828.0, '🥉'),
#         ('Hard Model (Curriculum)', 822.1, ''), ('AIMD-like', 328.2, ''),
#     ],
#     'Medium': [
#         ('Easy Model', 1152.8, '🏆'), ('Gradual Increase', 914.9, '🥈'), ('Hard Model (Curriculum)', 743.9, '🥉'),
#         ('Medium Model (Curriculum)', 727.9, ''), ('AIMD-like', 222.6, ''),
#     ],
#     'Hard': [
#         ('Easy Model', 975.4, '🏆'), ('Gradual Increase', 950.5, '🥈'), ('Hard Model (Curriculum)', 809.9, '🥉'),
#         ('Medium Model (Curriculum)', 699.9, ''), ('AIMD-like', 210.3, ''),
#     ]
# }


# --- 5. Streamlit App Functions (Includes Chart Logic) ---

@st.cache_data
def create_performance_matrix(results, policies):
    data = []
    for policy_name in policies:
        row = {'Policy': policy_name}
        for env_name in environments.keys():
            result = results[policy_name][env_name]
            row[env_name] = f"{result['mean']:.0f}±{result['std']:.0f}"
        data.append(row)
    return pd.DataFrame(data)

@st.cache_data
def get_ranking_data(rl_results, baseline_results, env_name):
    all_results = {}
    for model_name in rl_models:
        all_results[model_name] = {'score': rl_results[model_name][env_name]['mean'], 'type': 'RL Model'}
    for baseline_name in baselines_list:
        all_results[baseline_name] = {'score': baseline_results[baseline_name][env_name]['mean'], 'type': 'Baseline'}
        
    df = pd.DataFrame.from_dict(all_results, orient='index').reset_index().rename(columns={'index': 'Policy'})
    df = df.sort_values(by='score', ascending=False)
    return df
    
def create_ranking_chart(df, env_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = df['type'].map({'RL Model': 'darkgreen', 'Baseline': 'steelblue'})
    ax.barh(df['Policy'], df['score'], color=colors)
    ax.set_title(f'Top Policy Rankings on {env_name} Environment', fontsize=16, fontweight='bold')
    ax.set_xlabel('Average Reward')
    ax.invert_yaxis()
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='darkgreen', lw=4), Line2D([0], [0], color='steelblue', lw=4)]
    ax.legend(custom_lines, ['RL Model', 'Baseline'], loc='lower right')
    plt.tight_layout()
    return fig

def create_comprehensive_charts(rl_results, baseline_results):
    """Generates the 4-panel figure from evaluate2.py."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    env_names = list(environments.keys())
    
    # --- Plot 1: RL Models Heatmap ---
    ax = axes[0, 0]
    rl_matrix = np.array([[rl_results[model][env]['mean'] for env in env_names] for model in rl_models])
    im = ax.imshow(rl_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(env_names)))
    ax.set_yticks(range(len(rl_models)))
    ax.set_xticklabels(env_names)
    ax.set_yticklabels(rl_models)
    ax.set_title('RL Models Performance Heatmap', fontsize=14, fontweight='bold')
    for i in range(len(rl_models)):
        for j in range(len(env_names)):
            ax.text(j, i, f'{rl_matrix[i, j]:.0f}', ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax)
    
    # --- Plot 2: Performance by Environment (Bar chart) ---
    ax = axes[0, 1]
    x = np.arange(len(env_names))
    width = 0.25
    for i, model_name in enumerate(rl_models):
        means = [rl_results[model_name][env]['mean'] for env in env_names]
        ax.bar(x + i*width, means, width, label=model_name, alpha=0.8)
    ax.set_xlabel('Environment')
    ax.set_ylabel('Average Reward')
    ax.set_title('RL Models: Performance by Environment', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width*1)
    ax.set_xticklabels(env_names)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 3: Best RL vs Best Baseline ---
    ax = axes[1, 0]
    best_rl = [max([rl_results[m][env]['mean'] for m in rl_models]) for env in env_names]
    best_baseline = [max([baseline_results[b][env]['mean'] for b in baselines_list]) for env in env_names]
    
    x = np.arange(len(env_names))
    width = 0.35
    ax.bar(x - width/2, best_rl, width, label='Best RL Model', alpha=0.8, color='green')
    ax.bar(x + width/2, best_baseline, width, label='Best Baseline', alpha=0.8, color='orange')
    ax.set_xlabel('Environment')
    ax.set_ylabel('Average Reward')
    ax.set_title('Best RL vs Best Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 4: Variance comparison ---
    ax = axes[1, 1]
    model_names_short = [m.split()[0] for m in rl_models]
    variances_by_env = {env: [rl_results[m][env]['std'] for m in rl_models] for env in env_names}
    
    x = np.arange(len(model_names_short))
    for i, (env_name, variances) in enumerate(variances_by_env.items()):
        ax.bar(x + i*width - width, variances, width, label=f'{env_name} Env', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Performance Variance (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_short)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


# APP LAYOUT

st.set_page_config(layout="wide", page_title="CC RL Model Evaluation")

st.title("Network Congestion Control RL Policy Evaluation")
st.markdown("A comprehensive comparison of trained Reinforcement Learning (RL) models against various baseline congestion control policies across different network environments (Easy, Medium, Hard).")
# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
selected_env = st.sidebar.selectbox(
    "Select Environment for Chart Ranking",
    options=list(environments.keys()),
    index=2 # Default to Hard
)

# --- Performance Matrices ---
st.header("Performance Matrices (Average Reward +- Standard Deviation)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("RL Models Performance")
    rl_df = create_performance_matrix(rl_results, rl_models)
    st.dataframe(rl_df, hide_index=True)

with col2:
    st.subheader("Baseline Policies Performance")
    baseline_df = create_performance_matrix(baseline_results, baselines_list)
    st.dataframe(baseline_df, hide_index=True)

st.divider()

# --- Static Ranking Lists (Tables) ---
st.header("Policy Ranking Summary")
st.markdown("Top performing policies (RL and Baselines) across all three environments, ranked 1-5 by average reward.")

rank_cols = st.columns(3)

for i, (env_name, rankings) in enumerate(ranking_lists.items()):
    # Prepare data for DataFrame
    data = []
    for rank, (policy, score, icon) in enumerate(rankings):
        data.append({
            'Rank': f"{icon} {rank + 1}",
            'Policy': policy,
            'Reward': f"{score:.1f}"
        })
    
    ranking_df = pd.DataFrame(data)
    
    with rank_cols[i]:
        st.subheader(f"🥇 {env_name} Environment")
        st.dataframe(
            ranking_df, 
            hide_index=True,
            column_config={
                "Rank": st.column_config.TextColumn("Rank", width="small")
            }
        )

st.divider()

# --- Comprehensive Charts ---
st.header("📈 Curriculum Progression and Generalization Analysis")
st.markdown("Visualizing the cross-environment performance and stability (variance) of the RL models.")

comp_charts = create_comprehensive_charts(rl_results, baseline_results)
st.pyplot(comp_charts)

st.divider()

# --- Interactive Ranking Chart ---
st.header(f"📊 Interactive Policy Ranking on the **{selected_env}** Environment")
st.markdown("Full comparison of all policies (RL and Baselines) by their average reward on the selected environment.")

ranking_df = get_ranking_data(rl_results, baseline_results, selected_env)
ranking_chart = create_ranking_chart(ranking_df, selected_env)
st.pyplot(ranking_chart)