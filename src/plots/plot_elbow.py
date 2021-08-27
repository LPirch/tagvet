import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='generate elbow plots')
    parser.add_argument('infile', help='JSON file with clustering results')
    parser.add_argument('fig_dir', help='directory to store the figures in (optional)', default=None)
    args = parser.parse_args()
    return vars(args)


def load_json(infile):
    return json.load(open(infile, 'r'))


def load_results(result_file):
    all_results = load_json(result_file)['metrics']
    dataframes = {}
    rename_dict = {
        'ARI': 'Adjusted Rand Index',
        'silhouette': 'Mean Silhouette Score'
    }
    for method, results in all_results.items():
        params = sorted(results[0][0].keys())
        metrics = [m for m in sorted(results[0][1].keys()) if m not in params]
        columns = params + ['metric', 'score']
        data = []
        for res in results:
            for m in metrics:
                m_name = m
                if m_name in rename_dict:
                    m_name = rename_dict[m_name]
                row = []
                for p in params:
                    row.append(res[0][p])
                row.append(m_name)
                row.append(res[1][m])
                data.append(row)
        dataframes[method] = pd.DataFrame(data, columns=columns)
    return dataframes


def plot_elbow(dataframes, method, primary_param, secondary_param, metrics, fig_dir=None):
    df = dataframes[method]
    sns.set_style('darkgrid')
    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 10))
    fig.suptitle('{} Clustering'.format(method), fontsize=16)

    for i, m in enumerate(metrics):
        legend_style = 'full' if i == 0 else False
        sns.lineplot(data=df, x=primary_param, y=m, hue=secondary_param,
                     ax=axs[i], palette='dark', legend=legend_style)
        axs[i].axvline(x=30, linestyle='--', color='g', linewidth=0.8)
        axs[i].axvline(x=77, linestyle='--', color='g', linewidth=0.8)
        axs[i].axvline(x=130, linestyle='--', color='g', linewidth=0.8)
        axs[i].axvline(x=335, linestyle='--', color='r', linewidth=0.8)

    if True or fig_dir is None:
        plt.show()
    else:
        filename = os.path.join(fig_dir, 'elbow-{}.png'.format(method))
        plt.savefig(filename, dpi=300)

    plt.clf()


def plot_elbow_camera_ready(dataframes, method, primary_param, secondary_param, metrics, fig_dir=None):
    df = dataframes[method]
    df = df.loc[df[secondary_param] == 'complete']
    df = df.loc[df['metric'].isin(['Adjusted Rand Index', 'Mean Silhouette Score'])]
    df = df.loc[df['n_clusters'] <= 100]
    df = df.rename(columns={
        'n_clusters': 'Number of Clusters',
        'score': 'Score',
        'metric': 'Metric'
    })
    palette = list(sns.color_palette('dark', 2))
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    # fig.tight_layout(pad=1.5, w_pad=5)
    sns.lineplot(data=df.loc[df['Metric'] == 'Adjusted Rand Index'], legend=False, color=palette[0],
                 x='Number of Clusters', y='Score', ax=ax, palette='dark', label='Adjusted Rand Index')
    ax.set_ylabel('Adjusted Rand Index')
    ax2 = ax.twinx()
    sns.lineplot(data=df.loc[df['Metric'] == 'Mean Silhouette Score'], legend=False, color=palette[1],
                 x='Number of Clusters', y='Score', ax=ax2, palette='dark', label='Mean Silhouette Score')
    ax2.set_ylabel('Mean Silhouette Score')
    ax2.grid(None)
    ax.axvline(x=30, linestyle='--', color='grey', linewidth=0.8)
    """
    ax.axvline(x=77, linestyle='--', color='g', linewidth=0.8)
    ax.axvline(x=130, linestyle='--', color='g', linewidth=0.8)
    ax.axvline(x=335, linestyle='--', color='r', linewidth=0.8)
    """
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.88, top=0.95)
    ax.figure.legend(loc='lower right', bbox_to_anchor=(0.88, 0.15))
    if fig_dir is None:
        plt.show()
    else:
        filename = os.path.join(fig_dir, 'elbow-{}.eps'.format(method))
        plt.savefig(filename, dpi=600)

    plt.clf()


def main(infile, fig_dir, **cfg):
    os.makedirs(fig_dir, exist_ok=True)
    plot_confs = {
        'DBSCAN': {
            'primary_param': 'eps',
            'secondary_param': 'min_samples',
            'metrics': ['ARI', 'DBCV', 'silhouette', 'n_clusters', 'n_outliers']
        },
        'KMeans': {
            'primary_param': 'n_clusters',
            'secondary_param': None,
            'metrics': ['ARI', 'DBCV', 'silhouette']
        },
        'Agglomerative': {
            'primary_param': 'n_clusters',
            'secondary_param': 'linkage',
            'metrics': ['ARI', 'silhouette']
        }
    }
    dataframes = load_results(infile)
    for method, conf in plot_confs.items():
        if method in dataframes:
            # plot_elbow(dataframes, method, fig_dir=fig_dir, **conf)
            plot_elbow_camera_ready(dataframes, method, fig_dir=fig_dir, **conf)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
