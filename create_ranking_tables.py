# -*- coding: utf-8 -*- python.exe create_ranking_tables.py --data-root rankings --results-dir "C:\Users\USER\Desktop\ocr3" --python .venv\Scripts\python.exe
import os
import yaml
import pandas as pd
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_get_metrics(submission_path, data_root, python_exe):
    try:
        cmd = [python_exe, 'get_metrics.py', '--submission', submission_path, '--data-root', data_root]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            print(f"Ошибка: {submission_path}")
            if result.stderr:
                print(f"  Детали: {result.stderr.strip()}")
            return {}
        
        metrics = {}
        lines = result.stdout.split('\n')
        in_results = False
        
        for line in lines:
            line = line.strip()
            if 'DATASET' in line and 'CER' in line and 'WER' in line:
                in_results = True
                continue
            if in_results and line.startswith('-' * 10):
                if metrics:
                    break
                continue
            if in_results and line and not line.startswith('='):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        dataset_name = parts[0]
                        cer = float(parts[1])
                        wer = float(parts[2])
                        accuracy = float(parts[3])
                        metrics[dataset_name] = {'cer': cer, 'wer': wer, 'accuracy': accuracy}
                    except (ValueError, IndexError):
                        continue
        return metrics
    except Exception as e:
        print(f"Ошибка: {e}")
        return {}


def extract_dataset_and_model(filename, datasets):
    name = filename.replace('.csv', '')
    matched_dataset = None
    matched_length = 0
    
    for dataset_name in datasets.keys():
        if name.startswith(dataset_name + '_'):
            if len(dataset_name) > matched_length:
                matched_dataset = dataset_name
                matched_length = len(dataset_name)
    
    if matched_dataset:
        model_name = name[len(matched_dataset) + 1:]
        return matched_dataset, model_name
    return None, None


def collect_all_results(results_dir, data_root, datasets, python_exe):
    all_results = {dataset: {} for dataset in datasets.keys()}
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    total_files = len(result_files)
    
    print(f"Найдено {total_files} файлов с результатами")
    
    for idx, filename in enumerate(result_files, 1):
        filepath = os.path.join(results_dir, filename)
        dataset_name, model_name = extract_dataset_and_model(filename, datasets)
        
        if not dataset_name or not model_name:
            continue
        
        print(f"[{idx}/{total_files}] {dataset_name} / {model_name}")
        metrics = run_get_metrics(filepath, data_root, python_exe)
        
        if dataset_name in metrics:
            all_results[dataset_name][model_name] = metrics[dataset_name]
            m = metrics[dataset_name]
            print(f"  CER={m['cer']:.4f}, WER={m['wer']:.4f}, ACC={m['accuracy']:.4f}")
    
    return all_results


def calculate_rankings(df):
    if 'CER' in df.columns:
        df['Rank_CER'] = df['CER'].rank(method='min', ascending=True)
    if 'WER' in df.columns:
        df['Rank_WER'] = df['WER'].rank(method='min', ascending=True)
    if 'Accuracy' in df.columns:
        df['Rank_Accuracy'] = df['Accuracy'].rank(method='min', ascending=False)
    
    rank_columns = [col for col in df.columns if col.startswith('Rank_')]
    if rank_columns:
        df['Average_Rank'] = df[rank_columns].mean(axis=1)
        df['Final_Rank'] = df['Average_Rank'].rank(method='min', ascending=True)
    
    return df


def create_ranking_table(dataset_name, results, output_dir):
    if not results:
        print(f"Нет результатов")
        return None
    
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'Model'
    df = df.reset_index()
    df.rename(columns={'cer': 'CER', 'wer': 'WER', 'accuracy': 'Accuracy'}, inplace=True)
    
    df = calculate_rankings(df)
    
    if 'Average_Rank' in df.columns:
        df = df.sort_values('Average_Rank')
    
    output_columns = ['Final_Rank', 'Model', 'CER', 'WER', 'Accuracy', 'Average_Rank']
    output_columns = [col for col in output_columns if col in df.columns]
    df_output = df[output_columns].copy()
    
    for col in ['CER', 'WER', 'Accuracy', 'Average_Rank']:
        if col in df_output.columns:
            df_output[col] = df_output[col].round(4)
    
    if 'Final_Rank' in df_output.columns:
        df_output['Final_Rank'] = df_output['Final_Rank'].astype(int)
    
    output_path = os.path.join(output_dir, f"{dataset_name}_ranking.csv")
    df_output.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n{dataset_name}:")
    print(df_output.to_string(index=False))
    print(f"Сохранено: {output_path}\n")
    
    return df_output


# ============================================================
# README GENERATION
# ============================================================

def generate_readme(rankings_dir, output_path="README.md"):
    """Генерирует README.md с heatmap."""
    
    lines = []
    lines.append("# 🏆 AzbukaBoard — Cyrillic Handwriting OCR Leaderboard\n")
    lines.append("Benchmark для оценки моделей распознавания кириллического рукописного текста.\n")
    
    # Добавляем heatmap
    charts_dir = os.path.join(rankings_dir, "charts")
    if os.path.exists(charts_dir):
        lines.append("![Heatmap](rankings/charts/metrics_heatmap.png)\n")
    
    lines.append("---\n")
    
    # Добавляем информацию о метриках
    lines.append("## Metrics Description\n")
    lines.append("- **CER** (Character Error Rate) — Доля ошибочных символов. Чем меньше, тем лучше.\n")
    lines.append("- **WER** (Word Error Rate) — Доля ошибочных слов. Чем меньше, тем лучше.\n")
    lines.append("- **ACC** (Accuracy) — Доля полностью правильно распознанных строк. Чем больше, тем лучше.\n")
    lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"README обновлён: {output_path}")


# ============================================================
# HEATMAP CHARTS
# ============================================================

def plot_heatmap(ax, data, title, cmap):
    """Рисует heatmap с аннотациями."""
    # robust color scaling via percentiles
    vmin, vmax = np.percentile(data.values, [2, 98])
    
    ax.imshow(data.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=12)
    
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j, i, f"{data.values[i, j]:.4f}",
                ha="center", va="center",
                fontsize=8
            )


def plot_metrics_heatmap(rankings_dir, output_dir=None):
    """Строит heatmap для всех метрик."""
    
    if output_dir is None:
        output_dir = os.path.join(rankings_dir, "charts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем все ranking файлы
    ranking_files = [f for f in os.listdir(rankings_dir) if f.endswith('_ranking.csv')]
    ranking_files = [f for f in ranking_files if 'example' not in f.lower()]
    
    if not ranking_files:
        print("Нет ranking файлов для построения графиков")
        return
    
    # Сокращённые названия датасетов
    short_names = {
        'DonkeySmallOCR-Numbers-Printed-15random': 'DonkeyNumbers',
        'YeniseiGovReports-HWR': 'YeniseiGov-HWR',
        'YeniseiGovReports-PRT': 'YeniseiGov-PRT',
        'HandwrittenKazakhRussian': 'HandwrittenKZRU',
        'school_notebooks_RU': 'SchoolNotebooksRU',
        'RussianSchoolEssays': 'RussianSchoolEssays',
        'orig_cyrillic': 'OrigCyrillic',
    }
    
    dfs = {}
    datasets = []
    for filename in ranking_files:
        dataset_name = filename.replace('_ranking.csv', '')
        short_name = short_names.get(dataset_name, dataset_name)
        filepath = os.path.join(rankings_dir, filename)
        df = pd.read_csv(filepath)
        df.columns = [c.strip().replace("...", "") for c in df.columns]
        dfs[short_name] = df
        datasets.append(short_name)
    
    # Берём модели из первого датасета
    first_dataset = datasets[0]
    models = dfs[first_dataset]["Model"].tolist()
    
    # Создаём таблицы метрик
    acc = pd.DataFrame(index=models, columns=datasets, dtype=float)
    cer = pd.DataFrame(index=models, columns=datasets, dtype=float)
    wer = pd.DataFrame(index=models, columns=datasets, dtype=float)
    
    for d in datasets:
        for _, row in dfs[d].iterrows():
            acc.loc[row["Model"], d] = row["Accuracy"]
            cer.loc[row["Model"], d] = row["CER"]
            wer.loc[row["Model"], d] = row["WER"]
    
    # Mean-sort (independently)
    acc = acc.loc[acc.mean(axis=1).sort_values(ascending=False).index]
    cer = cer.loc[cer.mean(axis=1).sort_values().index]
    wer = wer.loc[wer.mean(axis=1).sort_values().index]
    
    # Append mean column
    acc["Mean"] = acc.mean(axis=1)
    cer["Mean"] = cer.mean(axis=1)
    wer["Mean"] = wer.mean(axis=1)
    
    # Final figure: 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(23, 7))
    
    plot_heatmap(axes[0], acc, "Accuracy (ACC ↑, mean-sorted)", cmap="viridis")
    plot_heatmap(axes[1], cer, "Character Error Rate (CER ↓, mean-sorted, robust)", cmap="viridis_r")
    plot_heatmap(axes[2], wer, "Word Error Rate (WER ↓, mean-sorted, robust)", cmap="viridis_r")
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "metrics_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap сохранён: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--results-dir', default='results_ocr')
    parser.add_argument('--output-dir', default='rankings')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--python', default='python')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = load_config(args.config)
    datasets = config.get('datasets', {})
    
    print(f"Датасетов: {len(datasets)}\n")
    
    # 1. Собираем результаты
    all_results = collect_all_results(args.results_dir, args.data_root, datasets, args.python)
    
    # 2. Создаём ranking таблицы
    for dataset_name in datasets.keys():
        results = all_results.get(dataset_name, {})
        if results:
            print(f"\n{'='*60}")
            print(f"{dataset_name}: {len(results)} моделей")
            print(f"{'='*60}")
            create_ranking_table(dataset_name, results, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Рейтинговые таблицы сохранены в: {args.output_dir}")
    print(f"{'='*60}")
    
    # 3. Генерируем heatmap
    print("\nГенерация heatmap...")
    plot_metrics_heatmap(args.output_dir)
    
    # 4. Генерируем README
    print("\nГенерация README...")
    generate_readme(args.output_dir)
    
    print(f"\n{'='*60}")
    print("Готово!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()