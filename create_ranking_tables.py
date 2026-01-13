# -*- coding: utf-8 -*-
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

def generate_dataset_table(df, dataset_name, config):
    """Генерирует Markdown-таблицу для одного датасета."""
    lines = []
    lines.append(f"## Dataset: {dataset_name}\n")
    
    # Добавляем информацию о датасете из конфига
    ds_config = config.get('datasets', {}).get(dataset_name, {})
    if ds_config.get('homepage') and ds_config['homepage'] != '-':
        lines.append(f"**Homepage:** [{ds_config['homepage']}]({ds_config['homepage']})\n")
    if ds_config.get('author') and ds_config['author'] != '-':
        lines.append(f"**Author:** {ds_config['author']}\n")
    if ds_config.get('license') and ds_config['license'] != '-':
        lines.append(f"**License:** {ds_config['license']}\n")
    lines.append("")
    
    # Заголовок таблицы
    lines.append("| Rank | Model | CER ↓ | WER ↓ | ACC ↑ | Avg Rank |")
    lines.append("|------|-------|-------|-------|-------|----------|")
    
    for _, row in df.iterrows():
        rank = int(row['Final_Rank']) if 'Final_Rank' in row else '-'
        model = row['Model']
        cer = f"{row['CER']:.4f}" if 'CER' in row else '-'
        wer = f"{row['WER']:.4f}" if 'WER' in row else '-'
        acc = f"{row['Accuracy']:.4f}" if 'Accuracy' in row else '-'
        avg_rank = f"{row['Average_Rank']:.2f}" if 'Average_Rank' in row else '-'
        
        lines.append(f"| {rank} | {model} | {cer} | {wer} | {acc} | {avg_rank} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_readme(rankings_dir, config, output_path="README.md", generate_charts=True):
    """Генерирует полный README.md файл из ranking таблиц."""
    
    # Сначала генерируем графики, если нужно
    charts_dir = os.path.join(rankings_dir, "charts")
    if generate_charts:
        print("Генерация графиков...")
        plot_radar_charts(rankings_dir, charts_dir)
    
    lines = []
    lines.append("# 🏆 AzbukaBoard — Cyrillic Handwriting OCR Leaderboard\n")
    lines.append("Benchmark для оценки моделей распознавания кириллического рукописного текста.\n")
    
    # Добавляем секцию с графиками в начале
    if os.path.exists(charts_dir):
        lines.append("## 📊 Comparison Charts\n")
        lines.append("### Accuracy Radar\n")
        lines.append("![Accuracy Radar](rankings/charts/radar_accuracy.png)\n")
        lines.append("### 1-CER Radar\n")
        lines.append("![1-CER Radar](rankings/charts/radar_1_cer.png)\n")
    
    lines.append("---\n")
    
    # Загружаем все ranking файлы
    ranking_files = [f for f in os.listdir(rankings_dir) if f.endswith('_ranking.csv')]
    
    # Исключаем example датасеты
    ranking_files = [f for f in ranking_files if 'example' not in f.lower()]
    
    all_dfs = {}
    for filename in sorted(ranking_files):
        dataset_name = filename.replace('_ranking.csv', '')
        filepath = os.path.join(rankings_dir, filename)
        df = pd.read_csv(filepath)
        all_dfs[dataset_name] = df
        
        # Генерируем таблицу для датасета
        table = generate_dataset_table(df, dataset_name, config)
        lines.append(table)
        lines.append("---\n")
    
    # Добавляем информацию о метриках
    lines.append("## 📖 Metrics Description\n")
    lines.append("- **CER** (Character Error Rate) — Доля ошибочных символов. Чем меньше, тем лучше.\n")
    lines.append("- **WER** (Word Error Rate) — Доля ошибочных слов. Чем меньше, тем лучше.\n")
    lines.append("- **ACC** (Accuracy) — Доля полностью правильно распознанных строк. Чем больше, тем лучше.\n")
    lines.append("- **Avg Rank** — Средний ранг по всем метрикам.\n")
    lines.append("")
    
    # Сохраняем README
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"README обновлён: {output_path}")
    return all_dfs


# ============================================================
# RADAR CHARTS
# ============================================================

def plot_radar_charts(rankings_dir, output_dir=None):
    """Строит radar-графики для сравнения моделей по датасетам."""
    
    if output_dir is None:
        output_dir = os.path.join(rankings_dir, "charts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем все ranking файлы
    ranking_files = [f for f in os.listdir(rankings_dir) if f.endswith('_ranking.csv')]
    ranking_files = [f for f in ranking_files if 'example' not in f.lower()]
    
    if not ranking_files:
        print("Нет ranking файлов для построения графиков")
        return
    
    dfs = {}
    for filename in ranking_files:
        dataset_name = filename.replace('_ranking.csv', '')
        # Сокращаем длинные названия для графика
        short_name = dataset_name.replace('DonkeySmallOCR-Numbers-Printed-15random', 'DonkeyOCR')
        short_name = short_name.replace('YeniseiGovReports-', 'Yenisei-')
        short_name = short_name.replace('HandwrittenKazakhRussian', 'KZ-RU')
        short_name = short_name.replace('school_notebooks_RU', 'SchoolNB')
        short_name = short_name.replace('RussianSchoolEssays', 'Essays')
        short_name = short_name.replace('orig_cyrillic', 'OrigCyr')
        
        filepath = os.path.join(rankings_dir, filename)
        dfs[short_name] = pd.read_csv(filepath)
    
    # Находим все модели которые есть во всех датасетах
    all_models = set(dfs[list(dfs.keys())[0]]['Model'].tolist())
    for df in dfs.values():
        all_models &= set(df['Model'].tolist())
    
    if not all_models:
        print("Нет моделей, представленных во всех датасетах")
        # Используем все уникальные модели
        all_models = set()
        for df in dfs.values():
            all_models |= set(df['Model'].tolist())
    
    models = sorted(list(all_models))
    labels = list(dfs.keys())
    
    # Геометрия радара
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # Безопасная метрика 1 - CER
    EPS = 1e-3
    def safe_1_minus_cer(cer):
        val = 1.0 - cer
        return max(EPS, min(1.0, val))
    
    # Функция для построения радара
    def plot_radar(metric, title, filename):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
        
        for idx, model in enumerate(models):
            values = []
            valid = True
            
            for dataset in labels:
                df = dfs[dataset]
                model_row = df[df['Model'] == model]
                
                if model_row.empty:
                    valid = False
                    break
                
                row = model_row.iloc[0]
                
                if metric == "ACC":
                    values.append(row["Accuracy"])
                elif metric == "1-CER":
                    values.append(safe_1_minus_cer(row["CER"]))
            
            if not valid:
                continue
                
            values += values[:1]
            
            ax.plot(angles, values, linewidth=1.5, alpha=0.8, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(title, pad=20, fontsize=14)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), frameon=False, fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График сохранён: {output_path}")
    
    # Строим графики
    plot_radar("ACC", "Accuracy (ACC) — Сравнение моделей", "radar_accuracy.png")
    plot_radar("1-CER", "Quality (1 − CER) — Сравнение моделей", "radar_1_cer.png")
    
    print(f"\nГрафики сохранены в: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--results-dir', default='results_ocr')
    parser.add_argument('--output-dir', default='rankings')
    parser.add_argument('--data-root', required=False, default=None)
    parser.add_argument('--python', default='python')
    parser.add_argument('--update-readme', action='store_true', help='Только обновить README из существующих rankings')
    parser.add_argument('--charts', action='store_true', help='Только построить графики из существующих rankings')
    parser.add_argument('--no-readme', action='store_true', help='Не генерировать README')
    parser.add_argument('--no-charts', action='store_true', help='Не генерировать графики')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Если только обновление README
    if args.update_readme:
        generate_readme(args.output_dir, config)
        return
    
    # Если только построение графиков
    if args.charts:
        plot_radar_charts(args.output_dir)
        return
    
    # Полный режим - нужен data-root
    if args.data_root is None:
        print("Ошибка: --data-root обязателен для полного запуска")
        print("Используйте --update-readme или --charts для работы с существующими rankings")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets = config.get('datasets', {})
    
    print(f"Датасетов: {len(datasets)}\n")
    
    all_results = collect_all_results(args.results_dir, args.data_root, datasets, args.python)
    
    for dataset_name in datasets.keys():
        results = all_results.get(dataset_name, {})
        if results:
            print(f"\n{'='*60}")
            print(f"{dataset_name}: {len(results)} моделей")
            print(f"{'='*60}")
            create_ranking_table(dataset_name, results, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Готово! Все рейтинговые таблицы сохранены в: {args.output_dir}")
    print(f"{'='*60}")
    
    # Генерируем README (включая графики) если не отключено
    if not args.no_readme:
        print(f"\n{'='*60}")
        print("Генерация README и графиков...")
        print(f"{'='*60}")
        generate_readme(args.output_dir, config, generate_charts=not args.no_charts)
    elif not args.no_charts:
        # Если README отключен, но графики нужны
        print(f"\n{'='*60}")
        print("Генерация графиков...")
        print(f"{'='*60}")
        plot_radar_charts(args.output_dir)


if __name__ == "__main__":
    main()
