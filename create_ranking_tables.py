# -*- coding: utf-8 -*-
import os
import yaml
import pandas as pd
import argparse
import subprocess


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_get_metrics(submission_path, data_root, python_exe):
    try:
        cmd = [python_exe, 'get_metrics.py', '--submission', submission_path, '--data-root', data_root]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            print(f"Ошибка: {submission_path}")
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
        return
    
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


if __name__ == "__main__":
    main()
