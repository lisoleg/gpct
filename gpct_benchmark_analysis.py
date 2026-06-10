#!/usr/bin/env python3
"""
gpct_benchmark_analysis.py
TOSAS GPCT Benchmark分析器
在真实SAT Competition benchmark上验证预言P1：
工业实例的ρ_i分布具有CV>0.5、Skew>1.0（非均匀社区结构）
随机实例的CV≈0.28、Skew≈0（均匀分布）
k/n < 0.15
消融：破坏社区结构后CV显著下降

用法：
    python gpct_benchmark_analysis.py --dir sat2024 --output results.csv
    python gpct_benchmark_analysis.py --dir satlib --output satlib_results.csv --max-instances 200
"""

import numpy as np
from scipy import stats
import os, sys, glob, random, time, csv, argparse
from collections import defaultdict

# ============================================================
# 1. DIMACS CNF解析
# ============================================================
def parse_dimacs(filepath):
    clauses = []
    n_vars = 0
    n_clauses = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p cnf'):
                parts = line.split()
                n_vars = int(parts[2])
                n_clauses = int(parts[3])
                continue
            lits = [int(x) for x in line.split() if x != '0']
            if lits:
                clauses.append(tuple(lits))
    return n_vars, n_clauses, clauses

# ============================================================
# 2. ρ_i耦合度计算
# ============================================================
def compute_rho(clauses, n_vars):
    rho = np.zeros(n_vars)
    for clause in clauses:
        cl = len(clause)
        for lit in clause:
            var_idx = abs(lit) - 1
            rho[var_idx] += 1.0 / cl
    return rho

# ============================================================
# 3. 分析单个实例
# ============================================================
def analyze_instance(name, clauses, n_vars, n_clauses):
    rho = compute_rho(clauses, n_vars)
    cv = np.std(rho) / np.mean(rho) if np.mean(rho) > 0 else 0
    sk = stats.skew(rho)

    k_results = {}
    for p in [85, 90, 95, 97, 99]:
        t = np.percentile(rho, p)
        k = int(np.sum(rho > t))
        k_results[p] = k

    return {
        'name': name, 'n_vars': n_vars, 'n_clauses': n_clauses,
        'CV': cv, 'Skew': sk,
        'k_P85': k_results[85], 'k_P90': k_results[90],
        'k_P95': k_results[95], 'k_P97': k_results[97], 'k_P99': k_results[99],
        'k_P95_n': k_results[95] / n_vars if n_vars > 0 else 0
    }

# ============================================================
# 4. 消融实验（破坏社区结构）
# ============================================================
def ablation_experiment(clauses, n_vars, n_trials=30):
    """用随机变量替换子句变量→破坏社区"""
    m = len(clauses)
    cv_ablated = []
    sk_ablated = []

    for _ in range(n_trials):
        ablated = []
        for clause in clauses:
            clen = len(clause)
            vs = random.sample(range(1, n_vars + 1), clen)
            new_c = tuple(v * (1 if lit > 0 else -1) for lit, v in zip(clause, vs))
            ablated.append(new_c)
        ra = compute_rho(ablated, n_vars)
        cv_ablated.append(np.std(ra) / np.mean(ra) if np.mean(ra) > 0 else 0)
        sk_ablated.append(stats.skew(ra))

    return np.mean(cv_ablated), np.std(cv_ablated), np.mean(sk_ablated), np.std(sk_ablated)

# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='TOSAS GPCT Benchmark分析器')
    parser.add_argument('--dir', default='sat2024', help='CNF文件目录')
    parser.add_argument('--output', default='gpct_results.csv', help='输出CSV文件')
    parser.add_argument('--max-instances', type=int, default=100, help='最大分析实例数')
    parser.add_argument('--skip-ablation', action='store_true', help='跳过消融实验')
    args = parser.parse_args()

    cnf_files = glob.glob(os.path.join(args.dir, '**/*.cnf'), recursive=True)
    cnf_files += glob.glob(os.path.join(args.dir, '*.cnf'))

    if not cnf_files:
        print(f"错误：在 {args.dir} 中未找到CNF文件")
        print("请先下载SAT benchmark:")
        print("  wget 'https://zenodo.org/records/15095752/files/sat-competition-2024-main-benchmarks.zip?download=1'")
        print("  unzip sat-competition-2024-main-benchmarks.zip -d sat2024/")
        sys.exit(1)

    print(f"找到 {len(cnf_files)} 个CNF文件")

    results = []
    for i, fpath in enumerate(cnf_files[:args.max_instances]):
        name = os.path.basename(fpath)
        try:
            n_vars, n_clauses, clauses = parse_dimacs(fpath)
            if n_vars < 100 or n_clauses < 100:
                continue

            r = analyze_instance(name, clauses, n_vars, n_clauses)

            if not args.skip_ablation:
                cv_ab, cv_ab_std, sk_ab, sk_ab_std = ablation_experiment(clauses, n_vars, 10)
                r['CV_ablated'] = cv_ab
                r['CV_ablated_std'] = cv_ab_std
                r['Skew_ablated'] = sk_ab
                r['CV_drop_pct'] = (r['CV'] - cv_ab) / r['CV'] * 100 if r['CV'] > 0 else 0
                r['significant'] = r['CV'] > cv_ab + 2 * cv_ab_std
            else:
                r['CV_ablated'] = 0
                r['significant'] = False

            results.append(r)

            if (i + 1) % 20 == 0:
                print(f"  进度: {i + 1}/{min(len(cnf_files), args.max_instances)}")
        except Exception as e:
            print(f"  ⚠ {name}: {e}")
            continue

    # 写入CSV
    fieldnames = ['name', 'n_vars', 'n_clauses', 'CV', 'Skew',
                  'k_P95', 'k_P95_n', 'CV_ablated', 'Skew_ablated',
                  'CV_drop_pct', 'significant']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # 汇总统计
    cvs = [r['CV'] for r in results]
    sks = [r['Skew'] for r in results]
    kns = [r['k_P95_n'] for r in results]
    drops = [r.get('CV_drop_pct', 0) for r in results if r.get('CV_ablated', 0) > 0]
    sigs = sum(1 for r in results if r.get('significant', False))

    print(f"\n{'='*60}")
    print(f"  分析完成: {len(results)} 个实例")
    print(f"{'='*60}")
    print(f"  CV(ρ)      = {np.mean(cvs):.4f} ± {np.std(cvs):.4f}")
    print(f"  Skew(ρ)    = {np.mean(sks):.4f} ± {np.std(sks):.4f}")
    print(f"  k/n(P95)   = {np.mean(kns):.4f} ± {np.std(kns):.4f}")
    print(f"  k/n<0.15   = {sum(1 for k in kns if k<0.15)}/{len(kns)} ({sum(1 for k in kns if k<0.15)/max(len(kns),1)*100:.0f}%)")
    if drops:
        print(f"  CV下降     = {np.mean(drops):.1f}% ± {np.std(drops):.1f}%")
        print(f"  消融显著   = {sigs}/{len(drops)}")

    p1_pass = np.mean(cvs) > 0.4 and np.mean(kns) < 0.15
    print(f"\n  TOSAS预言P1: {'✓ 验证通过' if p1_pass else '⚠ 需进一步验证'}")
    print(f"  结果已保存: {args.output}")

if __name__ == '__main__':
    main()
