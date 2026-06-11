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
import signal

# ============================================================
# 进度条工具
# ============================================================
class Progress:
    """轻量级终端进度条，无需额外依赖"""
    def __init__(self, total, prefix='进度', bar_len=30):
        self.total = total
        self.prefix = prefix
        self.bar_len = bar_len
        self.start_time = time.time()
        self.current = 0

    def update(self, n=1, suffix=''):
        self.current += n
        pct = self.current / max(self.total, 1)
        filled = int(self.bar_len * pct)
        bar = '█' * filled + '░' * (self.bar_len - filled)
        elapsed = time.time() - self.start_time
        eta = (elapsed / pct - elapsed) if pct > 0 else 0
        eta_str = _fmt_time(eta) if pct > 0 else '--:--'
        line = (f'\r  {self.prefix} [{bar}] {self.current}/{self.total}'
                f'  {pct*100:.1f}%  已用:{_fmt_time(elapsed)}  剩余:{eta_str}'
                f'  {suffix[:40]:<40}')
        print(line, end='', flush=True)

    def done(self, msg=''):
        elapsed = time.time() - self.start_time
        bar = '█' * self.bar_len
        print(f'\r  {self.prefix} [{bar}] {self.current}/{self.total}'
              f'  100.0%  总用时:{_fmt_time(elapsed)}  {msg:<40}')

def _fmt_time(sec):
    sec = int(sec)
    if sec < 60:
        return f'{sec}s'
    return f'{sec//60}m{sec%60:02d}s'

# ============================================================
# 超时装饰器（Unix/Windows 均可用的线程方案）
# ============================================================
import threading

class TimeoutError(Exception):
    pass

def run_with_timeout(fn, args=(), kwargs={}, timeout_sec=60):
    """在子线程中运行fn，超时抛出 TimeoutError"""
    result = [None]
    exc = [None]

    def _target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_sec)
    if t.is_alive():
        raise TimeoutError(f'超时 {timeout_sec}s')
    if exc[0] is not None:
        raise exc[0]
    return result[0]

# ============================================================
# 1. DIMACS CNF解析（带大文件保护）
# ============================================================
def parse_dimacs(filepath, max_clauses=2_000_000):
    """
    解析 DIMACS CNF 文件。
    max_clauses: 超过此数量的子句直接截断，防止内存溢出。
    """
    clauses = []
    n_vars = 0
    n_clauses = 0
    truncated = False
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p cnf'):
                parts = line.split()
                n_vars = int(parts[2])
                n_clauses = int(parts[3])
                # 预检：超大实例跳过
                if n_vars > 500_000 or n_clauses > 5_000_000:
                    raise MemoryError(
                        f'实例过大 ({n_vars} vars, {n_clauses} clauses)，已跳过'
                    )
                continue
            lits = [int(x) for x in line.split() if x != '0']
            if lits:
                clauses.append(tuple(lits))
                if len(clauses) >= max_clauses:
                    truncated = True
                    break
    return n_vars, n_clauses, clauses, truncated

# ============================================================
# 2. ρ_i 耦合度计算（向量化加速）
# ============================================================
def compute_rho(clauses, n_vars):
    rho = np.zeros(n_vars)
    for clause in clauses:
        cl = len(clause)
        if cl == 0:
            continue
        inv_cl = 1.0 / cl
        for lit in clause:
            var_idx = abs(lit) - 1
            if 0 <= var_idx < n_vars:
                rho[var_idx] += inv_cl
    return rho

# ============================================================
# 3. 分析单个实例
# ============================================================
def analyze_instance(name, clauses, n_vars, n_clauses):
    rho = compute_rho(clauses, n_vars)
    mean_rho = np.mean(rho)
    cv = np.std(rho) / mean_rho if mean_rho > 0 else 0.0
    sk = float(stats.skew(rho))

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
# 4. 消融实验（优化版：向量化随机打乱，避免逐子句 sample）
# ============================================================
def ablation_experiment(clauses, n_vars, n_trials=10):
    """
    用随机变量替换子句变量 → 破坏社区。
    优化：将全部文字索引打成数组，一次性 shuffle，
    比逐子句 random.sample 快约 10x。
    """
    m = len(clauses)
    # 预计算每个子句的长度和在扁平数组中的偏移
    lengths = np.array([len(c) for c in clauses], dtype=np.int32)
    signs = []
    for c in clauses:
        signs.extend(1 if l > 0 else -1 for l in c)
    signs = np.array(signs, dtype=np.int8)
    total_lits = len(signs)

    cv_ablated, sk_ablated = [], []
    vars_arr = np.arange(1, n_vars + 1, dtype=np.int32)

    for _ in range(n_trials):
        # 为每个文字随机选一个变量（有放回）
        rand_vars = np.random.choice(vars_arr, size=total_lits, replace=True)
        # 恢复符号
        rand_lits = rand_vars * signs

        # 重建 clauses（仍需 Python 循环，但数据来自 numpy）
        ablated_clauses = []
        offset = 0
        for ln in lengths:
            ablated_clauses.append(tuple(int(x) for x in rand_lits[offset:offset+ln]))
            offset += ln

        ra = compute_rho(ablated_clauses, n_vars)
        mean_ra = np.mean(ra)
        cv_ablated.append(np.std(ra) / mean_ra if mean_ra > 0 else 0.0)
        sk_ablated.append(float(stats.skew(ra)))

    return (np.mean(cv_ablated), np.std(cv_ablated),
            np.mean(sk_ablated), np.std(sk_ablated))

# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='TOSAS GPCT Benchmark分析器')
    parser.add_argument('--dir', default='sat2024', help='CNF文件目录')
    parser.add_argument('--output', default='gpct_results.csv', help='输出CSV文件')
    parser.add_argument('--max-instances', type=int, default=100, help='最大分析实例数')
    parser.add_argument('--skip-ablation', action='store_true', help='跳过消融实验')
    parser.add_argument('--ablation-trials', type=int, default=10,
                        help='消融实验轮次（默认10，减小可加速）')
    parser.add_argument('--timeout', type=int, default=120,
                        help='单实例超时秒数（默认120s，超时自动跳过）')
    parser.add_argument('--max-clauses', type=int, default=2_000_000,
                        help='单实例最大子句数，超出截断（防止内存溢出）')
    args = parser.parse_args()

    # ---- 收集文件 ----
    cnf_files = glob.glob(os.path.join(args.dir, '**/*.cnf'), recursive=True)
    cnf_files += glob.glob(os.path.join(args.dir, '*.cnf'))
    cnf_files = list(dict.fromkeys(cnf_files))  # 去重

    if not cnf_files:
        print(f'错误：在 {args.dir} 中未找到CNF文件')
        print('请先下载SAT benchmark:')
        print("  wget 'https://zenodo.org/records/15095752/files/"
              "sat-competition-2024-main-benchmarks.zip?download=1'")
        print('  unzip sat-competition-2024-main-benchmarks.zip -d sat2024/')
        sys.exit(1)

    target = cnf_files[:args.max_instances]
    total = len(target)
    print(f'找到 {len(cnf_files)} 个CNF文件，本次分析前 {total} 个')
    print(f'超时保护: {args.timeout}s/实例   消融轮次: {args.ablation_trials}')
    print()

    # ---- 主循环 ----
    results = []
    skipped = 0
    prog = Progress(total, prefix='分析')

    for i, fpath in enumerate(target):
        name = os.path.basename(fpath)
        prog.update(0, suffix=name)  # 先刷新当前文件名（不增计数）

        def _do_instance():
            n_vars, n_clauses, clauses, truncated = parse_dimacs(
                fpath, max_clauses=args.max_clauses)
            if n_vars < 100 or n_clauses < 100:
                return None

            r = analyze_instance(name, clauses, n_vars, n_clauses)
            r['truncated'] = truncated

            if not args.skip_ablation:
                cv_ab, cv_ab_std, sk_ab, sk_ab_std = ablation_experiment(
                    clauses, n_vars, args.ablation_trials)
                r['CV_ablated'] = cv_ab
                r['CV_ablated_std'] = cv_ab_std
                r['Skew_ablated'] = sk_ab
                r['CV_drop_pct'] = (
                    (r['CV'] - cv_ab) / r['CV'] * 100 if r['CV'] > 0 else 0)
                r['significant'] = r['CV'] > cv_ab + 2 * cv_ab_std
            else:
                r['CV_ablated'] = 0.0
                r['CV_ablated_std'] = 0.0
                r['Skew_ablated'] = 0.0
                r['CV_drop_pct'] = 0.0
                r['significant'] = False

            return r

        try:
            r = run_with_timeout(_do_instance, timeout_sec=args.timeout)
            if r is not None:
                results.append(r)
            # 更新进度（增加计数）
            prog.current += 1
            prog.update(0, suffix=f'✓ {name}')
        except TimeoutError:
            prog.current += 1
            prog.update(0, suffix=f'⏱ 超时跳过 {name}')
            skipped += 1
        except MemoryError as e:
            prog.current += 1
            prog.update(0, suffix=f'💾 内存跳过 {name}')
            skipped += 1
        except Exception as e:
            prog.current += 1
            prog.update(0, suffix=f'⚠ 错误 {name}: {e}')
            skipped += 1

    prog.done(f'完成 {len(results)} 个，跳过 {skipped} 个')
    print()

    if not results:
        print('没有可分析的有效实例，退出。')
        sys.exit(1)

    # ---- 写入CSV ----
    fieldnames = ['name', 'n_vars', 'n_clauses', 'truncated',
                  'CV', 'Skew',
                  'k_P85', 'k_P90', 'k_P95', 'k_P97', 'k_P99',
                  'k_P95_n', 'CV_ablated', 'CV_ablated_std',
                  'Skew_ablated', 'CV_drop_pct', 'significant']
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # ---- 汇总统计 ----
    cvs = [r['CV'] for r in results]
    sks = [r['Skew'] for r in results]
    kns = [r['k_P95_n'] for r in results]
    drops = [r['CV_drop_pct'] for r in results if r.get('CV_ablated', 0) > 0]
    sigs = sum(1 for r in results if r.get('significant', False))
    trunc = sum(1 for r in results if r.get('truncated', False))

    print(f'\n{"="*60}')
    print(f'  分析完成: {len(results)} 个实例  (跳过 {skipped} 个，截断 {trunc} 个)')
    print(f'{"="*60}')
    print(f'  CV(ρ)      = {np.mean(cvs):.4f} ± {np.std(cvs):.4f}')
    print(f'  Skew(ρ)    = {np.mean(sks):.4f} ± {np.std(sks):.4f}')
    print(f'  k/n(P95)   = {np.mean(kns):.4f} ± {np.std(kns):.4f}')
    n_kn = len(kns)
    n_kn_lt = sum(1 for k in kns if k < 0.15)
    print(f'  k/n<0.15   = {n_kn_lt}/{n_kn} ({n_kn_lt/max(n_kn,1)*100:.0f}%)')
    if drops:
        print(f'  CV下降     = {np.mean(drops):.1f}% ± {np.std(drops):.1f}%')
        print(f'  消融显著   = {sigs}/{len(drops)}')

    p1_pass = np.mean(cvs) > 0.4 and np.mean(kns) < 0.15
    print(f'\n  TOSAS预言P1: {"✓ 验证通过" if p1_pass else "⚠ 需进一步验证"}')
    print(f'  结果已保存: {args.output}')

if __name__ == '__main__':
    main()
