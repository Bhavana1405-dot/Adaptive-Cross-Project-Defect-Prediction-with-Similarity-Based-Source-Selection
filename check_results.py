import csv

rows = []
with open('results.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({k: float(v) if k != 'project' else v for k, v in row.items()})

f1s  = [r['f1_mean']  for r in rows]
aucs = [r['auc_mean'] for r in rows]
mccs = [r['mcc_mean'] for r in rows]

f1_mean  = sum(f1s)  / len(f1s)
auc_mean = sum(aucs) / len(aucs)
mcc_mean = sum(mccs) / len(mccs)

print('='*65)
print('CURRENT RESULTS  (results.csv -- 30 trials per project)')
print('='*65)
print('  F1  (mean across %d projects): %.4f' % (len(rows), f1_mean))
print('  AUC (mean across %d projects): %.4f' % (len(rows), auc_mean))
print('  MCC (mean across %d projects): %.4f' % (len(rows), mcc_mean))

# --- Compare vs base paper
baseline = {'f1': 0.627, 'auc': 0.663, 'mcc': 0.245}
our      = {'f1': f1_mean, 'auc': auc_mean, 'mcc': mcc_mean}

print()
print('Vs TriStage-CPDP (base paper):')
for m in ['f1', 'auc', 'mcc']:
    d = (our[m] - baseline[m]) / baseline[m] * 100
    arrow = '/\\' if d > 0 else '\\/'
    print('  %s: Ours=%.3f  Paper=%.3f  %s%.1f%%' % (
        m.upper(), our[m], baseline[m], arrow, abs(d)))

# --- Per-project table
print()
print('  %-20s %7s %7s %7s' % ('Project', 'F1', 'AUC', 'MCC'))
print('  ' + '-'*44)
for r in rows:
    print('  %-20s %7.3f %7.3f %7.3f' % (
        r['project'], r['f1_mean'], r['auc_mean'], r['mcc_mean']))

# --- Win/loss vs paper F1
paper_f1 = {
    'ant': 0.632, 'camel-1.0': 0.472, 'camel-1.2': 0.496,
    'camel-1.4': 0.481, 'camel-1.6': 0.517, 'ivy-1.1': 0.604,
    'ivy-1.4': 0.377, 'ivy-2.0': 0.547, 'log4j-1.0': 0.557,
    'log4j-1.1': 0.620, 'log4j-1.2': 0.831, 'lucene-2.0': 0.667,
    'lucene-2.2': 0.665, 'lucene-2.4': 0.693, 'poi-1.5': 0.718,
    'poi-2.0': 0.501, 'poi-2.5': 0.757, 'poi-3.0': 0.791,
    'xalan-2.4': 0.463, 'xalan-2.5': 0.592, 'xalan-2.6': 0.701,
    'xalan-2.7': 0.793, 'xerces-1.2': 0.393, 'xerces-1.3': 0.502,
    'xerces-1.4': 0.903,
}
print()
print('Per-project F1 vs TriStage-CPDP paper:')
print('  %-20s %7s %7s %8s  %s' % ('Project', 'Ours', 'Paper', 'Diff', 'Win?'))
print('  ' + '-'*54)
wins, losses = [], []
for r in rows:
    p = paper_f1.get(r['project'])
    if p is None:
        continue
    diff = r['f1_mean'] - p
    sym = 'WIN' if diff >= 0 else 'LOSS'
    print('  %-20s %7.3f %7.3f  %+7.3f  %s' % (
        r['project'], r['f1_mean'], p, diff, sym))
    if diff >= 0:
        wins.append(r['project'])
    else:
        losses.append(r['project'])

print()
print('  Beats paper on F1: %d/%d projects' % (len(wins), len(wins)+len(losses)))
print('  WINS  :', wins)
print('  LOSSES:', losses)

# Previous (before CHMSR changes) numbers for reference
prev = {'f1': 0.534, 'auc': 0.704, 'mcc': 0.243}
print()
print('='*65)
print('CHANGE SUMMARY  (before CHMSR changes  vs  NOW)')
print('='*65)
for m in ['f1', 'auc', 'mcc']:
    old = prev[m]
    new = our[m]
    d = (new - old) / old * 100
    arrow = '/\\' if d > 0 else '\\/'
    print('  %s: Before=%.3f  Now=%.3f  %s%.1f%%' % (
        m.upper(), old, new, arrow, abs(d)))
