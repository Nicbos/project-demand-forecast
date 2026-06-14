"""
model_analysis.py — Full per-model analysis of CatBoost H1–H7 demand forecasting models.

Анализ для каждой из 7 моделей (горизонты h1–h7):
  1. Gain (PredictionValuesChange) — встроенная важность CatBoost
  2. SHAP values (на сэмпле) — направленность и магнитуда влияния
  3. Permutation importance (на сэмпле) — model-agnostic важность
  4. Interaction scores — сила взаимодействий между фичами
  5. Residual analysis — распределение ошибок, гетероскедастичность
  6. Segment errors — ошибки по департаментам/форматам/городам
  7. Calibration check — predicted vs actual по децилям
  8. PDP (Partial Dependence) — влияние фичи на предсказание
  9. ALE (Accumulated Local Effects) — как PDP, но устойчив к корреляциям
  +  Cross-horizon heatmap — сводная таблица важности по всем горизонтам

Все результаты сохраняются в test/analysis/.
"""
import os, json, gc, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────
# Auto-detect Kaggle vs local
if os.path.exists('/kaggle'):
    TEST_DIR = '/kaggle/working/test'
else:
    TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
OUT_DIR  = os.path.join(TEST_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = list(range(1, 8))
SAMPLE_SHAP    = 20_000
SAMPLE_PERM    = 50_000
SAMPLE_INTER   = 15_000
SAMPLE_RESID   = 100_000
SAMPLE_PDP     = 15_000       # строк для PDP/ALE
PDP_GRID       = 25           # точек в сетке PDP
PDP_TOP_N      = 12           # сколько top фичей анализировать
ALE_BINS       = 20
RANDOM_STATE   = 42

CAT_COLS = [
    'store_id', 'item_id', 'dept_name', 'class_name',
    'subclass_name', 'item_type', 'date_of_week',
    'format', 'city', 'division',
]
EXCLUDE = {'date', 'horizon', 'target'}

# ─────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────

def load_model_and_data(h: int):
    model = CatBoostRegressor()
    model.load_model(os.path.join(TEST_DIR, f'catboost_h{h}.cbm'))
    df = pd.read_parquet(os.path.join(TEST_DIR, f'train_h{h}.parquet'))
    return model, df


def prepare_pool(df: pd.DataFrame, model: CatBoostRegressor):
    feature_names = model.feature_names_
    if not feature_names:
        feature_names = [c for c in df.columns if c not in EXCLUDE]

    cat_features = [c for c in CAT_COLS if c in feature_names]
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype(str)

    X = df[feature_names]
    y = df['target'].values
    cat_indices = [i for i, c in enumerate(feature_names) if c in cat_features]
    pool = Pool(data=X, label=y, cat_features=cat_indices)
    return X, y, pool, feature_names, cat_features


def _cat_indices(feature_names):
    return [i for i, c in enumerate(feature_names) if c in CAT_COLS]


# ─────────────────────────────────────────────────────────────────────────
#  1. GAIN
# ─────────────────────────────────────────────────────────────────────────

def gain_importance(model, feature_names):
    imp = model.get_feature_importance(type='PredictionValuesChange')
    return pd.DataFrame({'feature': feature_names, 'gain': imp}).sort_values('gain', ascending=False)


# ─────────────────────────────────────────────────────────────────────────
#  2. SHAP
# ─────────────────────────────────────────────────────────────────────────

def shap_importance(model, pool, feature_names):
    n = min(SAMPLE_SHAP, pool.num_row())
    idx = np.random.RandomState(RANDOM_STATE).choice(pool.num_row(), size=n, replace=False)
    shap_vals = model.get_feature_importance(type='ShapValues', data=pool.slice(idx))
    shap_matrix = shap_vals[:, :-1]
    base_value = shap_vals[0, -1]

    df = pd.DataFrame({
        'feature': feature_names,
        'shap_mean_abs': np.abs(shap_matrix).mean(axis=0),
        'shap_mean': shap_matrix.mean(axis=0),
        'shap_std': shap_matrix.std(axis=0),
    }).sort_values('shap_mean_abs', ascending=False)
    return df, shap_matrix, base_value


# ─────────────────────────────────────────────────────────────────────────
#  3. PERMUTATION
# ─────────────────────────────────────────────────────────────────────────

def permutation_importance_fn(model, X, y, feature_names):
    n = min(SAMPLE_PERM, len(y))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(y), size=n, replace=False)
    X_sample = X.iloc[idx].copy()
    y_sample = y[idx]

    cat_idx = _cat_indices(feature_names)
    pool_full = Pool(data=X_sample, cat_features=cat_idx)
    baseline_pred = model.predict(pool_full)
    baseline_rmse = np.sqrt(mean_squared_error(y_sample, baseline_pred))

    results = []
    for i, col in enumerate(feature_names):
        X_perm = X_sample.copy()
        X_perm[col] = np.random.RandomState(RANDOM_STATE + i).permutation(X_perm[col].values)
        pool_perm = Pool(data=X_perm, cat_features=cat_idx)
        perm_pred = model.predict(pool_perm)
        perm_rmse = np.sqrt(mean_squared_error(y_sample, perm_pred))
        delta = perm_rmse - baseline_rmse
        results.append({'feature': col, 'perm_delta_rmse': delta,
                        'perm_pct': delta / baseline_rmse * 100})

    return pd.DataFrame(results).sort_values('perm_delta_rmse', ascending=False)


# ─────────────────────────────────────────────────────────────────────────
#  4. INTERACTION
# ─────────────────────────────────────────────────────────────────────────

def interaction_scores(model, pool, feature_names):
    n = min(SAMPLE_INTER, pool.num_row())
    idx = np.random.RandomState(RANDOM_STATE).choice(pool.num_row(), size=n, replace=False)
    scores = model.get_feature_importance(type='Interaction', data=pool.slice(idx))
    scores = np.asarray(scores)
    nf = len(feature_names)

    # CatBoost returns either:
    #   (nf, nf)  square matrix       or
    #   (n_pairs, 3)  triplets (idx1, idx2, score)
    if scores.ndim == 2 and scores.shape[1] == 3:
        # triplet form: (feature1, feature2, score)
        pairs = []
        for row in scores:
            i, j, v = int(row[0]), int(row[1]), float(row[2])
            if i < nf and j < nf and i != j:
                pairs.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'interaction': v,
                })
        return pd.DataFrame(pairs).sort_values('interaction', ascending=False)
    else:
        # square matrix form
        pairs = []
        for i in range(nf):
            for j in range(i + 1, nf):
                pairs.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'interaction': float(scores[i, j]),
                })
        return pd.DataFrame(pairs).sort_values('interaction', ascending=False)


# ─────────────────────────────────────────────────────────────────────────
#  5. RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────

def residual_analysis(model, pool, y_sample, pred_sample, feature_names, df_sample):
    residuals = y_sample - pred_sample
    results = {
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(np.std(residuals)),
        'residual_skew': float(pd.Series(residuals).skew()),
        'residual_kurtosis': float(pd.Series(residuals).kurtosis()),
        'pct_within_1sigma': float((np.abs(residuals) < np.std(residuals)).mean() * 100),
        'pct_within_2sigma': float((np.abs(residuals) < 2 * np.std(residuals)).mean() * 100),
        'overprediction_pct': float((pred_sample > y_sample).mean() * 100),
        'underprediction_pct': float((pred_sample < y_sample).mean() * 100),
    }
    results['hetero_corr_pred_absres'] = float(
        np.corrcoef(pred_sample, np.abs(residuals))[0, 1])

    corr_with_abs_res = {}
    for col in feature_names:
        if col in df_sample.columns and df_sample[col].dtype in ('float32','float64','int64','int8'):
            c = np.corrcoef(df_sample[col].values.astype(float), np.abs(residuals))[0, 1]
            if not np.isnan(c):
                corr_with_abs_res[col] = abs(c)
    top = sorted(corr_with_abs_res.items(), key=lambda x: x[1], reverse=True)[:10]
    results['top_features_correlated_with_abs_error'] = top
    return results


# ─────────────────────────────────────────────────────────────────────────
#  6. SEGMENT ERRORS
# ─────────────────────────────────────────────────────────────────────────

def segment_errors(df_sample, y_sample, pred_sample):
    df_e = df_sample.copy()
    df_e['abs_error'] = np.abs(y_sample - pred_sample)
    df_e['error'] = y_sample - pred_sample
    df_e['target'] = y_sample
    df_e['pred'] = pred_sample

    segments = {}
    for col in ['dept_name', 'format', 'city', 'date_of_week']:
        if col not in df_e.columns:
            continue
        grp = df_e.groupby(col).agg(
            count=('abs_error', 'count'),
            mae=('abs_error', 'mean'),
            rmse=('error', lambda x: np.sqrt((x**2).mean())),
            mean_target=('target', 'mean'),
            mean_pred=('pred', 'mean'),
            bias=('error', 'mean'),
        ).sort_values('count', ascending=False)
        segments[col] = grp
    return segments


# ─────────────────────────────────────────────────────────────────────────
#  7. CALIBRATION
# ─────────────────────────────────────────────────────────────────────────

def calibration_check(y_sample, pred_sample):
    df_c = pd.DataFrame({'actual': y_sample, 'pred': pred_sample})
    df_c['pred_bin'] = pd.qcut(df_c['pred'], q=10, duplicates='drop')
    cal = df_c.groupby('pred_bin', observed=False).agg(
        count=('actual', 'count'),
        mean_actual=('actual', 'mean'),
        mean_pred=('pred', 'mean'),
        std_actual=('actual', 'std'),
    ).reset_index()
    cal['bias'] = cal['mean_pred'] - cal['mean_actual']
    cal['pred_bin'] = cal['pred_bin'].astype(str)
    return cal


# ─────────────────────────────────────────────────────────────────────────
#  8. PDP (Partial Dependence Plot)
# ─────────────────────────────────────────────────────────────────────────

def compute_pdp(model, pool, feature_names, X_sample, cat_features, top_features):
    n = min(SAMPLE_PDP, pool.num_row())
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_sample), size=n, replace=False)
    base_X = X_sample.iloc[idx].copy()
    cat_idx = _cat_indices(feature_names)

    pdp_results = {}
    for feat in top_features:
        if feat not in base_X.columns:
            continue

        is_cat = feat in cat_features
        col_data = base_X[feat]

        if is_cat:
            grid = col_data.value_counts().head(PDP_GRID).index.tolist()
        else:
            lo, hi = col_data.quantile(0.01), col_data.quantile(0.99)
            grid = np.linspace(float(lo), float(hi), PDP_GRID)

        means = []
        for val in grid:
            X_temp = base_X.copy()
            if is_cat:
                X_temp[feat] = str(val)
            else:
                X_temp[feat] = val
            p = Pool(data=X_temp, cat_features=cat_idx)
            preds = model.predict(p)
            means.append(float(np.mean(preds)))

        pdp_results[feat] = {
            'grid': list(grid) if not is_cat else [str(v) for v in grid],
            'mean_pred': means,
            'is_cat': is_cat,
        }

    return pdp_results


# ─────────────────────────────────────────────────────────────────────────
#  9. ALE (Accumulated Local Effects)
# ─────────────────────────────────────────────────────────────────────────

def compute_ale(model, pool, feature_names, X_sample, cat_features, top_features):
    n = min(SAMPLE_PDP, pool.num_row())
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_sample), size=n, replace=False)
    base_X = X_sample.iloc[idx].copy()
    cat_idx = _cat_indices(feature_names)

    ale_results = {}
    for feat in top_features:
        if feat not in base_X.columns:
            continue

        is_cat = feat in cat_features

        if is_cat:
            top_cats = base_X[feat].value_counts().head(ALE_BINS).index.tolist()
            cat_means = {}
            for cat_val in top_cats:
                mask = base_X[feat] == cat_val
                if mask.sum() == 0:
                    continue
                X_temp = base_X[mask].copy()
                p = Pool(data=X_temp, cat_features=cat_idx)
                preds = model.predict(p)
                cat_means[str(cat_val)] = float(np.mean(preds))
            if cat_means:
                grand_mean = np.mean(list(cat_means.values()))
                cat_means = {k: v - grand_mean for k, v in cat_means.items()}
            ale_results[feat] = {
                'grid': list(cat_means.keys()),
                'ale': list(cat_means.values()),
                'is_cat': True,
            }
        else:
            col_vals = base_X[feat].values.astype(float)
            lo, hi = float(np.quantile(col_vals, 0.01)), float(np.quantile(col_vals, 0.99))
            bin_edges = np.linspace(lo, hi, ALE_BINS + 1)

            ale = [0.0]
            bin_centers = []
            for k in range(ALE_BINS):
                lo_edge, hi_edge = bin_edges[k], bin_edges[k+1]
                center = (lo_edge + hi_edge) / 2
                bin_centers.append(center)
                in_bin = (col_vals >= lo_edge) & (col_vals < hi_edge)
                if k == ALE_BINS - 1:
                    in_bin = (col_vals >= lo_edge) & (col_vals <= hi_edge)
                if in_bin.sum() < 2:
                    ale.append(ale[-1])
                    continue
                X_lo = base_X[in_bin].copy()
                X_lo[feat] = lo_edge
                p_lo = Pool(data=X_lo, cat_features=cat_idx)
                pred_lo = model.predict(p_lo)
                X_hi = base_X[in_bin].copy()
                X_hi[feat] = hi_edge
                p_hi = Pool(data=X_hi, cat_features=cat_idx)
                pred_hi = model.predict(p_hi)
                delta = float(np.mean(pred_hi - pred_lo))
                ale.append(ale[-1] + delta)
            ale_arr = np.array(ale[1:])
            ale_arr = ale_arr - np.mean(ale_arr)
            ale_results[feat] = {
                'grid': bin_centers,
                'ale': ale_arr.tolist(),
                'is_cat': False,
            }

    return ale_results


# ─────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    all_gain        = {}
    all_shap        = {}
    all_perm        = {}
    all_interaction = {}
    all_residuals   = {}
    all_segments    = {}
    all_calibration = {}
    all_pdp         = {}
    all_ale         = {}
    metrics         = {}

    for h in HORIZONS:
        print(f'\n{"="*60}')
        print(f'  HORIZON H={h}')
        print(f'{"="*60}')

        model, df = load_model_and_data(h)
        X, y, pool, feature_names, cat_features = prepare_pool(df, model)

        print(f'  Rows: {len(y):,}  |  Features: {len(feature_names)}  |  Cat features: {len(cat_features)}')

        # 1. Gain
        print('  [1/9] Gain importance...')
        df_gain = gain_importance(model, feature_names)
        all_gain[h] = df_gain
        top_gain = df_gain['feature'].head(PDP_TOP_N).tolist()

        # 2. SHAP
        print('  [2/9] SHAP values...')
        df_shap, shap_matrix, base_value = shap_importance(model, pool, feature_names)
        all_shap[h] = df_shap
        np.savez_compressed(os.path.join(OUT_DIR, f'shap_raw_h{h}.npz'),
                            shap=shap_matrix, features=feature_names, base=base_value)

        # 3. Permutation
        print('  [3/9] Permutation importance...')
        df_perm = permutation_importance_fn(model, X, y, feature_names)
        all_perm[h] = df_perm

        # 4. Interactions
        print('  [4/9] Interaction scores...')
        df_inter = interaction_scores(model, pool, feature_names)
        all_interaction[h] = df_inter

        # Prepare sample for residual + segment + calibration
        n_res = min(SAMPLE_RESID, len(y))
        idx_res = np.random.RandomState(RANDOM_STATE).choice(len(y), size=n_res, replace=False)
        pool_res = pool.slice(idx_res)
        pred_res = model.predict(pool_res)
        y_res = y[idx_res]
        df_res = df.iloc[idx_res].reset_index(drop=True)

        # 5. Residual analysis
        print('  [5/9] Residual analysis...')
        res_info = residual_analysis(model, pool_res, y_res, pred_res, feature_names, df_res)
        all_residuals[h] = res_info

        # 6. Segment errors
        print('  [6/9] Segment error analysis...')
        seg = segment_errors(df_res, y_res, pred_res)
        all_segments[h] = seg

        # 7. Calibration
        print('  [7/9] Calibration check...')
        cal = calibration_check(y_res, pred_res)
        all_calibration[h] = cal

        # 8. PDP
        print(f'  [8/9] PDP for top-{PDP_TOP_N} features...')
        pdp = compute_pdp(model, pool, feature_names, X, cat_features, top_gain)
        all_pdp[h] = pdp

        # 9. ALE
        print(f'  [9/9] ALE for top-{PDP_TOP_N} features...')
        ale = compute_ale(model, pool, feature_names, X, cat_features, top_gain)
        all_ale[h] = ale

        metrics[h] = {
            'rmse': float(np.sqrt(mean_squared_error(y_res, pred_res))),
            'mae': float(mean_absolute_error(y_res, pred_res)),
            'wmape': float(np.abs(y_res - pred_res).sum() / (np.abs(y_res).sum() + 1e-8)),
        }
        print(f'  Metrics: RMSE={metrics[h]["rmse"]:.4f}, MAE={metrics[h]["mae"]:.4f}, '
              f'WMAPE={metrics[h]["wmape"]:.4f}')
        print(f'  Residuals: mean={res_info["residual_mean"]:.4f}, σ={res_info["residual_std"]:.4f}, '
              f'overpred={res_info["overprediction_pct"]:.1f}%, underpred={res_info["underprediction_pct"]:.1f}%')

        del model, df, X, y, pool
        gc.collect()

    # ── Save ────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  Saving results to {OUT_DIR}')
    print(f'{"="*60}')

    for h in HORIZONS:
        all_gain[h].to_csv(os.path.join(OUT_DIR, f'gain_h{h}.csv'), index=False)
        all_shap[h].to_csv(os.path.join(OUT_DIR, f'shap_h{h}.csv'), index=False)
        all_perm[h].to_csv(os.path.join(OUT_DIR, f'permutation_h{h}.csv'), index=False)
        all_interaction[h].to_csv(os.path.join(OUT_DIR, f'interaction_h{h}.csv'), index=False)
        all_calibration[h].to_csv(os.path.join(OUT_DIR, f'calibration_h{h}.csv'), index=False)
        with open(os.path.join(OUT_DIR, f'pdp_h{h}.json'), 'w') as f:
            json.dump(all_pdp[h], f, indent=2, default=str)
        with open(os.path.join(OUT_DIR, f'ale_h{h}.json'), 'w') as f:
            json.dump(all_ale[h], f, indent=2, default=str)

    pd.DataFrame(all_residuals).T.to_csv(os.path.join(OUT_DIR, 'residuals_summary.csv'))
    with open(os.path.join(OUT_DIR, 'residuals.json'), 'w') as f:
        json.dump(all_residuals, f, indent=2, default=str)

    for h in HORIZONS:
        for seg_col, seg_df in all_segments[h].items():
            seg_df.to_csv(os.path.join(OUT_DIR, f'segment_{seg_col}_h{h}.csv'))

    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── Cross-horizon heatmap ──────────────────────────────────────────
    print('\n\n' + '='*80)
    print('  CROSS-HORIZON FEATURE IMPORTANCE HEATMAP')
    print('='*80)

    all_features = sorted(set.union(*[set(df['feature']) for df in all_gain.values()]))

    gain_matrix = {}
    for feat in all_features:
        gain_matrix[feat] = {}
        for h in HORIZONS:
            row = all_gain[h][all_gain[h]['feature'] == feat]
            gain_matrix[feat][f'h{h}'] = float(row['gain'].iloc[0]) if len(row) else 0.0
    gain_hm = pd.DataFrame(gain_matrix).T
    gain_hm['avg_gain'] = gain_hm.mean(axis=1)
    gain_hm = gain_hm.sort_values('avg_gain', ascending=False)
    gain_hm.to_csv(os.path.join(OUT_DIR, 'cross_horizon_gain_heatmap.csv'))

    shap_matrix_hm = {}
    for feat in all_features:
        shap_matrix_hm[feat] = {}
        for h in HORIZONS:
            row = all_shap[h][all_shap[h]['feature'] == feat]
            shap_matrix_hm[feat][f'h{h}'] = float(row['shap_mean_abs'].iloc[0]) if len(row) else 0.0
    shap_hm = pd.DataFrame(shap_matrix_hm).T
    shap_hm['avg_shap'] = shap_hm.mean(axis=1)
    shap_hm = shap_hm.sort_values('avg_shap', ascending=False)
    shap_hm.to_csv(os.path.join(OUT_DIR, 'cross_horizon_shap_heatmap.csv'))

    perm_matrix = {}
    for feat in all_features:
        perm_matrix[feat] = {}
        for h in HORIZONS:
            row = all_perm[h][all_perm[h]['feature'] == feat]
            perm_matrix[feat][f'h{h}'] = float(row['perm_delta_rmse'].iloc[0]) if len(row) else 0.0
    perm_hm = pd.DataFrame(perm_matrix).T
    perm_hm['avg_perm'] = perm_hm.mean(axis=1)
    perm_hm = perm_hm.sort_values('avg_perm', ascending=False)
    perm_hm.to_csv(os.path.join(OUT_DIR, 'cross_horizon_perm_heatmap.csv'))

    # ── Print summaries ────────────────────────────────────────────────
    print('\nGain importance across horizons (top-25):')
    print(gain_hm.head(25).to_string())

    print('\n\nSHAP |mean| across horizons (top-25):')
    print(shap_hm.head(25).to_string())

    print('\n\n' + '='*80)
    print('  TOP-10 GAIN PER HORIZON')
    print('='*80)
    for h in HORIZONS:
        print(f'\n─── H{h} ───')
        print(all_gain[h].head(10).to_string(index=False))

    print('\n\n' + '='*80)
    print('  TOP-10 SHAP PER HORIZON')
    print('='*80)
    for h in HORIZONS:
        print(f'\n─── H{h} ───')
        print(all_shap[h].head(10).to_string(index=False))

    print('\n\n' + '='*80)
    print('  TOP-10 INTERACTIONS PER HORIZON')
    print('='*80)
    for h in HORIZONS:
        print(f'\n─── H{h} ───')
        print(all_interaction[h].head(10).to_string(index=False))

    print('\n\n' + '='*80)
    print('  RESIDUAL ANALYSIS ACROSS HORIZONS')
    print('='*80)
    res_df = pd.DataFrame(all_residuals).T
    print(res_df[['residual_mean', 'residual_std', 'residual_skew',
                   'overprediction_pct', 'underprediction_pct',
                   'hetero_corr_pred_absres']].to_string())

    print('\n\n' + '='*80)
    print('  FEATURES MOST CORRELATED WITH |ERROR|')
    print('='*80)
    for h in HORIZONS:
        print(f'\n─── H{h} ───')
        for feat, corr in all_residuals[h]['top_features_correlated_with_abs_error'][:5]:
            print(f'  {feat}: {corr:.4f}')

    print('\n\n' + '='*80)
    print('  CALIBRATION BIAS PER DECILE')
    print('='*80)
    for h in HORIZONS:
        print(f'\n─── H{h} ───')
        print(all_calibration[h][['pred_bin', 'count', 'mean_actual', 'mean_pred', 'bias']].to_string(index=False))

    print(f'\n\n✅ DONE. All results saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
