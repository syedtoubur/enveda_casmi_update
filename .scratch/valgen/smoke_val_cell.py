import os, sys, pickle, tempfile, pandas as pd
sys.path.insert(0, 'C:/casmi/lines/exploit/build'); import patch_harness_dump as phd
T = tempfile.mkdtemp(); W = os.path.join(T, 'work')
vk = [r for r in pickle.load(open('C:/Users/h/OneDrive/Desktop/ENVEDA_casmi/val_scripts/val_key.pkl', 'rb')) if r['cohort'] == 'C2'][:3]
vq = pd.read_parquet('C:/Users/h/OneDrive/Desktop/ENVEDA_casmi/val_scripts/val.parquet')
vq = vq[vq.molecule_id.isin([r['molecule_id'] for r in vk])]
q = vq.groupby('molecule_id').precursor_mz.median()
rows = []                                                  # train: truth rows (some other adduct), a leftover, decoys
for i, r in enumerate(vk):
    rows += [dict(normalized_smiles=r['smiles'], inchikey14=r['inchikey14'], adduct='[M+H]+', precursor_mz=q[r['molecule_id']]),
             dict(normalized_smiles=r['smiles'], inchikey14=r['inchikey14'], adduct='[M+Na]+', precursor_mz=q[r['molecule_id']] + 21.98)]
rows += [dict(normalized_smiles='CCO', inchikey14='LFQSCWFLJHTTHZ', adduct='[M+H]+', precursor_mz=47.05)] * 4
tr = pd.DataFrame(rows)
ex = [dict(r, ban_rows=[2 * i] if i else [1]) for i, r in enumerate(vk)]   # ban only one truth row each
pickle.dump(ex, open(os.path.join(T, 'val_key.pkl'), 'wb'))
vq.to_parquet(os.path.join(T, 'val.parquet')); vq.head(1).to_parquet(os.path.join(T, 'test.parquet'))
tr.to_parquet(os.path.join(T, 'train.parquet'))
src = phd.VAL_CELL.replace('@SHARD@', "'C2'").replace('/kaggle/working/val_comp', W.replace('\\', '/'))
g = dict(find=lambda n: os.path.join(T, n))
exec(compile(src, 'VAL_CELL', 'exec'), g)
out = pd.read_parquet(os.path.join(W, 'train.parquet'))
print('train rows kept', len(out), 'truth smiles left', int(out.normalized_smiles.isin([r['smiles'] for r in vk]).sum()))

