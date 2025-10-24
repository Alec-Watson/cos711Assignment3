import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

print("=" * 70)
print("MULTI-LABEL DATASET: LOAD, CLEAN & SPLIT (60/20/20)")
print("=" * 70)

print("\n[1] Loading combined_dataset.csv...")
df = pd.read_csv('combined_dataset.csv')
print(f"✅ Loaded {len(df)} samples")

print("\n[2] Data Cleaning...")
initial_count = len(df)

critical_cols = ['filename', 'RA', 'Dec', 'label1']
missing_critical = df[critical_cols].isna().any(axis=1)
if missing_critical.sum() > 0:
    print(f"⚠️  Removing {missing_critical.sum()} rows with missing critical data")
    df = df[~missing_critical].reset_index(drop=True)

duplicates = df.duplicated(subset=['filename'])
if duplicates.sum() > 0:
    print(f"⚠️  Removing {duplicates.sum()} duplicate filenames")
    df = df[~duplicates].reset_index(drop=True)

removed = initial_count - len(df)
if removed == 0:
    print(f"✅ No issues found - all {len(df)} samples are clean")
else:
    print(f"✅ Cleaned: {removed} rows removed, {len(df)} samples remaining")

print("\n[3] Creating multi-label format...")
print(f"ℹ️  Note: NaN in label2/label3 is expected (single-label samples)")
labels_list = []
for _, row in df.iterrows():
    sample_labels = [row['label1']]
    if pd.notna(row['label2']):
        sample_labels.append(row['label2'])
    if pd.notna(row['label3']):
        sample_labels.append(row['label3'])
    labels_list.append(sample_labels)

all_labels = sorted(set(label for labels in labels_list for label in labels))
print(f"✅ Found {len(all_labels)} unique labels: {all_labels}")

mlb = MultiLabelBinarizer(classes=all_labels)
label_matrix = mlb.fit_transform(labels_list)

for i, label in enumerate(all_labels):
    df[f'label_{label}'] = label_matrix[:, i]

print(f"✅ Created binary label columns (shape: {label_matrix.shape})")

print("\nLabel distribution:")
for label in all_labels:
    count = label_matrix[:, all_labels.index(label)].sum()
    print(f"  {label:25s}: {count:4d} ({count/len(df)*100:5.2f}%)")

print(f"\nMulti-label samples: {sum(1 for l in labels_list if len(l) > 1)}")

print("\n[4] Splitting dataset (60/20/20) with stratification...")
RANDOM_STATE = 42

print("ℹ️  Using label1 for stratification to ensure balanced distribution")
train_df, temp_df = train_test_split(
    df, 
    test_size=0.40, 
    random_state=RANDOM_STATE, 
    shuffle=True,
    stratify=df['label1']
)
print(f"✅ Train: {len(train_df)} samples (60.0%)")
print(f"   Temp:  {len(temp_df)} samples (40.0%)")

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50, 
    random_state=RANDOM_STATE, 
    shuffle=True,
    stratify=temp_df['label1']
)
print(f"✅ Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"✅ Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

print("\nLabel distribution across splits:")
print(f"{'Label':<25s} {'Train':<12s} {'Val':<12s} {'Test':<12s}")
print("-" * 65)
for label in sorted(all_labels):
    train_count = (train_df['label1'] == label).sum() + (train_df['label2'] == label).sum() + (train_df['label3'] == label).sum()
    val_count = (val_df['label1'] == label).sum() + (val_df['label2'] == label).sum() + (val_df['label3'] == label).sum()
    test_count = (test_df['label1'] == label).sum() + (test_df['label2'] == label).sum() + (test_df['label3'] == label).sum()
    print(f"{label:<25s} {train_count:<4d} ({train_count/len(train_df)*100:5.1f}%)  {val_count:<4d} ({val_count/len(val_df)*100:5.1f}%)  {test_count:<4d} ({test_count/len(test_df)*100:5.1f}%)")

print("\n[5] Saving to combined_dataset_split.csv...")
df['split'] = 'train'
df.loc[val_df.index, 'split'] = 'val'
df.loc[test_df.index, 'split'] = 'test'

df.to_csv('combined_dataset_split.csv', index=False, float_format="%.10f")
print("✅ Saved combined_dataset_split.csv")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Unique labels: {len(all_labels)}")
print(f"\nColumns in output:")
print(f"  - filename, RA, Dec, label1, label2, label3")
print(f"  - label_<class> (binary 0/1 for each of {len(all_labels)} classes)")
print(f"  - split (train/val/test)")
print("\n✅ Ready for multi-label training!")
