import os
import re
import pandas as pd

UNL_PATH = "unl/unl_PNG"
TEST_MATCHED_FILENAMES_PATH = "test_matched_filenames.txt"
OUTPUT_PATH = "unlabelled_dataset.csv"

def extract_coordinates_from_filename(filename):
    match = re.findall(r"[-+]?\d*\.\d+|\d+", filename)
    if len(match) >= 2:
        try:
            ra = float(match[0])
            dec = float(match[1])
            return ra, dec
        except ValueError:
            return None, None
    return None, None


def load_excluded_filenames(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: {filepath} not found. No files will be excluded.")
        return set()
    
    with open(filepath, 'r') as f:
        excluded = set(line.strip() for line in f if line.strip())
    
    return excluded


def main():
    print("🔧 Loading excluded test filenames...")
    excluded_filenames = load_excluded_filenames(TEST_MATCHED_FILENAMES_PATH)
    print(f"✅ Loaded {len(excluded_filenames)} test filenames to exclude.")
    
    print(f"\n📂 Scanning unlabelled images in {UNL_PATH}...")
    if not os.path.exists(UNL_PATH):
        print(f"❌ ERROR: Path {UNL_PATH} does not exist!")
        return
    
    unlabelled_data = []
    image_files = [f for f in os.listdir(UNL_PATH) if f.lower().endswith(".png")]
    
    excluded_count = 0
    processed_count = 0
    
    for filename in image_files:
        if filename in excluded_filenames:
            excluded_count += 1
            continue
        
        ra, dec = extract_coordinates_from_filename(filename)
        
        if ra is not None and dec is not None:
            unlabelled_data.append({
                "filename": filename,
                "RA": ra,
                "Dec": dec
            })
            processed_count += 1
            
            if processed_count % 1000 == 0:
                print(f"   Processed {processed_count} images...")
    
    print(f"✅ Processed {processed_count} unlabelled images.")
    print(f"🚫 Excluded {excluded_count} test images.")
    
    unlabelled_df = pd.DataFrame(unlabelled_data)
    unlabelled_df.to_csv(OUTPUT_PATH, index=False, float_format="%.10f")
    
    print(f"\n💾 Saved unlabelled dataset to: {OUTPUT_PATH}")
    print(f"📊 Total unlabelled images (excluding test): {len(unlabelled_df)}")


if __name__ == "__main__":
    main()
