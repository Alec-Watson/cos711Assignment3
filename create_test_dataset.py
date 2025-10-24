import os
import re
import pandas as pd
import numpy as np

UNL_PATH = "unl/unl_PNG"
TEST_CSV_PATH = "test.csv"
OUTPUT_PATH = "test_dataset.csv"
MATCHED_FILENAMES_PATH = "test_matched_filenames.txt"

def angular_distance(ra1, dec1, ra2, dec2):
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)
    
    delta_ra = ra2_rad - ra1_rad
    
    cos_angle = (np.sin(dec1_rad) * np.sin(dec2_rad) +
                 np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(delta_ra))
    
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    distance = np.arccos(cos_angle)
    
    return np.degrees(distance)


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


def find_closest_match(target_ra, target_dec, unlabelled_images):
    min_distance = float('inf')
    best_match = None
    
    for filename, ra, dec in unlabelled_images:
        distance = angular_distance(target_ra, target_dec, ra, dec)
        if distance < min_distance:
            min_distance = distance
            best_match = filename
    
    return best_match, min_distance


def main():
    print("🔧 Loading test.csv...")
    test_df = pd.read_csv(TEST_CSV_PATH, header=None, names=["RA", "Dec"])
    print(f"✅ Loaded {len(test_df)} test coordinates.")
    
    print(f"\n📂 Scanning unlabelled images in {UNL_PATH}...")
    if not os.path.exists(UNL_PATH):
        print(f"❌ ERROR: Path {UNL_PATH} does not exist!")
        return
    
    # Extract coordinates from all unlabelled images
    unlabelled_images = []
    image_files = [f for f in os.listdir(UNL_PATH) if f.lower().endswith(".png")]
    
    for filename in image_files:
        ra, dec = extract_coordinates_from_filename(filename)
        if ra is not None and dec is not None:
            unlabelled_images.append((filename, ra, dec))
    
    print(f"✅ Found {len(unlabelled_images)} unlabelled images with valid coordinates.")
    
    print(f"\n🔍 Matching test coordinates to unlabelled images (enforcing unique matches)...")
    matched_data = []
    matched_filenames = []
    used_images = set()
    
    for idx, row in test_df.iterrows():
        target_ra = row["RA"]
        target_dec = row["Dec"]
        
        available_images = [(f, ra, dec) for f, ra, dec in unlabelled_images if f not in used_images]
        
        if not available_images:
            print(f"❌ [{idx+1}/{len(test_df)}] No available images left for ({target_ra:.4f}, {target_dec:.4f})")
            continue
        
        best_filename, distance = find_closest_match(target_ra, target_dec, available_images)
        
        if best_filename:
            matched_data.append({
                "filename": best_filename,
                "RA": target_ra,
                "Dec": target_dec
            })
            matched_filenames.append(best_filename)
            used_images.add(best_filename)
            print(f"✅ [{idx+1}/{len(test_df)}] Matched ({target_ra:.4f}, {target_dec:.4f}) -> {best_filename} (Δ={distance:.8f})")
        else:
            print(f"❌ [{idx+1}/{len(test_df)}] No match found for ({target_ra:.4f}, {target_dec:.4f})")
    
    test_dataset_df = pd.DataFrame(matched_data)
    test_dataset_df.to_csv(OUTPUT_PATH, index=False, float_format="%.10f")
    print(f"\n💾 Saved test dataset to: {OUTPUT_PATH}")
    print(f"📊 Total test images matched: {len(test_dataset_df)}")
    
    with open(MATCHED_FILENAMES_PATH, 'w') as f:
        for filename in matched_filenames:
            f.write(filename + '\n')
    print(f"💾 Saved matched filenames to: {MATCHED_FILENAMES_PATH}")


if __name__ == "__main__":
    main()
