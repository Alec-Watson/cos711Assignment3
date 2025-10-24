import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

TYP_PATH = "typ/typ_PNG"
EXO_PATH = "exo/exo_PNG"
LABELS_PATH = "labels.csv"
OUTPUT_PATH = "combined_dataset.csv"

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


def match_label(ra, dec, labels_df):
    temp_df = labels_df.copy()
    temp_df["distance"] = temp_df.apply(
        lambda row: angular_distance(ra, dec, row["RA"], row["Dec"]), 
        axis=1
    )
    best_match = temp_df.loc[temp_df["distance"].idxmin()]
    
    return best_match


def process_images_in_folder(folder_path, labels_df, category_label):
    data = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]

    for filename in image_files:
        ra, dec = extract_coordinates_from_filename(filename)
        if ra is None or dec is None:
            print(f"⚠️ Could not extract coordinates from: {filename}")
            continue

        match = match_label(ra, dec, labels_df)

        if match is not None:
            labels = []
            if pd.notna(match["Label1"]):
                labels.append(match["Label1"])
            if pd.notna(match["Label2"]):
                labels.append(match["Label2"])
            if pd.notna(match["Label3"]):
                labels.append(match["Label3"])
            
            label_str = ", ".join(labels)
            
            data.append({
                "filename": filename,
                "RA": match["RA"], 
                "Dec": match["Dec"],  
                "label1": match["Label1"] if pd.notna(match["Label1"]) else None,
                "label2": match["Label2"] if pd.notna(match["Label2"]) else None,
                "label3": match["Label3"] if pd.notna(match["Label3"]) else None,
                "distance": match["distance"],
                "source_type": category_label
            })
            print(f"✅ Matched {filename} -> {label_str} (Δ={match['distance']:.8f})")
        else:
            data.append({
                "filename": filename,
                "RA": ra,
                "Dec": dec,
                "label1": "Unmatched",
                "label2": None,
                "label3": None,
                "distance": None,
                "source_type": category_label
            })
            print(f"❌ No match found for {filename}")

    return pd.DataFrame(data)

def main():
    print("🔧 Loading labels.csv...")
    labels_df = pd.read_csv(
        LABELS_PATH,
        header=None,
        names=["RA", "Dec", "Label1", "Label2", "Label3"],
        dtype={"RA": float, "Dec": float, "Label1": str, "Label2": str, "Label3": str}
    )
    labels_df = labels_df.dropna(subset=["RA", "Dec", "Label1"])

    print(f"Loaded {len(labels_df)} labels.")

    print("\nProcessing typical images...")
    typ_df = process_images_in_folder(TYP_PATH, labels_df, "typical")

    print("\nProcessing exotic images...")
    exo_df = process_images_in_folder(EXO_PATH, labels_df, "exotic")

    combined_df = pd.concat([typ_df, exo_df], ignore_index=True)

    combined_df.to_csv(OUTPUT_PATH, index=False, float_format="%.10f")

    print(f"\nSaved combined dataset to: {OUTPUT_PATH}")
    print(f"Total images processed: {len(combined_df)}")


if __name__ == "__main__":
    main()
