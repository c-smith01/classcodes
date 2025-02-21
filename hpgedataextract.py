import os
import re
import pandas as pd

def extract_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract dead time
    dead_time_match = re.search(r'Dead Time\s+:\s+([\d\.]+)\s+%', content)
    dead_time = float(dead_time_match.group(1)) if dead_time_match else None
    
    # Extract peak energies and total counts from "Peak Analysis Report"
    peak_data = re.findall(r'\d+\s+\d+-\s+\d+\s+\d+\.\d+\s+(\d+\.\d+)\s+\d+\.\d+\s+(\d+\.\d+)', content)
    if peak_data:
        energies = [float(p[0]) for p in peak_data]
        counts = [float(p[1]) for p in peak_data]
        avg_energy = sum(energies) / len(energies) if energies else None
        total_counts = sum(counts) if counts else None
    else:
        avg_energy, total_counts = None, None
    
    # Extract isotope activities from "Interference Corrected Report"
    activity_data = re.findall(r'([A-Z]+-\d+)\s+\d+\.\d+E?[\+\-]?\d*\s+(\d+\.\d+E?[\+\-]?\d*)', content)
    isotope_activities = {iso: float(act) for iso, act in activity_data}
    
    # Store results
    return {
        "Filename": os.path.basename(file_path),
        "Dead Time (%)": dead_time,
        "Avg Energy (keV)": avg_energy,
        "Total Counts": total_counts,
        "Isotope Activities": isotope_activities
    }

def process_folder(folder_path, output_csv):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Process only text files
            file_path = os.path.join(folder_path, filename)
            extracted_data = extract_data_from_file(file_path)
            data_list.append(extracted_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_list)
    
    # Expand isotope activity dictionary into separate columns
    isotope_df = df.pop("Isotope Activities").apply(pd.Series)
    df = pd.concat([df, isotope_df], axis=1)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

# Example usage
# process_folder("path/to/folder", "output_data.csv")
