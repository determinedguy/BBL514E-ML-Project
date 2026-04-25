import subprocess
import pandas as pd
from pathlib import Path

def extract_features_from_pcap(pcap_path: Path, output_csv_path: Path) -> pd.DataFrame:
    """
    Takes a raw .pcap file uploaded via the web interface, 
    runs it through the cicflowmeter Python library, 
    and returns a pandas DataFrame ready for the ML pipeline.
    """
    print(f"Running CICFlowMeter on {pcap_path}...")
    
    # The cicflowmeter python library usually runs via CLI command
    # e.g., cicflowmeter -f input.pcap -c output.csv
    try:
        subprocess.run(
            ["cicflowmeter", "-f", str(pcap_path), "-c", str(output_csv_path)],
            check=True
        )
        print("Extraction successful.")
        
        # Load the newly extracted features
        df = pd.read_csv(output_csv_path)
        return df
        
    except subprocess.CalledProcessError as e:
        print(f"Error during feature extraction: {e}")
        return pd.DataFrame()