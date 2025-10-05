from fastapi import APIRouter, HTTPException
import pandas as pd

router = APIRouter()

# Dataset configurations
DATASET_CONFIG = {
    "k2": {
        "file": "datasets/training_sets/K2 Planets and Candidates-Processed.csv",
        "disposition_column": "disposition",
        "confirmed_values": ["CONFIRMED"],
        "false_positive_values": ["FALSE POSITIVE"],
    },
    "kepler": {
        "file": "datasets/training_sets/Kepler Objects of Interest - Filtered.csv",
        "disposition_column": "koi_disposition",
        "confirmed_values": ["CONFIRMED"],
        "false_positive_values": ["FALSE POSITIVE"],
    },
    "tess": {
        "file": "datasets/training_sets/TESS Objects of Interest - Filtered.csv",
        "disposition_column": "tfopwg_disp",
        "confirmed_values": ["KP"],  # KP = Known Planet
        "false_positive_values": ["FP"],  # FP = False Positive
    },
}


@router.get("/simple-analytics/{dataset}")
def get_simple_analytics(dataset: str):
    """
    Get simple analytics for a given dataset (k2, tess, or kepler).
    Returns the number of confirmed planets and false positives.
    """
    # Normalize dataset name to lowercase
    dataset = dataset.lower()

    # Check if dataset is valid
    if dataset not in DATASET_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset. Choose from: {', '.join(DATASET_CONFIG.keys())}",
        )

    config = DATASET_CONFIG[dataset]

    try:
        # Read the CSV file
        df = pd.read_csv(config["file"])

        # Get the disposition column
        disposition_col = config["disposition_column"]

        if disposition_col not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Column '{disposition_col}' not found in dataset",
            )

        # Count confirmed planets
        confirmed_count = df[disposition_col].isin(config["confirmed_values"]).sum()

        # Count false positives
        false_positive_count = (
            df[disposition_col].isin(config["false_positive_values"]).sum()
        )

        return {
            "dataset": dataset,
            "confirmed_planets": int(confirmed_count),
            "false_positives": int(false_positive_count),
            "total_records": len(df),
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Dataset file not found: {config['file']}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing dataset: {str(e)}"
        )
