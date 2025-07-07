from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.stats import norm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend origin in production (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_value_below_label(df, label):
    """Search for an uppercase label and return the value directly below it"""
    uppercase_label = label.upper()
    for i, row in df.iterrows():
        for j, cell in enumerate(row):
            if str(cell).strip().upper() == uppercase_label:
                # Return the value in the next row, same column
                if i + 1 < len(df):
                    return df.iloc[i + 1, j]
    return None

@app.post("/calculate_cohensd")
async def calculate_cohensd(file: UploadFile = File(...)):
    try:
        print('hello2')
        # Read the Excel file
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents), sheet_name="Cohens d", header=None)
        
        # Extract values by searching for uppercase labels
        mean1 = find_value_below_label(df, "Mean 1")
        std_dev1 = find_value_below_label(df, "Std. Dev.1") or find_value_below_label(df, "Std. Dev1")
        n1 = find_value_below_label(df, "N1")
        
        mean2 = find_value_below_label(df, "Mean 2")
        std_dev2 = find_value_below_label(df, "Std. Dev.2") or find_value_below_label(df, "Std. Dev2")
        n2 = find_value_below_label(df, "N2")
        
        conf_level = 0.95
        
        # Validate we got all required values
        required = {
            "MEAN 1": mean1, "STD DEV1": std_dev1, "N1": n1,
            "MEAN 2": mean2, "STD DEV2": std_dev2, "N2": n2
        }
        
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"Missing required values: {', '.join(missing)}")

        # Convert to appropriate types
        mean1, mean2 = float(mean1), float(mean2)
        std_dev1, std_dev2 = float(std_dev1), float(std_dev2)
        n1, n2 = int(n1), int(n2)
        conf_level = float(conf_level)

        # Calculate statistics
        mean_diff = mean1 - mean2
        pooled_var = ((n1-1)*std_dev1**2 + (n2-1)*std_dev2**2)/(n1+n2-2)
        pooled_std = np.sqrt(pooled_var)
        cohens_d = mean_diff / pooled_std
        
        # Confidence intervals
        za2 = norm.ppf(1 - (1 - conf_level)/2)
        d_lower = cohens_d - za2 * (std_dev1/np.sqrt(n1))
        d_upper = cohens_d + za2 * (std_dev2/np.sqrt(n2))
        
        # Effect size interpretation
        def cohen_interpretation(d):
            if d <= -0.8: return "LARGE -ve effect"
            elif d <= -0.5: return "MODERATE -ve effect"
            elif d <= -0.2: return "SMALL -ve effect"
            elif d < 0.2: return "0 or near zero effect"
            elif d < 0.5: return "SMALL +ve effect"
            elif d < 0.8: return "MODERATE +ve effect"
            else: return "LARGE +ve effect"
            
        def wolf_interpretation(d):
            if d <= -0.5: return "EDUCATIONAL -ve effect"
            elif d <= -0.25: return "PRACTICAL/CLINICAL -ve effect"
            elif d < 0.25: return "0 or near 0 effect"
            elif d < 0.5: return "PRACTICAL/CLINICAL +ve effect"
            else: return "EDUCATIONAL +ve effect"
        
        # Prepare response
        response = {
            "input_values": {
                "mean1": mean1,
                "std_dev1": std_dev1,
                "n1": n1,
                "mean2": mean2,
                "std_dev2": std_dev2,
                "n2": n2,
                "confidence_level": conf_level
            },
            "calculated_values": {
                "mean_difference": mean_diff,
                "pooled_variance": pooled_var,
                "pooled_standard_deviation": pooled_std,
                "cohens_d": cohens_d,
                "confidence_interval": {
                    "lower": d_lower,
                    "upper": d_upper
                },
                "effect_size_interpretation": {
                    "cohen": cohen_interpretation(cohens_d),
                    "wolf": wolf_interpretation(cohens_d)
                }
            },
            "status": "success"
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(e),
                "status": "error"
            }
        )