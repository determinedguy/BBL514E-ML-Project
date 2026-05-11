# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from src.backend_predict import predict_traffic
import uvicorn

app = FastAPI(title="Intrusion Detection ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------

@app.post("/api/predict")
async def upload_and_predict(file: UploadFile = File(...)):
    # 1. Security check: make sure they actually uploaded a CSV
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # 2. Save the uploaded CSV temporarily to the server
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 3. Feed it to the ML Engine!
        print(f"Processing {file.filename} through the ensemble...")
        results_df = predict_traffic(temp_file_path)
        
        # 4. Convert the results to a dictionary so it can be sent as JSON
        result_json = results_df.to_dict(orient="records") # type: ignore
        return {"filename": file.filename, "predictions": result_json}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 5. Clean up: Delete the temporary file so the server doesn't get cluttered
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# This allows you to run the file directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)