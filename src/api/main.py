from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import plotly.express as px
import pandas as pd
from src.utils.pdf_generator import generate_pdf

app = FastAPI()
templates = Jinja2Templates(directory="src/api/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Sample data for analytics
    data = {
        "diploma_type": ["real", "fake", "real", "fake"],
        "authenticity_score": [0.95, 0.2, 0.85, 0.3],
        "features_detected": [5, 1, 4, 2]
    }
    df = pd.DataFrame(data)
    
    # Generate plot
    fig = px.bar(df, x="diploma_type", y="authenticity_score", color="diploma_type", barmode="group")
    graph_html = fig.to_html(full_html=False)
    
    return templates.TemplateResponse("dashboard.html", {"request": request, "graph_html": graph_html})

@app.get("/generate_report", response_class=FileResponse)
async def generate_report():
    data = {
        "authenticity_score": 0.95,
        "features_detected": 5
    }
    output_path = "report.pdf"
    generate_pdf(data, output_path)
    return FileResponse(output_path, media_type='application/pdf', filename="report.pdf")