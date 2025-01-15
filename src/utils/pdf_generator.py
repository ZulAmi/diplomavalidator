from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader
import os

def generate_pdf(data, output_path):
    env = Environment(loader=FileSystemLoader("src/utils/templates"))
    template = env.get_template("report.html")
    html_content = template.render(data=data)
    HTML(string=html_content).write_pdf(output_path)