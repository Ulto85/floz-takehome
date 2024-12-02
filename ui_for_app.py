import gradio as gr
from crawler import Crawler
from dotenv import load_dotenv
import os
import json

# Load the API key
load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")

# Function to run the crawler and save the output
def run_crawler(prompt, website, depth):
    try:
        depth = int(depth)  # Ensure depth is an integer
        client = Crawler(prompt, website, api_key, depth, "json")
        output = client.execute().content
        output_data = json.loads(output)
        
        # Save the output to a file
        file_name = "crawler_output.json"
        with open(file_name, "w") as file:
            json.dump(output_data, file, indent=4)
        
        return "✅ Crawl completed successfully! Download the file below.", file_name
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Gradio Interface
def create_interface():
    with gr.Blocks(css="""
        #title {
            font-size: 28px;
            font-weight: bold;
            color: #3c7df0;
            text-align: center;
            margin-bottom: 20px;
        }
        #subtitle {
            font-size: 16px;
            color: #555;
            text-align: center;
            margin-bottom: 40px;
        }
        .gr-textbox {
            background-color: #f7f7f9;
            border: 1px solid #ddd;
        }
        .gr-button {
            background-color: #3c7df0;
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        .gr-button:hover {
            background-color: #2f66c2;
        }
    """) as interface:
        # Title and Description
        gr.Markdown("<div id='title'>🌐 Web Crawler Interface</div>")
        gr.Markdown("<div id='subtitle'>Configure your crawler below and download the results.</div>")

        with gr.Row():
            # Input fields
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter the prompt for the crawler", lines=2)
                website = gr.Textbox(label="Website URL", placeholder="Enter the website URL", lines=1)
                depth = gr.Number(label="Depth", value=1, precision=0)
            
            # Run button
            run_button = gr.Button("🚀 Run Crawler")

        # Output fields
        result_message = gr.Textbox(label="Result", interactive=False, lines=2)
        file_download = gr.File(label="Download Output File", visible=True)

        # Functionality
        def on_click(prompt, website, depth):
            message, file_path = run_crawler(prompt, website, depth)
            return message, file_path if file_path else None

        # Link the button to the crawler function
        run_button.click(on_click, inputs=[prompt, website, depth], outputs=[result_message, file_download])

    return interface

# Launch the Interface

interface = create_interface()
interface.launch()
