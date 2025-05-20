import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/docs"  # Change this if deployed backend URL differs

def upload_file(file):
    if file is None:
        return "Please upload a file first."
    files = {"file": (file.name, file, file.type)}
    res = requests.post(f"{API_URL}/upload", files=files)
    if res.status_code == 200:
        return f"Uploaded {file.name}"
    else:
        return f"Upload failed: {res.json().get('detail', '')}"

def extract_embed(file):
    if file is None:
        return "Please upload a file first."
    files = {"file": (file.name, file, file.type)}
    res = requests.post(f"{API_URL}/extract_embed", files=files)
    if res.status_code == 200:
        paragraphs = res.json().get("paragraphs", 0)
        return f"Extraction complete! Paragraphs embedded: {paragraphs}"
    else:
        return f"Extraction failed: {res.json().get('detail', '')}"

def search(query, search_mode):
    if not query:
        return "Please enter a query."
    if search_mode == "Basic Search":
        res = requests.get(f"{API_URL}/search", params={"query": query})
        if res.status_code == 200:
            results = res.json().get("results", [])
            if not results:
                return "No relevant results found."
            output = ""
            for r in results:
                output += f"**File**: {r['file']}  \n**Page**: {r['page']}  \n**Paragraph ID**: {r['paragraph_id']}  \n**Score**: {r['score']:.4f}\n> {r['text']}\n\n---\n\n"
            return output
        else:
            return "Search failed."
    else:
        res = requests.get(f"{API_URL}/search_clustered", params={"query": query, "top_k": 10, "num_clusters": 3})
        if res.status_code == 200:
            themes = res.json().get("themes", [])
            if not themes:
                return "No themes found for this query."
            output = ""
            for theme in themes:
                output += f"### {theme['label']}\n"
                for r in theme["results"]:
                    output += f"**File**: {r['file']}  \n**Page**: {r['page']}  \n**Paragraph ID**: {r['paragraph_id']}\n> {r['text']}\n\n"
                output += "---\n\n"
            return output
        else:
            return "Search failed."

with gr.Blocks() as demo:
    gr.Markdown("# 📄 AI-Powered Document Chatbot")

    with gr.Row():
        file_input = gr.File(label="Upload a document (PDF, JPG, PNG)", file_types=[".pdf", ".jpg", ".jpeg", ".png"])
        upload_btn = gr.Button("Upload")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

    upload_btn.click(upload_file, inputs=file_input, outputs=upload_status)

    with gr.Row():
        extract_btn = gr.Button("Extract & Embed")
        extract_status = gr.Textbox(label="Extraction Status", interactive=False)

    extract_btn.click(extract_embed, inputs=file_input, outputs=extract_status)

    gr.Markdown("## Ask a question 📌")
    query_input = gr.Textbox(label="Enter your query")

    search_mode = gr.Radio(choices=["Basic Search", "Theme Clustered Search"], label="Select Search Mode", value="Basic Search")

    search_btn = gr.Button("Search")
    search_output = gr.Markdown()

    search_btn.click(search, inputs=[query_input, search_mode], outputs=search_output)

demo.launch()
