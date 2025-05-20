import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Document Chatbot", layout="centered")
st.title("📄 AI-Powered Document Chatbot")

# File upload
st.header("1. Upload a document")
uploaded_file = st.file_uploader("Choose a file (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Upload"):
        with st.spinner("Uploading..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            res = requests.post(f"{API_URL}/upload", files=files)

        if res.status_code == 200:
            st.success(f"Uploaded {uploaded_file.name}")
        else:
            st.error(res.json().get("detail", "Upload failed"))

# Extract + Embed
if uploaded_file:
    if st.button("Extract & Embed"):
        with st.spinner("Extracting and embedding..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            res = requests.post(f"{API_URL}/extract_embed", files=files)

        if res.status_code == 200:
            st.success(f"Extraction complete! Paragraphs embedded: {res.json()['paragraphs']}")
        else:
            st.error(res.json().get("detail", "Extraction failed"))

# Search interface
st.header("2. Ask a question 📌")
query = st.text_input("Enter your query:")

# Add option to select search type
search_type = st.radio("Select Search Mode:", ("Basic Search", "Theme Clustered Search"))

if query and st.button("Search"):
    with st.spinner("Searching..."):
        if search_type == "Basic Search":
            res = requests.get(f"{API_URL}/search", params={"query": query})
        else:
            res = requests.get(f"{API_URL}/search_clustered", params={"query": query, "top_k": 10, "num_clusters": 3})

    if res.status_code == 200:
        data = res.json()

        if search_type == "Basic Search":
            results = data.get("results", [])
            if not results:
                st.info("No relevant results found.")
            else:
                st.subheader("Top Results")
                for result in results:
                    st.markdown(f"""
                    **File**: {result['file']}  
                    **Page**: {result['page']}  
                    **Paragraph ID**: {result['paragraph_id']}  
                    **Score**: {result['score']:.4f}

                    > {result['text']}
                    ---
                    """)
        else:
            themes = data.get("themes", [])
            if not themes:
                st.info("No themes found for this query.")
            else:
                st.subheader("Themes Found")
                for theme in themes:
                    st.markdown(f"### {theme['label']}")
                    for result in theme["results"]:
                        st.markdown(f"""
                        **File**: {result['file']}  
                        **Page**: {result['page']}  
                        **Paragraph ID**: {result['paragraph_id']}

                        > {result['text']}
                        """)
                    st.markdown("---")
    else:
        st.error("Search failed.")
