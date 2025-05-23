import os
import requests
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from crewai import Agent, Task, Crew
from typing import List, Dict

# Configuration
os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"  # Replace with your actual key

# Initialize Ollama
try:
    llm = OllamaLLM(model="llava:13b", num_gpu=1)
    analyst_llm = ChatOpenAI(model="ollama/qwen2.5:latest", base_url="http://localhost:11434", api_key="ollama")
except Exception as e:
    st.error(f"Failed to initialize Ollama: {str(e)}")
    st.error("Make sure Ollama is running and the model is downloaded (ollama pull llava:13b)")
    st.stop()

# Set up Streamlit
st.set_page_config(page_title="Image Similarity Search", layout="wide")
st.title("🔍 Image Similarity Search Engine")

def save_uploaded_file(uploaded_file):
    """Save uploaded file in the current directory with its original name and extension."""
    try:
        file_name = uploaded_file.name
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        print(f"Saved image to: {file_path}")  # Debug logging
        return file_name  # Return only the filename and extension
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None


def analyze_image_with_llava(file_name: str) -> str:
    """Analyze image with Llava model using file path."""
    
    # Try file path first
    template_file = """
    You are an expert image analysis model. Analyze the provided image and describe its content in detail, focusing on the following aspects:
    - **Objects**: Identify the primary objects, including their shapes, textures, and specific features (e.g., "a red pleated skirt" instead of "clothing").
    - **Background**: Describe relevant background elements that contribute to the scene’s context (e.g., indoor setting, outdoor landscape).
    - **Persons**: Note the presence of people, specifying if they appear to be men or women (if identifiable) and their count.
    - **Body Posture**: Describe the posture or pose of any persons (e.g., standing, sitting, walking).
    - **Colors**: Highlight dominant colors of objects, clothing, and background.

    Provide a concise and accurate description suitable for generating search keywords to find visually similar images.

    Image path: {image_path}

    Return a structured description addressing each aspect above. Avoid including irrelevant details unless they are critical to the scene.
"""
    
    print(f"Analyzing image with LLaVA using file path: {file_name}")
    prompt = ChatPromptTemplate.from_template(template=template_file)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"image_path": f"./{file_name}"})
        print(f"LLaVA raw response (file path): {response}")  # Debug logging
        # Check if response is relevant
        if not response or "failed" in response.lower() or len(response.strip()) < 20:
            st.warning("Image analysis with file path produced an unclear result")
        else:
            return response.strip()
    except Exception as e:
        st.warning(f"Error analyzing image with file path: {str(e)}")

def search_similar_images(query: str) -> List[Dict]:
    """Search for similar images using SerperAPI."""
    url = "https://google.serper.dev/images"
    payload = {"q": query, "num": 10}
    headers = {
        "X-API-KEY": os.environ["SERPER_API_KEY"],
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("images", [])
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

def generate_search_keywords(image_description: str) -> str:
    """Generate search keywords using CrewAI agents."""
    if image_description == "Analysis failed" or "failed" in image_description.lower():
        return "Failed to generate keywords"
    
    try:
        print(f"Image description for keyword generation: {image_description}")
        analyst = Agent(
            role='Senior Image Analyst',
            goal='Generate the most effective search terms to find similar images',
            backstory="""You're an expert at analyzing visual content and creating search queries
            that will return the most relevant similar images. You understand color theory,
            composition, and visual patterns.""",
            allow_delegation=False,
            llm=analyst_llm,
            verbose=True
        )

        keyword_task = Task(
            description=f"""Analyze this image description and generate the best search queries
            to find visually similar images online. Provide 3-5 specific search terms that
            would return the most relevant results. Focus on the main objects and their specific features (e.g., "red pleated skirt" instead of "clothing").
            
            Image Description:
            {image_description}""",
            agent=analyst,
            expected_output="A bulleted list of 3-5 specific search queries for finding similar images."
        )

        crew = Crew(
            agents=[analyst],
            tasks=[keyword_task],
            verbose=True
        )

        result = crew.kickoff()
        if hasattr(result, 'tasks_output') and result.tasks_output:
            output = result.tasks_output[0].raw
        else:
            output = str(result)
        print(f"Generated keywords: {output}")
        return output
    except Exception as e:
        st.error(f"Keyword generation failed: {str(e)}")
        return "Failed to generate keywords"

def display_results(original_image, description: str, keywords: str, similar_images: List[Dict]):
    """Display results in a structured layout."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("Image Analysis")
        st.text_area("Description", value=description, height=200, disabled=True)
        st.subheader("Suggested Search Keywords")
        st.write(keywords)
    
    st.subheader("Similar Images Found")
    if not similar_images:
        st.warning("No similar images found. Try different keywords.")
        return
    
    cols = st.columns(4)
    for idx, img in enumerate(similar_images[:8]):
        try:
            img_url = img.get("imageUrl", "")
            if img_url:
                with cols[idx % 4]:
                    st.image(img_url, use_container_width=True)
                    st.caption(img.get("title", "")[:50] + "...")
        except Exception as e:
            st.error(f"Error displaying image {idx}: {str(e)}")
            continue

def main():
    uploaded_file = st.file_uploader("Upload an image to find similar photos", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded file to the current directory with original name
        file_name = save_uploaded_file(uploaded_file)
        if not file_name:
            st.error("Failed to save image. Please try again.")
            return
        
        try:
            with st.spinner("Analyzing image..."):
                uploaded_file.seek(0)
                description = analyze_image_with_llava(file_name)
                
                if "failed" in description.lower():
                    manual_keywords = st.text_input("Image analysis failed. Enter search keywords manually:", "similar image")
                    keywords = manual_keywords
                    keyword_list = [manual_keywords]
                else:
                    keywords = generate_search_keywords(description)
                    keyword_list = [k.strip().lstrip('-').strip() for k in keywords.split("\n") if k.strip().startswith('-')]
                
                with st.spinner("Searching for similar images..."):
                    search_query = keyword_list[0] if keyword_list else "similar image"
                    similar_images = search_similar_images(search_query)
                
                display_results(uploaded_file, description, keywords, similar_images)
        finally:
            # Optionally remove the file after processing
            # os.remove(os.path.join(os.getcwd(), file_name))
            pass

if __name__ == "__main__":
    main()