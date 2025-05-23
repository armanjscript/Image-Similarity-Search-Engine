# Image Similarity Search Engine

## Description

This project is an Image Similarity Search Engine built with Python, utilizing [Streamlit](https://streamlit.io/) for the user interface and integrating with AI models via [Ollama](https://ollama.com/) and [CrewAI](https://www.crewai.com/) for advanced image analysis and keyword generation. The application allows users to upload an image, analyze its content, generate relevant search keywords, and find visually similar images online using the [SerperAPI](https://serper.dev/). It demonstrates the power of combining local AI models with web APIs to create a seamless user experience for image-based searches, making it easier for users to find what they're looking for without needing to manually craft search queries.

## Technologies Used

- **Python**: The primary programming language for the project.
- **Streamlit**: A Python library for building interactive web applications.
- **Ollama**: A framework for running local AI models, such as Llava and Qwen, for image analysis and keyword generation.
- **CrewAI**: A library for creating AI agents to automate tasks, such as generating search keywords.
- **SerperAPI**: An external API for performing image searches based on generated keywords.
- **Langchain**: Used for prompt templates and output parsers to structure AI interactions.

## Installation

To run this project, you need to have Python installed on your machine, along with Ollama for running local AI models. Additionally, you'll need a SerperAPI key for image search functionality.

1. **Install Python and necessary libraries:**
   - Ensure you have Python 3.8 or higher installed.
   - Install required libraries using pip:
     ```bash
     pip install streamlit requests langchain-core langchain-ollama crewai
     ```
   - Note: The project uses Langchain for prompt templates and output parsers, but it does not require `langchain-openai` since it uses Ollama locally.

2. **Set up Ollama:**
   - Install Ollama from their [official website](https://ollama.com/) or using your package manager.
   - Download the required models (`llava:13b` and `qwen2.5`) using Ollama:
     ```bash
     ollama pull llava:13b
     ollama pull qwen2.5
     ```
   - Start Ollama if it's not already running:
     ```bash
     ollama serve
     ```

3. **Obtain SerperAPI Key:**
   - Sign up for a SerperAPI account at [SerperAPI](https://serper.dev/) and get your API key.

4. **Set Environment Variables:**
   - Set the `SERPER_API_KEY` environment variable with your API key.
     - On Linux/Mac:
       ```bash
       export SERPER_API_KEY=your_api_key_here
       ```
     - On Windows:
       ```bash
       set SERPER_API_KEY=your_api_key_here
       ```

**Note:** Running AI models locally can be resource-intensive. Ensure your machine has sufficient CPU, GPU, and memory to handle the workload.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/image-similarity-search-engine.git
   ```

2. Navigate to the project directory:
   ```bash
   cd image-similarity-search-engine
   ```

3. Run the application:
   ```bash
   streamlit run main.py
   ```

4. Upload an image in the Streamlit interface to see the similar images found online.

## Features

- Upload an image and get a detailed description using AI.
- Automatically generate search keywords based on the image content.
- Find and display visually similar images from the web.
- User-friendly interface built with Streamlit.
- Handles errors gracefully, with fallback options for manual keyword input.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. If you encounter any issues or have suggestions for improvement, feel free to open an issue or reach out.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or feedback, please contact me at [your_email@example.com].