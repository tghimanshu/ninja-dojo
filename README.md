# Vistar - Your Marketing Companion

Vistar is a Streamlit-based application developed for Tata Motors. It leverages the power of Generative AI to assist marketing teams in creating detailed customer personas, storyboards, scene images, and high-quality videos using Google's Veo and Imagen models.

## Purpose

The primary goal of Vistar is to streamline the marketing content creation process. By automating the generation of personas and visual assets, it allows marketing professionals to focus on strategy and creativity. Specifically, it focuses on Tata Motors' Platform Agnostic TCU offerings.

## Features

*   **Persona Generation**: Automatically generates detailed customer personas based on input descriptions, including profile name, needs, pain points, and motivators.
*   **Storyboard Creation**: Creates a structured storyboard with scene descriptions based on the generated persona and a core narrative.
*   **Scene Image Generation**: Generates high-quality images for each scene in the storyboard using Google's Imagen model.
*   **Video Generation**: Converts text prompts and scene images into videos using Google's Veo model (Imagen 2/3 and Veo 2.0).
*   **Audio/Video Processing**: Utilities for merging audio with images, creating zoom effects, and handling video outputs.

## Setup

### Prerequisites

*   Python 3.8+
*   A Google Cloud Platform (GCP) project with Vertex AI and Cloud Storage enabled.
*   `gcloud` CLI installed and authenticated.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration:**
    *   Create a `.env` file in the root directory (or ensure environment variables are set) with the following:
        ```
        PROJECT_ID=your-project-id
        LOCATION=us-central1
        BUCKET_NAME=your-gcs-bucket-name
        OUTPUT_FOLDER=your-output-folder-path/
        ```
    *   Ensure you have a `credentials.json` file for a Service Account with appropriate permissions (Vertex AI User, Storage Object Admin) in the root directory, or ensure your environment is authenticated via `gcloud auth application-default login`.

## Usage

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

2.  **Using the Interface:**
    *   **Persona Generation**: Enter persona details and click "Generate Persona".
    *   **Storyboard Generation**: Enter a core narrative and click "Generate STORYBOARD".
    *   **Scene Image Generation**: Review the scenes and click "Generate Scene Image" to visualize the storyboard.
    *   **Video Generation**: (Optional/Advanced) Use the "Generate Video" button to create videos from the selected scenes and images.

## Project Structure

*   `app.py`: The main entry point for the Streamlit application. Handles UI and workflow logic.
*   `utils/helper_funcs.py`: Contains utility functions for API calls (Vertex AI, GCS), image/video processing (MoviePy), and other helper tasks.
*   `utils/constants.py`: Stores prompt templates and other constant values used for AI generation.
*   `images/`: Directory for storing generated images (locally).
*   `requirements.txt`: List of Python dependencies.

## Technologies Used

*   **Streamlit**: For the web user interface.
*   **Google Vertex AI**: For text generation (Gemini), image generation (Imagen), and video generation (Veo).
*   **Google Cloud Storage**: For storing generated assets.
*   **MoviePy**: For programmatic video editing and processing.
