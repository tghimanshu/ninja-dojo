# Future Plan (Phase 2)

This document outlines the proposed enhancements and features for the next phase of the Vistar project. Having completed Phase 1 (Core Functionality & Documentation), Phase 2 will focus on scalability, user experience, and advanced content capabilities.

## 1. User Experience & Interface Enhancements

*   **Interactive Storyboard Editor**: Allow users to manually edit the generated storyboard JSON directly in the UI before generating images. This includes drag-and-drop reordering of scenes.
*   **Prompt Engineering UI**: meaningful UI controls to tweak generation parameters (temperature, token limit, style presets) without modifying code.
*   **Progress Tracking**: Implement more granular progress bars for long-running operations like video generation and batch image processing.
*   **Gallery View**: A dedicated gallery page to browse, filter, and download previously generated personas, storyboards, and videos.

## 2. Advanced AI Capabilities

*   **Custom Model Fine-tuning**: Explore fine-tuning Gemini or Imagen models on Tata Motors' specific brand assets and tone of voice for more consistent on-brand generation.
*   **Audio Generation Integration**: Fully integrate the Text-to-Speech (TTS) capabilities (currently present in helper functions but not fully utilized in the main flow) to generate voiceovers for the videos automatically.
*   **Multi-Language Support**: Implement the translation features (commented out in code) to support generating marketing content in multiple languages automatically.

## 3. Infrastructure & Backend

*   **Asynchronous Task Queue**: Move long-running generation tasks (video rendering) to a background worker queue (e.g., Celery or Cloud Tasks) to prevent the Streamlit interface from freezing or timing out.
*   **Database Integration**: Replace the current file-based storage (local JSONs and GCS blobs) with a structured database (e.g., Firestore or PostgreSQL) to manage user sessions, project history, and metadata more robustly.
*   **User Authentication**: Implement proper user authentication (e.g., Google OAuth) to secure the application and allow for multi-user support with private workspaces.

## 4. Code Quality & CI/CD

*   **Unit & Integration Tests**: Develop a comprehensive test suite (using `pytest`) to cover all helper functions and API integrations.
*   **CI/CD Pipeline**: Set up a pipeline (e.g., GitHub Actions or Cloud Build) to automatically run tests and deploy the Streamlit app to Cloud Run or App Engine upon pushing changes.
*   **Configuration Management**: Move hardcoded configuration (like Project IDs) entirely to environment variables or a secure secret manager.

## 5. Analytics

*   **Usage Tracking**: Implement analytics to track which features are used most, generation success rates, and API cost estimation per user/project.
