import streamlit as st
from utils.helper_funcs import generate, generate_image
from utils.constants import PERSONA_PROMPT, STORYBOARD_PROMPT, SCENE_PROMPT
import time
import json
import os
import google.auth
import google.auth.transport.requests
import subprocess
import requests

PROJECT_ID = "dark-torch-384306"  # @param {type:"string"}
video_model = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/veo-2.0-generate-001"
prediction_endpoint = f"{video_model}:predictLongRunning"
fetch_endpoint = f"{video_model}:fetchPredictOperation"

st.title("Vistar - Your Marketing Companion")
st.write(
    "Welcome to Vistar, your marketing companion for Tata Motors. This app is designed to help you generate detailed customer personas based on the provided information. Please enter the details below to get started."
)

persona_json = {
    "profile_name": "Mr. Vikram Sharma",
    "potential_vehicle_fit": "Tata Motors 1518 Tipper Truck with Platform Agnostic TCU",
    "core_needs": "Reliable and efficient transportation for construction materials, real-time vehicle tracking and data analysis for improved logistics and investment optimization,  robust security features to protect his assets.",
    "business_type": "Construction and Infrastructure materials supply and trading;  Significant private investments in various sectors.",
    "pain_points": "High fuel costs, lack of real-time visibility into fleet operations leading to inefficiencies,  difficulty in managing driver behavior and maintenance schedules, concerns about vehicle security and theft, and lack of data-driven insights for optimizing business decisions.",
    "motivators": "Improved operational efficiency, reduced fuel consumption, enhanced security for his assets,  data-driven insights for better investment decisions,  strong brand reputation of Tata Motors and the advanced features of the Platform Agnostic TCU (especially its integration capabilities with existing business software).  Return on investment (ROI) from improved efficiency and reduced operational costs.",
    "created_at": "2024-10-27T10:30:00Z",
    "created_by": "A. Kapoor",
}
title = st.text_input("Enter the title of the video:", "Vikram Sharma")
persona_details = st.text_area("Enter the persona details here:", height=200)

if st.button("Generate Persona"):
    if persona_details:
        with st.spinner("Generating persona..."):
            response = generate(PERSONA_PROMPT.format(persona_details=persona_details))
            st.success("Persona generated successfully!")
            st.write(response)
        persona_json = response
    else:
        st.warning("Please enter the persona details before generating.")
else:
    st.write(persona_json)


core_narrative = st.text_area("Enter the core narrative here:", height=200)

scenes = [
    {
        "scene": 1,
        "duration": 7,
        "video_prompt": "Open on Vikram Sharma (40s, confident, slightly stressed) in his office reviewing a delivery schedule on a tablet.  The screen shows late deliveries and increased fuel consumption.  A worried expression crosses his face.",
    },
    {
        "scene": 2,
        "duration": 6,
        "video_prompt": "Quick cuts showing various perspectives of a Tata 1518 Tipper truck on a busy highway. One shot shows a driver speeding; another shows aggressive overtaking.  A final shot shows the truck arriving late at a construction site, causing delays.",
    },
    {
        "scene": 3,
        "duration": 7,
        "video_prompt": "Vikram is now looking at a dashboard on his tablet displaying data from the Tata Motors Platform Agnostic TCU.  The dashboard shows real-time location, speed, and driver behavior. He notices the speeding incident and harsh braking.  His expression shifts to one of understanding and relief.",
    },
    {
        "scene": 4,
        "duration": 7,
        "video_prompt": "Close-up on the tablet screen, highlighting the TCU's features: speed alerts, harsh braking notifications, and location tracking. A voiceover explains how these features improve safety, reduce fuel consumption, and increase efficiency.",
    },
    {
        "scene": 5,
        "duration": 7,
        "video_prompt": "Vikram is now smiling, looking confident. He's reviewing improved delivery times and reduced fuel costs on his tablet.  The background subtly shows a Tata Motors logo.",
    },
    {
        "scene": 6,
        "duration": 7,
        "video_prompt": "A montage of shots showing the Tata 1518 Tipper truck driving smoothly and safely, arriving on time, and delivering goods efficiently.  Upbeat background music plays.",
    },
    {
        "scene": 7,
        "duration": 7,
        "video_prompt": 'Vikram shakes hands with a satisfied construction site manager.  Text overlay: "Tata Motors Platform Agnostic TCU: Driving Safety, Efficiency, and ROI."  The Tata Motors logo appears prominently.',
    },
]

if st.button("Generate STORYBOARD"):
    with st.spinner("Generating storyboard..."):
        response = generate(
            STORYBOARD_PROMPT.format(
                persona_details=persona_json, core_narrative=core_narrative
            ),
        )
        st.success("Storyboard generated successfully!")
        st.write(
            STORYBOARD_PROMPT.format(
                persona_details=persona_json, core_narrative=core_narrative
            ),
        )
        st.write(response)
    scene = response
else:
    st.write(scenes)

if st.button("Generate Scene Image"):
    for i, scene in enumerate(scenes):
        with st.spinner(f"Generating scene {i + 1}..."):
            st.write(f"Scene {i + 1} details: {scene.get('video_prompt')}")
            response = generate_image(
                SCENE_PROMPT.format(
                    scene_details=json.dumps(scene.get("video_prompt"))
                ),
                i,
                title,
            )
        st.success("Scene image generated successfully!")
        for img in response:
            st.image(img, caption=f"Generated Scene Image {i + 1}")
else:
    for i in range(0, 7):
        st.write(f"Scene {i + 1} images:")
        st.write(f"Scene {scenes[i]['video_prompt']}")
        st.image(f"images/{i}/0.png")
        st.image(f"images/{i}/1.png")
        st.image(f"images/{i}/2.png")
        st.image(f"images/{i}/3.png")
selected_scenes = [
    {
        "scene": 1,
        "image": 1,
        "video_prompt": "Open on Vikram Sharma (40s, confident, slightly stressed) in his office reviewing a delivery schedule on a tablet.  The screen shows late deliveries and increased fuel consumption.  A worried expression crosses his face.",
    },
    {
        "scene": 2,
        "image": 2,
        "video_prompt": "Quick cuts showing various perspectives of a Tata 1518 Tipper truck on a busy highway. One shot shows a driver speeding; another shows aggressive overtaking.  A final shot shows the truck arriving late at a construction site, causing delays.",
    },
    {
        "scene": 3,
        "image": 1,
        "video_prompt": "Vikram is now looking at a dashboard on his tablet displaying data from the Tata Motors Platform Agnostic TCU.  The dashboard shows real-time location, speed, and driver behavior. He notices the speeding incident and harsh braking.  His expression shifts to one of understanding and relief.",
    },
    {
        "scene": 4,
        "image": 2,
        "video_prompt": "Close-up on the tablet screen, highlighting the TCU's features: speed alerts, harsh braking notifications, and location tracking. A voiceover explains how these features improve safety, reduce fuel consumption, and increase efficiency.",
    },
    {
        "scene": 5,
        "image": 1,
        "video_prompt": "Vikram is now smiling, looking confident. He's reviewing improved delivery times and reduced fuel costs on his tablet.  The background subtly shows a Tata Motors logo.",
    },
    {
        "scene": 6,
        "image": 1,
        "video_prompt": "A montage of shots showing the Tata 1518 Tipper truck driving smoothly and safely, arriving on time, and delivering goods efficiently.  Upbeat background music plays.",
    },
    {
        "scene": 7,
        "image": 1,
        "video_prompt": 'Vikram shakes hands with a satisfied construction site manager.  Text overlay: "Tata Motors Platform Agnostic TCU: Driving Safety, Efficiency, and ROI."  The Tata Motors logo appears prominently.',
    },
]


def send_request_to_google_api(api_endpoint, data=None):
    """
    Sends an HTTP request to a Google API endpoint.

    Args:
        api_endpoint: The URL of the Google API endpoint.
        data: (Optional) Dictionary of data to send in the request body (for POST, PUT, etc.).

    Returns:
        The response from the Google API.
    """

    # Get access token calling API
    creds, project = google.auth.default(quota_project_id=PROJECT_ID)
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    access_token = creds.token

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(api_endpoint, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


def compose_videogen_request(
    prompt,
    image_uri,
    gcs_uri,
    seed,
    aspect_ratio,
    sample_count,
    enable_prompt_rewriting,
):
    instance = {"prompt": prompt}
    if image_uri:
        instance["image"] = {"gcsUri": image_uri, "mimeType": "png"}
    request = {
        "instances": [instance],
        "parameters": {
            "storageUri": gcs_uri,
            "sampleCount": sample_count,
            "seed": seed,
            "aspectRatio": aspect_ratio,
            "enablePromptRewriting": enable_prompt_rewriting,
        },
    }
    return request


def fetch_operation(lro_name):
    request = {"operationName": lro_name}
    # The generation usually takes 2 minutes. Loop 30 times, around 5 minutes.
    for i in range(30):
        resp = send_request_to_google_api(fetch_endpoint, request)
        if "done" in resp and resp["done"]:
            return resp
        time.sleep(10)


def text_to_video(prompt, seed, aspect_ratio, sample_count, output_gcs, enable_pr):
    req = compose_videogen_request(
        prompt, None, output_gcs, seed, aspect_ratio, sample_count, enable_pr
    )
    resp = send_request_to_google_api(prediction_endpoint, req)
    print(resp)
    return fetch_operation(resp["name"])


def image_to_video(
    prompt, image_gcs, seed, aspect_ratio, sample_count, output_gcs, enable_pr
):
    req = compose_videogen_request(
        prompt, image_gcs, output_gcs, seed, aspect_ratio, sample_count, enable_pr
    )
    resp = send_request_to_google_api(prediction_endpoint, req)
    print(resp)
    return fetch_operation(resp["name"])


def show_video(op, input_file, prompt):
    print(op)
    my_path = input_file.split("/")[-1].split(".")[0]
    if not os.path.exists("output/" + my_path + "/" + prompt + "/"):
        os.makedirs("output/" + my_path + "/" + prompt + "/")
    if op["response"]:
        for video in op["response"]["videos"]:
            gcs_uri = video["gcsUri"]
            file_name = gcs_uri.split("/")[-1]
            # open(f"{i}.txt", "w").write(gcs_uri + "\n")
            # !gsutil cp {gcs_uri} {file_name}
            print(
                "google",
                "storage",
                "cp",
                gcs_uri,
                f"output/{my_path}/{prompt}/{file_name}",
            )
            path = f"output/tm/{gcs_uri.split('tm/')[1]}"
            subprocess.call(["gcloud", "storage", "cp", gcs_uri, path])

            # media.show_video(media.read_video(file_name), height=500)


inputs = [
    {
        "prompt": f"""
        As a professional ad video, create high speed motion of people moving around in style.
        {scene['video_prompt']}
        """,
        "image_gcs": f"gs://ninja-dojo/images/{scene['scene']}/{scene['image']}.png",
        "seed": 7,
        "aspect_ratio": "16:9",
        "sample_count": 4,
        "output_gcs": f"gs://ninja-dojo/output/ss/{scene['scene']}/",
        "rewrite_prompt": True,
    }
    for scene in selected_scenes
]

if st.button("Generate Video"):
    for input in inputs:
        op = image_to_video(
            input["prompt"],
            input["image_gcs"],
            input["seed"],
            input["aspect_ratio"],
            input["sample_count"],
            input["output_gcs"],
            input["rewrite_prompt"],
        )
        show_video(op, input["image_gcs"], input["prompt"])

    # ref: https://www.gumlet.com/learn/ffmpeg-extract-frames/
    # ffmpeg -i sample_1.mp4 -c:v png output_frame%04d.png
