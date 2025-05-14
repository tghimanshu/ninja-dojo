"""Synthesizes speech from the input string of text or ssml.
Make sure to be working in a virtual environment.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""

# from google.cloud import texttospeech_v1, storage, translate_v3
from google.cloud import storage

# from google.cloud import translate_v3
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from google.auth import load_credentials_from_file
from datetime import datetime, timezone, timedelta
import re
import json
import os
import shutil
import streamlit as st
import time

from moviepy import *
import math
from PIL import Image
import numpy
import glob

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

import json
import csv
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID", "dark-torch-384306")
LOCATION = os.environ.get("LOCATION", "us-central1")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "transcoder-app-streamlit")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "hackathon/")

if os.path.exists("credentials.json"):
    credentials, project = load_credentials_from_file("credentials.json")
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials,
    )
    storage_client = storage.Client.from_service_account_json("credentials.json")
else:
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
    )
    storage_client = storage.Client()

model = GenerativeModel("gemini-1.5-flash-002")
imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
bucket = storage_client.get_bucket(BUCKET_NAME)


def generate(instructions):
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
    ]
    response = model.generate_content(
        instructions,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )

    return response.text


def generate_scene_image(prompt, title, language, audio_file):
    images = imagen_model.generate_images(
        prompt=prompt,
        number_of_images=4,
        aspect_ratio="1:1",
        safety_filter_level="block_few",
        person_generation="allow_adult",
    )
    scenes_path = f"{OUTPUT_FOLDER}{title}/00/{language}/scenes"
    if not os.path.exists(scenes_path):
        os.makedirs(scenes_path)
    all_images = []
    for i, image in enumerate(images):
        image.save(f"{scenes_path}/{audio_file}-{str(i).zfill(2)}.png")
        bucket.blob(
            f"{scenes_path}/{audio_file}-{str(i).zfill(2)}.png"
        ).upload_from_filename(f"{scenes_path}/{audio_file}-{str(i).zfill(2)}.png")
        all_images.append(f"{scenes_path}/{audio_file}-{str(i).zfill(2)}.png")
    return all_images


def check_scene_images(title, language, audio_file):
    if not os.path.exists(
        f"{OUTPUT_FOLDER}{title}/00/{language}/scenes/{audio_file}-00.png"
    ):
        return False
    return f"{OUTPUT_FOLDER}{title}/00/{language}/scenes/{audio_file}-00.png"


def generate_character_images(characters, title, retries=0):
    all_characters_images = {}
    for character in characters:
        images = imagen_model.generate_images(
            prompt=character["character_design"],
            number_of_images=4,
            aspect_ratio="1:1",
            safety_filter_level="block_few",
            person_generation="allow_adult",
        )

        character_path = (
            f"{OUTPUT_FOLDER}{title}/images/characters/{character['name'].lower()}/"
        )

        if not os.path.exists(character_path):
            os.makedirs(character_path)

        character_images = []

        for i, image in enumerate(images):
            character_file = (
                f"{character_path}/{str(i).zfill(2)}{character['name'].lower()}.png"
            )
            image.save(character_file)
            bucket.blob(character_file).upload_from_filename(character_file)
            character_images.append(character_file)

        if character["name"] not in all_characters_images:
            all_characters_images[character["name"]] = character_images

    do_retry = False
    for a in all_characters_images:
        if len(all_characters_images[a]) == 0:
            do_retry = True
    if retries < 1 and do_retry:
        return generate_character_images(characters, title, retries + 1)
    return all_characters_images


def generate_image(prompt, scene_no, title, retries=0):
    images = imagen_model.generate_images(
        prompt=prompt,
        number_of_images=4,
        aspect_ratio="1:1",
        safety_filter_level="block_few",
        person_generation="allow_adult",
    )
    if not os.path.exists(f"images/{title}/{scene_no}"):
        os.makedirs(f"images/{title}/{scene_no}")
    all_images = []
    for i, image in enumerate(images):
        image.save(f"images/{title}/{scene_no}/{i}.png")
        bucket.blob(f"images/{title}/{scene_no}/{i}.png").upload_from_filename(
            f"images/{title}/{scene_no}/{i}.png"
        )
        all_images.append(f"images/{title}/{scene_no}/{i}.png")
    if len(all_images) == 0:
        if retries < 2:
            return generate_image(prompt, scene_no, title, retries + 1)
        else:
            return []
    return all_images


def get_audiobook_images(story_title, story_details):
    if not os.path.exists(f"{OUTPUT_FOLDER}{story_title}/images"):
        os.mkdir(f"{OUTPUT_FOLDER}{story_title}/images")
    for img in story_details["images"]:
        if not os.path.exists(img):
            bucket.blob(img).download_to_filename(img)


def download_chapter(chapter):
    if not os.path.exists(chapter) and bucket.blob(chapter).exists():
        bucket.blob(chapter).download_to_filename(chapter)


def get_audio_paths(story_title, chapter, language, outputType="poster"):
    # Replace with the folder containing your audio files
    audio_folder = f"{OUTPUT_FOLDER}{story_title}/{chapter}/{language.lower()}"
    # Replace with the desired output path
    output_path = f"{OUTPUT_FOLDER}{story_title}/{chapter}/{language.lower()}"

    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    if not os.path.exists(f"{audio_folder}/audio_clips"):
        os.makedirs(f"{audio_folder}/audio_clips")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if outputType == "poster":
        return audio_folder, f"{output_path}/poster.mp4"
    elif outputType == "scenes":
        return audio_folder, f"{output_path}/scenes.mp4"
    elif outputType == "veo":
        return audio_folder, f"{output_path}/veo.mp4"
    else:
        return audio_folder, f"{output_path}/poster.mp4"


def format_markdown(response_text):
    formatted_text = re.sub(r"xml", "", response_text)
    formatted_text = re.sub(r"json", "", response_text)
    formatted_text = re.sub(r"```", "", formatted_text)
    return formatted_text


def zoom_in_effect(clip, zoom_ratio=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        new_size = [
            math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
            math.ceil(img.size[1] * (1 + (zoom_ratio * t))),
        ]

        # The new dimensions must be even.
        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)

        img = img.resize(new_size, Image.LANCZOS)

        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)

        img = img.crop([x, y, new_size[0] - x, new_size[1] - y]).resize(
            base_size, Image.LANCZOS
        )

        result = numpy.array(img)
        img.close()

        return result

    return clip.transform(effect)


def merge_audio_with_image(
    image_path, audio_folder, output_path, dialogue_count, gap_ms=20
):
    """
    Merges all audio files in a folder with an image to create a video.

    Args:
        image_path: Path to the image file (e.g., "storypic.png").
        audio_folder: Path to the folder containing audio files (e.g., "orig").
        output_path: Path to save the output video file (e.g., "output.mp4").
        gap_ms: Duration of the gap between audio clips in milliseconds.
    """

    # try:
    # Load the image
    image_clips = []
    for image_clip in image_path:
        image_clips.append(ImageClip(image_clip))
    # image_clip = ImageClip(image_path)  # Initial duration, will be adjusted

    audio_clips = []
    total_audio_duration = 0

    mp3_files = []
    for i in range(0, dialogue_count):
        file_path = os.path.join(audio_folder, f"{str(i).zfill(2)}.mp3")

        if os.path.exists(file_path):
            mp3_files.append(file_path)

    # Create a list to store audio clips
    audio_clips = []

    # Add each MP3 file with a 500ms gap
    for mp3_file in mp3_files:
        audio_clip = AudioFileClip(mp3_file)
        audio_clips.append(audio_clip)
        audio_clips.append(
            AudioFileClip(mp3_file).subclipped(0, gap_ms / 1000)
        )  # 500ms silence

    # print("Audio Clips", audio_clips)
    # Concatenate all audio clips
    final_audio = concatenate_audioclips(audio_clips)

    # Set the image duration to match the total audio duration
    duration_image_clips = []
    for image_clip in image_clips:
        duration_image_clips.append(
            zoom_in_effect(
                image_clip.with_duration(final_audio.duration / len(image_clips)), 0.04
            )
        )
    # image_clip = image_clip.with_duration(final_audio.duration) #adjust image duration

    # Combine image and audio
    print("DURATION IMAGE CLIPS: ", duration_image_clips)
    video_clip = concatenate_videoclips(duration_image_clips)
    video_clip = video_clip.with_audio(final_audio)
    # video_clip = image_clip.with_audio(final_audio)

    # Write the video to a file
    video_clip.write_videofile(
        output_path, codec="libx264", audio_codec="aac", fps=24, logger=None, threads=64
    )

    print(f"Video created successfully at: {output_path}")

    # if os.path.exists(output_path):
    #     with open(output_path, "rb") as file:
    #         st.video(file.read())
    # else:
    #     time.sleep(10)
    # with open(output_path, "rb") as file:
    #     st.video(file.read())

    blob = bucket.blob(output_path)
    blob.upload_from_filename(output_path)

    print("Files", blob.path)
    path = bucket.get_blob(output_path).generate_signed_url(
        expiration=datetime.now(timezone.utc) + timedelta(minutes=30)
    )

    # return f"gs://{video_bucket_name}/{output_path}"
    return path, output_path
    # return output_path

    # except Exception as e:
    #     print(f"An error occurred: {e}")


def get_folder_path(story_title):
    return f"{OUTPUT_FOLDER}{story_title}"


# def create_audio(target_fol, dialJSON):
#     # Instantiates a client
#     client = texttospeech_v1.texttospeech_v1Client()
#     dialogueCount = 0

#     # Select the type of audio file you want returned
#     audio_config = texttospeech_v1.AudioConfig(
#         audio_encoding=texttospeech_v1.AudioEncoding.MP3
#     )
#     try:
#         if os.path.exists(target_fol):
#             shutil.rmtree(target_fol)

#         os.mkdir(target_fol)
#         data = dialJSON

#         audio_clips = []

#         for i, dialogue in enumerate(data["dialogues"]):
#             speaker = dialogue.get("speaker", "Unknown Speaker")
#             voiceName = dialogue.get("voice", "Unknown Voice")
#             language_code = dialogue.get("language_code", "Unknown Voice")
#             speech = dialogue.get("speech", "No speech provided")

#             print(
#                 f"count: {i}, Speaker: {speaker}, Voice: {voiceName},  Speech: {speech}\n"
#             )

#             voice = texttospeech_v1.VoiceSelectionParams(
#                 language_code=language_code, name=voiceName
#             )

#             # Set the text input to be synthesized
#             synthesis_input = texttospeech_v1.SynthesisInput(ssml=speech)

#             # Perform the text-to-speech request on the text input with the selected
#             # voice parameters and audio file type
#             response = client.synthesize_speech(
#                 input=synthesis_input, voice=voice, audio_config=audio_config
#             )

#             dialogueCount = i

#             audio_file = target_fol + f"/{str(i).zfill(2)}.mp3"

#             # The response's audio_content is binary.
#             with open(audio_file, "wb") as out:
#                 # Write the response to the output file.
#                 out.write(response.audio_content)
#             bucket.blob(audio_file).upload_from_filename(audio_file)

#             audio_clips.append(AudioFileClip(audio_file))

#         final_audio_file = re.sub("audio_clips", "final_audio.mp3", target_fol)
#         final_audio = concatenate_audioclips(audio_clips)
#         final_audio.write_audiofile(final_audio_file)

#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#     except KeyError as e:
#         print(f"Error: Missing key in JSON data: {e}")


# def create_multiaudio(target_fol, dialJSON):
#     try:
#         if os.path.exists(target_fol):
#             shutil.rmtree(target_fol)

#         os.mkdir(target_fol)
#         data = json.loads(dialJSON)

#         # Instantiates a client
#         client = texttospeech_v1.texttospeech_v1Client()

#         turns = []

#         for i, dialogue in enumerate(data["dialogues"]):
#             turns.append(
#                 texttospeech_v1.MultiSpeakerMarkup.Turn(
#                     text=dialogue.get("speech"),
#                     speaker=dialogue.get("voice", "Unknown Voice"),
#                 )
#             )

#         # turns = turns[6:13]
#         multi_speaker_markup = texttospeech_v1.MultiSpeakerMarkup(turns=turns)

#         # Set the text input to be synthesized
#         synthesis_input = texttospeech_v1.SynthesisInput(
#             multi_speaker_markup=multi_speaker_markup
#         )

#         # Build the voice request, select the language code ('en-US') and the voice
#         voice = texttospeech_v1.VoiceSelectionParams(
#             language_code="en-US", name="en-US-Studio-MultiSpeaker"
#         )

#         # Select the type of audio file you want returned
#         audio_config = texttospeech_v1.AudioConfig(
#             audio_encoding=texttospeech_v1.AudioEncoding.MP3
#         )

#         # Perform the text-to-speech request on the text input with the selected
#         # voice parameters and audio file type
#         response = client.synthesize_speech(
#             input=synthesis_input, voice=voice, audio_config=audio_config
#         )

#         # The response's audio_content is binary.
#         with open(target_fol + "/multi_output.mp3", "wb") as out:
#             # Write the response to the output file.
#             out.write(response.audio_content)
#             print('Audio content written to file "multi_output.mp3"')
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#     except KeyError as e:
#         print(f"Error: Missing key in JSON data: {e}")


def is_cached_content(story_title):
    details_file_path = f"{OUTPUT_FOLDER}{story_title}/details.json"

    if not os.path.exists(f"{OUTPUT_FOLDER}{story_title}"):
        os.makedirs(f"{OUTPUT_FOLDER}{story_title}")

    if os.path.exists(details_file_path):
        story_details = json.load(open(details_file_path, "r"))
        return True, story_details

    if bucket.blob(details_file_path).exists():
        data = bucket.blob(
            f"{OUTPUT_FOLDER}{story_title}/details.json"
        ).download_as_text()
        story_details = json.loads(data)
        return True, story_details
    return False, {}


def save_audiobook(story_title, story_details):
    details_file_path = f"{OUTPUT_FOLDER}{story_title}/details.json"
    open(details_file_path, "w").write(json.dumps(story_details))
    final_file = bucket.blob(details_file_path)
    final_file.upload_from_filename(details_file_path)


def get_stories():
    stories = bucket.list_blobs(prefix=OUTPUT_FOLDER)
    story_options = []
    for story in stories:
        temp = re.sub(OUTPUT_FOLDER, "", story.name)
        temp = temp.split("/")[0]
        if temp not in story_options:
            story_options.append(temp)
    return story_options


def merge_audio_with_scene_images(
    title, language, audio_folder, output_path, dialogue_count, gap_ms=20
):
    """
    Merges all audio files in a folder with an image to create a video.

    Args:
        image_path: Path to the image file (e.g., "storypic.png").
        audio_folder: Path to the folder containing audio files (e.g., "orig").
        output_path: Path to save the output video file (e.g., "output.mp4").
        gap_ms: Duration of the gap between audio clips in milliseconds.
    """

    # try:
    audio_clips = []
    image_files = []
    total_audio_duration = 0

    mp3_files = []
    for i in range(0, dialogue_count):
        file_path = os.path.join(audio_folder, f"{str(i).zfill(2)}.mp3")
        bucket.blob(file_path).upload_from_filename(file_path)
        image_path = (
            f"{OUTPUT_FOLDER}{title}/00/{language}/scenes/{str(i).zfill(2)}.mp3-00.png"
        )
        if os.path.exists(file_path):
            mp3_files.append(file_path)
            if os.path.exists(image_path):
                image_files.append(image_path)
            else:
                image_files.append(f"{OUTPUT_FOLDER}{title}/images/storypic0.png")

    # Create a list to store audio clips
    audio_clips = []
    image_clips = []

    # Add each MP3 file with a 500ms gap
    for mp3_file, img_file in zip(mp3_files, image_files):
        audio_clip = AudioFileClip(mp3_file)
        img_clip = ImageClip(img_file)
        img_clip = img_clip.with_duration(audio_clip.duration)
        img_clip = img_clip.with_effects([vfx.CrossFadeIn(1)])
        img_clip = zoom_in_effect(img_clip)

        audio_clips.append(audio_clip)
        audio_clips.append(
            AudioFileClip(mp3_file).subclipped(0, gap_ms / 1000)
        )  # 500ms silence
        image_clips.append(img_clip)

    # Concatenate all audio clips
    final_audio = concatenate_audioclips(audio_clips)
    final_img = concatenate_videoclips(image_clips)

    # Set the image duration to match the total audio duration
    # duration_image_clips = []
    # for image_clip in image_clips:
    #     duration_image_clips.append(image_clip.with_duration(final_audio.duration / len(image_clips)))
    # image_clip = image_clip.with_duration(final_audio.duration) #adjust image duration

    # Combine image and audio
    # video_clip = concatenate_videoclips(duration_image_clips)
    video_clip = final_img.with_audio(final_audio)
    # video_clip = image_clip.with_audio(final_audio)

    # Write the video to a file
    # final_audio.write_audiofile(output_path.replace("mp4", "mp3")) # Use a suitable codec like libx264
    video_clip.write_videofile(
        output_path, codec="libx264", audio_codec="aac", fps=24, logger=None, threads=64
    )

    print(f"Chapter created successfully at: {output_path}")

    # if os.path.exists(output_path):
    #     with open(output_path, "rb") as file:
    #         st.video(file.read())
    # else:
    #     time.sleep(10)
    # with open(output_path, "rb") as file:
    #     st.video(file.read())

    blob = bucket.blob(output_path)
    blob.upload_from_filename(output_path)

    print("Files", blob.path)
    path = bucket.get_blob(output_path).generate_signed_url(
        expiration=datetime.now(timezone.utc) + timedelta(minutes=30)
    )

    # return f"gs://{video_bucket_name}/{output_path}"
    return path, output_path
    # return output_path

    # except Exception as e:
    #     print(f"An error occurred: {e}")


# def translate_text(
#     text: str = "YOUR_TEXT_TO_TRANSLATE",
#     language_code: str = "fr",
# ) -> translate_v3.TranslationServiceClient:
#     """Translating Text from English.
#     Args:
#         text: The content to translate.
#         language_code: The language code for the translation.
#             E.g. "fr" for French, "es" for Spanish, etc.
#             #neural_machine_translation_model
#             Available languages: https://cloud.google.com/translate/docs/languages
#     """

#     client = translate_v3.TranslationServiceClient()
#     parent = f"projects/{PROJECT_ID}/locations/global"
#     # Translate text from English to chosen language
#     # Supported mime types: # https://cloud.google.com/translate/docs/supported-formats
#     response = client.translate_text(
#         contents=[text],
#         target_language_code=language_code,
#         parent=parent,
#         mime_type="text/plain",
#         source_language_code="en-US",
#     )

#     # Display the translation for each input text provided
#     for translation in response.translations:
#         print(f"Translated text: {translation.translated_text}")
#     # Example response:
#     # Translated text: Bonjour comment vas-tu aujourd'hui?

#     return response


def download_video(gcs_path, local_path):
    print(" ----------- ")
    print(gcs_path)
    print(local_path)
    bucket.blob(gcs_path).download_to_filename(local_path)
