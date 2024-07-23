import streamlit as st
import tempfile
import os
import zipfile
import torch
from io import BytesIO
from pydub import AudioSegment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, AudioFileClip
import typing as T

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


# Constants
MaxFileSize = 200000000  # max file size allowed: 200MB
MaxAudioDuration = 10  # max audio duration allowed: 10 seconds


def streamlit_header():
    st.title('Video Maker')


def streamlit_footer():
    st.write('Made by [Katana Iryna](https://www.linkedin.com/in/iryna-katana/)')


def upload_video_file():
    """
    Allow the user to upload a video file and validate its size.

    Returns:
        uploaded_file: The uploaded video file if valid, else None.
    """
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], disabled=st.session_state.get('processing', False))
    if uploaded_file is not None:
        if uploaded_file.size > MaxFileSize:
            st.write("File size too large. Please upload a file smaller than 200MB")
            return None
    return uploaded_file


def select_clip_count(video_path):
    """
    Allow the user to select the number of clips to split the video into.

    min_value is the duration of the video divided by MaxAudioDuration
    max_value is the duration of the video
    """
    duration = VideoFileClip(video_path).duration
    min_clips = int(duration / MaxAudioDuration)
    clip_count = st.number_input("Enter the number of clips to split the video into", min_value=min_clips, max_value=int(duration), value=min_clips, help="The number of clips can’t be more than the video duration and each clip duration is less than 10 seconds")
    return clip_count


def select_clip_for_audio(clip_count):
    """
    Allow the user to select which clip to add the generated audio track to.

    max_value is the number of clips
    """
    clip_number = st.number_input("Enter the clip number to add the generated audio track", min_value=1, max_value=clip_count, value=1)
    return clip_number


def select_column_count(clip_count):
    """
    Allow the user to select the number of columns to display the clips.

    If the number of clips is less than 3, the default number of columns is 1.
    Else, the default number of columns is 3.
    """
    default_columns = 3 if clip_count >= 3 else 1
    column_count = st.number_input(
        "Enter the number of columns to display clips",
        min_value=1,
        max_value=clip_count,
        value=default_columns
    )
    return column_count


def select_device():
    """
    Allow the user to select the compute device to use (CPU, CUDA, MPS).

    Returns:
        device: The selected compute device.
    """
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"

    device_options = ["cuda", "cpu", "mps"]
    device = st.selectbox(
        "Device",
        options=device_options,
        index=device_options.index(default_device),
        help="Which compute device to use. CUDA is recommended.",
    )
    assert device is not None

    return device


def split_video(video_path, clip_count, output_dir):
    """
    Split the video into the specified number of clips.

    Args:
        video_path: Path to the uploaded video file.
        clip_count: Number of clips to split the video into.
        output_dir: Directory to save the split clips.

    Returns:
        clips: List of paths to the split video clips.
    """
    video = VideoFileClip(video_path)
    duration = video.duration
    clip_duration = duration / clip_count
    clips = []
    
    for i in range(clip_count):
        start_time = i * clip_duration
        end_time = (i + 1) * clip_duration if (i + 1) * clip_duration < duration else duration
        output_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)
        clips.append(output_path)
    
    return clips


def convert_duration_to_ms(duration_seconds):
    """
    Convert duration from seconds to milliseconds and round up to nearest multiple of 8.

    Args:
        duration_seconds: Duration in seconds.

    Returns:
        duration_ms: Duration in milliseconds, rounded up to nearest multiple of 8.
    """
    duration_ms = int(duration_seconds * 100)
    if duration_ms % 8 != 0:
        duration_ms = ((duration_ms // 8) + 1) * 8
    return duration_ms


def generate_audio(inputs, duration, output_dir):
    """
    Generate audio based on the provided inputs and parameters.

    Args:
        inputs: Dictionary containing input parameters for audio generation.
        duration: Duration of the audio in seconds.
        output_dir: Directory to save the generated audio.

    Returns:
        file_path: Path to the generated audio file.
    """
    extension = "wav"
    checkpoint = "riffusion/riffusion-model-v1"

    # Extract parameters from inputs
    prompt = inputs['prompt']
    negative_prompt = inputs['negative_prompt']
    starting_seed = inputs['starting_seed']
    device = inputs['device']
    num_inference_steps = inputs['num_inference_steps']
    guidance = inputs['guidance']
    scheduler = inputs['scheduler']
    use_20k = inputs['use_20k']

    # Set spectrogram parameters based on use_20k
    if use_20k:
        params = SpectrogramParams(
            min_frequency=10,
            max_frequency=20000,
            sample_rate=44100,
            stereo=True,
        )
    else:
        params = SpectrogramParams(
            min_frequency=0,
            max_frequency=10000,
            stereo=False,
        )

    seed = starting_seed
    duration = convert_duration_to_ms(duration)  # Convert duration to milliseconds

    st.write(f"#### Generating Riff with Seed-{seed}")

    # Generate spectrogram image from text prompt
    image = streamlit_util.run_txt2img(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance=guidance,
        negative_prompt=negative_prompt,
        seed=seed,
        width=duration,
        height=512,
        checkpoint=checkpoint,
        device=device,
        scheduler=scheduler,
    )

    st.image(image)  # Display the generated image

    # Convert spectrogram image to audio segment
    segment = streamlit_util.audio_segment_from_spectrogram_image(
        image=image,
        params=params,
        device=device,
    )

    file_path = os.path.join(output_dir, f"generated_audio.wav")
    
    with open(file_path, "wb") as f:
        segment.export(f, format=extension)
    
    return file_path    


def add_audio_to_clip(video_path, audio_path):
    """
    Add the generated audio to the specified video clip.

    Args:
        video_path: Path to the video clip.
        audio_path: Path to the generated audio file.

    Returns:
        output_path: Path to the final video with the added audio.
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    audio_segment = AudioSegment.from_wav(audio_path)

    # Trim the audio segment to match the exact video duration
    audio_segment = audio_segment[:int(video_clip.duration * 1000)]

    otput_path = audio_path.replace(".wav", "_new.wav")
    audio_segment.export(otput_path, format="wav")

    # Load the new audio clip
    audio_clip = AudioFileClip(otput_path)

    # Combine the video and audio clips
    final_clip = video_clip.set_audio(audio_clip)

    output_path = video_path.replace(".mp4", "_with_audio.mp4")
    final_clip.write_videofile(output_path)

    video_clip.close()
    audio_clip.close()
    final_clip.close()
    return output_path


def create_zip(clips):
    """
    Create a ZIP file containing the provided video clips.

    Args:
        clips: List of paths to video clips.

    Returns:
        zip_buffer: In-memory ZIP file buffer containing the video clips.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for clip in clips:
            with open(clip, 'rb') as f:
                zf.writestr(os.path.basename(clip), f.read())
    zip_buffer.seek(0)
    return zip_buffer


def main():
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    streamlit_header()

    uploaded_file = upload_video_file()
    
    if uploaded_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())  # Save the uploaded file to a temporary directory
            st.write(f"Uploaded file: {uploaded_file.name}")

            clip_count = select_clip_count(video_path)
            column_count = select_column_count(clip_count)
            clip_number = select_clip_for_audio(clip_count)
            with st.form("Inputs"):
                
                prompt = st.text_input("Prompt", key="prompt", help="Enter a text prompt for generating the audio.")

                with st.expander("Additional Parameters"):
                    negative_prompt = st.text_input("Negative Prompt", help="Specify a negative prompt to avoid certain outputs.")

                    row = st.columns(4)
                    starting_seed = T.cast(
                        int,
                        row[0].number_input(
                            "Seed",
                            value=42,
                            help="Change this to generate different variations.",
                        ),
                    )
                    device = select_device()
                    num_inference_steps = T.cast(int, st.number_input("Inference Steps", value=50, help="Number of steps for the model to iterate over."))
                    guidance = st.number_input("Guidance", value=7.0, help="How much the model listens to the text prompt.")
                    scheduler = st.selectbox(
                        "Scheduler",
                        options=streamlit_util.SCHEDULER_OPTIONS,
                        index=0,
                        help="Which diffusion scheduler to use."
                    )
                    assert scheduler is not None

                    use_20k = st.checkbox("Use 20kHz", value=False, help="Generate audio with a higher frequency range.")

                submit_button = st.form_submit_button(f"Split video and Generate audio for {clip_number} clip", type="primary")

            inputs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "starting_seed": starting_seed,
                "device": device,
                "num_inference_steps": num_inference_steps,
                "guidance": guidance,
                "scheduler": scheduler,
                "use_20k": use_20k
            }

            if submit_button:
                if not prompt:
                    st.error("Prompt is required.")
                    return    
                
                st.session_state.processing = True

                with st.spinner("Splitting video..."):
                    clips = split_video(video_path, clip_count, temp_dir)  # Split the video into clips

                duration = VideoFileClip(clips[clip_number-1]).duration
                audio_path = generate_audio(inputs, duration, temp_dir)  # Generate audio for the selected clip
                if audio_path:
                    with st.spinner("Adding audio to the selected clip..."):
                        final_clip_path = add_audio_to_clip(clips[clip_number-1], audio_path)  # Add audio to the selected clip
                        clips[clip_number-1] = final_clip_path
                        st.success("Audio added successfully!")   

                st.write(f"Generated clips (clip №{clip_number} contains generated audio):")
                cols = st.columns(column_count)

                for i, clip in enumerate(clips):
                    with cols[i % column_count]:
                        st.video(clip)  # Display the video clips
                        
                zip_buffer = create_zip(clips)  # Create a ZIP file containing the video clips
                st.download_button("Download All Clips as ZIP", zip_buffer, file_name="clips.zip")

                st.session_state.processing = False

    streamlit_footer()

if __name__ == '__main__':
    main()
