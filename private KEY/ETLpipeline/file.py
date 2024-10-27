import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

huggingface_api_token = os.getenv("huggingface_api")
if huggingface_api_token is None:
    raise ValueError("The API token was not found. Please check your .env file.")

device = -1  # Use -1 for CPU, or set to 0 for GPU

# Initialize the summarization pipeline
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    device=device
)

# Function to chunk text
def chunk_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_length:  # Adjust as necessary for your tokenization method
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to summarize large text
def summarize_large_text(text, summarizer):
    chunks = chunk_text(text, max_length=512)  # Adjust max_length based on model constraints
    intermediate_summaries = []

    # Summarize each chunk
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        intermediate_summaries.append(summary[0]['summary_text'])

    # Combine intermediate summaries
    combined_summary = " ".join(intermediate_summaries)

    # Optionally summarize the combined summary for brevity
    final_summary = summarizer(combined_summary, max_length=130, min_length=30, do_sample=False)
    return final_summary[0]['summary_text']

# Streamlit app layout
st.title("YouTube Video Transcript Summarizer")
st.write("Enter a YouTube video URL to get the summarized transcript.")

video_url = st.text_input("YouTube Video URL:")

if st.button("Get Summary"):
    if video_url:
        # Extract the video ID from the URL
        video_id = video_url.split("v=")[-1]
        if "&" in video_id:
            video_id = video_id.split("&")[0]

        try:
            # Get the transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            formatted_text = " ".join([entry["text"] for entry in transcript])

            # Generate the summary
            summary = summarize_large_text(formatted_text, summarizer)

            # Display the summary
            st.subheader("Summary:")
            st.write(summary)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid YouTube video URL.")
