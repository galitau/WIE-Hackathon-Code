import re
from openai import OpenAI
from huggingface_hub import InferenceClient
from deepgram import DeepgramClient

AUDIO_URL = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"

def main():
    try:
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        response = deepgram.listen.v1.media.transcribe_url(
            url=AUDIO_URL,
            model="nova-3",
            language="en",
        )

        # Print the full response object
        print(response)

        # Or access the transcript directly
        print("\nTranscript:")
        print(response.results.channels[0].alternatives[0].transcript)

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()

# User Input
#story = input("Enter your story: ")
story = response

#  Summarize the story using a text-to-text model 
text_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="MyKey",
)

summary_prompt = (
    "Summarize this story into a clear, realistic narrative from a single person's perspective. "
    "Remove filler words and dialogue from other people. Keep it max 3 sentences. Keep only the main events and feelings: \n\n"
    f"{story}"
)

summary_response = text_client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[{"role": "user", "content": summary_prompt}],
)

# Get the cleaned summary
summary_text = summary_response.choices[0].message.content
print("\n--- Summary ---")
print(summary_text)

# Split the summary into comic panel sentences
panel_sentences = summary_text.split('. ')
# panel_sentences = [sentence.strip() for sentence in panel_sentences if sentence.strip()]

print("\n--- Comic Panels ---")
for i, sentence in enumerate(panel_sentences, 1):
    print(f"Panel {i}: {sentence}")

# Generate an image for each panel
image_client = InferenceClient(
    provider="fal-ai",
    api_key="MyKey",
)

for i, panel_text in enumerate(panel_sentences, start=1):
    image_prompt = f"Gemerate a comic book looking image that depicts the following scenario when walking to school as a kid: {panel_text}"
    image = image_client.text_to_image(
        image_prompt,
        model="black-forest-labs/FLUX.1-dev",
    )
    image_path = f"comic_panel_{i}.png"
    image.save(image_path)
    print(f"Saved {image_path}")