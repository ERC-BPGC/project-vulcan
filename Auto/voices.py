import pyttsx3

def list_and_set_female_voice():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    # List all available voices with their IDs
    for idx, voice in enumerate(voices):
        print(f"Voice {idx}: {voice.name}, ID: {voice.id}")

    # Try to find a voice with "female" in the name, or select manually
    for voice in voices:
        if "female" in voice.name.lower() or "feminine" in voice.name.lower():  # Adjust based on inspection
            engine.setProperty('voice', voice.id)
            print(f"Selected Female Voice: {voice.name}")
            break
    else:
        print("Female voice not found. Using default voice.")

    # Test the selected voice
    engine.say("Hello, I am using a female voice.")
    engine.runAndWait()

list_and_set_female_voice()
