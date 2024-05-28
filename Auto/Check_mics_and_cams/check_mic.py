import speech_recognition as sr

# Get a list of available microphones
microphones = sr.Microphone.list_microphone_names()

# Print the list of microphones
for i, mic_name in enumerate(microphones):
    print(f"{i + 1}. {mic_name}")
