import speech_recognition as sr
import os
from gtts import gTTS
import playsound
import numpy as np
import time
import openai
import pyttsx3

# Set OpenAI API key
openai.api_key = ''


def text_to_speech(text):
    tts = gTTS(text=text, lang='en')  
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    

def get_speech_input():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please speak:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand what you said.")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""


def process_input(input_text):
    try:
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": input_text}],
            max_tokens=50
        )
        output_text = response.choices[0].message['content'].strip()
        return output_text
    except Exception as e:
        print("Error occurred while processing input:", e)
        return "Sorry, I encountered an error."



def main():
    hi_detected = False
    
    while True:
    
        user_input = get_speech_input()
        if not user_input:
            continue
        
        if not hi_detected:
            if "hi" in user_input.lower():
                hi_detected = True
                hi_response = "Hi! How can I assist you today?"
                print("System:", hi_response)
                text_to_speech(hi_response)
                continue
            else:
                print("Please say 'hi' to start the conversation.")
                continue

        processed_output = process_input(user_input)
        print("GPT output:", processed_output) 
        text_to_speech(processed_output)
       
        if "bye" in user_input.lower():
           
            break

        time.sleep(2)

if __name__ == "__main__":
    main()
