from conversation import get_speech_input, text_to_speech, process_input
import time


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