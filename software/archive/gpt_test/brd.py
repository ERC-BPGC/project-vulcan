import os
from Bard import Chatbot

# token = os.environ.get('bard')

token = ""

bot = Chatbot(token)

while True:
    #query = input("Enter prompt : ")
    query = "hi"
    if query == "quit":
        break

    output = bot.ask(query)['content']
    print(output)
