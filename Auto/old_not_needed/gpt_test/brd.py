import os
from Bard import Chatbot

# token = os.environ.get('bard')

token = "WwgUKHp2_WOi3nApv48FZ099HacULOchHt5J-PhvSNHA11tQ7EZ2WIVZlMgn9HUtmh3K_Q."

bot = Chatbot(token)

while True:
    #query = input("Enter prompt : ")
    query = "hi"
    if query == "quit":
        break

    output = bot.ask(query)['content']
    print(output)
