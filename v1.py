from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from asr import takeCommand
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import playsound
import torch
import wikipedia
import os
import processor
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
import wikipedia as wiki
from transformers import pipeline

def question_answer(query):
    model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

    tokenizer.encode(query, truncation=True, padding=True)

    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    context = wiki.summary(query)


    ans = nlp({
        'question': query,
        'context': context
    })

    return ans




url = "https://api.au-syd.text-to-speech.watson.cloud.ibm.com/instances/22d50268-b127-44fc-b4f0-200cf587e53d"
apikey = "uTtX5kOZgL4CRYi6cgNnHYuNxwUT-7M885La5xZ9dBa7"

def speak(text):
    authenticator = IAMAuthenticator(apikey)
    tts = TextToSpeechV1(authenticator=authenticator)
    tts.set_service_url(url)

    with open('./speech.mp3', 'wb') as audio_file:
        res = tts.synthesize(text, accept='audio/mp3', voice='en-GB_JamesV3Voice').get_result()
        audio_file.write(res.content)


        playsound.playsound('speech.mp3')


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

def jarvis_response(query):

    for step in range(1):
        # encode the new user input, add the eos_token and return a tensor in Pytorch

        new_user_input_ids = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)


        response = " {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))


    return str(response)

def wolfram(query):
    client = 'WJU3L2-QAGA3PJGEU'
    question = query
    client = wolframalpha.Client('WJU3L2-QAGA3PJGEU')
    res = client.query(question)
    answer = next(res.results).text

    return answer



def respond(query):
    response = ""

    pred = processor.chatbot_response(query)

    if processor.chatbot_response(query) == "QA":
        response = question_answer(query)

    elif processor.chatbot_response(query) == "Essay":
        response = "essay"


    elif processor.chatbot_response(query) == "Wolfram":
        response = wolfram(query)
        print(response)

    elif processor.chatbot_response(query) == "other":
        response = jarvis_response(query)

    else:
        response = processor.chatbot_response(query)

    return response



while True:
    transcript = takeCommand()

    speak(respond(transcript))
