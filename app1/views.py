from multiprocessing import AuthenticationError
from django.shortcuts import redirect, render
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse
from django.contrib.auth import authenticate,login
from django.contrib.auth.forms import AuthenticationForm
import openai
from django.conf import settings
from app1.backends import EmailBackend

import os
import sys
import json

from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI 
from langchain.chat_models import ChatOpenAI

#api_key="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"
# os.environ["OPENAI_API_KEY"] = "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"
#query = sys.argv[1]


# openai.api_key="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"
"""
def ask_openai(message):
    response=openai.Completion.create(
        model="text-davinci-003",
        prompt=message,
        max_tokens=150,
        n=1,
        stop=None,
        tempetature=0.7,
    )
    answer=response.choices[0].text.strip()
    return answer

bot = ChatBot('chatbot', read_only=False, logic_adapters=['chatterbot.logic.BestMatch'])"""


    





# Create your views here.
def home(request):
    if request.method=="POST":
        username=request.POST['username']
        email=request.POST['email']
        password=request.POST['pass']

        myuser=User.objects.create_user(username,email,password)
        myuser.save()
        messages.success(request,"your account has been created")
        return redirect('home')
    
    return render(request, "home.html")

def s(request):
    return render(request, "s.css")


def register(request):
    if request.method=="POST":
        user=request.POST['username']
        email=request.POST['email']
        password=request.POST['pass1']
        if User.objects.filter(username=user).exists():
            messages.error(request, "Username already exists.")
            return redirect('register')
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect('register')

        try:
            myuser = User.objects.create_user(user, email, password     )
            myuser.save()
            messages.success(request, "Your account has been created successfully.")
            return redirect('login')
        except Exception as e:
            messages.error(request, f"Error creating account: {e}")
            return redirect('register')
    else:
        return render(request, "register.html")

def chat(request):
    return render(request, "chat.html")
 
backend = EmailBackend()
def login_user(request):
    if request.method=="POST":
        email=request.POST.get('email')
        pass1=request.POST.get('pass1')
        print(email,pass1)

        user=backend.authenticate(request, email=email, password=pass1)
        print(user)
        if user is not None:
            login(request,user)
            messages.error(request, f"successfully logged in")
            return redirect('chat')
        else:
             messages.error(request, f"Email and password incorrect ")
            
    return render(request, "login.html")

def specific(request):
    return HttpResponse("list1")


'''loader = DirectoryLoader(".", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])'''


# Load data from intents.json
intents_file_path = os.path.join(settings.BASE_DIR, 'app1', 'data', 'intents.json')

with open(intents_file_path, 'r') as file:
    intents_data = json.load(file)
    


# Function to check if the query matches any pattern in intents
# def get_intent_response(query):
#     for intent in intents_data['intents']:
#         if any(pattern in query for pattern in intent['patterns']):
#             return intent['responses']
#     return HttpResponse("please check your internet connection")

import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

def preprocess_text(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    tokens=word_tokenize(text)
    tokens=[word for word in tokens if word not in stop_words]
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return ''.join(tokens)

for intent in intents_data['intents']:
    intent['patterns']=[preprocess_text(pattern) for pattern in intent['patterns']]


# Function to check if the query matches any pattern in intents
def get_intent_response(query):
    preprocessed_query=preprocess_text(query)
    for intent in intents_data['intents']:
        if any(preprocessed_query in pattern for pattern in intent['patterns']):
            return intent['responses']
        return HttpResponse("Sorry!...for the inconvineance please do contact our college phone.no: 08256-236961,236621 between 9am to 5pm or you can also visit our college website <a target=\"_blank\" href=\"https://sdmit.in/\">here</a>")
    return HttpResponse("sorry! I am not able to understand please rephrase it.")
# Function to get response from ChatGPT
def get_gpt_response(query,index):
    return index.query(query, llm=ChatOpenAI())

# Get the query from command line arguments


# Get response from intents

def chat_response(prompt):
    # response=openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role":"user","content":prompt}]
    # )
    query = prompt
    intent_response = get_intent_response(query)

    if intent_response:
        print(intent_response)
        return intent_response
    # else:
    #     # Load index for ChatGPT
    #     loader = DirectoryLoader(".", glob="*.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])
        
    #     # Get response from ChatGPT
    #     gpt_response = get_gpt_response(query,index)
    #     print(gpt_response)


    #     return gpt_response

    # texts = []
    # for intent in intents_data['intents']:
    #     texts.extend(intent['patterns'])

   
    # return index.query(prompt, llm=ChatOpenAI())


#Load index for ChatGPT
loader = DirectoryLoader(".", glob="*.txt")
#index = VectorstoreIndexCreator().from_loaders([loader])

'''def get_intent_response(query):
    for intent in intents_data['intents']:
        if any(pattern in query for pattern in intent['patterns']):
            return intent['responses']
    return None

# Function to get response from ChatGPT
def get_gpt_response(query):
    return index.query(query, llm=ChatOpenAI())

# Function to get combined response
def get_combined_response(query):
    intent_response = get_intent_response(query)
    if intent_response:
        return intent_response
    else:
        gpt_response = get_gpt_response(query)
        return gpt_response'''

def getResponse(request):

    userMessage = request.GET.get('userMessage')
    if userMessage.lower() in ["quit","exit","bye"]:
        return HttpResponse("Thank you")
    
    # Save the user query to the database
    query = UserQuery(query_text=userMessage)
    query.save()
    
    response=chat_response(userMessage)

    # response += intents_data
       
    return HttpResponse(response)


from django.views.decorators.csrf import csrf_exempt
from .models import UserQuery

@csrf_exempt
def getResponse(request):
    userMessage = request.GET.get('userMessage')

    if userMessage.lower() in ["quit","exit","bye"]:
        return HttpResponse("Thank you")

    # Save the user query to the database
    query = UserQuery(query_text=userMessage)
    query.save()

    # Generate a response (this is a simple example, replace with actual chatbot logic)
    response = chat_response(userMessage)


    return HttpResponse(response)


