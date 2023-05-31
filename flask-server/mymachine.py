from flask import Flask, jsonify,request
from dotenv import load_dotenv
load_dotenv()
from twitchio.ext import commands
import os
import asyncio
from threading import Thread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, AutoModel, AdamW,pipeline
from transformers import BertTokenizer, BertForSequenceClassification
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
import json
import requests
import objsize
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import requests
from requests.adapters import HTTPAdapter
import pickle
import socketio as sio
from google.cloud import storage
import twitchio
import concurrent.futures
import nest_asyncio
from aiohttp import web
from aiohttp import ClientSession
from transformers import get_scheduler
from transformers import set_seed
from torch.utils.data import DataLoader
from datetime import timedelta
from http import cookies
from threading import Lock
import time
import portalocker
from torch.nn.parallel import DataParallel
# import fcntl

# logging.basicConfig(level=logging.INFO)

codes = []


# Use the output of the SimpleCookie object as an HTTP header

nest_asyncio.apply()


socketio=sio.AsyncServer(cors_allowed_origins="*")

app = web.Application()

socketio.attach(app)
# app=sio.ASGIApp(socketio)


# app=Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'

user_sessions = {}

CHANNELS=["eg103"]

bot=""
# CORS(app)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120, ping_interval=20)

# https://nightbot.tv/login/twitch/callback?code=orc3v6b0ryiosg0gd9mj2n4kenrz88&scope=clips:edit%20channel:edit:commercial%20channel:manage:broadcast%20channel:manage:redemptions%20channel:read:editors%20channel:read:redemptions%20channel:read:subscriptions%20channel:read:vips%20moderation:read%20user:read:email%20user:edit:broadcast&state=d7985a705d069277a91ed1ec4f85d530404f70ca5de8577cda884e5d1d779f65
# https://id.twitch.tv/oauth2/authorize?client_id=jwk4v6qrymwe3odjf13oaif4&response_type=code&force_verify=true&scope=clips%3Aedit%20channel%3Aedit%3Acommercial%20channel%3Amanage%3Abroadcast%20channel%3Amanage%3Aredemptions%20channel%3Aread%3Aeditors%20channel%3Aread%3Aredemptions%20channel%3Aread%3Asubscriptions%20channel%3Aread%3Avips%20moderation%3Aread%20user%3Aread%3Aemail%20user%3Aedit%3Abroadcast&state=248dbc2351e07623d04a76056fb02184c7787247e933209fb604ce4c9125c91e&redirect_uri=https%3A%2F%2Fnightbot.tv%2Flogin%2Ftwitch%2Fcallback
# https://id.twitch.tv/oauth2/authorize?client_id=q6ccgfkr2dcjbgw3ud05m2a4k11oxd&redirect_uri=http://localhost:3000/Loadingtoken&response_type=token&scope=chat:edit+chat:read+user_read+channel:moderate+moderation:read+moderator:manage:banned_users&force_verify=true%&state=248dbc2351e07623d04a76056fb02184c7787247e933209fb604ce4c9125c91e"

# Refresh the access token using the refresh token

event_happened=False
token=""

lock = Lock()
lock2=Lock()


# -----------------THIS CLASS IS TO CONVERT MY DATA TO A DATASET-----------------------------------------------------------------------------------------------
class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels=None):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels:
                item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.encodings["input_ids"])

#---------------------------------------------THIS IS USED TO TRAIN THE MODEL-----------------------------------------------------------------------------------------
def trainmodel(x, y, username,code):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model =getmodelfromcode(code)

    print("training....")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # X_train, _, y_train, _ = train_test_split(x, y, train_size=1)
    X_train=x
    y_train=y

    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=500)
        #X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=500)

    train_dataset = Dataset(X_train_tokenized, y_train)
    #val_dataset = Dataset(X_val_tokenized, y_val)

    def compute_metrics(p):
        print(type(p))
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pr8ed=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            

    # optimizer = AdamW(model.parameters(),lr=5e-5)
    args = TrainingArguments(
        output_dir=username + "/CustomModel/",
        num_train_epochs=10,
        per_device_train_batch_size=10,
        optim = "adamw_torch"
        # num_trainers=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        # data_collator=data_collator,
        train_dataset=train_dataset,
        #eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    torch.save({
            'model_state_dict': model.state_dict(),
    }, username + "/CustomModel/checkpoint.pth")
    # trainer.save_model(username+'/CustomModel')

    changemodelwithcode(code,model)


#----------------------------------FORGET ABOUT THIS PART----------------------------------------------------------------------------------------------------
    

def train_and_update_banmodel(banmodel,textdata,data,code,username):
    optimizer = torch.optim.AdamW(banmodel.parameters(),lr=4e-5)
    
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=2
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    banmodel.to(device)

    # input_ids = tokenizer.encode(str(textdata), add_special_tokens=True, truncation=True, max_length=500)
    # input_ids = torch.tensor(input_ids).unsqueeze(0)

    input_ids = torch.tensor([tokenizer.encode(str(textdata), add_special_tokens=True)])

    labels = torch.tensor([int(data)])

    banmodel.train(mode=True)
    for i in range(2):
        outputs = banmodel(input_ids, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("SAVING NOW")
    with lock:
        torch.save({
            'model_state_dict': banmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, username + "/banmodel/checkpoint.pth")
        # banmodel.save_pretrained(username+"/banmodel")


    changebanmodelwithcode(code,banmodel)
    print("CHANGED BANMODEL WITH CODE")



@socketio.on("selectedoption")
async def handleselect(sid,data, textdata,code):
    print(textdata)

    username=getusernamefromcode(code)
    banmodel=getbanmodelfromcode(code)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, train_and_update_banmodel, banmodel, textdata, data,code,username)

    await socketio.emit("finishedtrainban",room=username)
    print("TRAINED BANN")

    

# -----------------------------------USE THIS TO GET A NEW REFRESH TOKEN-------------------------------------------------------------------------------------

# response = requests.post("https://id.twitch.tv/oauth2/token", params={
#     "client_id": os.environ["CLIENT_ID"],
#     "client_secret": "moa9wyp354a19uj6einp42n7vp7krt",
#     "code": "i933x4u7n7wmkolqrtunagid3tuotz",
#     "grant_type": "authorization_code",
#     "redirect_uri": "http://localhost:3000/Loadingtoken" 
# }



# # Parse the response JSON to get the access token and refresh token.
# response_json = response.json()
# print(response_json)
# access_token = response_json["access_token"]
# refresh_token = response_json["refresh_token"]

# ----------------------------------------------------------------------------------------------------------------------------

def refresh_access_token(refresh):
    client_secret = "moa9wyp354a19uj6einp42n7vp7krt"

    token_url = "https://id.twitch.tv/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": os.environ["CLIENT_ID"],
        "client_secret": client_secret,
        "refresh_token":refresh
    }

    response = requests.post(token_url, data=data)
    print(response.json())


    new_access_token = response.json()["access_token"]
    expirestime=response.json()["expires_in"]


    return new_access_token, expirestime


access_token, expiration_time = refresh_access_token(os.environ["REFRESH_TOKEN"])

async def main():
    global access_token
    global expiration_time
    
    bot = commands.Bot(
        token=access_token,
        client_id=os.environ['CLIENT_ID'],
        nick = "testtwitchmodaicool",
        prefix="!",
        initial_channels=["eg103"]
    )


    @bot.event()
    async def event_token_expired():
        global codes

        access_token, expiration_time = refresh_access_token(os.environ["REFRESH_TOKEN"])
        CHANNELS=[code["username"] for code in codes]

        os.environ["TMI_TOKEN"]=access_token
        
        bot = commands.Bot(
            token=access_token,
            client_id=os.environ['CLIENT_ID'],
            nick = "testtwitchmodaicool",
            prefix="!",
            initial_channels=CHANNELS
        )
        await bot.connect()

    os.environ['TMI_TOKEN']=access_token
    asyncio.create_task(bot.start())

    # while True:
    #     await asyncio.sleep(expiration_time)
    
    #     access_token, expiration_time = refresh_access_token(os.environ["REFRESH_TOKEN"])
        
    #     print("NEW TOKEN")
    #     os.environ["TMI_TOKEN"]=access_token

    #     CHANNELS=[code["username"] for code in codes]

    #     bot = commands.Bot(
    #         token=access_token,
    #         client_id=os.environ['CLIENT_ID'],
    #         nick = "testtwitchmodaicool",
    #         prefix="!",
    #         initial_channels=CHANNELS
    #     )

    #     await bot.connect()


count=0




@socketio.on("disconnect")
async def test(sid):
    global count
    global bot
    count-=1

    print(socketio.rooms(sid))
    for room in socketio.rooms(sid):
        if room != sid:
            print("ROOM:" +room)
            await bot.part_channels([room])
        print(room)
        socketio.leave_room(sid, room)

    print(socketio.rooms(sid))

    # remove the user id from array if exist
    print("Amount connected: "+ str(count))
    


@socketio.on("sendbackends")
async def trainit(sid,data,code):

    print("yo")
    
    x=[]
    y=[]


    for i in range(9):
        x.append(data[i]['userstext'])
        y.append(data[i]['label'])
    
    username=getusernamefromcode(code)

    # await trainmodel(x, y, username, code)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, trainmodel, x, y, username, code)
   
    x=[]
    y=[]



def getbanmodelfromusername(username):
    banmodel=None
    for code in codes:
        if code["username"] == username:
            banmodel = code["banmodel"]
            break
    return banmodel

def getmodelfromusername(username):
    model=None
    for user in codes:
        if user["username"] == username:
            model = user["model"]
            break
    return model

def changebanmodelwithcode(code,banmodel):
    for code in codes:
        if code["code"] == code:
            code["banmodel"] = banmodel
            break


def getbanmodelfromcode(code):
    banmodel = None  # default value if code not found
    for code_dict in codes:
        if code_dict["code"] == code:
            banmodel = code_dict["banmodel"]
            break
    return banmodel


def changemodelwithcode(code,model):
    for code_dict in codes:
        if code_dict['code'] == code:
            code_dict['model'] = model
            break

def getmodelfromcode(code):
    model=None
    for code_dict in codes:
        if code_dict["code"] == code:
            model = code_dict["model"]
            break
    return model


def checkifcodealreadyin(code):
    for code_info in codes:
        if code_info['code'] == code:
            return True
    return False


def remove_dict_by_username(username):
    for d in codes:
        if d['username'] == username:
            codes.remove(d)


def change_id_by_code(code_val, new_id_val):
    for d in codes:
        if d['code'] == code_val:
            d['id'] = new_id_val
            return True  # return True if code_val is found and id is updated
    return False 

def getusernamefromcode(code):
    username=None
    for d in codes:
        if d["code"] == code:
            username = d["username"]
            break
    return username

def getuseridfromcode(code):
    id=None
    for d in codes:
        if d["code"] == code:
            id = d["id"]
            break
    return id

def getrefreshfromcode(code):
    id=None
    for d in codes:
        if d["code"] == code:
            id = d["refresh_token"]
            break
    return id
    

    

# number.index(minvalue)
# @torch.no_grad()
def predict_label(text,username):
    # input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    model=getmodelfromusername(username)
    model.eval()
    input_ids=tokenizer(text, padding=True, truncation=True, max_length=500, return_tensors="pt")
    logits = model(**input_ids)[0]
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    return probs

# @torch.no_grad()
def predict_ban(text,username):
    banmodel=getbanmodelfromusername(username)
    banmodel.eval()
    input_ids = tokenizer(text, padding=True, truncation=True, max_length=500, return_tensors="pt")
    logits = banmodel(**input_ids)[0]
    probs = torch.nn.functional.softmax(logits, dim=1)
   
    # So when chooses ban I want to train then: (bans show up as yellow) I want to show both the 10 probs and the likely prob
    return probs

# Function to get the most uncertain examples
def get_uncertain_examples(data, num_examples, code):
    
    min_chat = None
    min_maxitem = float('inf')
    username = getusernamefromcode(code)

    for item in data:
        maxitem = predict_label(item['chat'],username).max().item()
        if maxitem < min_maxitem:
            min_chat = item['chat']
            min_maxitem = predict_label(item['chat'],username).max().item()

    print(min_chat)
    return min_chat
    # This will output "asd", which is the chat with the least "maxitem" value in the input array.

# ---------------------------------------------------------------------------------------------------------------------------------------------------------



# so token expired and function didnt happen 

def get_id_from_username(username):
    for code_info in codes:
        if code_info['username'] == username:
            return code_info['id']
    return None


    
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------



def get_model(username):
    if os.path.exists(username):

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        banmodel = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=10)
        
        checkpoint2= torch.load(username+"/CustomModel/checkpoint.pth")
        model.load_state_dict(checkpoint2['model_state_dict'])

        checkpoint = torch.load(username + "/banmodel/checkpoint.pth")
        banmodel.load_state_dict(checkpoint['model_state_dict'])

    
        print("Loaded model for user", username)
        
    else:

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        banmodel = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=10)
        
        os.makedirs(username + "/CustomModel", exist_ok=True)
        os.makedirs(username + "/banmodel", exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
        }, username + "/CustomModel/checkpoint.pth")

        torch.save({
            'model_state_dict': banmodel.state_dict(),
        }, username + "/banmodel/checkpoint.pth")

        print("Created new model for user", username)

    return model, banmodel



@socketio.on("hasconnected")
async def hascon(sid,data):
    global codes
    global bot
    # must close other tabs
    username=""
    if checkifcodealreadyin(data)==False:

        async with ClientSession() as session:
            async with session.post("https://id.twitch.tv/oauth2/token", params={
                "client_id": os.environ["CLIENT_ID"],
                "client_secret": "moa9wyp354a19uj6einp42n7vp7krt",
                "code": data,
                "grant_type": "authorization_code",
                "redirect_uri": "http://localhost:3000/Loadingtoken" 
            }) as response:
                response_json = await response.json()

        # Parse the response JSON to get the access token and refresh token.
        print(response_json)

        access_token = response_json["access_token"]
        refresh_token = response_json["refresh_token"]

        headers = {
            'Authorization': f'Bearer {access_token}',
            'Client-ID': os.environ["CLIENT_ID"]
        }
        
        async with ClientSession(headers=headers) as session:
            async with session.get('https://api.twitch.tv/helix/users') as response:
                response2 = await response.json()

        username = response2['data'][0]['login']
        id=response2['data'][0]['id']

        print(username)

        # incase left and made a new code
        remove_dict_by_username(username)

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            model,banmodel= await loop.run_in_executor(pool,get_model,username)
        

        code_info={"code":data, "username":username, "id":id,"refresh_token": refresh_token, "model":model, "banmodel":banmodel}
        socketio.enter_room(sid,username)

        codes.append(code_info)
        # print(codes)

    else:
    # incase refresh
        for code_info in codes:
            if code_info['code'] == data:
                socketio.enter_room(sid,code_info['username'])
                # code_info['id'].append(sid)
                username=code_info['username']
                break
        
        print("code already here")
        # print(codes)

    # if bot.get_channel(username):
    #     print("alreadyin")
    # else:
    await bot.join_channels([username])

    await socketio.emit("urusername", username, room=username)
    

    



# botrun()

token=""

@socketio.on('connect')
async def handle_connect(sid,environ):
    global bot
    global event_happened
    global token
    global count
    
    if event_happened==False or token!=os.environ["TMI_TOKEN"]:
        token=os.environ["TMI_TOKEN"]
        event_happened=True

        print("connecting new event token ")

        CHANNELS=[code["username"] for code in codes]

        bot = commands.Bot(
            token=os.environ["TMI_TOKEN"],
            client_id=os.environ['CLIENT_ID'],
            nick = "testtwitchmodaicool",
            prefix="!",
            initial_channels=CHANNELS
        )
        
    
        @bot.event()
        async def event_message(ctx):

            probs= predict_label(ctx.content,ctx.channel.name)

            print(predict_label(ctx.content,ctx.channel.name))
            print(float(probs[0][1].item()))

            percentone=float(probs[0][0].item())
            percenttwo=float(probs[0][1].item())

            if percentone<percenttwo:
                color ="lime"
            else:
                color="red"

            theban=predict_ban(ctx.content,ctx.channel.name)
            print(theban)
            softmax_array = theban.detach().numpy()[0]
            print(softmax_array)
            banmax = theban.argmax().item()

            if color=="red":
                banprob=softmax_array
            else:
                banprob="no"

            print(ctx.channel.name)
            # print(id)

            newtwitchtext={"username": ctx.author.name,"chat":ctx.content, "color": color, "probability": str(percentone) + " " + str(percenttwo), "banprobability": str(banprob), "banmax": str(banmax)}
            # await socketio.emit("thetwitchchat",newtwitchtext, room=ctx.channel.name)
            await socketio.emit("thetwitchchat", newtwitchtext, room=ctx.channel.name)

        @bot.event()
        async def event_token_expired():
            global codes

            print("TOKEN EXPIRED")

            access_token, expiration_time = refresh_access_token(os.environ["REFRESH_TOKEN"])
            CHANNELS=[code["username"] for code in codes]

            os.environ["TMI_TOKEN"]=access_token
            
            bot = commands.Bot(
                token=access_token,
                client_id=os.environ['CLIENT_ID'],
                nick = "testtwitchmodaicool",
                prefix="!",
                initial_channels=CHANNELS
            )

        await bot.connect()


    
    
    count+=1
    print('Amount connected: '+str(count))





 

@socketio.on('ban_user')
async def baner(sid,data,code):
    global bot

    duration = data['duration']
    username = data['username']

    # print(str(duration))
    id =getuseridfromcode(code)
    urusername=getusernamefromcode(code)
    urrefreshtoken=getrefreshfromcode(code)

    uraccesstoken,time=refresh_access_token(urrefreshtoken)

    print(uraccesstoken)
    print(id)
    print(urusername)
   
    headers = {
        # for this tmi token can use to get for the bot because expire
        'Authorization': f'Bearer {os.environ["TMI_TOKEN"]}',
        'Client-ID': os.environ['CLIENT_ID']
    }

    token_url = "https://id.twitch.tv/oauth2/token"

    async with ClientSession(headers=headers) as session:
        url = "https://id.twitch.tv/oauth2/token"
        async with session.post(url, params={
            "grant_type": "refresh_token",
            "client_id": os.environ["CLIENT_ID"],
            "client_secret": os.environ["SECRET"],
            "refresh_token":urrefreshtoken
        }) as resp:
            response2 = await resp.json()

    # data = {
    #     "grant_type": "refresh_token",
    #     "client_id": os.environ["CLIENT_ID"],
    #     "client_secret": os.environ["SECRET"],
    #     "refresh_token":os.environ["REFRESH_TOKEN"]
    # }

    # response2 = requests.post(token_url, data=data)
    print(response2)

    uraccesstoken = response2["access_token"]

    user_id = id

    url = f"https://api.twitch.tv/helix/users?login={str(username)}"
        
    # response = requests.get(url, headers=headers)
    async with ClientSession(headers=headers) as session:
        url = f"https://api.twitch.tv/helix/users?login={str(username)}"
        async with session.get(url, verify_ssl=True) as resp:
            data = await resp.json()

    # data = response.json()
    ban_id = data["data"][0]["id"]

    print(ban_id)
    print(data['data'][0]['login'])

    
    partuser = bot.create_user(user_id,urusername)

    durations = {
        '1 min': 60,
        '5 mins': 300,
        '10 mins': 600,
        '30 min': 1800,
        '1 hour': 3600,
        '6 hours': 21600,
        '12 hours': 43200,
        '1 day': 86400,
        '1 week': 604800,
        'forever': 0
    }
        
        # Get the ban duration in seconds
    testdur = durations.get(str(duration))

    print(testdur)
    if duration==0:
        await partuser.ban_user(uraccesstoken,user_id,ban_id,"AI HAS BANNED U LOL NOOB")
    else:
        await partuser.timeout_user(uraccesstoken,user_id,ban_id,testdur,"AI HAS BANNED U LOL NOOB")
    

    


                                         
@socketio.on('send_data')
async def handle_send_data(sid,data,code):
    uncertain_examples = get_uncertain_examples(data, 1,code)
    username =getusernamefromcode(code)
    await socketio.emit('send_data', uncertain_examples, room=username)
    # await socketio.disconnect(sid)



model_lock = asyncio.Lock()

def train_and_update_model(model, parseddata, code, username,number):

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=2
    )

    input_ids = torch.tensor([tokenizer.encode(str(parseddata), add_special_tokens=True)])
    labels = torch.tensor([number])

    # del banmodel

    model.train(mode=True)

    for i in range(2):
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    with lock:
        # model.save_pretrained(username + "/CustomModel")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, username + "/CustomModel/checkpoint.pth")
     

    # with lock:
    #     model = BertForSequenceClassification.from_pretrained(username + '/CustomModel')

    changemodelwithcode(code, model)
    print("CHANGED MODEL WITH CODE")


@socketio.on('yes')
async def handletheyes(sid,data, mytwitch, code):

    model=getmodelfromcode(code)

    # print(mytwitcvbnh)
    
    # print(data)
    parseddata = json.loads(data)
    mytwitch = [x for x in mytwitch if x["chat"] != json.loads(data)]

    # ***************HERE REMOVE EVERY INSTACNE OF PARSED DATA ****************
    
    uncertain_examples = get_uncertain_examples(mytwitch, 1,code)

    username=getusernamefromcode(code)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, train_and_update_model, model, parseddata, code, username,1)
    
    await socketio.emit("send_data", uncertain_examples,room=username)

    print("MODEL HAS BEEN TRAINED")

    

@socketio.on("no")
async def handletheno(sid,data,mytwitch,code):

    model=getmodelfromcode(code)

    parseddata = json.loads(data)
    mytwitch = [x for x in mytwitch if x["chat"] != json.loads(data)]

    username=getusernamefromcode(code)

    uncertain_examples = get_uncertain_examples(mytwitch, 1,code)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, train_and_update_model, model, parseddata, code, username,0)
    
    await socketio.emit("send_data", uncertain_examples,room=username)

    print("MODEL HAS BEEN TRAINED")


    
    




# and then have a button that says "roam free" that just shows the chat that is bad

#get things in chat then fine tune with the bad things(use command for user to select from chat) 
# and use good things as active learning


async def timer():
    web.run_app(app, port=5000)



if __name__ =='__main__': 
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    thread1 = Thread(target=loop.run_until_complete, args=(main(),))
    thread1.start()

    asyncio.run(timer())
    



    # app.run(debug=True)
    
# first (0,1)
    

    # im thinking in the beginning give me 5 things you LOVE to see from the chat and 5 thing you HATE to see from the chat
    # If you have multiple GPUs, you can specify which one to use by passing the GPU index as an argument to the method, e.g. model.to("cuda:0").

