import React, { useState,useEffect } from 'react'
import { Routes, Route } from 'react-router-dom';
import "./App.css"
import "./index.js"
import io from 'socket.io-client';
import Cookies from 'js-cookie';

// import {connectSocket, disconnectSocket, getsocket} from "./socket.js"
const socket=io("http://localhost:5000", {autoConnect:false},  {query:{x:42}})


// const socket= getsocket()

function App(){
  socket.connect()
  
  const [inputData, setInputData]=useState('')
  const [isTrue, setIsTrue] = useState(true);
  const [count, setCount] = useState(1);
  const [thearray, setArray] = useState([]);
  const [prompt, setprompt] = useState("Please enter something in chat");
  const [chat, setchat] = useState([]);
  const [twitchchat, settwitchchat]= useState([])
  const [username,setusername]=useState("Loading and initializing...")
  const [banmodelbool, setbanmodelbool]= useState(true)
  const [modelbool, setmodelbool]= useState(true)


  let isalreadyconnect=false
  
  useEffect(() => {

    socket.on('send_data', data => {
        console.log(data)
        setprompt('"'+ data+ '"')
        document.getElementById("tgood").style.display="inline-block"
        document.getElementById("tbad").style.display="inline-block"
    });
    
    socket.on("connect", (data) => {
      // console.log("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
      console.log("test")
      if(!isalreadyconnect){
        socket.emit("hasconnected", Cookies.get("mycode"))
      }

      isalreadyconnect=true


    });

    socket.on("disconnect",()=>{
      // alert("a")
    })

    socket.on("urusername",(data)=>{
      setusername(data)
    })
    
    socket.on("finishedtrainban",()=>{
      setbanmodelbool(true)
    })

    socket.on("finishedtrain",()=>{
      setmodelbool(true)
    })
  
    
    // const intervalId = setInterval(requestData, 4000);
    

    const sendDataToBackend = (data) => {
      socket.emit("sendbackends", data,Cookies.get("mycode"))
    }

    if(count==10){
      sendDataToBackend(thearray)
      alert("now training ur model")
      let progressBar = document.querySelector(".progress-bar");
      let progressContainer = document.querySelector(".progress-container");  
      let progressPercent = 10;
      progressBar.style.width = progressPercent + "%";
      progressBar.style.backgroundColor = "#a86ff7";
      setCount(1)
    }
  
    
    socket.on("thetwitchchat", (data)=>{
      console.log(data)
      if (twitchchat.length < 40) {
        settwitchchat([...twitchchat, data]);
      } else {
        // If length exceeds MAX_TWITCH_CHAT, remove first element using slice method
        settwitchchat([...twitchchat.slice(1), data]);
      }
    })

    return () => {
      // clearInterval(intervalId);
    };

  });
  
  if(!Cookies.get("mycode")){
    window.location.href="http://localhost:3000/login"
  }



  const activelab = (e)=>{
    document.getElementById("myform").style.display="none"
    document.getElementById("secondpart").style.display="block"
    document.getElementById("twitchchat").style.display="none"
  }

  const twitchch = (e)=>{
    document.getElementById("myform").style.display="none"
    document.getElementById("secondpart").style.display="none"
    document.getElementById("twitchchat").style.display="block"
  }

  const badgood=(e)=>{
    document.getElementById("myform").style.display="block"
    document.getElementById("secondpart").style.display="none"
    document.getElementById("twitchchat").style.display="none"
  }
  
  const justfornow = (e)=>{
    e.preventDefault()
    socket.emit("send_data",twitchchat,Cookies.get("mycode"))
  }

  
  function selectOption(data,index) {
    if(banmodelbool==true){
      setbanmodelbool(false)
      const dropdown = document.getElementById(`myDropDown-${index}`);
      const selectedOption = dropdown.options[dropdown.selectedIndex].value;
      document.getElementById(`myspan-${index}`).style.display="none"
      socket.emit("selectedoption", selectedOption, data, Cookies.get("mycode"))
    }else{
      alert("Model is still training.....")
    }
  }


  const thisyes = (e)=>{
    e.preventDefault()
    const realprompt = prompt.substring(1, prompt.length - 1);
    socket.emit("yes", prompt, twitchchat, Cookies.get("mycode"))
    settwitchchat(prevChat => prevChat.filter(chatObj => chatObj.chat !== realprompt));
    document.getElementById("tgood").style.display="none"
    document.getElementById("tbad").style.display="none"
    console.log(twitchchat)
  }

  const thisno = (e)=>{
    e.preventDefault()
    const realprompt = prompt.substring(1, prompt.length - 1);
    socket.emit("no",prompt,twitchchat, Cookies.get("mycode"))
    settwitchchat(prevChat => prevChat.filter(chatObj => chatObj.chat !== realprompt));
    document.getElementById("tgood").style.display="none"
    document.getElementById("tbad").style.display="none"
    console.log(twitchchat)
  }
  
  
  const handleSubmit = (event) => {
      event.preventDefault()
      setCount(count+1)


      let progressBar = document.querySelector(".progress-bar");
      let progressPercent = (count+1) * 10;
      progressBar.style.width = progressPercent + "%";
      progressBar.style.backgroundColor = "#a86ff7";

      if(isTrue){

        document.getElementById("myH1").innerHTML = "BAD THING";
        document.getElementById("myH1").style.color="red"
        document.getElementById("greenthetext").id="redthetext"
        setArray([... thearray, {userstext: inputData, label: 1}])
        setIsTrue(false)
      } else{

        document.getElementById("redthetext").id="greenthetext"
        document.getElementById("myH1").innerHTML = "GOOD THING";
        document.getElementById("myH1").style.color="lime"
        setArray([... thearray,{userstext: inputData, label: 0}])
        setIsTrue(true)
      }        
  };

  function findindex(valueToFind){

    const dropdown = document.getElementById("myDropDown");
    let selectedOptionText=-1 

    let myarr=["1 min","5 mins","10 mins","30 mins","1 hour","6 hours","12 hours","1 day","1 week", "forever"]

    return myarr[valueToFind]

  }

  function handleBanClick(username, duration) {
    
    socket.emit('ban_user', { username, duration }, Cookies.get("mycode"));
  }

  const bc=new BroadcastChannel('logout')
  bc.addEventListener("message",(event)=>{
      if(event.data=="logout"){
        document.cookie = 'mycode=; expires=Thu, 01 Jan 1970 00:00:01 GMT';
        window.location.href="http://localhost:3000/login"
      }
  })

  const logout =()=>{
    document.cookie="mycode=; expires=Thu, 01 Jan 1970 00:00:01 GMT"
    bc.postMessage('logout');
    // socket.emit("removearraywiththiscode", Cookies.get("mycode"))
    // nmight store codes that will not get used for a long time because will never log out
    window.location.href="http://localhost:3000/login"
    
  }


  

  
  

  return (
    // <html>
    <div>
      <div id='box'>
        <button class="navbut" onClick={activelab} style={{color:"black"}}>Active Labelling</button>
        <button class="navbut" onClick={twitchch} style={{color:"purple"}}> Twitch Chat</button>
        <button class="navbut" id="half-red-half-green" onClick={badgood} >Bad and Good</button>
        <button onClick={logout}>
          logout
        </button>
        <div style={{color:"white"}}>
          {username}
        </div>
      </div>

      
      <form onSubmit={handleSubmit} class="body" id='myform'>
        <div class="progress-container">
          <div class="progress-bar"></div>
        </div>
        <br></br>
        <br></br>

        <h1 id='myH1'>GOOD THING</h1>
        <br></br>
        <label id ="mylabel">
          <input
            type="text"
            value={inputData}
            id="greenthetext"
            onChange={(e) => setInputData(e.target.value)}
           placeholder="Your Phrase"/>
        </label>
       
        
        <br></br>
        <br></br>
        <div >
          {count} /10
        </div>
      </form>
      <form id = "secondpart">
        <br></br>
        <br></br>
          <button onClick={justfornow} style={{fontSize:"1.2em", width:"200px"}}>Get The Uncertain</button>
          <br></br>
          <br></br>
          
          <h1 id='disprompt'>{prompt}</h1>
          <br></br>
          <br></br>
          <div id='yingyang'>
            <button onClick={thisyes} style={{marginRight:"10px"}} id = "tgood" class="button-73">GOOD</button>
            <button onClick={thisno} style={{marginLeft:"10px"}} id ="tbad" class="button-73">BAD</button>
          </div>
      </form>
      <form id='twitchchat'>
        <br></br>
        <h1 style={{color:"white"}} id="myH1">TWITCH CHAT HERE</h1>
        <br></br>
        <div id = "thetext" style={{ overflow: 'auto', maxHeight: '72.5vh', margin:"auto",borderColor:"black"}}>
          {twitchchat.map((chatObj, index) => (
    <div key={index} id="individualchat" style={{ color: chatObj.color}}>

      <p style={{ color: 'white', display: 'inline-block',margin: 0 }}>"{chatObj.username}"  </p>:  {chatObj.chat} {chatObj.probability}

      {chatObj.color==='red' &&
      <span>
        <span id={`myspan-${index}`}>
            <select id={`myDropDown-${index}`} style={{marginLeft:"10px"}}>
              <option value="0">1 min</option>
              <option value="1">5 mins</option>
              <option value="2">10 mins</option>
              <option value="3">30 mins</option>
              <option value="4">1 hours</option>
              <option value="5">6 hours</option>
              <option value="6">12 hours</option>
              <option value="7">1 day</option>
              <option value="8">1 week</option>
              <option value="9">forever</option>
            </select>
              
            <button type="button" style={{marginLeft:"10px"}} onClick={()=>selectOption(chatObj.chat, index)}>SELECT BAN </button>
          
            <div style={{color:"yellow"}}>
                {chatObj.banprobability}
            </div>

            <div style={{display: "flex" ,justifyContent: "center"}}>
              <div style={{color:"greenyellow"}}>
                  {findindex(chatObj.banmax)}
              </div>

              <button type="button" style={{marginLeft:"10px", backgroundColor:"red", color:"black", borderColor:"white"}} onClick={()=>{
                handleBanClick(chatObj.username, findindex(chatObj.banmax))
              }}>
                  actually ban?
              </button>
            </div>
        </span> 

          
        </span>
      }
      
      <hr></hr>
    </div>
    ))}
        </div>
      </form>
     
    </div>
  );
}
export {App}
