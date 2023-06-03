
import React, { useEffect, useState } from "react";
import myImage from './attachment_125459655.jpg';
import Cookies from 'js-cookie';
const CryptoJS = require('crypto-js');


export function Login(){

    function generateRandomState() {
        var stateArray = CryptoJS.lib.WordArray.random(16);
        return stateArray.toString(CryptoJS.enc.Hex);
    }
      
    var state = generateRandomState();

    document.body.style.backgroundColor = "white";

    const [accessToken, setAccessToken] = useState(null);

    const logi=new BroadcastChannel('login')

    logi.addEventListener("message",(event)=>{
        if(event.data=="login"){
            window.location.href="http://streamingbear.up.railway.app"
        }
    })

    if (Cookies.get("mycode")) {
        logi.postMessage("login")
        window.location.href="http://streamingbear.up.railway.app"
        
    }

    const login = (e)=>{
        e.preventDefault()
        if(!Cookies.get("mycode")){
            window.location.href=`https://id.twitch.tv/oauth2/authorize?client_id=q6ccgfkr2dcjbgw3ud05m2a4k11oxd&redirect_uri=http://streamingbear.up.railway.app/Loadingtoken&response_type=code&scope=chat:edit+chat:read+user_read+channel:moderate+moderation:read+moderator:manage:banned_users&force_verify=true&state=${state}`
        }else{
            window.location.href="http://streamingbear.up.railway.app"
        }
    }  

    // https://id.twitch.tv/oauth2/authorize?client_id=q6ccgfkr2dcjbgw3ud05m2a4k11oxd&redirect_uri=http://localhost:3000/Loadingtoken&response_type=token&scope=chat:edit+chat:read+user_read+channel:moderate+moderation:read+moderator:manage:banned_users&force_verify=true
    // alright to change url do what night bot did which is bring to another page then form that page get the url and send to backend and also change link to be localhost:3000

    return(
    <div>
         
         <img src={myImage}></img>
        {/* <button onClick={login}>Login with twitch</button> */}
        <div class="wrap">
            <button class="button" onClick={login}>Login with twitch</button>
        </div>
      
    </div>
    )
}
