import { json } from 'react-router-dom';
import io from 'socket.io-client';
import React, { useState,useEffect } from 'react'
import "./App.css"
import Cookies from 'js-cookie';


export function Loadingtoken(){

    const [time, settimeout]=useState(0)
    document.body.style.backgroundColor="black"

    const hash = window.location.hash;

    

    useEffect(()=>{
        // const socket=io("http://localhost:5000")

        const queryString = window.location.search;
        const urlParams = new URLSearchParams(queryString);
        const code = urlParams.get('code');

        Cookies.set('mycode', code);

        // socket.emit("bottoken", code)

        window.location.href="http://localhost:3000"

        // socket.on("redirect",()=>{
        //     Cookies.set('mycode', code);
        //     socket.disconnect()
        // })

    
        
        
     

        // return()=>{
        //     socket.disconnect()
        // }

    })
    

    return(
   
        <h1 class="rainbow rainbow_text_animated">.......</h1>

    )

}