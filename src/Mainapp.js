import {Routes} from "react-router-dom"
import { Route } from "react-router-dom";

import {App} from './App';
import { Login } from './Login'
import {Loadingtoken} from './Loadingtoken'
import "./App.css"

export default function Mainapp(){
    return(
        <div>
        <Routes>
            <Route exact path="/" element={<App />}/>
            <Route exact path="/login" element={<Login />}/>  
            <Route path="/Loadingtoken" element={<Loadingtoken />}/>  
        </Routes>
        
       </div>
        
    )
}

