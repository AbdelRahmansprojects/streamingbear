import io from "socket.io-client";

let socket;

export const connectSocket = () => {
  socket = io("http://localhost:5000");
  // Add event listeners or emit events here
};

export const disconnectSocket = () => {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
};

export const getsocket=()=>{
    return socket
}