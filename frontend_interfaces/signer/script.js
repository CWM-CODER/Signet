const FLASK_SERVER_URL = "http://127.0.0.1:5000/predict";
const RESET_URL = "http://127.0.0.1:5000/reset";
const SUPABASE_URL = "https://hhhbpzitdosieuafbfmc.supabase.co";
const SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhoaGJweml0ZG9zaWV1YWZiZm1jIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkwMTMwMjksImV4cCI6MjA4NDU4OTAyOX0.3sZPV5-tWIAfxP6V8QEzGii2NOogfWyfmpX4fZfqgP0";

const MY_SECRET_CODE = "Hello MST";

const video = document.getElementById('video');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const letterDisplay = document.getElementById('predictedLetter');
const bufferDisplay = document.getElementById('wordBuffer');
const chatBox = document.getElementById('chat-box');
const jsonDisplay = document.getElementById('json-display');
const modeButton = document.getElementById('modeToggle');


let lastSentMessage = "";
let currentMode = "letters";
let isSecure = false;

let supabaseClient = null;
try {
    if (window.supabase) {
        supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);
        console.log("Supabase initialized.");
    } else {
        console.warn("Supabase library not loaded.");
    }
} catch (e) {
    console.error("Supabase Init Error:", e);
}


if (bufferDisplay) {
    bufferDisplay.innerText = "🔒 LOCKED. Waiting for Receiver...";
    bufferDisplay.style.color = "orange";
}

const handshakeInterval = setInterval(async () => {
    if (!supabaseClient) return;

    const timeCheck = new Date(Date.now() - 10000).toISOString();

    const { data } = await supabaseClient
        .from('access_codes')
        .select('*')
        .gt('created_at', timeCheck)
        .order('created_at', { ascending: false })
        .limit(1);

    if (data && data.length > 0) {
        const receivedCode = data[0].code_text;
        console.log("Receiver entered:", receivedCode);

        if (receivedCode === MY_SECRET_CODE) {
            console.log("🔓 Code Matched! Starting App...");
            clearInterval(handshakeInterval);
            isSecure = true;

            startApp();

        } else {
            console.warn("Incorrect Code Attempt:", receivedCode);
            if (bufferDisplay.innerText !== "❌ INCORRECT CODE") {
                alert("❌ Connection Failed: Receiver entered incorrect code!");
                bufferDisplay.innerText = "❌ INCORRECT CODE";
                bufferDisplay.style.color = "red";
            }
        }
    }
}, 2000);
function startApp() {

    if (bufferDisplay) {
        bufferDisplay.innerText = "CONNECTED SECURELY";
        bufferDisplay.style.color = "#00ff00";
    }

    setupCamera();

    if (supabaseClient) {
        console.log("Listening for messages...");

        supabaseClient.channel('public:conversation_logs')
            .on('postgres_changes', {
                event: 'INSERT',
                schema: 'public',
                table: 'conversation_logs'
            }, (payload) => {
                console.log("New DB Entry:", payload.new);
                displayMessage(payload.new);
            })
            .subscribe();
    }

    setInterval(function () {
        if (!isSecure) return;

        if (video.readyState === 4) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.5);

            fetch(FLASK_SERVER_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: imageData,
                    mode: currentMode,
                    room_code: MY_SECRET_CODE
                })
            })
                .then(response => response.json())
                .then(data => {
                    if (letterDisplay) letterDisplay.innerText = data.predicted_char || "--";

                    if (data.current_word && data.current_word.length > 0) {
                        bufferDisplay.innerText = `Typing: ${data.current_word}`;
                        bufferDisplay.style.color = "#ffff00";
                    }
                    else if (data.buffer_count) {
                        bufferDisplay.innerText = `Buffer: ${data.buffer_count}`;
                        bufferDisplay.style.color = "white";
                    }

                    if (data.processed_image) {
                        const imageObj = new Image();
                        imageObj.onload = function () { ctx.drawImage(imageObj, 0, 0, canvas.width, canvas.height); };
                        imageObj.src = data.processed_image;
                    }

                    if (data.final_message) {
                        handleAutoSendSuccess(data.final_message);
                    }
                })
                .catch(err => {
                    if (jsonDisplay && jsonDisplay.style.color !== "orange") {
                        jsonDisplay.innerText = "Backend Disconnected: " + err.message;
                        jsonDisplay.style.color = "orange";
                    }
                });
        }
    }, 100);
}



function setupCamera() {
    if (jsonDisplay) jsonDisplay.innerText = "Status: Initializing Camera...";

    navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" }
    })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
            if (jsonDisplay) {
                jsonDisplay.innerText = "Status: Camera ON. Secure Tunnel Active.";
                jsonDisplay.style.color = "yellow";
            }
        })
        .catch(function (err) {
            console.error(err);
            if (jsonDisplay) {
                jsonDisplay.innerText = "CAMERA ERROR: " + err.message;
                jsonDisplay.style.color = "red";
            }
        });
}

function toggleMode() {
    if (currentMode === "letters") {
        currentMode = "words";
        if (modeButton) modeButton.innerText = "Mode: Speaking (Words)";
        if (letterDisplay) letterDisplay.innerText = "--";
        if (bufferDisplay) bufferDisplay.innerText = "Switching to Words...";
    } else {
        currentMode = "letters";
        if (modeButton) modeButton.innerText = "Mode: Spelling (Letters)";
        if (letterDisplay) letterDisplay.innerText = "--";
        if (bufferDisplay) bufferDisplay.innerText = "Switching to Letters...";
    }
    fetch(RESET_URL, { method: 'POST' });
}

if (modeButton) modeButton.addEventListener('click', toggleMode);

function handleAutoSendSuccess(msg) {
    if (msg === lastSentMessage) return;
    lastSentMessage = msg;
    bufferDisplay.style.color = "#00ff00";
    bufferDisplay.innerText = `Sent: "${msg}"`;
    setTimeout(() => {
        bufferDisplay.style.color = "white";
        bufferDisplay.innerText = "Buffer: 0 / 10";
    }, 2000);
}

function displayMessage(msg) {
    const msgDiv = document.createElement('div');
    const waiter = document.querySelector('.waiting-msg');
    if (waiter) waiter.remove();

    if (msg.sender_role === 'receiver') {
        msgDiv.className = 'msg receiver';
        msgDiv.innerText = msg.message_content;
    } else {
        msgDiv.className = 'msg signer';
        msgDiv.innerText = "You: " + msg.message_content;
    }

    if (chatBox) {
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}