


const SUPABASE_URL = "https://hhhbpzitdosieuafbfmc.supabase.co";
const SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhoaGJweml0ZG9zaWV1YWZiZm1jIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkwMTMwMjksImV4cCI6MjA4NDU4OTAyOX0.3sZPV5-tWIAfxP6V8QEzGii2NOogfWyfmpX4fZfqgP0";


const chatContainer = document.getElementById('chat-container');
const ttsToggle = document.getElementById('ttsToggle');
const languageSelect = document.getElementById('languageSelect');
const replyInput = document.getElementById('replyInput');
const statusDisplay = document.getElementById('connection-status'); 

function updateStatus(text, color) {
    if (statusDisplay) {
        statusDisplay.innerText = text;
        statusDisplay.style.color = color;
    }
}

let client = null;
let MY_ROOM_CODE = null;

if (window.supabase) {
    client = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);
    console.log("Receiver: Connected to Supabase");
    
    MY_ROOM_CODE = prompt("🔒 Enter Connection Code to join:");

    if (MY_ROOM_CODE) {
        sendHandshake(MY_ROOM_CODE);
        
        updateStatus("🟢 Connected: " + MY_ROOM_CODE, "#00ff00"); 

        startAppLogic();

    } else {
        console.warn("No code entered.");
        document.body.innerHTML = "<h1 style='color:red; text-align:center; margin-top:50px;'>❌ Access Denied: No Code Entered</h1>";
    }

} else {
    console.error("Receiver: Supabase library not found.");
    updateStatus("🔴 Offline", "red");
}

async function sendHandshake(code) {
    if (!client) return;
    try {
        await client.from('access_codes').insert({
            code_text: code
        });
        console.log("Handshake sent to DB:", code);
    } catch(e) {
        console.error("Handshake Error:", e);
    }
}

function startAppLogic() {
    if (client) {
        console.log("Listening for conversation...");

        client.channel('public:conversation_logs') 
            .on('postgres_changes', { 
                event: 'INSERT', 
                schema: 'public', 
                table: 'conversation_logs'
            }, async (payload) => {
                const newRow = payload.new;
                console.log("Message Received:", newRow); 

                const sysMsg = document.querySelector('.system-msg');
                if (sysMsg) sysMsg.remove();

                if (newRow.sender_role === 'signer') {
                    const originalText = newRow.message_content;
                    const targetLang = languageSelect ? languageSelect.value : 'en';
                    
                    let finalText = originalText;
                    if (targetLang !== 'en') {
                        finalText = await translateText(originalText, targetLang);
                    }
                    
                    addMessageToScreen(finalText, 'signer');
                    if (ttsToggle && ttsToggle.checked) {
                        speak(finalText, targetLang);
                    }
                } 
                else if (newRow.sender_role === 'receiver') {
                    addMessageToScreen(newRow.message_content, 'receiver');
                }
            })
            .subscribe((status) => {
                if (status === 'SUBSCRIBED') {
                    console.log("✅ Subscribed to chat channel.");
                }
            });
    }

    if (replyInput) {
        replyInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendReply();
        });
    }
}

async function sendReply() {
    if (!replyInput) return;
    const text = replyInput.value.trim();
    if (!text) return;

    if (client) {
        try {
            const { error } = await client.from('conversation_logs').insert({
                sender_role: 'receiver',
                message_content: text
            });

            if (error) {
                console.error("Supabase Error:", error);
                alert("Error: " + error.message);
            }
        } catch (err) {
            console.error("Send failed:", err);
        }
    }
    replyInput.value = ""; 
}

function addMessageToScreen(text, role) {
    if (!chatContainer) return;

    const div = document.createElement('div');
    div.className = `msg ${role}`; 
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const label = role === 'signer' ? 'Signer' : 'You';
    
    div.innerHTML = `
        <div class="msg-content">
            <strong>${label}</strong>: ${text}
            <span class="timestamp" style="font-size:0.7em; color:#ccc; margin-left:8px;">${time}</span>
        </div>
    `;
    chatContainer.appendChild(div); 
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function speak(text, lang) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel(); 
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang; 
        utterance.rate = 1.0; 
        window.speechSynthesis.speak(utterance);
    }
}

async function translateText(text, targetLang) {
    try {
        const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=en|${targetLang}`;
        const res = await fetch(url);
        const data = await res.json();
        return (data && data.responseData) ? data.responseData.translatedText : text;
    } catch (err) {
        return text; 
    }
}