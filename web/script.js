document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messagesContainer');
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const modelSelect = document.getElementById('modelSelect');

    let ws;
    let currentBotMessage = null;
    let currentBotContent = "";

    function connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

        ws.onopen = () => {
            console.log('Connected to Keith Gateway');
            document.querySelector('.status-indicator').classList.add('online');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'progress') {
                if (!currentBotMessage) {
                    currentBotMessage = createMessageElement('bot-msg');
                    currentBotContent = "";
                    currentBotMessage.progressLogs = [];
                }
                if (!currentBotMessage.progressLogs) currentBotMessage.progressLogs = [];
                currentBotMessage.progressLogs.push(data.content);

                let logsHtml = currentBotMessage.progressLogs.map(log => {
                    let color = "#a78bfa";
                    let bg = "rgba(139, 92, 246, 0.1)";
                    let border = "#8b5cf6";

                    if (log.includes("[IMPLEMENTER]")) {
                        color = "#fb923c";
                        bg = "rgba(249, 115, 22, 0.1)";
                        border = "#ea580c";
                    } else if (log.includes("[REVIEWER]")) {
                        color = "#4ade80";
                        bg = "rgba(74, 222, 128, 0.1)";
                        border = "#16a34a";
                    } else if (log.includes("[UI_AGENT]")) {
                        color = "#f472b6";
                        bg = "rgba(236, 72, 153, 0.1)";
                        border = "#db2777";
                    } else if (log.includes("🧬 Spawning")) {
                        color = "#38bdf8";
                        bg = "rgba(56, 189, 248, 0.1)";
                        border = "#0ea5e9";
                    } else if (log.includes("[THOUGHT]")) {
                        color = "#f8fafc";
                        bg = "rgba(30, 41, 59, 0.6)";
                        border = "#64748b";
                        // Remove the [THOUGHT] prefix for cleaner display 
                        log = log.replace("[THOUGHT] \\n", "").replace("[THOUGHT] \n", "").replace("[THOUGHT]", "");
                    }
                    return `<div class="progress-log" style="font-family: 'JetBrains Mono', monospace; font-size: 0.8em; color: ${color}; margin-bottom: 6px; padding: 6px 10px; background: ${bg}; border-radius: 6px; border-left: 2px solid ${border};">${log}</div>`;
                }).join('');

                if (currentBotContent === "") {
                    currentBotMessage.querySelector('.message-content').innerHTML = logsHtml + '<div class="typing-indicator" id="typing-indicator" style="margin-top: 10px;"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
                }
                scrollToBottom();
            } else if (data.type === 'bot_stream') {
                if (!currentBotMessage) {
                    currentBotMessage = createMessageElement('bot-msg');
                }

                // Parse logs and body
                const logParts = data.content.split('\n\n---BOT---\n\n');
                let logsRaw = logParts.length > 1 ? logParts[0] : '';
                let currentBotContent = logParts.length > 1 ? logParts[1] : data.content;

                const logsHtml = logsRaw.split('\n').filter(l => l.trim().length > 0).map(log => {
                    let color = "#94a3b8";
                    let bg = "rgba(255, 255, 255, 0.03)";
                    let border = "#475569";

                    if (log.includes("[IMPLEMENTER]")) {
                        color = "#fb923c";
                        bg = "rgba(249, 115, 22, 0.1)";
                        border = "#ea580c";
                    } else if (log.includes("[REVIEWER]")) {
                        color = "#4ade80";
                        bg = "rgba(74, 222, 128, 0.1)";
                        border = "#16a34a";
                    } else if (log.includes("[UI_AGENT]")) {
                        color = "#f472b6";
                        bg = "rgba(236, 72, 153, 0.1)";
                        border = "#db2777";
                    } else if (log.includes("🧬 Spawning")) {
                        color = "#38bdf8";
                        bg = "rgba(56, 189, 248, 0.1)";
                        border = "#0ea5e9";
                    } else if (log.includes("[THOUGHT]")) {
                        color = "#f8fafc";
                        bg = "rgba(30, 41, 59, 0.6)";
                        border = "#64748b";
                        log = log.replace("[THOUGHT] \\n", "").replace("[THOUGHT] \n", "").replace("[THOUGHT]", "");
                    }
                    return `<div class="progress-log" style="font-family: 'JetBrains Mono', monospace; font-size: 0.8em; color: ${color}; margin-bottom: 6px; padding: 6px 10px; background: ${bg}; border-radius: 6px; border-left: 2px solid ${border};">${log}</div>`;
                }).join('');

                // Render logs above the actual markdown response
                currentBotMessage.querySelector('.message-content').innerHTML = (logsHtml ? logsHtml + '<div style="margin-top: 12px;">' : '') + imageHtml + marked.parse(currentBotContent) + (logsHtml ? '</div>' : '');
                scrollToBottom();
            } else if (data.type === 'bot_chunk') {
                if (!currentBotMessage) {
                    currentBotMessage = createMessageElement('bot-msg');
                    currentBotContent = "";
                    currentBotMessage.progressLogs = [];
                }
                currentBotContent += data.content;

                let logsHtml = (currentBotMessage.progressLogs || []).map(log => {
                    let color = "#a78bfa";
                    let bg = "rgba(139, 92, 246, 0.1)";
                    let border = "#8b5cf6";

                    if (log.includes("[IMPLEMENTER]")) {
                        color = "#fb923c";
                        bg = "rgba(249, 115, 22, 0.1)";
                        border = "#ea580c";
                    } else if (log.includes("[REVIEWER]")) {
                        color = "#4ade80";
                        bg = "rgba(74, 222, 128, 0.1)";
                        border = "#16a34a";
                    } else if (log.includes("[UI_AGENT]")) {
                        color = "#f472b6";
                        bg = "rgba(236, 72, 153, 0.1)";
                        border = "#db2777";
                    } else if (log.includes("🧬 Spawning")) {
                        color = "#38bdf8";
                        bg = "rgba(56, 189, 248, 0.1)";
                        border = "#0ea5e9";
                    } else if (log.includes("[THOUGHT]")) {
                        color = "#f8fafc";
                        bg = "rgba(30, 41, 59, 0.6)";
                        border = "#64748b";
                        log = log.replace("[THOUGHT] \\n", "").replace("[THOUGHT] \n", "").replace("[THOUGHT]", "");
                    }
                    return `<div class="progress-log" style="font-family: 'JetBrains Mono', monospace; font-size: 0.8em; color: ${color}; margin-bottom: 6px; padding: 6px 10px; background: ${bg}; border-radius: 6px; border-left: 2px solid ${border};">${log}</div>`;
                }).join('');

                // Render logs above the actual markdown response
                currentBotMessage.querySelector('.message-content').innerHTML = (logsHtml ? logsHtml + '<div style="margin-top: 12px;">' : '') + marked.parse(currentBotContent) + (logsHtml ? '</div>' : '');
                scrollToBottom();
            } else if (data.type === 'bot_done') {
                currentBotMessage = null; // Reset for next message
            } else if (data.type === 'error') {
                const errorMsg = createMessageElement('system-msg');
                errorMsg.querySelector('.message-content').innerHTML = `<span style="color: #ef4444">Error: ${data.content}</span>`;
                scrollToBottom();
                currentBotMessage = null;
            }
        };

        ws.onclose = () => {
            console.log('Disconnected. Reconnecting in 3s...');
            document.querySelector('.status-indicator').classList.remove('online');
            setTimeout(connect, 3000);
        };
    }

    function createMessageElement(type) {
        const div = document.createElement('div');
        div.className = `message ${type}`;

        // Remove typing indicator if exists
        const typing = document.getElementById('typing-indicator');
        if (typing) typing.remove();

        div.innerHTML = `
            <div class="message-content">
                ${type === 'bot-msg' ? '<div class="typing-indicator" id="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>' : ''}
            </div>
        `;
        messagesContainer.appendChild(div);
        return div;
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const content = messageInput.value.trim();
        if (!content || !ws || ws.readyState !== WebSocket.OPEN) return;

        // Add user message
        const userMsg = createMessageElement('user-msg');
        userMsg.querySelector('.message-content').textContent = content;

        // Add loading indicator for bot
        createMessageElement('bot-msg');

        // Send to WebSocket
        ws.send(JSON.stringify({
            type: 'user_msg',
            content: content,
            model: modelSelect.value
        }));

        messageInput.value = '';
        messageInput.style.height = 'auto'; // Reset height
        scrollToBottom();
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Enter to submit (Shift+Enter for newline)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Chrono-Matrix Polling Engine
    function pollChronoJobs() {
        fetch('/api/jobs')
            .then(res => res.json())
            .then(jobs => {
                const list = document.getElementById('chronoList');
                if (!list) return;

                if (!jobs || jobs.length === 0) {
                    list.innerHTML = `
                        <li class="agent-item" style="opacity: 0.5;">
                            <span class="agent-icon">🕰️</span>
                            <div class="agent-info">
                                <span class="agent-name">No Active Alarms</span>
                            </div>
                        </li>
                    `;
                    return;
                }

                let html = '';
                jobs.forEach(job => {
                    let typeBadge = job.type === 'one-off' ? '1x' : `∞ (${job.interval_minutes}m)`;
                    let color = job.type === 'one-off' ? '#f59e0b' : '#3b82f6';

                    html += `
                        <li class="agent-item">
                            <span class="agent-icon">⚡</span>
                            <div class="agent-info" style="width: 100%;">
                                <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                                    <span class="agent-name" style="font-size: 0.85em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px;" title="${job.task_prompt}">
                                        ${job.task_prompt}
                                    </span>
                                    <span style="font-size: 0.7em; background: ${color}; color: white; padding: 2px 5px; border-radius: 4px; font-weight: bold;">
                                        ${typeBadge}
                                    </span>
                                </div>
                                <span class="agent-status" style="font-size: 0.75em; opacity: 0.7; margin-top: 3px; display: block;">
                                    ID: ${job.id.substring(4, 10)}
                                </span>
                            </div>
                        </li>
                    `;
                });
                list.innerHTML = html;
            })
            .catch(err => console.error("Chrono fetch error:", err));
    }

    // Initialize
    connect();
    pollChronoJobs();
    setInterval(pollChronoJobs, 5000); // Sweep every 5s
});
