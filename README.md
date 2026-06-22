# ACROBOT-2.2
A RAG based chatbot for Acropolis College.

# Features

- Speech-to-Speech AI chatbot
- Hindi + English support
- RAG based
- Web search fallback
- Wake word support
- Smart silence detection
- Real-time voice interaction

---

# Follow these steps to run

## Step 1: Create a virtual environment

```bash
python -m venv acrobot-env
```

---

## Step 2: Activate the virtual environment

### Windows

```bash
acrobot-env\Scripts\activate
```

### Linux / Mac

```bash
source acrobot-env/bin/activate
```

---

## Step 2.1: Windows PowerShell Fix

If activation is blocked on Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```bash
acrobot-env\Scripts\activate
```

---

## Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# Step 4: Install dependencies

```bash
python -m pip install groq edge-tts pygame sounddevice soundfile numpy python-dotenv pymupdf requests scikit-learn
```

---

# Step 4.1: Install offline library
```bash
sudo apt install espeak espeak-data libespeak-dev
pip install pyttsx3

```

---

# Step 5: Create .env file

Create a `.env` file in the project folder and add:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

# Step 6: Run the chatbot

```bash
python acrobot.py
```

---

# Wake Words

You can activate the chatbot using:

- Hello
- Hey
- Hello Acrobot
- Acrobot

---

# Notes

- Internet connection is required
- First startup may take some time
- Works best with a microphone and speaker


# Acrobot-Service-rpi4

## Step 1 — Copy all 4 files to your Pi project folder:
```
# destination: /home/test/Desktop/Acrobot-2.2/
```

## Step 2 — Run the service setup (as user test, no sudo):
```
cd /home/test/Desktop/Acrobot-2.2
chmod +x setup_service.sh
./setup_service.sh
```

## Step 3 — Reboot and verify the bot starts on its own:
```
sudo reboot
# wait 30 seconds, then:
./check_status.sh
```

## Step 4 — Run lockdown (with sudo), and when it asks about SSH choose option 2:
```
sudo ./lockdown.sh
# When it asks "Choose SSH policy [1/2/3]" → type 2
# This keeps SSH ON but disables password login
```
