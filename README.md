# MM2-Prosjekt-Wav-Animation

For å initialisere kode:

linux:

```bash
python .m venv .env # Hvis du ikke har lagd environment enda
source .env/bin/activate
```

windows:

```powershell
python -m venv .env # Hvis du ikke har lagd environment enda
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass # Hvis du ikke får kjørt neste kommando
. ./env/Scripts/activate
```

For Windows, hvis

```powershell
ffmpeg -version
```

ikke virker, kjør denne:

```powershell
winget install "ffmpeg (Essentials Build)"
```

Deretter kjør denne (linux og windows)

```bash
pip install -r requirements.txt
```

Deretter kan du kjøre programmet slik:

```bash
python animate-wave.py -i [INPUT_FILE] -o [OUTPUT_FILE] -f [FPS]
```
