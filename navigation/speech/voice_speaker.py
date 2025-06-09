import os
import asyncio
import tempfile
import sounddevice as sd
import edge_tts
import sys

def find_device_index_by_name(keyword, kind='output'):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if keyword.lower() in dev['name'].lower():
            if kind == 'output' and dev['max_output_channels'] > 0:
                return i
    return None

class VoiceSpeaker:
    def __init__(self, speaker_name="pulse"):
        self.speaker_index = find_device_index_by_name(speaker_name, kind='output')
        if self.speaker_index is None:
            raise ValueError(f"Không tìm thấy loa nào chứa '{speaker_name}'!")
        print(f"🔊 Speaker index (PulseAudio): {self.speaker_index}")

    def speak(self, text):
        if not text:
            print("❌ Không có văn bản để nói.")
            return
        async def run_tts():
            try:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice="vi-VN-HoaiMyNeural",
                    rate="+30%"
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    await communicate.save(f.name)
                    os.system(f"ffplay -nodisp -autoexit -loglevel quiet {f.name} &")
            except Exception as e:
                print(f"⚠️ Lỗi TTS hoặc phát âm: {e}")
        asyncio.run(run_tts())

speaker_service = VoiceSpeaker(speaker_name="pulse")
