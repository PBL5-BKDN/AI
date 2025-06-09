import os
import asyncio
import tempfile
import sounddevice as sd
import speech_recognition as sr
import edge_tts


def find_device_index_by_name(keyword, kind='input'):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if keyword.lower() in dev['name'].lower():
            if kind == 'input' and dev['max_input_channels'] > 0:
                return i
            elif kind == 'output' and dev['max_output_channels'] > 0:
                return i
    return None


class VoiceService:
    def __init__(self, mic_name="USB Composite", speaker_name="pulse"):
        # Chọn PulseAudio nếu đang dùng hệ thống hỗ trợ mixing
        self.mic_index = find_device_index_by_name(mic_name, kind='input')
        self.speaker_index = find_device_index_by_name(speaker_name, kind='output')
        self.recognizer = sr.Recognizer()

        if self.mic_index is None:
            raise ValueError(f"Không tìm thấy micro nào chứa '{mic_name}'!")
        if self.speaker_index is None:
            raise ValueError(f"Không tìm thấy loa nào chứa '{speaker_name}'!")

        print(f"🎤 Mic index: {self.mic_index}")
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
                    # Dùng ffplay để phát qua PulseAudio, hỗ trợ playback song song
                    os.system(f"ffplay -nodisp -autoexit -loglevel quiet {f.name} &")

            except Exception as e:
                print(f"⚠️ Lỗi TTS hoặc phát âm: {e}")

        asyncio.run(run_tts())

    def recognize_speech(self):
        try:
            with sr.Microphone(device_index=self.mic_index, sample_rate=48000) as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("🎙️ Đang lắng nghe ...")
                audio = self.recognizer.listen(source)

            print("🧠 Đang xử lý âm thanh ...")
            return self.recognizer.recognize_google(audio, language="vi-VN")

        except sr.UnknownValueError:
            print("❌ Không thể nhận diện giọng nói!")
        except sr.RequestError:
            print("❌ Lỗi kết nối đến dịch vụ nhận diện!")
        except Exception as e:
            print(f"⚠️ Lỗi khi thu âm: {e}")

        return None


# Dùng 1 phần tên thiết bị là được
voice_service = VoiceService(mic_name="USB Composite Device", speaker_name="pulse")
