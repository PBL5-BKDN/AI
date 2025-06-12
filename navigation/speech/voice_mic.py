from pydub import AudioSegment
import sounddevice as sd
import speech_recognition as sr
import tempfile
import edge_tts

print(sd.query_devices())

def find_device_index_by_name(keyword, kind='input'):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if keyword.lower() in dev['name'].lower():
            if kind == 'input' and dev['max_input_channels'] > 0:
                return i
    return None

class VoiceMic:
    def __init__(self, mic_name):
        self.mic_index = find_device_index_by_name(mic_name, kind='input')
        self.recognizer = sr.Recognizer()
        if self.mic_index is None:
            raise ValueError(f"Không tìm thấy micro nào chứa '{mic_name}'!")
        print(f"🎤 Mic index: {self.mic_index}")

    def recognize_speech(self):
        try:
            with sr.Microphone(device_index=self.mic_index, sample_rate=48000) as source:
                # print("⏳ Đang điều chỉnh nhiễu nền...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)  # tăng lên 1.5s hoặc 2s
                print("🎙️ Đang lắng nghe ...")
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
            print("🧠 Đang xử lý âm thanh ...")
            text = self.recognizer.recognize_google(audio, language="vi-VN")
            print(f"[MIC] Nhận diện: {text}")
            return text
        except sr.UnknownValueError:
            None
            # print("❌ Không thể nhận diện giọng nói!")
        except sr.RequestError:
            print("❌ Lỗi kết nối đến dịch vụ nhận diện!")
        except Exception as e:
            None
            # print(f"⚠️ Lỗi khi thu âm: {e}")
        return None

async def run_tts(text):
    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice="vi-VN-NamMinhNeural",
            rate="+20%"
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            await communicate.save(f.name)
            # Chuyển mp3 sang wav
            sound = AudioSegment.from_mp3(f.name)
            wav_file = f.name.replace(".mp3", ".wav")
            sound.export(wav_file, format="wav")
            # Phát bằng sounddevice
            import soundfile as sf
            data, fs = sf.read(wav_file, dtype='float32')
            sd.play(data, fs)
            sd.wait()
    except Exception as e:
        print(f"⚠️ Lỗi TTS hoặc phát âm: {e}")


