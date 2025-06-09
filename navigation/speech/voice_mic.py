import sounddevice as sd
import speech_recognition as sr

def find_device_index_by_name(keyword, kind='input'):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if keyword.lower() in dev['name'].lower():
            if kind == 'input' and dev['max_input_channels'] > 0:
                return i
    return None

class VoiceMic:
    def __init__(self, mic_name="USB Composite"):
        self.mic_index = find_device_index_by_name(mic_name, kind='input')
        self.recognizer = sr.Recognizer()
        if self.mic_index is None:
            raise ValueError(f"Không tìm thấy micro nào chứa '{mic_name}'!")
        print(f"🎤 Mic index: {self.mic_index}")

    def recognize_speech(self):
        try:
            with sr.Microphone(device_index=self.mic_index, sample_rate=48000) as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("🎙️ Đang lắng nghe ...")
                audio = self.recognizer.listen(source)
            print("🧠 Đang xử lý âm thanh ...")
            text = self.recognizer.recognize_google(audio, language="vi-VN")
            print(f"[MIC] Nhận diện: {text}")
            return text
        except sr.UnknownValueError:
            print("❌ Không thể nhận diện giọng nói!")
        except sr.RequestError:
            print("❌ Lỗi kết nối đến dịch vụ nhận diện!")
        except Exception as e:
            print(f"⚠️ Lỗi khi thu âm: {e}")
        return None

mic_service = VoiceMic(mic_name="USB Composite Device")
   
