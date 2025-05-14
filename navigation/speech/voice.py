"""
Voice module for speech recognition and text-to-speech functionality
"""
import speech_recognition as sr
import pygame
from gtts import gTTS
import io

class VoiceService:
    def __init__(self):
        pygame.mixer.init()
        self.recognizer = sr.Recognizer()
        print(len(sr.Microphone.list_microphone_names()), "microphones found")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
        
    def speak(self, text):
        """
        Convert text to speech and play it
        
        Args:
            text (str): Text to be spoken
        """
        if not text:
            print("Không có văn bản để nói.")
            return
        try:
            tts = gTTS(text=text, lang='vi')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Lỗi TTS hoặc phát âm: {e}")
    
    def recognize_speech(self):
        """
        Recognize speech from microphone
        
        Returns:
            str: Recognized text or None if recognition failed
        """
        with sr.Microphone(device_index=11) as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Đang lắng nghe từ mic ...")
            audio = self.recognizer.listen(source)
            
        try:
            return self.recognizer.recognize_google(audio, language="vi-VN")
        except sr.UnknownValueError:
            print("Không thể nhận diện giọng nói!")
        except sr.RequestError:
            print("Lỗi kết nối đến dịch vụ nhận diện giọng nói!")
        return None