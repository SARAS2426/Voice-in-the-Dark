
import speech_recognition
import pyttsx


speech_engine = pyttsx.init('sapi5') # see http://pyttsx.readthedocs.org/en/latest/engine.html#pyttsx.init
rate = speech_engine.getProperty('rate')
speech_engine.setProperty('rate', rate - 2000)

def speak(text):
	speech_engine.say(text)
	speech_engine.runAndWait()


recognizer = speech_recognition.Recognizer()

def listen():
	with speech_recognition.Microphone() as source:
		recognizer.adjust_for_ambient_noise(source)
		audio = recognizer.listen(source)

	try:
		return recognizer.recognize_google(audio)
	except speech_recognition.UnknownValueError: 
		print("Could not understand audio")
	except speech_recognition.RequestError as e:
		print("Recog Error; {0}".format(e))

	return "Other Error"

if __name__ == "__main__":
	
	
	print("What are you looking for?")
	speak("What are you looking for?")
	
	obj = listen()
	print(obj)
	print("Great! Click a picture so that I can find it for you!")
	speak("Great! Click a picture so that I can find it for you!")
	
	