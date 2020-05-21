from playsound import playsound

def playA(gesture=0):
    if gesture < 2:
        playsound("808-conga-19.wav", False)
    elif gesture == 3:
        playsound("808-cymbal-06.wav", False)
    else:
        playsound("808-cymbal-25.wav", False)
    
def playB(gesture=0):
    if gesture < 2:
        playsound('808-cymbal-11.wav', False)
    elif gesture == 3:
        playsound("808-hat-13.wav", False)
    else:
        playsound("808-hat-18.wav", False)

def playC(gesture=0):
    if gesture < 2:
        playsound('808-cymbal-21.wav', False)
    elif gesture == 3:
        playsound("808-hat-09.wav", False)
    else:
        playsound("808-hat-17.wav", False)

def playD(gesture=0):
    if gesture < 2:
        playsound('808-hat-01.wav', False)
    elif gesture == 3:
        playsound("808-tom-08.wav", False)
    else:
        playsound("808-tom-27.wav", False)

def playE(gesture=0):
    if gesture < 2:
        playsound('808-tom-02.wav', False)
    elif gesture == 3:
        playsound("808-clap-08.wav", False)
    else:
        playsound("808-kick-05.wav", False)

def playF(gesture=0):
    if gesture < 2:
        playsound('808-tom-18.wav', False)
    elif gesture == 3:
        playsound("808-kick-29.wav", False)
    else:
        playsound("808-conga-25.wav", False)
