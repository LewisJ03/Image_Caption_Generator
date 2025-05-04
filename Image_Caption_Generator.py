import random as rand
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def menu():
    mainMenu=int(input("type 1 for a random image or type 2 to enter a specific file location"))
    match mainMenu:
        case 1:
            randImage()
        case 2:
            upload()
        case _:
            menu()

def randImage():
    fileDir='.\\archive\\Images\\'
    folderFiles=[f for f in os.listdir(fileDir) if os.path.isfile(os.path.join(fileDir, f))]#creates list of all files in folder
    randFile=rand.choice(folderFiles)
    imageDir=os.path.join(fileDir,randFile)
    os.startfile(imageDir)#opens file of random image
    captionCheck(randFile)
    captionGenerator(imageDir)
    menu()

def upload():
    try:
        imageDir=input("Copy image directory here")
        caption=input("please enter a descriptive sentence about the image")
        os.startfile(imageDir)
        captionGenerator(imageDir)
        menu()
    except:
        print("File cannot be found")
        menu()

def captionGenerator(imageDir):
    processor= BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")#loads pretrained Blip processor
    model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")#loads blip model

    rawImage=Image.open(imageDir).convert("RGB")#converts image to required format
    
    input=processor(rawImage,return_tensors="pt")

    output=model.generate(**input)
    print(f"\nThis is an image of:{processor.decode(output[0],skip_special_tokens=True)}\n")

def captionCheck(randFile):
    with open('archive\captions.txt','r') as file:
        for lineNum, line in enumerate(file, start=2):#first line is name so skip
            if randFile in line.strip():#checks for file name in caption file
                caption=(line.strip()).replace((randFile+","),"")
                caption=caption.replace(" .","")#removes all but caption from descriptive string
    print(caption)
menu()