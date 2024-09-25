import random as rand
import os

def menu():
    mainMenu=int(input("type 1 for a random image or type 2 to enter a specific file location"))
    match mainMenu:
        case 1:
            randImage()  
        case 2:
            imageDir=input("Copy image directory here")
        case _:
            menu()

def randImage():
    fileDir='.\\archive\\Images\\'
    folderFiles=[f for f in os.listdir(fileDir) if os.path.isfile(os.path.join(fileDir, f))]#creates list of all files in folder
    randFile=rand.choice(folderFiles)
    fullFile=os.path.join(fileDir,randFile)
    os.startfile(fullFile)#opens file of random image

randImage()