import os, torch, requests, random as rand
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import models, transforms

def menu():
    mainMenu=int(input("\ntype 1 for a random image, type 2 to enter a specific file location or type 3 to exit\n"))
    match mainMenu:
        case 1:
            randImage()
        case 2:
            upload()
        case 3:
            quit()
        case _:
            menu()

def randImage():
    fileDir='.\\archive\\Images\\'#change here for archive directotry changes
    folderFiles=[f for f in os.listdir(fileDir) if os.path.isfile(os.path.join(fileDir, f))]#creates list of all files in folder
    randFile=rand.choice(folderFiles)
    imageDir=os.path.join(fileDir,randFile)

    os.startfile(imageDir)#opens file of random image
    realCaption=captionCheck(randFile)
    modelSwitch(imageDir,realCaption)
    menu()

def upload():
    try:
        imageDir=input("\nCopy image directory here\n")
        realCaption=input("\nplease enter a descriptive sentence about the image\n")
        print(realCaption)#takes image directory and makes the user input a caption about the image

        os.startfile(imageDir)
        modelSwitch(imageDir,realCaption)
        menu()
    except:
        print("\nFile cannot be found")
        menu()

def modelSwitch(imageDir,realCaption):
    modelMenu=int(input("\ntype 1 for or type 2 for Blip\n"))#user can choose which model to use
    match modelMenu:
        case 1:
            inceptionv3(imageDir,realCaption)#inception and blip for encode-decode(actual module path)
        case 2:
            blip(imageDir)#accurate for testing/ all in one model
        case _:
            menu()

def blip(imageDir):
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
                realCaption=caption.replace(" .","")#removes all but caption from descriptive string
    print("\n",realCaption,"\n")
    return realCaption

def inceptionv3(imageDir,realCaption):
    model = models.inception_v3(pretrained=True)#setting up model
    model.eval()
    image = Image.open(imageDir)

    reSize = transforms.Compose([
    transforms.Resize(299),transforms.CenterCrop(299),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) # resizes to fit inceptionv3 requirements

    tensor = reSize(image)
    unSqueezed = tensor.unsqueeze(0)#correcting format

    with torch.no_grad():
        outputs = model(unSqueezed)

    probability = torch.nn.functional.softmax(outputs[0], dim=0)
    predictions = 3#top 3 predictions
    values, indices = torch.topk(probability, predictions)

    _, predicted_class = torch.max(outputs, 1)

    LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    labels = requests.get(LABELS_URL).json()#returns a json string with probabilties of predictions

    captionWords=[]
    for i in range(predictions):
        idx = indices[i].item()
        label = labels[str(idx)][1]
        captionWords.append(label)#saves top predictions to list
        
    decoderBlip(captionWords,realCaption)

def decoderBlip(captionWords,realCaption): 
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")#setting up links for model

    prompt=f"A scene with{', '.join(captionWords)}.Describe this scene:"
    fakeImage = Image.new("RGB", (224, 224), color="white")#fake image needed for blip

    input=processor(text=prompt,images=fakeImage,return_tensors="pt")#needed inputs for model to work

    with torch.no_grad():
        output = model.generate(**input,max_length=50,num_return_sequences=1)

        caption=processor.decode(output[0],skip_special_tokens=True)
        descriptors=caption.split(':')
        descriptVerify=descriptors[1]
        if descriptVerify=="":#loops until returns no empty
            decoderBlip(captionWords,realCaption)
        print("\n",caption,"\n")
    captionVerify(descriptVerify,realCaption)


def captionVerify(descriptVerify,realCaption):
    descriptArr=descriptVerify.split()

    count=0
    for x in range(len(descriptArr)):
        if descriptArr[x] in realCaption:
            count+=1
    
    print("prompt accuracy:",count)

menu()