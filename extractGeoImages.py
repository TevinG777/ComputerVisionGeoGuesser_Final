import os
import pyautogui
import asyncio
import cv2
import time
import keyboard
import numpy as np
import pyperclip
import threading 


mapTempate = "DataSet\KeyImages\mapName_testing.png"
rounds = 1000

def checkMapNameOnScreen(MapLocation = None):
    
    
    while MapLocation is None:
        # Verify that the map name is on the screen via a template match
        MapLocation = pyautogui.locateOnScreen(mapTempate, confidence=0.9)
        print("Waiting for map name to appear on screen...")

def findCurrentImageNumber():
    # Open the Dataset/Images directory and find the highest numbered image
    imageDir = "DataSet/Images"
    imageFiles = os.listdir(imageDir)
    imageNumbers = []
    
    # Extract the image numbers from the filenames
    for imageFile in imageFiles:
        
        # Check if the file is a PNG image
        if imageFile.endswith(".png"):
            imageNumber = int(imageFile.split("image")[-1].split("_")[0].split(".")[0])
            imageNumbers.append(imageNumber)
            
    if imageNumbers:
        return max(imageNumbers) + 1
    else:
        return 1
    
def genImageName(imageNumber, globalImgNum):
    # Use the provided imageNumber as the changing suffix
    imgName = "image" + str(globalImgNum) + "_" + str(imageNumber) + ".png"

    return imgName

def takeScreenshotAndSave(imageNumber, globalImgNum):
    # Take a screenshot of the current screen
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Change the pixels it a 250x250 square at the top right corner black
    screenshot[0:115, screenshot.shape[1]-500:screenshot.shape[1]] = 0
                              
    # Save the image the the DataSet/Images directory
    cv2.imwrite(os.path.join("DataSet/Images", genImageName(imageNumber, globalImgNum)), screenshot)
    
def makeGuess():
    # Click in the bottom center of the screen to open the guess menu
    screenWidth, screenHeight = pyautogui.size()
    pyautogui.click(screenWidth - 100 , screenHeight - 100)
    time.sleep(1)
    
    # Seach for the space bar image on screen and click it
    spaceBarTemplate = "DataSet/KeyImages/spaceBar.png"
    spaceBarLocation = None
    
    while spaceBarLocation is None:
        try:
            spaceBarLocation = pyautogui.locateOnScreen(spaceBarTemplate, confidence=0.8)
        except:
            time.sleep(1)
        print("Searching for space bar on screen...")
    
    # Click the center of the found space bar location
    spaceBarCenter = pyautogui.center(spaceBarLocation)
    pyautogui.click(spaceBarCenter)
    
    # wait for 1 second
    time.sleep(1)
    
    # Template match for the flag and then click it
    flagImg = "DataSet/KeyImages/locationFlag.png"
    
    locationFlag = None
    while locationFlag is None:
        try:
            locationFlag = pyautogui.locateOnScreen(flagImg, confidence=0.95)
        except:
            time.sleep(1)
        print("Searching for location flag on screen...")
        
    # Click the center of the found flag location
    flagCenter = pyautogui.center(locationFlag)
    pyautogui.click(flagCenter)
    
    # Seach for the GoogleAtmessage and click it, grab the values and the parse it for the @ smybol, read the values after the @
    # up until you reach the second , and then print the values
    iterations = 0
    
    googleAtTemplate = "DataSet/KeyImages/googleAtMessage.png"
    googleAtLocation = None
    while googleAtLocation is None:
        try:
            googleAtLocation = pyautogui.locateOnScreen(googleAtTemplate, confidence=0.9)
        except:
            iterations += 1
            
            # If more than 5 iterations, click the flag again to reopen the menu
            if iterations > 5:
                # Search for the flag again
                flagCenter = pyautogui.locateOnScreen(flagImg, confidence=0.95)
                pyautogui.click(pyautogui.center(flagCenter))
                
                iterations = 0
            time.sleep(1)
        print("Searching for Google At message on screen...")
        
    # Click the center of the found Google At location
    googleAtCenter = pyautogui.center(googleAtLocation)
    pyautogui.click(googleAtCenter)
    
    # Grab the text text that is in the input box
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.5)  
    
    clipboardText = pyperclip.paste()
    
    # Parse the clipboard text for the @ symbol
    atIndex = clipboardText.find('viewpoint=')
    if atIndex != 0:
        # Read from the index of viewpoint= and then up until &heading
        endIndex = clipboardText.find('&heading', atIndex)
        coordinates = clipboardText[atIndex + 10:endIndex]
        
        print("Extracted Coordinates: ", coordinates)

        lat, lon = coordinates.split("%2C")
        
        print("Extracted Coordinates: ", lat + "," + lon)
        
        # hit crtl+w to close the guess menu
        pyautogui.hotkey('ctrl', 'w')
        time.sleep(1)
        
        # Click in the bottom center of the screen to refocus
        pyautogui.click(screenWidth / 2, screenHeight - 300)
        time.sleep(1)
        
    return lat, lon 

def hitNextButton():
    nextButtonTemplate = "DataSet/KeyImages/NextButton.png"
        
    nextButtonLocation = None
    while nextButtonLocation is None:
        try:
            nextButtonLocation = pyautogui.locateOnScreen(nextButtonTemplate, confidence=0.9)
        except:
            time.sleep(1)
        print("Searching for Next Button on screen...")
        
    # Click the center of the found Next Button location
    nextButtonCenter = pyautogui.center(nextButtonLocation)
    pyautogui.click(nextButtonCenter)
    time.sleep(0.5)
    
    # Click where the next button used to be to refocus
    pyautogui.click(nextButtonCenter)
    time.sleep(1)
    
    # Click again to take care of playagin case
    pyautogui.click(nextButtonCenter)
    time.sleep(0.5)
    pyautogui.click(nextButtonCenter)
    time.sleep(0.5)
    
def emergencyStop():
    print("Press 'q' to stop the program...")
    while True:
        if keyboard.is_pressed('q'):
            print("Emergency stop activated. Exiting program...")
            os._exit(0)   # kills the whole process immediately
        time.sleep(0.1)

        
    

def main():
    globalImgNum = findCurrentImageNumber()
    
    # Start emergency stop watcher in a background thread
    stop_thread = threading.Thread(target=emergencyStop, daemon=True)
    stop_thread.start()
    
    # if the user presses 'r', start the process
    print("Press 'r' to start the image extraction process...")
    while True:
        if keyboard.is_pressed('r'):
            print("Starting image extraction process...")
            break
        time.sleep(0.1)


    
    for imagNum in range(globalImgNum, globalImgNum + rounds):

        # Initialize map location
        mapLocation = None

        # Wait until the map name appears on screen
        checkMapNameOnScreen(mapLocation)

        for i in range(3):
            takeScreenshotAndSave(i, imagNum)

            # HOLD A for 2 seconds to rotate
            pyautogui.keyDown('a')
            time.sleep(1.2)

            # Release A key
            pyautogui.keyUp('a')

        # Make the guess
        lat, lon  = makeGuess()

        # Save to a text file with the image name but .txt extension and first line lat second line lon
        annotationsDir = "DataSet/Annotations"
        coordsFileName = os.path.join(annotationsDir, "image" + str(imagNum) + "_coords.txt")
        
        with open(coordsFileName, 'w') as f:
            f.write(lat + "\n")
            f.write(lon + "\n")
            
        # After processing one image click the next button to go to the next image
        hitNextButton()
        

    
        
if __name__ == "__main__":
    
    main()
    