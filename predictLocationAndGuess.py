import os
import time
import threading
from pathlib import Path

import cv2
import keyboard
import numpy as np
import pyautogui
import torch
from PIL import Image

from dataset import get_transforms
from model import GeoLocalizationModel
from plot_on_russia_map import plot_point_on_map, latlon_to_pixels

MAP_TEMPLATE = "DataSet/KeyImages/mapName.png"
ROUNDS = 1000
IMAGE_SIZE_MODEL = 256  # use 256x256 resized inputs (Lanczos)
GUESS_DIR = Path("DataSet/Guesses")
GUESS_COORDS_DIR = GUESS_DIR / "Guess_LatLon"
GUESS_MAP_DIR = GUESS_DIR / "Guess_Maps"
MAP_IMAGE_PATH = Path("DataSet/KeyImages/russia_map.png")
CHECKPOINT_PATH = Path("best_model.pth")


def checkMapNameOnScreen(map_location=None):
    """Wait until the map name template appears on screen."""
    while map_location is None:
        map_location = pyautogui.locateOnScreen(MAP_TEMPLATE, confidence=0.9)
        print("Waiting for map name to appear on screen...")


def findCurrentImageNumber():
    """Find the next image index based on existing guess files."""
    GUESS_DIR.mkdir(parents=True, exist_ok=True)
    image_files = os.listdir(GUESS_DIR)
    image_numbers = []

    for image_file in image_files:
        if image_file.endswith(".png"):
            try:
                image_number = int(
                    image_file.split("image")[-1].split("_")[0].split(".")[0]
                )
                image_numbers.append(image_number)
            except ValueError:
                continue

    return max(image_numbers) + 1 if image_numbers else 1


def genImageName(image_number, global_img_num):
    return f"image{global_img_num}_{image_number}.png"


def takeScreenshotAndSave(image_number, global_img_num):
    """
    Capture a screenshot, mask UI noise, resize to 256x256 (Lanczos),
    save it, and return a PIL image.
    """
    GUESS_DIR.mkdir(parents=True, exist_ok=True)

    screenshot = pyautogui.screenshot()
    bgr_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Mask top-right HUD area to reduce noise for the model
    bgr_img[0:115, bgr_img.shape[1] - 500 : bgr_img.shape[1]] = 0

    resized_bgr = cv2.resize(bgr_img, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    out_path = GUESS_DIR / genImageName(image_number, global_img_num)
    cv2.imwrite(str(out_path), resized_bgr)

    rgb_img = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)


def load_model(checkpoint_path=CHECKPOINT_PATH):
    """Load the trained model from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoLocalizationModel(pretrained=False)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path} on {device}")
    return model, device


def get_base_map_dimensions():
    """Load the base Russia map and return (width, height)."""
    img = cv2.imread(str(MAP_IMAGE_PATH))
    if img is None:
        raise FileNotFoundError(f"Could not load map image at {MAP_IMAGE_PATH}")
    return img.shape[1], img.shape[0]


def save_prediction_map(lat: float, lon: float, img_num: int):
    """Use plot_on_russia_map to store the predicted point visualization."""
    GUESS_MAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GUESS_MAP_DIR / f"image{img_num}_map.png"
    try:
        plot_point_on_map(lat, lon, str(out_path))
    except Exception as exc:
        print(f"Failed to save prediction map: {exc}")
    return out_path


def save_screen_overlay(click_x: float, click_y: float, img_num: int):
    """Capture the current screen and draw the click point for debugging."""
    screenshot = pyautogui.screenshot()
    bgr_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    cv2.circle(bgr_img, (int(click_x), int(click_y)), 10, (0, 0, 255), thickness=3)

    overlay_path = GUESS_MAP_DIR / f"image{img_num}_screen_click.png"
    GUESS_MAP_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), bgr_img)
    return overlay_path


def perform_guess_click(lat: float, lon: float, img_num: int, base_map_size: tuple[int, int]):
    """Locate the map on screen, map coords to pixels, click, and save overlay."""
    base_w, base_h = base_map_size

    # Find map on screen
    map_location = None
    print("Searching for Russia map on screen to place guess...")
    while map_location is None:
        try:
            map_location = pyautogui.locateOnScreen(str(MAP_IMAGE_PATH), confidence=0.7)
        except Exception:
            time.sleep(0.5)
        if map_location is None:
            time.sleep(0.5)

    scale_x = map_location.width / base_w
    scale_y = map_location.height / base_h

    x_px, y_px = latlon_to_pixels(lat, lon, base_w, base_h)
    click_x = map_location.left + x_px * scale_x
    click_y = map_location.top + y_px * scale_y

    overlay_path = save_screen_overlay(click_x, click_y, img_num)
    pyautogui.click(click_x, click_y)
    print(f"Auto-placed guess at ({click_x:.1f}, {click_y:.1f}); overlay saved to {overlay_path}")


def wait_for_guess_and_click(lat: float, lon: float, img_num: int, base_map_size: tuple[int, int]):
    """Wait for 'g' to place the guess on the map (or 'n' to skip auto-click)."""
    print("Press 'g' to place guess on the map (auto-click). Press 'n' to skip.")
    while True:
        if keyboard.is_pressed("g"):
            perform_guess_click(lat, lon, img_num, base_map_size)
            return
        if keyboard.is_pressed("n"):
            print("Skipping auto-click for this round.")
            return
        time.sleep(0.1)


def capture_views(global_img_num, transform):
    """
    Capture three views, apply transforms, and return a tensor of shape (3, 3, H, W).
    """
    views = []
    for i in range(3):
        pil_image = takeScreenshotAndSave(i, global_img_num)
        views.append(transform(pil_image))

        # Rotate view between captures to mimic training data variety
        pyautogui.keyDown("a")
        time.sleep(1.2)
        pyautogui.keyUp("a")

    return torch.stack(views)


def predict_coordinates(model, device, view_tensor):
    """Run the model on a single location tensor and return (lat, lon)."""
    with torch.no_grad():
        batch = view_tensor.unsqueeze(0).to(device)  # (1, 3, 3, H, W)
        prediction = model(batch)[0].detach().cpu().tolist()

    lat, lon = prediction
    return lat, lon


def hitNextButton():
    next_button_template = "DataSet/KeyImages/NextButton.png"

    next_button_location = None
    while next_button_location is None:
        try:
            next_button_location = pyautogui.locateOnScreen(
                next_button_template, confidence=0.9
            )
        except:
            time.sleep(1)
        print("Searching for Next Button on screen...")

    next_button_center = pyautogui.center(next_button_location)
    pyautogui.click(next_button_center)
    time.sleep(0.5)

    pyautogui.click(next_button_center)
    time.sleep(1)

    # Double extra clicks to handle play-again prompts
    pyautogui.click(next_button_center)
    time.sleep(0.5)
    pyautogui.click(next_button_center)
    time.sleep(0.5)


def wait_for_next_prompt():
    """
    Block until the user chooses an action:
    - 'n' => proceed to hit Next
    - 'm' => skip hitting Next (in case you've already advanced)
    - 'q' => handled by emergencyStop, but we also break defensively
    """
    print("Press 'n' for Next, 'm' to skip Next (already clicked), or 'q' to stop...")
    while True:
        if keyboard.is_pressed("n"):
            return "next"
        if keyboard.is_pressed("m"):
            return "skip"
        if keyboard.is_pressed("q"):
            return "quit"
        time.sleep(0.1)


def emergencyStop():
    print("Press 'q' to stop the program...")
    while True:
        if keyboard.is_pressed("q"):
            print("Emergency stop activated. Exiting program...")
            os._exit(0)  # kills the whole process immediately
        time.sleep(0.1)


def main():
    # Use 256px resized inputs (matching saved guess images)
    transform = get_transforms(split="test", image_size=IMAGE_SIZE_MODEL)
    model, device = load_model()
    base_map_size = get_base_map_dimensions()
    global_img_num = findCurrentImageNumber()
    GUESS_DIR.mkdir(parents=True, exist_ok=True)
    GUESS_COORDS_DIR.mkdir(parents=True, exist_ok=True)
    GUESS_MAP_DIR.mkdir(parents=True, exist_ok=True)

    # Start emergency stop watcher in a background thread
    stop_thread = threading.Thread(target=emergencyStop, daemon=True)
    stop_thread.start()

    

    for img_num in range(global_img_num, global_img_num + ROUNDS):
        
        print("Press 'r' to start the prediction process...")
        while True:
            if keyboard.is_pressed("r"):
                print("Starting prediction process...")
                break
            time.sleep(0.1)
        
        
        checkMapNameOnScreen()

        view_tensor = capture_views(img_num, transform)
        lat, lon = predict_coordinates(model, device, view_tensor)

        print(f"Predicted Coordinates -> Lat: {lat:.6f}, Lon: {lon:.6f}")

        coords_path = GUESS_COORDS_DIR / f"image{img_num}_coords.txt"
        with open(coords_path, "w") as f:
            f.write(f"{lat}\n{lon}\n")

        save_prediction_map(lat, lon, img_num)

        wait_for_guess_and_click(lat, lon, img_num, base_map_size)

        action = wait_for_next_prompt()
        if action == "next":
            hitNextButton()
        elif action == "skip":
            print("Skipping Next click for this round.")
        else:
            print("Stopping by user request.")
            break


if __name__ == "__main__":
    main()
