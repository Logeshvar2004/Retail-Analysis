import argparse
import json
import os
from typing import Any, Optional, Tuple, List, Dict

import cv2
import numpy as np

KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")

THICKNESS = 2
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR colors
WINDOW_NAME = "Draw Zones"
POLYGONS: List[List[Tuple[int, int]]] = [[]]
NAMES: List[str] = []

current_mouse_position: Optional[Tuple[int, int]] = None

def resolve_source(source_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(source_path):
        return None

    cap = cv2.VideoCapture(source_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    return frame

def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global current_mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        POLYGONS[-1].append((x, y))

def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
    global POLYGONS, current_mouse_position
    image[:] = original_image.copy()
    for idx, polygon in enumerate(POLYGONS):
        color = COLORS[idx % len(COLORS)] if idx < len(POLYGONS) - 1 else (255, 255, 255)

        if len(polygon) > 1:
            for i in range(1, len(polygon)):
                cv2.line(
                    img=image,
                    pt1=polygon[i - 1],
                    pt2=polygon[i],
                    color=color,
                    thickness=THICKNESS,
                )
            if idx < len(POLYGONS) - 1:
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=polygon[0],
                    color=color,
                    thickness=THICKNESS,
                )
        if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=current_mouse_position,
                color=color,
                thickness=THICKNESS,
            )
    cv2.imshow(WINDOW_NAME, image)

def close_and_finalize_polygon(image: np.ndarray, original_image: np.ndarray) -> None:
    if len(POLYGONS[-1]) > 2:
        cv2.line(
            img=image,
            pt1=POLYGONS[-1][-1],
            pt2=POLYGONS[-1][0],
            color=(255, 255, 255),
            thickness=THICKNESS,
        )
        # Prompt for zone name
        zone_name = input("Enter name for this zone: ")
        NAMES.append(zone_name)
    POLYGONS.append([])
    image[:] = original_image.copy()
    redraw_polygons(image)
    cv2.imshow(WINDOW_NAME, image)

def redraw_polygons(image: np.ndarray) -> None:
    for idx, polygon in enumerate(POLYGONS[:-1]):
        if len(polygon) > 1:
            color = COLORS[idx % len(COLORS)]
            for i in range(len(polygon) - 1):
                cv2.line(
                    img=image,
                    pt1=polygon[i],
                    pt2=polygon[i + 1],
                    color=color,
                    thickness=THICKNESS,
                )
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=polygon[0],
                color=color,
                thickness=THICKNESS,
            )

def save_polygons_to_json(polygons: List[List[Tuple[int, int]]], names: List[str], target_path: str):
    data_to_save: List[Dict[str, Any]] = []
    for polygon, name in zip(polygons, names):
        data_to_save.append({"name": name, "points": polygon})
    with open(target_path, "w") as f:
        json.dump(data_to_save, f, indent=4)

def main(source_path: str, zone_configuration_path: str) -> None:
    global current_mouse_position
    original_image = resolve_source(source_path=source_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    image = original_image.copy()
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ENTER or key == KEY_NEWLINE:
            close_and_finalize_polygon(image, original_image)
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            current_mouse_position = None
        elif key == KEY_SAVE:
            save_polygons_to_json(POLYGONS[:-1], NAMES, zone_configuration_path)
            print(f"Polygons saved to {zone_configuration_path}")
        redraw(image, original_image)
        if key == KEY_QUIT:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively draw polygons on images or video frames and save the annotations.")
    parser.add_argument("--source", type=str, required=True, help="Path to the source image or video file for drawing polygons.")
    parser.add_argument("--zone", type=str, required=True, help="Path where the polygon annotations will be saved as a JSON file.")
    arguments = parser.parse_args()
    
    main(
        source_path=arguments.source,
        zone_configuration_path=arguments.zone,
    )
