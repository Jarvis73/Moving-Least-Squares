import cv2
import sys
import os
import re
import numpy as np
from img_utils import (
    mls_affine_deformation, 
    mls_similarity_deformation, 
    mls_rigid_deformation
)

from PIL import Image

global deformation_output

class ControlPoints:
    def __init__(self):
        self.points = []
        self.original_points = []
        self.selected_point = None
        self.dragging = False
        self.dragged_points = set()
        self.insert_index = 0
        self.mouse_position = (0, 0)

    def dragging_point(self, point):
            x, y = point
            px, py = self.pending_point
            threshold = 5  # Adjust the threshold if necessary
            return abs(x - px) > threshold or abs(y - py) > threshold

    def add_point(self, point):
        self.points.append(point)
        self.original_points.append(point)

    def remove_selected_point(self):
        if self.selected_point is not None:
            self.points.pop(self.selected_point)
            self.original_points.pop(self.selected_point)
            self.dragged_points.discard(self.selected_point)
            self.selected_point = None

    def select_point(self, point, max_distance=10):
        for i, p in enumerate(self.points):
            if np.linalg.norm(np.array(p) - np.array(point)) <= max_distance:
                self.selected_point = i
                return True
        return False
    
    def unselect_point(self):
        self.selected_point = None

    def update_point(self, point):
        if self.selected_point is not None:
            self.points[self.selected_point] = point
            self.dragged_points.add(self.selected_point)

    def deselect_point(self):
        self.selected_point = None

    def is_selected(self, point):
        return self.selected_point is not None and self.points[self.selected_point] == point

    def has_been_dragged(self, index):
        return index in self.dragged_points

    def get_original_point(self, index):
        return self.original_points[index]

    # debug function
    def insert_point(self, x, y):
        self.points.append((x, y))
        self.original_points.append((x, y))


def draw_image_with_points(image, control_points):
    def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=5):
        dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((1 - r) * pt1[0] + r * pt2[0])
            y = int((1 - r) * pt1[1] + r * pt2[1])
            pts.append((x, y))
        for p in pts[::2]:
            cv2.circle(img, p, thickness, color, -1)

    new_image = image.copy()
    for idx, point in enumerate(control_points.points):
        color = (0, 0, 255)  # Red for non-dragged points
        if idx in control_points.dragged_points:
            color = (128, 0, 128)  # Dark purple for dragged points
            # Draw dotted line between initial and current location
            draw_dotted_line(new_image, control_points.original_points[idx], point, (0, 0, 0), 1, 5)

        if idx == control_points.selected_point:
            color = (255, 0, 0)  # Blue for selected point

        cv2.circle(new_image, point, 5, color, -1)
    
    x, y = control_points.mouse_position
    cv2.putText(new_image, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return new_image


def mouse_callback(event, x, y, flags, param):
    image, control_points = param

    control_points.mouse_position = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        point_selected = control_points.select_point((x, y))
        if point_selected:
            if control_points.selected_point is not None:
                original_point = control_points.get_original_point(control_points.selected_point)
                current_point = control_points.points[control_points.selected_point]
                print(f"Original point: {original_point} -- Current point: {current_point}")
        else:
            control_points.unselect_point()  # Unselect the current point if clicked outside the selection zone
        control_points.dragging = True
        control_points.pending_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if not control_points.select_point((x, y)):
            if control_points.pending_point is not None and not control_points.dragging_point((x, y)):
                control_points.add_point((x, y))
                control_points.select_point((x, y))
                control_points.pending_point = None
        control_points.dragging = False

    elif event == cv2.EVENT_MOUSEMOVE and control_points.dragging:
        if control_points.selected_point is not None:
            control_points.update_point((x, y))

    updated_image = draw_image_with_points(image, control_points)
    cv2.imshow(window_name, updated_image)


def clear_points(control_points):
    control_points.points = []
    control_points.original_points = []
    control_points.dragged_points = set()
    control_points.selected_point = None

def key_callback(image, control_points, file_path):
    key = cv2.waitKey(1) & 0xFF

    insert_points_list = [[209, 236], [258, 218], [221, 336], [283, 321]]

    if key == ord('q') or key == 27:  # 'q' or ESC key to quit
        return False

    if key == ord('d'):  # 'd' key to delete the selected point
        control_points.remove_selected_point()

    if key == ord('c'):  # 'c' key to clear all control points
        clear_points(control_points)
        
    if key == ord('a'):  # 'a' key to create an affine deformation
        if cv2.getWindowProperty("Deformation", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Deformation")

        control_points_list = [[point[1], point[0]] for point in control_points.points]
        original_points_list = [[point[1], point[0]] for point in control_points.original_points]

        demo(file_path, original_points_list, control_points_list, "affine")

    if key == ord('s'):  # 's' key to create a similarity deformation
        if cv2.getWindowProperty("Deformation", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Deformation")
        control_points_list = [[point[1], point[0]] for point in control_points.points]
        original_points_list = [[point[1], point[0]] for point in control_points.original_points]

        demo(file_path, original_points_list, control_points_list, "similarity")

    if key == ord('r'):  # 'r' key to create a rigid deformation
        if cv2.getWindowProperty("Deformation", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Deformation")
        control_points_list = [[point[1], point[0]] for point in control_points.points]
        original_points_list = [[point[1], point[0]] for point in control_points.original_points]

        demo(file_path, original_points_list, control_points_list, "rigid")
    
    if key == ord('i'):  # 'i' key to insert the next point from the list
        if control_points.insert_index < len(insert_points_list):
            x, y = insert_points_list[control_points.insert_index]
            control_points.insert_point(x, y)
            control_points.insert_index += 1

    
    if key == ord('w'):  # '1' key to save the displayed image in "Deformation" window
        if cv2.getWindowProperty("Deformation", cv2.WND_PROP_VISIBLE) >= 1:
            global deformation_output
            current_path = os.path.dirname(os.path.realpath(__file__))
            image_folder = os.path.join(current_path, "images")

            os.makedirs(image_folder, exist_ok=True)

            filenames = os.listdir(image_folder)
            deform_numbers = [int(re.findall(r'\d+', f)[0]) for f in filenames if f.startswith("deform_") and f.endswith(".jpg")]

            if not deform_numbers:
                last_deform_number = 0
            else:
                last_deform_number = max(deform_numbers)

            new_deform_number = last_deform_number + 1
            new_filename = f"deform_{new_deform_number}.jpg"

            # Save the image with the new filename
            new_filepath = os.path.join(image_folder, new_filename)
            cv2.imwrite(new_filepath, deformation_output)
            print(f"Image saved as {new_filepath}")

    updated_image = draw_image_with_points(image, control_points)
    cv2.imshow(window_name, updated_image)

    return True

def demo(image, p, q, mode):
    global deformation_output

    p = np.array(p)
    q = np.array(q)

    image = np.array(Image.open(image))
    
    height, width, _ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)
    
    if mode == "affine":
        affine = mls_affine_deformation(vy, vx, p, q, alpha=1)
        aug1 = np.ones_like(image)
        aug1[vx, vy] = image[tuple(affine)]
    elif mode == "similarity":
        similar = mls_similarity_deformation(vy, vx, p, q, alpha=1)
        aug1 = np.ones_like(image)
        aug1[vx, vy] = image[tuple(similar)]
    elif mode == "rigid":
        rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
        aug1 = np.ones_like(image)
        aug1[vx, vy] = image[tuple(rigid)]

    deformation_output = cv2.cvtColor(aug1, cv2.COLOR_RGB2BGR)
    cv2.imshow("Deformation", deformation_output) 

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Usage: python live_demo.py <image_path>")
        print("\nHotkeys:")
        print("  q or ESC - Quit ")
        print("  d - Delete the selected control point")
        print("  c - Clear all control points")
        print("  a - Create an affine deformation and display it in a separate window")
        print("  s - Create a similarity deformation and display it in a separate window")
        print("  r - Create a rigid deformation and display it in a separate window")
        print("  w - Write the last deformation to the images folder")
    else:
        file_path = sys.argv[1]
        image = cv2.imread(file_path)

        if image is None:
            print(f"Error: Could not load image from '{file_path}'.")
            sys.exit(1)

        window_name = "Moving Least Squares Demo"
        control_points = ControlPoints()

        height, width, _ = image.shape
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)


        cv2.setMouseCallback(window_name, mouse_callback, (image, control_points))
        cv2.imshow(window_name, image)

        while key_callback(image, control_points, file_path):
            pass

        cv2.destroyAllWindows()