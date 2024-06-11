import cv2
import numpy as np
from tkinter import Tk, Scale, HORIZONTAL, Button, Label, colorchooser, Toplevel, Frame, Scrollbar, Canvas, Entry, filedialog, StringVar, OptionMenu
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk

original_image = None

palette_settings = {
    'hue_shift': 0,
    'saturation_factor': 0.4,
    'brightness_beta': 0,
    'white_intensity': 1.0,
    'black_intensity': 0.1
}

def apply_magic_pro_filter(image, settings):
    if image is None or image.size == 0:
        return None

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, settings['gaussian_blur'], 0)

    # Convert to LAB color space to work with brightness
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=settings['clahe_clip_limit'], tileGridSize=settings['clahe_tile_grid_size'])
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Apply noise reduction
    denoised_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, settings['h'], settings['hForColorComponents'], settings['templateWindowSize'], settings['searchWindowSize'])

    # Increase contrast and brightness
    adjusted = cv2.convertScaleAbs(denoised_image, alpha=settings['contrast_alpha'], beta=settings['brightness_beta'])

    # Apply sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)

    # Convert to grayscale
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   settings['adaptive_thresh_block_size'], settings['adaptive_thresh_C'])

    # Use morphological operations to thicken the text
    kernel = np.ones(settings['dilate_kernel_size'], np.uint8)
    thickened = cv2.dilate(binary, kernel, iterations=settings['dilate_iterations'])

    # Merge thickened text back with the original color image
    colored_thickened = cv2.bitwise_and(sharpened, sharpened, mask=thickened)

    # Convert to HSV color space
    hsv = cv2.cvtColor(colored_thickened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Increase saturation
    s = cv2.multiply(s, settings['saturation_factor'])

    # Merge the channels back and convert to BGR
    hsv_colored = cv2.merge([h, s, v])
    final_image = cv2.cvtColor(hsv_colored, cv2.COLOR_HSV2BGR)

    """-----------thicken the font-----------"""

    # Apply additional filter to thicken the font further
    gray_final = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    binary_final = cv2.adaptiveThreshold(gray_final, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         settings['adaptive_thresh_block_size_2'], settings['adaptive_thresh_C_2'])
    kernel_final = np.ones(settings['dilate_kernel_size_2'], np.uint8)
    thickened_final = cv2.dilate(binary_final, kernel_final, iterations=settings['dilate_iterations_2'])

    final_image_colored_thickened = cv2.bitwise_and(final_image, final_image, mask=thickened_final)

    """------------black_point----------"""
    # Convert to LAB color space
    lab = cv2.cvtColor(final_image_colored_thickened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply Black Point adjustment
    black_point = settings['black_point']
    l = cv2.addWeighted(l, 1.0, l, 0, -black_point)

    lab = cv2.merge((l, a, b))
    final_image_colored_thickened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return final_image_colored_thickened

def apply_color_palette(image, palette_settings):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply hue shift
    h = (h.astype(int) + palette_settings.get('hue_shift', 0)) % 180
    h = h.astype(np.uint8)

    # Apply saturation factor
    s = cv2.multiply(s, palette_settings.get('saturation_factor', 1.0))

    # Merge the channels back and convert to BGR
    hsv_colored = cv2.merge([h, s, v])
    image = cv2.cvtColor(hsv_colored, cv2.COLOR_HSV2BGR)

    # Apply brightness adjustment
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=palette_settings.get('brightness_beta', 0))

    # Apply white intensity adjustment
    white_intensity = palette_settings.get('white_intensity', 1.0) or 1.0
    white_mask = cv2.inRange(image, np.array([200, 200, 200]), np.array([255, 255, 255]))
    if white_intensity != 1.0:
        image[white_mask != 0] = cv2.convertScaleAbs(image[white_mask != 0], alpha=white_intensity, beta=0)

    # Apply black intensity adjustment
    black_intensity = palette_settings.get('black_intensity', 1.0) or 1.0
    black_mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([50, 50, 50]))
    if black_intensity != 1.0:
        image[black_mask != 0] = cv2.convertScaleAbs(image[black_mask != 0], alpha=black_intensity, beta=0)

    return image

def update_image():
    global original_image
    if original_image is None:
        return
    settings = {
        'gaussian_blur': (blur_slider.get(), blur_slider.get()),
        'clahe_clip_limit': clahe_clip_limit_slider.get(),
        'clahe_tile_grid_size': (clahe_tile_grid_size_slider.get(), clahe_tile_grid_size_slider.get()),
        'contrast_alpha': contrast_alpha_slider.get(),
        'brightness_beta': brightness_beta_slider.get(),
        'adaptive_thresh_block_size': adaptive_thresh_block_size_slider.get() | 1,  # Ensure odd number
        'adaptive_thresh_C': adaptive_thresh_C_slider.get(),
        'dilate_kernel_size': (dilate_kernel_size_slider.get(), dilate_kernel_size_slider.get()),
        'dilate_iterations': dilate_iterations_slider.get(),
        'saturation_factor': saturation_factor_slider.get(),
        'h': h_slider.get(),
        'hForColorComponents': hForColorComponents_slider.get(),
        'templateWindowSize': templateWindowSize_slider.get(),
        'searchWindowSize': searchWindowSize_slider.get(),
        'adaptive_thresh_block_size_2': adaptive_thresh_block_size_2_slider.get() | 1,  # Ensure odd number
        'adaptive_thresh_C_2': adaptive_thresh_C_2_slider.get(),
        'dilate_kernel_size_2': (dilate_kernel_size_2_slider.get(), dilate_kernel_size_2_slider.get()),
        'dilate_iterations_2': dilate_iterations_2_slider.get(),
        'black_point': black_point_slider.get()
    }

    processed_image = apply_magic_pro_filter(original_image, settings)
    if processed_image is None:
        return

    # Apply color palette if specified
    if palette_settings:
        processed_image = apply_color_palette(processed_image, palette_settings)

    # Resize the image to fit within the display area
    display_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    display_image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=display_image)
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)

def open_image():
    global original_image
    file_path = askopenfilename()
    original_image = cv2.imread(file_path)
    update_image()

def save_image():
    file_path = asksaveasfilename(defaultextension=".jpg")
    processed_image = apply_magic_pro_filter(original_image, {
        'gaussian_blur': (blur_slider.get(), blur_slider.get()),
        'clahe_clip_limit': clahe_clip_limit_slider.get(),
        'clahe_tile_grid_size': (clahe_tile_grid_size_slider.get(), clahe_tile_grid_size_slider.get()),
        'contrast_alpha': contrast_alpha_slider.get(),
        'brightness_beta': brightness_beta_slider.get(),
        'adaptive_thresh_block_size': adaptive_thresh_block_size_slider.get(),
        'adaptive_thresh_C': adaptive_thresh_C_slider.get(),
        'dilate_kernel_size': (dilate_kernel_size_slider.get(), dilate_kernel_size_slider.get()),
        'dilate_iterations': dilate_iterations_slider.get(),
        'saturation_factor': saturation_factor_slider.get(),
        'h': h_slider.get(),
        'hForColorComponents': hForColorComponents_slider.get(),
        'templateWindowSize': templateWindowSize_slider.get(),
        'searchWindowSize': searchWindowSize_slider.get(),
        'adaptive_thresh_block_size_2': adaptive_thresh_block_size_2_slider.get(),
        'adaptive_thresh_C_2': adaptive_thresh_C_2_slider.get(),
        'dilate_kernel_size_2': (dilate_kernel_size_2_slider.get(), dilate_kernel_size_2_slider.get()),
        'dilate_iterations_2': dilate_iterations_2_slider.get(),
        'black_point': black_point_slider.get()
    })

    # Apply color palette if specified
    if palette_settings:
        processed_image = apply_color_palette(processed_image, palette_settings)

    cv2.imwrite(file_path, processed_image)

def choose_white_color():
    global palette_settings
    color_code = colorchooser.askcolor(title="Choose White Color")
    if color_code:
        white_color = color_code[0]
        palette_settings['white_intensity'] = white_color
        white_color_button.config(bg=color_code[1])

def choose_black_color():
    global palette_settings
    color_code = colorchooser.askcolor(title="Choose Black Color")
    if color_code:
        black_color = color_code[0]
        palette_settings['black_intensity'] = black_color
        black_color_button.config(bg=color_code[1])

def open_palette_settings():
    def apply_palette_settings():
        global palette_settings
        palette_settings = {
            'hue_shift': hue_shift_slider.get(),
            'saturation_factor': saturation_factor_slider_palette.get(),
            'brightness_beta': brightness_beta_slider_palette.get(),
            'white_intensity': white_intensity_slider.get(),
            'black_intensity': black_intensity_slider.get()
        }
        update_image()

    palette_window = Toplevel(root)
    palette_window.title("Color Palette Settings")

    hue_shift_slider = Scale(palette_window, from_=-180, to=180, orient=HORIZONTAL, label="Hue Shift")
    hue_shift_slider.set(palette_settings.get('hue_shift', 0))
    hue_shift_slider.grid(row=0, column=0, padx=5, pady=5)

    saturation_factor_slider_palette = Scale(palette_window, from_=0.1, to=5, resolution=0.1, orient=HORIZONTAL, label="Saturation Factor")
    saturation_factor_slider_palette.set(palette_settings.get('saturation_factor', 1.0))
    saturation_factor_slider_palette.grid(row=0, column=1, padx=5, pady=5)

    brightness_beta_slider_palette = Scale(palette_window, from_=-100, to=100, orient=HORIZONTAL, label="Brightness Beta")
    brightness_beta_slider_palette.set(palette_settings.get('brightness_beta', 0))
    brightness_beta_slider_palette.grid(row=1, column=0, padx=5, pady=5)

    white_intensity_slider = Scale(palette_window, from_=0.1, to=2, resolution=0.1, orient=HORIZONTAL, label="White Intensity")
    white_intensity_slider.set(palette_settings.get('white_intensity', 1.0))
    white_intensity_slider.grid(row=1, column=1, padx=5, pady=5)

    black_intensity_slider = Scale(palette_window, from_=0.1, to=2, resolution=0.1, orient=HORIZONTAL, label="Black Intensity")
    black_intensity_slider.set(palette_settings.get('black_intensity', 1.0))
    black_intensity_slider.grid(row=2, column=0, padx=5, pady=5)

    Button(palette_window, text="Apply", command=apply_palette_settings).grid(row=2, column=1, padx=5, pady=5)

def export_filter():
    model_name = model_name_var.get()
    if not model_name:
        print("Model name is required")
        return

    filter_settings = {
        'gaussian_blur': (blur_slider.get(), blur_slider.get()),
        'clahe_clip_limit': clahe_clip_limit_slider.get(),
        'clahe_tile_grid_size': (clahe_tile_grid_size_slider.get(), clahe_tile_grid_size_slider.get()),
        'contrast_alpha': contrast_alpha_slider.get(),
        'brightness_beta': brightness_beta_slider.get(),
        'adaptive_thresh_block_size': adaptive_thresh_block_size_slider.get() | 1,  # Ensure odd number
        'adaptive_thresh_C': adaptive_thresh_C_slider.get(),
        'dilate_kernel_size': (dilate_kernel_size_slider.get(), dilate_kernel_size_slider.get()),
        'dilate_iterations': dilate_iterations_slider.get(),
        'saturation_factor': saturation_factor_slider.get(),
        'h': h_slider.get(),
        'hForColorComponents': hForColorComponents_slider.get(),
        'templateWindowSize': templateWindowSize_slider.get(),
        'searchWindowSize': searchWindowSize_slider.get(),
        'adaptive_thresh_block_size_2': adaptive_thresh_block_size_2_slider.get() | 1,  # Ensure odd number
        'adaptive_thresh_C_2': adaptive_thresh_C_2_slider.get(),
        'dilate_kernel_size_2': (dilate_kernel_size_2_slider.get(), dilate_kernel_size_2_slider.get()),
        'dilate_iterations_2': dilate_iterations_2_slider.get(),
        'black_point': black_point_slider.get(),
        'color_palettes': {
            'custom': {
                'hue_shift': palette_settings['hue_shift'],
                'saturation_factor': palette_settings['saturation_factor'],
                'brightness_beta': palette_settings['brightness_beta'],
                'white_intensity': palette_settings['white_intensity'],
                'black_intensity': palette_settings['black_intensity']
            }
        }
    }

    # Read the existing constants_temp.py file
    with open('constants_temp.py', 'r') as file:
        lines = file.readlines()

    # Find the model's dictionary and update it
    model_found = False
    for i, line in enumerate(lines):
        if f"'{model_name}':" in line:
            model_found = True
            j = i + 1
            while lines[j].strip() != '},' and lines[j].strip() != '}':
                j += 1
            # Delete the old settings for the model
            del lines[i + 1:j]
            # Insert new settings
            insert_settings = [f"        '{key}': {value},\n" for key, value in filter_settings.items() if key != 'color_palettes']
            color_palette_lines = [f"        'color_palettes': {{\n"]
            for palette, settings in filter_settings['color_palettes'].items():
                color_palette_lines.append(f"            '{palette}': {{\n")
                color_palette_lines.extend([f"                '{k}': {v},\n" for k, v in settings.items()])
            #     color_palette_lines.append("            },\n")
            # color_palette_lines.append("        },\n")
            lines[i + 1:i + 1] = insert_settings + color_palette_lines
            break

    # If the model was not found, append it correctly
    if not model_found:
        lines.append(f"    '{model_name}': {{\n")
        lines.extend([f"        '{key}': {value},\n" for key, value in filter_settings.items() if key != 'color_palettes'])
        lines.append("        'color_palettes': {\n")
        for palette, settings in filter_settings['color_palettes'].items():
            lines.append(f"            '{palette}': {{\n")
            lines.extend([f"                '{k}': {v},\n" for k, v in settings.items()])
            lines.append("            },\n")
        lines.append("        },\n")
        lines.append("    },\n")

    # Write the updated settings back to constants_temp.py
    with open('constants_temp.py', 'w') as file:
        file.writelines(lines)

    print(f"Filter settings for model '{model_name}' have been updated.")



root = Tk()
root.title("Image Filter Adjuster")

# Create a main frame with horizontal layout
main_frame = Frame(root)
main_frame.pack(fill="both", expand=1)

# Create a frame for controls and pack it to the left
controls_frame = Frame(main_frame)
controls_frame.pack(side="left", fill="both", expand=1)

# Create a canvas and a vertical scrollbar to allow scrolling through the sliders
canvas = Canvas(controls_frame)
canvas.pack(side="left", fill="both", expand=1)

scrollbar = Scrollbar(controls_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create another frame inside the canvas
frame = Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

# # Add a label and entry for model name
# model_name_var = StringVar()
# model_name_label = Label(frame, text="Model Name")
# model_name_label.grid(row=11, column=0, padx=5, pady=5)
# model_name_entry = Entry(frame, textvariable=model_name_var)
# model_name_entry.grid(row=11, column=1, columnspan=2, padx=5, pady=5)

# Add a label and entry for model name
model_name_var = StringVar()
model_name_var.set("GLOBAL")  # Set default value
model_name_label = Label(frame, text="Model Name")
model_name_label.grid(row=11, column=0, padx=5, pady=5)
# Create a dropdown menu for model names
model_names = ['GLOBAL', 'SOPHACA', 'SPR', 'GPM', 'SOPHADIMS', 'RECAMED', 'COOPER']
model_name_dropdown = OptionMenu(frame, model_name_var, *model_names)
model_name_dropdown.grid(row=11, column=1, columnspan=2, padx=5, pady=5)

# Add a button to export the filter settings
export_button = Button(frame, text="Export Filter", command=export_filter)
export_button.grid(row=12, column=0, columnspan=3, padx=5, pady=5)

# Create sliders for each parameter
blur_slider = Scale(frame, from_=1, to=20, resolution=2, orient=HORIZONTAL, label="Gaussian Blur")
blur_slider.set(3)
blur_slider.grid(row=0, column=0, padx=5, pady=5)

clahe_clip_limit_slider = Scale(frame, from_=1, to=10, resolution=0.1, orient=HORIZONTAL, label="CLAHE Clip Limit")
clahe_clip_limit_slider.set(1.5)
clahe_clip_limit_slider.grid(row=0, column=1, padx=5, pady=5)

clahe_tile_grid_size_slider = Scale(frame, from_=1, to=100, orient=HORIZONTAL, label="CLAHE Tile Grid Size")
clahe_tile_grid_size_slider.set(20)
clahe_tile_grid_size_slider.grid(row=0, column=2, padx=5, pady=5)

contrast_alpha_slider = Scale(frame, from_=0.1, to=3, resolution=0.1, orient=HORIZONTAL, label="Contrast Alpha")
contrast_alpha_slider.set(1.1)
contrast_alpha_slider.grid(row=1, column=0, padx=5, pady=5)

brightness_beta_slider = Scale(frame, from_=-100, to=100, orient=HORIZONTAL, label="Brightness Beta")
brightness_beta_slider.set(30)
brightness_beta_slider.grid(row=1, column=1, padx=5, pady=5)

adaptive_thresh_block_size_slider = Scale(frame, from_=3, to=101, orient=HORIZONTAL, label="Adaptive Threshold Block Size")
adaptive_thresh_block_size_slider.set(85)
adaptive_thresh_block_size_slider.grid(row=1, column=2, padx=5, pady=5)

adaptive_thresh_C_slider = Scale(frame, from_=-100, to=100, orient=HORIZONTAL, label="Adaptive Threshold C")
adaptive_thresh_C_slider.set(150)
adaptive_thresh_C_slider.grid(row=2, column=0, padx=5, pady=5)

dilate_kernel_size_slider = Scale(frame, from_=1, to=10, orient=HORIZONTAL, label="Dilate Kernel Size")
dilate_kernel_size_slider.set(1)
dilate_kernel_size_slider.grid(row=2, column=1, padx=5, pady=5)

dilate_iterations_slider = Scale(frame, from_=1, to=10, orient=HORIZONTAL, label="Dilate Iterations")
dilate_iterations_slider.set(1)
dilate_iterations_slider.grid(row=2, column=2, padx=5, pady=5)

saturation_factor_slider = Scale(frame, from_=0.1, to=5, resolution=0.1, orient=HORIZONTAL, label="Saturation Factor")
saturation_factor_slider.set(3.0)
saturation_factor_slider.grid(row=3, column=0, padx=5, pady=5)

h_slider = Scale(frame, from_=0, to=100, orient=HORIZONTAL, label="Denoising Strength (h)")
h_slider.set(5)
h_slider.grid(row=3, column=1, padx=5, pady=5)

hForColorComponents_slider = Scale(frame, from_=0, to=100, orient=HORIZONTAL, label="Denoising Strength for Color Components (hForColorComponents)")
hForColorComponents_slider.set(0)
hForColorComponents_slider.grid(row=3, column=2, padx=5, pady=5)

templateWindowSize_slider = Scale(frame, from_=1, to=10, orient=HORIZONTAL, label="Template Window Size")
templateWindowSize_slider.set(7)
templateWindowSize_slider.grid(row=4, column=0, padx=5, pady=5)

searchWindowSize_slider = Scale(frame, from_=1, to=50, orient=HORIZONTAL, label="Search Window Size")
searchWindowSize_slider.set(1)
searchWindowSize_slider.grid(row=4, column=1, padx=5, pady=5)

adaptive_thresh_block_size_2_slider = Scale(frame, from_=3, to=101, orient=HORIZONTAL, label="Adaptive Threshold Block Size 2")
adaptive_thresh_block_size_2_slider.set(95)
adaptive_thresh_block_size_2_slider.grid(row=4, column=2, padx=5, pady=5)

adaptive_thresh_C_2_slider = Scale(frame, from_=-100, to=100, orient=HORIZONTAL, label="Adaptive Threshold C 2")
adaptive_thresh_C_2_slider.set(100)
adaptive_thresh_C_2_slider.grid(row=5, column=0, padx=5, pady=5)

dilate_kernel_size_2_slider = Scale(frame, from_=1, to=10, orient=HORIZONTAL, label="Dilate Kernel Size 2")
dilate_kernel_size_2_slider.set(1)
dilate_kernel_size_2_slider.grid(row=5, column=1, padx=5, pady=5)

dilate_iterations_2_slider = Scale(frame, from_=1, to=10, orient=HORIZONTAL, label="Dilate Iterations 2")
dilate_iterations_2_slider.set(1)
dilate_iterations_2_slider.grid(row=5, column=2, padx=5, pady=5)

black_point_slider = Scale(frame, from_=0, to=100, orient=HORIZONTAL, label="Black Point")
black_point_slider.set(10)
black_point_slider.grid(row=6, column=0, padx=5, pady=5)

# Add color picker buttons for white and black intensities
white_color_button = Button(frame, text="Choose White Color", command=choose_white_color)
white_color_button.grid(row=6, column=1, padx=5, pady=5)

black_color_button = Button(frame, text="Choose Black Color", command=choose_black_color)
black_color_button.grid(row=6, column=2, padx=5, pady=5)

# Add button to open color palette settings
Button(frame, text="Open Color Palette Settings", command=open_palette_settings).grid(row=7, column=0, columnspan=3, padx=5, pady=5)

# Add buttons to open and save images
Button(frame, text="Open Image", command=open_image).grid(row=8, column=0, columnspan=3, padx=5, pady=5)
Button(frame, text="Save Image", command=save_image).grid(row=9, column=0, columnspan=3, padx=5, pady=5)

# Add Apply button to apply changes
apply_button = Button(frame, text="Apply", command=update_image)
apply_button.grid(row=10, column=0, columnspan=3, padx=5, pady=5)

# Create a frame for the image and pack it to the right
image_frame = Frame(main_frame)
image_frame.pack(side="right", fill="both", expand=1)

# Add a label to display the image
image_label = Label(image_frame)
image_label.pack(fill="both", expand=1)

# Initialize color variables
white_color = (255, 255, 255)
black_color = (0, 0, 0)

# Initialize palette settings
palette_settings = {
    'hue_shift': 0,
    'saturation_factor': 0.4,
    'brightness_beta': 0,
    'white_intensity': 1.0,
    'black_intensity': 0.1
}

root.mainloop()