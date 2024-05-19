from asyncio.windows_events import NULL
import tkinter as tk
import filters as flt
import numpy as np
import cv2 as cv
import os 
import matplotlib.pyplot as plt
from tkinter import filedialog, OptionMenu, messagebox
from PIL import Image, ImageTk

class ImageWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")

        # Initial main window size
        self.window_width = 980
        self.window_height = 520

        # Initial size and position of image display area
        self.img_display_width = int(self.window_width * 0.75)
        self.img_display_height = int(self.window_height * 0.7)
        self.img_display_x = 10
        self.img_display_y = 10

        # Initial size and position of the load button
        self.load_button_x = 10
        self.load_button_y = self.img_display_height + 20

        # Create main canvas
        self.canvas = tk.Canvas(master, width=self.window_width, height=self.window_height, bg="white")
        self.canvas.pack()

        # Create rectangle space for image display
        self.img_display = self.canvas.create_rectangle(self.img_display_x, self.img_display_y,
                                                        self.img_display_x + self.img_display_width,
                                                        self.img_display_y + self.img_display_height,
                                                        outline="black")

        self.canvas.bind("<Button-1>", self.hide_slider)
        # Button to load image
        self.load_image_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_image_button.place(x=self.load_button_x, y=self.load_button_y)

        # Button to Original image
        self.original_image_button = tk.Button(master, text="Original image", command=self.original_image)
        self.original_image_button.place(x=self.load_button_x, y=self.load_button_y + 40)

        # Filter options
        self.f_options = ["Filters", "1-LPF", "2-HPF", "3-Mean", "4-Median"]
        self.filter_var = tk.StringVar()
        self.filter_var.set(self.f_options[0])  # Default option
        self.filter_menu = OptionMenu(master, self.filter_var, *self.f_options, command=self.on_option_selected)
        self.filter_menu.place(x=self.img_display_x + self.img_display_width + 20, y=self.img_display_y, width=200)

        # Edge Detection options
        self.edge_options = ["Edge detection", "1-Prewitt", "2-Sobel", "3-Roberts"]
        self.edge_detection_var = tk.StringVar()
        self.edge_detection_var.set(self.edge_options[0])  # Default option
        self.edge_detection_menu = OptionMenu(master, self.edge_detection_var, *self.edge_options, command=self.on_option_selected)
        self.edge_detection_menu.place(x=self.img_display_x + self.img_display_width + 20, y=self.img_display_y + 40, width=200)

        # Morphological operation options
        self.morph_options = ["Morphological operation", "1-Erosion", "2-Dilation", "3-Open", "4-Close"]
        self.morph_var = tk.StringVar()
        self.morph_var.set(self.morph_options[0])  # Default option
        self.morph_menu = OptionMenu(master, self.morph_var, *self.morph_options, command=self.on_option_selected)
        self.morph_menu.place(x=self.img_display_x + self.img_display_width + 20, y=self.img_display_y + 80, width=200)

        # Button "Hough Transforming" beside the space
        self.hough_button = tk.Button(master, text="Hough Transform", width=10, height=1, command=self.apply_hough)
        self.hough_button.place(x=self.img_display_x + self.img_display_width + 20, y=self.img_display_y + 120, width=200)

        # Segmentation options
        self.options = ["Segmentation", "1-Split & Merge", "2-Thresholding"]
        self.seg_var = tk.StringVar()
        self.seg_var.set(self.options[0])  # Default option
        self.seg_menu = OptionMenu(master, self.seg_var, *self.options, command=self.on_option_selected)
        self.seg_menu.place(x=self.img_display_x + self.img_display_width + 20, y=self.img_display_y + 160, width=200)

        # Display default image
        self.default_image_path = "default-image.png"
        self.is_default_image = True
        try:
            self.image = Image.open(self.default_image_path)
            self.image_width, self.image_height = self.image.size
            self.load_default_image(self.default_image_path)
        except Exception as e:
            print(f"Error loading default image: {e}")

    def original_image(self):
        if not self.is_default_image:
            self.display_image(self.image)
        self.reset_ui()
        

    def on_option_selected(self, option):
        if self.is_default_image:
            messagebox.showinfo("Info", "Please load an image to apply operations.")
            return
        # Reset all selections
        self.filter_var.set(self.f_options[0])
        self.edge_detection_var.set(self.edge_options[0])
        self.morph_var.set(self.morph_options[0])
        self.seg_var.set(self.options[0])
        
        
        # Remove any existing slider and label
        self.hide_slider(None)

        if option in self.f_options and option != "Filters":
            self.filter_var.set(option)
            if option != "2-HPF":
                self.slider = tk.Scale(self.master, from_=1, to=20, orient="horizontal", command=self.update_filter)
                self.slider.set(5)
                self.toggle_slider()
            self.apply_filter(option)
        elif option in self.edge_options and option != "Edge detection":
            self.edge_detection_var.set(option)
            self.apply_edge_detection(option)
        elif option in self.morph_options and option != "Morphological operation":
            self.morph_var.set(option)
            self.slider = tk.Scale(self.master, from_=1, to=20, orient="horizontal", command=self.update_morphological_operation)
            self.slider.set(5)
            self.toggle_slider()
        elif option in self.options and option != "Segmentation":
            self.seg_var.set(option)
            if option == "2-Thresholding":
                self.slider = tk.Scale(self.master, from_=1, to=255, orient="horizontal", command=self.update_thresholding)
                self.slider.set(127)
                self.toggle_slider("Value", 500)
            else:
                self.apply_segmentation("1-Split & Merge")
             

    def load_default_image(self, path):
        image = Image.open(path)
        image = image.resize((self.img_display_width, self.img_display_height))
        img_display_center_x = self.img_display_x
        img_display_center_y = self.img_display_y
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(img_display_center_x, img_display_center_y, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def load_image(self):
        initial_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Test photos")  
        file_path = filedialog.askopenfilename(initialdir=initial_dir)
        if file_path:
            self.image = Image.open(file_path)
            self.image_width, self.image_height = self.image.size
            aspect_ratio = self.image_width / self.image_height

            if self.image_width > self.img_display_width or self.image_height > self.img_display_height:
                if aspect_ratio > 1:
                    self.image_width = self.img_display_width
                    self.image_height = int(self.image_width / aspect_ratio)
                    if self.image_height > self.img_display_height:
                        self.image_height = self.img_display_height
                else:
                    self.image_height = self.img_display_height
                    self.image_width = int(self.image_height * aspect_ratio)
                    if self.image_width > self.img_display_width:
                        self.image_width = self.img_display_width

                self.image = self.image.resize((self.image_width, self.image_height))
            self.is_default_image = False

            self.display_image(self.image)
            self.reset_ui()

    def display_image(self, image):
        photo = ImageTk.PhotoImage(image)
        img_display_center_x = self.img_display_x + (self.img_display_width - self.image_width) / 2
        img_display_center_y = self.img_display_y + (self.img_display_height - self.image_height) / 2
        self.canvas.create_image(img_display_center_x, img_display_center_y, anchor=tk.NW, image=photo)
        self.canvas.image = photo


    def toggle_slider(self, value="Kernel size", X=460):
        self.slider.place(x=self.load_button_x + 540, y=self.load_button_y + 5, width=200)
        if hasattr(self, 'text_label'):
            self.text_label.destroy()
        self.text_label = tk.Label(self.master, text=f"{value}", bg="lightgray", font=("Arial", 12))
        self.text_label.place(x=self.load_button_x + X, y=self.load_button_y + 5)

    def hide_slider(self, event):
        if hasattr(self, 'slider') and self.slider.winfo_ismapped():
            self.slider.place_forget()
            self.text_label.destroy()

    def update_filter(self, value):
        filter_name = self.filter_var.get()
        if filter_name == "1-LPF":
            self.apply_filter("1-LPF", int(value))
        elif filter_name == "3-Mean":
            self.apply_filter("3-Mean", int(value))
        elif filter_name == "4-Median":
            self.apply_filter("4-Median", int(value))

    def update_morphological_operation(self, value):
        morph_name = self.morph_var.get()
        if morph_name == "1-Erosion":
            self.apply_morphological_operation("1-Erosion", int(value))
        elif morph_name == "2-Dilation":
            self.apply_morphological_operation("2-Dilation", int(value))
        elif morph_name == "3-Open":
            self.apply_morphological_operation("3-Open", int(value))
        elif morph_name == "4-Close":
            self.apply_morphological_operation("4-Close", int(value))

    def update_thresholding(self, value):
        self.apply_segmentation("2-Thresholding", int(value))

    def apply_filter(self, filter_name, kernel_size=5):
        result_image=NULL
        if self.image:
            image_array = np.array(self.image)
            if filter_name == "1-LPF":
                result_image = flt.apply_lpf(image_array, kernel_size)
            elif filter_name == "2-HPF":
                result_image = flt.apply_hpf(image_array)
            elif filter_name == "3-Mean":
                result_image = flt.apply_mean(image_array, kernel_size)
            elif filter_name == "4-Median":
                result_image = flt.apply_median(image_array, kernel_size)

            self.display_image(Image.fromarray(result_image))

    def apply_edge_detection(self, edge_name):
        if self.image:
            result_image=NULL
            if edge_name == "1-Prewitt":
                result_image = flt.apply_prewitt(np.array(self.image))
            elif edge_name == "2-Sobel":
                result_image = flt.apply_sobel(np.array(self.image))
            elif edge_name == "3-Roberts":
                result_image = flt.apply_roberts(np.array(self.image))

            self.display_image(Image.fromarray(result_image))

    def apply_morphological_operation(self, morph_name, kernel_size=5):
        if self.image:
            result_image=NULL
            image_array = np.array(self.image)
            if morph_name == "1-Erosion":
                result_image = flt.get_erosion(image_array, kernel_size)
            elif morph_name == "2-Dilation":
                result_image = flt.get_dilation(image_array, kernel_size)
            elif morph_name == "3-Open":
                result_image = flt.get_open(image_array, kernel_size)
            elif morph_name == "4-Close":
                result_image = flt.get_close(image_array, kernel_size)

            self.display_image(Image.fromarray(result_image))

    def apply_hough(self):
        if self.is_default_image:
            messagebox.showinfo("Info", "Please load an image to apply operations.")
            return
        self.reset_ui()
        result_image = flt.get_hough_transform(np.array(self.image))
        self.display_image(Image.fromarray(result_image))

    def apply_segmentation(self, seg_name, value=5):
        result_image=NULL
        if self.image:
            if seg_name == "1-Split & Merge":
                result_image = flt.get_seg_split_and_merge(self.image)
            elif seg_name == "2-Thresholding":
                result_image = flt.get_seg_threshold(np.array(self.image), value)

            self.display_image(Image.fromarray(result_image))

    def reset_ui(self):
        # Reset filter, edge detection, morph operation, and segmentation options to default
        self.filter_var.set(self.f_options[0])
        self.edge_detection_var.set(self.edge_options[0])
        self.morph_var.set(self.morph_options[0])
        self.seg_var.set(self.options[0])

        # Hide the slider if it is visible
        if hasattr(self, 'slider') and self.slider.winfo_ismapped():
            self.slider.place_forget()
            self.text_label.destroy()
    


def main():
    root = tk.Tk()
    app = ImageWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()
