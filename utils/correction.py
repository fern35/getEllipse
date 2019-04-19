from tkinter import Tk, Button, messagebox, Toplevel
from tkinter.filedialog import asksaveasfile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import interpolate
import os
import json
import numpy as np


def interpolate_plot(X):
    tck1, u = interpolate.splprep([X[:, 0], X[:, 1]], s=0)
    unew = np.arange(0, 1.0, 0.01)
    out = interpolate.splev(unew, tck1)
    return np.vstack(out).T


def fit_ellipse(cont, method):
    x = cont[:, 0]
    y = cont[:, 1]

    x = x[:, None]
    y = y[:, None]

    D = np.hstack([x * x, x * y, y * y, x, y, np.ones(x.shape)])
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))

    if method == 1:
        n = np.argmax(np.abs(E))
    else:
        n = np.argmax(E)
    a = V[:, n]

    # -------------------Fit ellipse-------------------
    b, c, d, f, g, a = a[1] / 2., a[2], a[3] / 2., a[4] / 2., a[5], a[0]
    num = b * b - a * c
    cx = (c * d - b * f) / num
    cy = (a * f - b * d) / num

    angle = 0.5 * np.arctan(2 * b / (a - c)) * 180 / np.pi
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    a = np.sqrt(abs(up / down1))
    b = np.sqrt(abs(up / down2))

    # ---------------------Get path---------------------
    ell = Ellipse((cx, cy), a * 2., b * 2., angle)
    ell_coord = ell.get_verts()

    params = [cx, cy, a, b, angle]
    return params, ell_coord


def plot_ellipse(cx, cy, a, b, angle, ax):
    e1 = Ellipse((cx, cy), a, b,
                 angle=angle, linewidth=2, fill=False, zorder=2)
    ax.add_patch(e1)
    # long
    angle_rad = e1.angle / 180 * np.pi
    C = np.array([-e1.width / 2 * np.cos(angle_rad) + e1.center[0], -e1.height / 2 * np.sin(angle_rad) + e1.center[1]])
    D = np.array([e1.width / 2 * np.cos(angle_rad) + e1.center[0], e1.height / 2 * np.sin(angle_rad) + e1.center[1]])
    # short
    B = np.array([-e1.height / 2 * np.sin(angle_rad) + e1.center[0], e1.width / 2 * np.cos(angle_rad) + e1.center[1]])
    A = np.array([e1.height / 2 * np.sin(angle_rad) + e1.center[0], -e1.width / 2 * np.cos(angle_rad) + e1.center[1]])
    # center = (cx, cy)

    ax.scatter(A[0], A[1], label="A")
    ax.scatter(B[0], B[1], label="B")
    ax.scatter(C[0], C[1], label="C")
    ax.scatter(D[0], D[1], label="D")
    plt.legend()
    plt.axis("equal")
    plt.show()
    return e1


def get_ellipse_extremities(ellipse):
    # define main_axis and secondary_axis

    C = np.array([-ellipse.height / 2 * np.sin(ellipse.angle * np.pi / 180) + ellipse.center[0],
                  ellipse.height / 2 * np.cos(ellipse.angle * np.pi / 180) + ellipse.center[1]])
    D = np.array([ellipse.height / 2 * np.sin(ellipse.angle * np.pi / 180) + ellipse.center[0],
                  -ellipse.height / 2 * np.cos(ellipse.angle * np.pi / 180) + ellipse.center[1]])
    B = np.array([-ellipse.width / 2 * np.cos(ellipse.angle * np.pi / 180) + ellipse.center[0],
                  -ellipse.width / 2 * np.sin(ellipse.angle * np.pi / 180) + ellipse.center[1]])
    A = np.array([ellipse.width / 2 * np.cos(ellipse.angle * np.pi / 180) + ellipse.center[0],
                  ellipse.width / 2 * np.sin(ellipse.angle * np.pi / 180) + ellipse.center[1]])

    extremas = np.stack([A, B, C, D], axis=-1).T

    # define main axis based on extremas
    main_axis = np.argmin(extremas[:, 0])
    return extremas, main_axis


def get_fixed_and_updated_point_based_on_main_axis(ellipse, main_axis, extremity_type):
    extremas, _ = get_ellipse_extremities(ellipse)
    if main_axis == 1:
        fixed_pt = extremas[extremity_type % 2]
        updated_pt = extremas[(extremity_type + 1) % 2]
    if main_axis == 3:
        fixed_pt = extremas[(extremity_type) % 2 + 2]
        updated_pt = extremas[(extremity_type + 1) % 2 + 2]
    if main_axis == 0:
        fixed_pt = extremas[(extremity_type + 1) % 2]
        updated_pt = extremas[(extremity_type) % 2]
    if main_axis == 2:
        fixed_pt = extremas[(extremity_type + 1) % 2 + 2]
        updated_pt = extremas[(extremity_type) % 2 + 2]
    return fixed_pt, updated_pt


def update_ellipse_extremity(x, y, ellipse, main_axis, extremity_type, short_axis=False, center_fixed=False,
                             axis_fixed=False):
    extremas, _ = get_ellipse_extremities(ellipse)
    updated_pt = np.array([x, y])
    if main_axis == 1:
        fixed_pt = extremas[extremity_type % 2]
        if center_fixed:
            fixed_pt -= updated_pt - ellipse.center
        diff = updated_pt - fixed_pt
        if short_axis:
            height = np.linalg.norm(diff)
            width = ellipse.width
        else:
            width = np.linalg.norm(diff)
            height = ellipse.height
    elif main_axis == 3:
        fixed_pt = extremas[extremity_type % 2 + 2]
        if center_fixed:
            fixed_pt -= updated_pt - ellipse.center
        diff = updated_pt - fixed_pt
        if short_axis:
            width = np.linalg.norm(diff)
            height = ellipse.height
        else:
            width = ellipse.width
            height = np.linalg.norm(diff)
    elif main_axis == 0:
        fixed_pt = extremas[(extremity_type + 1) % 2]
        if center_fixed:
            fixed_pt -= updated_pt - ellipse.center
        diff = updated_pt - fixed_pt
        if short_axis:
            height = np.linalg.norm(diff)
            width = ellipse.width
        else:
            width = np.linalg.norm(diff)
            height = ellipse.height
    elif main_axis == 2:
        fixed_pt = extremas[(extremity_type + 1) % 2 + 2]
        if center_fixed:
            fixed_pt -= updated_pt - ellipse.center
        diff = updated_pt - fixed_pt
        if short_axis:
            width = np.linalg.norm(diff)
            height = ellipse.height
        else:
            width = ellipse.width
            height = np.linalg.norm(diff)
    if axis_fixed:
        angle = ellipse.angle
    else:
        angle = np.arctan(diff[1] / diff[0]) * 180 / np.pi

    if center_fixed:
        center = ellipse.center
    else:
        center = (updated_pt + fixed_pt) / 2
    # print("fixed pt", fixed_pt)
    # print("updated_pt", updated_pt)
    # print("center", center)
    # print("width", width)
    # print("height", height)
    # print("angle", angle)
    return (center[0], center[1]), width, height, angle, fixed_pt, updated_pt


class InteractiveEllipseCorrection:
    def __init__(self, master, f, ax, ellipses=[], savepath=None):

        self.stop_sign =False

        self.master = master
        self.savepath = savepath

        self.key = False
        self.click = -1

        self.pos_canvas = None
        self.prev_pos_canvas = None

        self.f = f
        self.ax = ax

        self.canvas = FigureCanvasTkAgg(self.f, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.ellipses = [ellipse for ellipse in ellipses]
        self.ellipse_points = [ellipse.get_verts().copy() for ellipse in self.ellipses]
        for ellipse in self.ellipses:
            self.ax.add_patch(ellipse)

        self.fixed_pt, = self.ax.plot([], [], marker="o", color="g")
        self.updated_pt, = self.ax.plot([], [], marker="o", color="r")

        self.canvas.callbacks.connect('button_press_event', self.on_click)
        self.canvas.callbacks.connect('motion_notify_event', self.motion)
        self.canvas.callbacks.connect('key_press_event', self.on_key_pressed)
        self.canvas.callbacks.connect('key_release_event', self.on_key_released)

        button_save = Button(master, text="Save", command=self.save)
        # button_save.grid(row=0, column=0)
        button_save.pack()
        button_next = Button(master, text="Next image", command=self.on_closing)
        # button_next.grid(row=0, column=1)
        button_next.pack()
        button_quit = Button(master, text="Quit", command=self.quit)
        # button_quit.grid(row=0, column=3)
        button_quit.pack()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def quit(self):
        self.stop_sign = True
        self.on_closing()

    def find_closest_ellipse(self, x, y):
        if len(self.ellipses) == 0:
            self.closest_ellipse_idx = -1
            return self.closest_ellipse_idx

        distances = []
        for ellipse in self.ellipses:
            distance = (x - ellipse.center[0]) ** 2 + (y - ellipse.center[1]) ** 2
            distances.append(distance)

        self.closest_ellipse_idx = np.argmin(distances)
        current_ellipse = self.ellipses[self.closest_ellipse_idx]
        self.extremas, self.main_ellipse_axis = get_ellipse_extremities(current_ellipse)

        return self.closest_ellipse_idx

    def update_ellipse(self, x, y):
        # 4 scenari:
        # click = -1: do nothing
        # click = 0: move left point of ellipse most horizontal axis
        # click = 1: move right point / horizontal axis of ellipse
        # click = 2: define other axis length

        # click = -1
        if self.click == -1: return

        current_ellipse = self.ellipses[self.closest_ellipse_idx]
        if self.click <= 1:
            center, width, height, angle, fixed_pt, updated_pt = update_ellipse_extremity(x, y, current_ellipse,
                                                                                          self.main_ellipse_axis,
                                                                                          (self.click + 1) % 2)
            self.fixed_pt.set_xdata(fixed_pt[0])
            self.fixed_pt.set_ydata(fixed_pt[1])
            self.updated_pt.set_xdata(updated_pt[0])
            self.updated_pt.set_ydata(updated_pt[1])

        # click = 1: change short axis length
        if self.click == 2:
            center, width, height, angle, _, _ = update_ellipse_extremity(x, y, current_ellipse,
                                                                          self.main_ellipse_axis, 1,
                                                                          short_axis=True,
                                                                          center_fixed=True,
                                                                          axis_fixed=True)

        current_ellipse.center = center
        current_ellipse.width = width
        current_ellipse.height = height
        current_ellipse.angle = angle

        self.canvas.draw()

    def motion(self, event):
        if event.inaxes is None: return
        x = event.xdata
        y = event.ydata

        if self.prev_pos_canvas is not None:
            self.x = x
            self.y = y

            if self.click != -1:
                self.update_ellipse(x, y)

                self.canvas.draw()
        elif self.pos_canvas is None:
            self.pos_pixel = self.master.winfo_pointerxy()
            self.pos_canvas = (x, y)
        elif self.prev_pos_canvas is None:
            self.prev_pos_canvas = self.pos_canvas
            self.prev_pos_pixel = self.pos_pixel
            self.pos_pixel = self.master.winfo_pointerxy()
            if self.pos_pixel[0] == self.prev_pos_pixel[0] or self.pos_pixel[1] == self.prev_pos_pixel[1]:
                self.prev_pos_canvas = None
                return
            self.pos_canvas = (x, y)
            self.coeff_x = (self.pos_pixel[0] - self.prev_pos_pixel[0]) / (
            (self.pos_canvas[0] - self.prev_pos_canvas[0]))
            self.coeff_y = (self.pos_pixel[1] - self.prev_pos_pixel[1]) / (
            (self.pos_canvas[1] - self.prev_pos_canvas[1]))
            self.ori_pixel = (self.pos_pixel[0] - self.coeff_x * self.pos_canvas[0],
                              self.pos_pixel[1] - self.coeff_y * self.pos_canvas[1])

    def on_click(self, event):
        if event.inaxes is None: return
        if self.prev_pos_canvas is None: return

        if self.click == -1:
            self.find_closest_ellipse(self.x, self.y)
        if event.button == 1:
            self.on_click_left(event)

    def on_click_left(self, event):
        if self.key:
            self.click = -1
        else:
            self.click = (self.click + 1) % 3

        current_ellipse = self.ellipses[self.closest_ellipse_idx]

        # get ellipse current point position at click
        fixed_pt, updated_pt = get_fixed_and_updated_point_based_on_main_axis(current_ellipse, (
                    self.main_ellipse_axis + self.click // 2 * 2) % 4, (self.click + 1) % 2)
        x_click_ellipse = int(updated_pt[0])
        y_click_ellipse = int(updated_pt[1])

        import ctypes

        SetCursorPos = ctypes.windll.user32.SetCursorPos
        SetCursorPos(int(x_click_ellipse * self.coeff_x + self.ori_pixel[0]),
                     int(y_click_ellipse * self.coeff_y + self.ori_pixel[1]))

        self.ellipse_points[self.closest_ellipse_idx] = current_ellipse.get_verts().copy()

    def on_key_pressed(self, event):
        self.key = (event.key == 'control')
        if self.key:
            print("key pressed")

    def on_key_released(self, event):
        if event.key == 'control':
            self.key = False
            print("key released")

    def save(self):
        if self.savepath is None:
            savepath = asksaveasfile()
        else:
            savepath = self.savepath
        os.makedirs(os.path.dirname(self.savepath), exist_ok=True)
        print("saving ellipse at location", self.savepath)

        ellipse_points_list = \
        [(ellipse._center[0], ellipse._center[1], ellipse.width, ellipse.height, ellipse.angle) for ellipse in
         self.ellipses][0]
        # ellipse_points_list = [list(pt.flatten()) for pt in self.ellipse_points]

        print(ellipse_points_list)
        json.dump(ellipse_points_list, open(savepath, "w"), indent=True, sort_keys=4)

    def on_closing(self):
        confirm_save = messagebox.askyesnocancel("Quit", "Do you want to save annotations before exiting?")
        if confirm_save is None:
            return
        if confirm_save:
            self.save()
        self.master.destroy()


def call_interactive_ellipse_correction(img, savepath):
    # X1 = np.concatenate([np.arange(10),np.arange(10)]).reshape(2,10).T

    a = img.shape[1] // 4
    b = img.shape[0] // 4
    center = np.array([img.shape[1], img.shape[0]]) // 2
    phi = 0

    # print("original angle of rotation", phi)
    # print("center", center)
    # print("axes", a, b)

    root = Tk()
    # root = Toplevel()
    f, ax = plt.subplots()
    ax.imshow(img)
    # ax.set_axis_off()plot

    ellipse1 = Ellipse(center, a * 2, b * 2, phi, fill=False, linewidth=2)

    corr = InteractiveEllipseCorrection(root, f, ax, ellipses=[ellipse1], savepath=savepath)
    root.mainloop()
    return corr.stop_sign

