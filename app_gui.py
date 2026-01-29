import tkinter as tk
from tkinter import filedialog, messagebox
from core.registry import BLOCKS

class VisionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Vision Pipeline Application")
        self.geometry("500x400")

        tk.Label(self, text="Vision Experiments", font=("Arial", 16)).pack(pady=10)

        self.exp_var = tk.StringVar(self)
        self.exp_menu = tk.OptionMenu(
            self,
            self.exp_var,
            *[f"{k}. {v[0]}" for k, v in BLOCKS.items()]
        )
        self.exp_menu.pack(pady=10)

        tk.Button(self, text="Select Input Image", command=self.select_image).pack(pady=5)
        tk.Button(self, text="Run Experiment", command=self.run_experiment).pack(pady=10)

        self.input_path = None

    def select_image(self):
        self.input_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )

    def run_experiment(self):
        if not self.exp_var.get():
            messagebox.showerror("Error", "Select an experiment")
            return

        if not self.input_path:
            messagebox.showerror("Error", "Select an input image")
            return

        exp_key = self.exp_var.get().split(".")[0]
        _, run_func = BLOCKS[exp_key]

        try:
            result = run_func(self.input_path, "data/output")
            messagebox.showinfo("Success", str(result))
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = VisionApp()
    app.mainloop()
