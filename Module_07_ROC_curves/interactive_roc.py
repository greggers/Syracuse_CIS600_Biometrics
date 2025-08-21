import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class InteractiveROCApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive ROC Curve with Normal Distributions")

        # Default parameters
        self.mean_pos = 1.0
        self.std_pos = 0.75
        self.mean_neg = -1.0
        self.std_neg = 0.5

        # UI frame for inputs
        input_frame = ttk.Frame(master)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Positive class mean input
        ttk.Label(input_frame, text="Mean (Positive):").grid(row=0, column=0, sticky=tk.W)
        self.mean_pos_var = tk.DoubleVar(value=self.mean_pos)
        self.mean_pos_entry = ttk.Entry(input_frame, width=7, textvariable=self.mean_pos_var)
        self.mean_pos_entry.grid(row=0, column=1, padx=5)

        # Positive class std dev input
        ttk.Label(input_frame, text="Std Dev (Positive):").grid(row=1, column=0, sticky=tk.W)
        self.std_pos_var = tk.DoubleVar(value=self.std_pos)
        self.std_pos_entry = ttk.Entry(input_frame, width=7, textvariable=self.std_pos_var)
        self.std_pos_entry.grid(row=1, column=1, padx=5)

        # Negative class mean input
        ttk.Label(input_frame, text="Mean (Negative):").grid(row=0, column=2, sticky=tk.W)
        self.mean_neg_var = tk.DoubleVar(value=self.mean_neg)
        self.mean_neg_entry = ttk.Entry(input_frame, width=7, textvariable=self.mean_neg_var)
        self.mean_neg_entry.grid(row=0, column=3, padx=5)

        # Negative class std dev input
        ttk.Label(input_frame, text="Std Dev (Negative):").grid(row=1, column=2, sticky=tk.W)
        self.std_neg_var = tk.DoubleVar(value=self.std_neg)
        self.std_neg_entry = ttk.Entry(input_frame, width=7, textvariable=self.std_neg_var)
        self.std_neg_entry.grid(row=1, column=3, padx=5)

        # Update button
        update_btn = ttk.Button(input_frame, text="Update Distributions", command=self.update_distributions)
        update_btn.grid(row=0, column=4, rowspan=2, padx=10)

        # Initialize matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.95, hspace=0.3)

        # Create canvas for embedding matplotlib in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize slider for threshold
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        self.slider = Slider(ax_slider, 'Threshold', -5, 5, valinit=0, valstep=0.01)
        self.slider.on_changed(self.update_plot_threshold)

        # Initialize point_plot and text to None
        self.point_plot = None
        self.text = None

        # Prepare x-axis for normal distributions
        self.x = np.linspace(-6, 6, 1000)

        # Initially prepare data and draw plots
        self.prepare_data()
        self.draw_plots()
        self.update_plot_threshold(self.slider.val)

    def prepare_data(self):
        # Fetch means and std devs from UI
        try:
            self.mean_pos = float(self.mean_pos_var.get())
            self.std_pos = float(self.std_pos_var.get())
            self.mean_neg = float(self.mean_neg_var.get())
            self.std_neg = float(self.std_neg_var.get())
            if self.std_pos <= 0 or self.std_neg <= 0:
                raise ValueError("Std dev must be positive")
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            # Don't update data if invalid input
            return

        # Calculate pdfs
        self.pdf_pos = (1 / (self.std_pos * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((self.x - self.mean_pos) / self.std_pos) ** 2)
        self.pdf_neg = (1 / (self.std_neg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((self.x - self.mean_neg) / self.std_neg) ** 2)

        # Generate sample scores for ROC
        np.random.seed(42)  # for reproducibility
        self.scores = np.concatenate([
            np.random.normal(self.mean_pos, self.std_pos, 1000),
            np.random.normal(self.mean_neg, self.std_neg, 1000)])
        self.labels = np.concatenate([np.ones(1000), np.zeros(1000)])

        # Range for threshold slider depends on data range
        self.min_score = np.min(self.scores)
        self.max_score = np.max(self.scores)
        self.slider.valmin = self.min_score
        self.slider.valmax = self.max_score
        self.slider.ax.set_xlim(self.min_score, self.max_score)

        # Precompute ROC curve
        thresholds = np.linspace(self.min_score, self.max_score, 500)
        self.tprs = []
        self.fprs = []
        for thresh in thresholds:
            tpr, fpr, _, _ = self.compute_roc(thresh)
            self.tprs.append(tpr)
            self.fprs.append(fpr)

    def compute_roc(self, threshold):
        TP = np.sum(self.scores[self.labels == 1] >= threshold)
        FP = np.sum(self.scores[self.labels == 0] >= threshold)
        FN = np.sum(self.scores[self.labels == 1] < threshold)
        TN = np.sum(self.scores[self.labels == 0] < threshold)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        return TPR, FPR, TP, FP

    def draw_plots(self):
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        # Top plot: distributions
        self.ax1.plot(self.x, self.pdf_pos, label=f'Positive class (μ={self.mean_pos}, σ={self.std_pos})', color='blue')
        self.ax1.plot(self.x, self.pdf_neg, label=f'Negative class (μ={self.mean_neg}, σ={self.std_neg})', color='red')
        self.ax1.set_title('Class Score Distributions')
        self.ax1.set_xlabel('Score')
        self.ax1.set_ylabel('Probability Density')
        self.ax1.legend()
        self.ax1.grid(True)

        # Bottom plot: ROC curve
        self.ax2.plot(self.fprs, self.tprs, label='ROC Curve', color='green')
        self.ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        self.ax2.set_title('ROC Curve')
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.legend()
        self.ax2.grid(True)

        # Initialize point_plot and text if needed
        if self.point_plot is None or self.text is None:
            self.point_plot, = self.ax2.plot([], [], 'ro')
            self.text = self.ax2.text(0.6, 0.1, '', bbox=dict(facecolor='white', alpha=0.8))

        self.canvas.draw_idle()

    def update_plot_threshold(self, val):
        threshold = self.slider.val
        TPR, FPR, TP, FP = self.compute_roc(threshold)
        if self.point_plot is not None and self.text is not None:
            self.point_plot.set_data([FPR], [TPR])
            self.text.set_text(f'Threshold: {threshold:.2f}\nTPR: {TPR:.2f} (TP={TP})\nFPR: {FPR:.2f} (FP={FP})')
            self.canvas.draw_idle()

    def update_distributions(self):
        self.prepare_data()
        self.draw_plots()
        self.update_plot_threshold(self.slider.val)

if __name__ == '__main__':
    root = tk.Tk()
    app = InteractiveROCApp(root)
    root.mainloop()