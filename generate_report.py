"""Generate the AI 100 Midterm PDF report."""

from fpdf import FPDF
import json
import os

OUTPUT_DIR = "outputs"
REPORT_PATH = os.path.join(OUTPUT_DIR, "AI100_Midterm_Report.pdf")


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "AI 100 Midterm Report - Kreesh Patel", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(230, 230, 240)
        self.cell(0, 10, f"  {num}. {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(4)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 11)
        x = self.get_x()
        self.cell(8, 6, "- ")
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bold_label(self, label, value):
        self.set_font("Helvetica", "B", 11)
        self.cell(self.get_string_width(label) + 2, 6, label)
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, value)
        self.ln(1)


def build_report():
    with open(os.path.join(OUTPUT_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title page ──────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 26)
    pdf.cell(0, 14, "MNIST Handwritten Digit Classification", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "with Convolutional Neural Networks", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 10, "AI 100 -Midterm Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 13)
    pdf.cell(0, 8, "Kreesh Patel", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, "GitHub: github.com/kreeshpatel/ai100-mnist-cnn-classification", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Section 1: Problem Definition & Dataset ─────────────────
    pdf.add_page()
    pdf.section_title("1", "Problem Definition and Dataset Curation")

    pdf.body_text(
        "The goal of this project is to build a deep learning model that can accurately classify "
        "handwritten digits (0 through 9) from grayscale images. Handwritten digit recognition is "
        "a foundational problem in computer vision and serves as an excellent benchmark for evaluating "
        "neural network architectures."
    )

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Dataset: MNIST", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.body_text(
        "The MNIST (Modified National Institute of Standards and Technology) dataset is one of the "
        "most widely used benchmarks in machine learning. It was originally compiled by Yann LeCun "
        "and colleagues and contains handwritten digits collected from Census Bureau employees and "
        "high school students."
    )

    pdf.bold_label("Training set: ", "60,000 images")
    pdf.bold_label("Test set: ", "10,000 images")
    pdf.bold_label("Image size: ", "28 x 28 pixels, single-channel grayscale")
    pdf.bold_label("Classes: ", "10 (digits 0-9)")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Preprocessing", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.body_text(
        "Each image is converted to a PyTorch tensor and normalized using the dataset's global mean "
        "(0.1307) and standard deviation (0.3081). This normalization centers pixel values around zero "
        "and helps the network converge faster during training. No data augmentation was applied, as "
        "the standard MNIST benchmark does not typically require it to achieve high accuracy."
    )

    # ── Section 2: Deep Learning Models ─────────────────────────
    pdf.add_page()
    pdf.section_title("2", "Deep Learning Model")

    pdf.body_text(
        "The model is a Convolutional Neural Network (CNN) implemented in PyTorch. CNNs are the "
        "standard architecture for image classification tasks because their convolutional layers "
        "can automatically learn spatial features such as edges, curves, and shapes directly from "
        "raw pixel data."
    )

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Architecture (MNISTNet)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Courier", "", 10)
    arch_text = (
        "Input (1 x 28 x 28)\n"
        "  -> Conv2d(1, 32, kernel=3, padding=1) -> ReLU -> MaxPool(2)    [32x14x14]\n"
        "  -> Conv2d(32, 64, kernel=3, padding=1) -> ReLU -> MaxPool(2)   [64x7x7]\n"
        "  -> Flatten                                                      [3136]\n"
        "  -> Linear(3136, 128) -> ReLU\n"
        "  -> Linear(128, 10)                                              [10 logits]"
    )
    pdf.multi_cell(0, 5, arch_text)
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    pdf.body_text(
        "The first convolutional layer extracts 32 low-level feature maps (edges, simple patterns). "
        "The second convolutional layer builds 64 higher-level feature maps from those. Each "
        "convolutional layer is followed by a ReLU activation for non-linearity and a 2x2 max-pooling "
        "layer that halves the spatial dimensions and provides translation invariance. The output is "
        "flattened into a 3,136-dimensional vector and passed through two fully connected layers, "
        "producing 10 raw logits (one per digit class)."
    )

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Training Configuration", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.bold_label("Loss function: ", "CrossEntropyLoss (combines LogSoftmax + NLLLoss)")
    pdf.bold_label("Optimizer: ", "Adam (lr = 0.001)")
    pdf.bold_label("Batch size: ", "64")
    pdf.bold_label("Epochs: ", "5")
    pdf.bold_label("Random seed: ", "42 (for reproducibility)")
    pdf.bold_label("Device: ", "CPU")
    pdf.ln(2)

    pdf.body_text(
        "Adam was chosen as the optimizer because it adapts learning rates per parameter, which "
        "generally leads to faster convergence than standard SGD on this type of task. "
        "CrossEntropyLoss is the standard choice for multi-class classification problems."
    )

    # ── Section 3: Results ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("3", "Results")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Training Performance", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Results table
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(70, 70, 120)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(30, 8, "Epoch", border=1, align="C", fill=True)
    pdf.cell(50, 8, "Training Loss", border=1, align="C", fill=True)
    pdf.cell(50, 8, "Training Accuracy", border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)

    for i in range(metrics["epochs"]):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 250)
        pdf.cell(30, 7, str(i + 1), border=1, align="C", fill=fill)
        pdf.cell(50, 7, f"{metrics['train_losses'][i]:.4f}", border=1, align="C", fill=fill)
        pdf.cell(50, 7, f"{metrics['train_accuracies'][i]:.2f}%", border=1, align="C", fill=fill)
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 100, 0)
    pdf.cell(0, 10, f"Final Test Accuracy: {metrics['test_accuracy']:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.body_text(
        "The model converges rapidly, achieving over 95% training accuracy after just the first epoch. "
        "By epoch 5, training accuracy reaches 99.54% and the test accuracy is 99.02%, demonstrating "
        "strong generalization with minimal overfitting."
    )

    # Loss curve
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Training Loss Curve", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    loss_img = os.path.join(OUTPUT_DIR, "loss_curve.png")
    if os.path.exists(loss_img):
        pdf.image(loss_img, x=25, w=160)
    pdf.ln(4)

    pdf.body_text(
        "The loss decreases sharply after the first epoch and continues to decline steadily, "
        "indicating effective learning without signs of divergence."
    )

    # Accuracy curve
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Training Accuracy Curve", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    acc_img = os.path.join(OUTPUT_DIR, "acc_curve.png")
    if os.path.exists(acc_img):
        pdf.image(acc_img, x=25, w=160)
    pdf.ln(4)

    pdf.body_text(
        "Training accuracy climbs quickly to above 98% by epoch 2 and plateaus near 99.5% by epoch 5, "
        "showing the model has effectively learned the digit patterns."
    )

    # Confusion matrix
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Confusion Matrix (Test Set)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    cm_img = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    if os.path.exists(cm_img):
        pdf.image(cm_img, x=30, w=150)
    pdf.ln(4)

    pdf.body_text(
        "The confusion matrix shows the model classifies nearly all test digits correctly. "
        "The diagonal is dominant with very few off-diagonal misclassifications. "
        "The most common errors involve visually similar digit pairs such as 4/9 and 3/5, "
        "which is expected and consistent with human confusion patterns."
    )

    # ── Section 4: Lessons & Experience ─────────────────────────
    pdf.add_page()
    pdf.section_title("4", "Lessons and Experience Learned")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key Takeaways", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    lessons = [
        (
            "CNNs are well-suited for image tasks: ",
            "Convolutional layers automatically learn spatial features (edges, curves, digit shapes) "
            "without manual feature engineering. This is a major advantage over traditional machine "
            "learning approaches that require hand-crafted features."
        ),
        (
            "Data normalization matters: ",
            "Normalizing inputs to zero mean and unit variance helped the model train faster and "
            "more stably. Without normalization, gradient magnitudes can vary wildly across features."
        ),
        (
            "Simple architectures can be highly effective: ",
            "A two-layer CNN with only ~400K parameters achieved 99.02% test accuracy on MNIST. "
            "This demonstrates that more complex models are not always necessary -choosing the right "
            "architecture for the problem is more important than adding layers."
        ),
        (
            "Reproducibility is important: ",
            "Setting a random seed ensured consistent results across runs, making it easier to "
            "compare experiments and debug issues during development."
        ),
        (
            "PyTorch provides a clean workflow: ",
            "The combination of Dataset/DataLoader for data handling, nn.Module for model definition, "
            "and autograd for backpropagation made the end-to-end implementation straightforward."
        ),
    ]

    for bold_part, normal_part in lessons:
        pdf.set_font("Helvetica", "", 11)
        x = pdf.get_x()
        pdf.cell(8, 6, "- ")
        pdf.set_font("Helvetica", "B", 11)
        pdf.write(6, bold_part)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, normal_part)
        pdf.ln(3)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Future Improvements", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    improvements = [
        "Add dropout layers to reduce overfitting on more complex datasets.",
        "Experiment with data augmentation (rotation, scaling) to improve robustness.",
        "Try deeper architectures or batch normalization for potentially higher accuracy.",
        "Extend the project to harder datasets like Fashion-MNIST or CIFAR-10.",
    ]

    for item in improvements:
        pdf.bullet(item)

    # ── Save ────────────────────────────────────────────────────
    pdf.output(REPORT_PATH)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
