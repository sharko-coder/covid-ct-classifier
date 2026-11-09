import numpy as np
import torch
import cv2
import streamlit as st
from PIL import Image
from torchvision import transforms

from model import get_model
from data import MEAN, STD

CLASSES = ["COVID", "non-COVID"]
CKPT = "outputs/resnet_mixed_best.pt"
THRESHOLD = 0.12  # P(COVID) above this is flagged; chosen on validation data

st.set_page_config(page_title="COVID-19 CT Classifier", layout="centered")


@st.cache_resource
def load_model():
    model = get_model("resnet", pretrained=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    store = {}
    layer = model.layer4[-1]
    layer.register_forward_hook(lambda m, i, o: store.update(a=o))
    layer.register_full_backward_hook(lambda m, gi, go: store.update(g=go[0]))
    return model, store


TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict_with_cam(model, store, img):
    x = TF(img).unsqueeze(0)
    out = model(x)
    probs = torch.softmax(out, dim=1)[0].detach().numpy()
    cls = 0 if probs[0] > THRESHOLD else 1  # 0 = COVID flagged

    model.zero_grad()
    out[0, cls].backward()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["a"]).sum(1)).squeeze().detach().numpy()
    cam = cv2.resize(cam / (cam.max() + 1e-8), (224, 224))

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]
    base = np.array(img.resize((224, 224)))
    overlay = (0.55 * base + 0.45 * heat).astype("uint8")
    return cls, probs, overlay


st.title("COVID-19 CT Scan Classifier")
st.warning(
    "**Educational demo only. Not a medical device.** This model was trained on two "
    "public datasets and has not been clinically validated. It must not be used to "
    "make health decisions. Real diagnosis requires PCR testing and a clinician."
)
st.caption("Privacy: please do not upload real patient scans. Use public or sample images only.")

model, store = load_model()
file = st.file_uploader("Upload a chest CT slice (PNG/JPG)", type=["png", "jpg", "jpeg"])

if file is not None:
    img = Image.open(file).convert("RGB")
    cls, probs, overlay = predict_with_cam(model, store, img)

    col1, col2 = st.columns(2)
    col1.image(img.resize((224, 224)), caption="Uploaded scan")
    col2.image(overlay, caption="Grad-CAM: regions the model focused on")

    if cls == 0:
        st.error("Flagged: pattern the model associates with COVID-19")
    else:
        st.success("Not flagged by the model")
    st.progress(float(probs[0]), text=f"P(COVID) = {probs[0]:.1%} (flag threshold {THRESHOLD:.0%})")
    st.progress(float(probs[1]), text=f"P(non-COVID) = {probs[1]:.1%}")
    st.caption(
        "The flag threshold is deliberately low to favor catching COVID cases, so "
        "false alarms are expected. The probabilities are not calibrated. Heatmaps "
        "show where the model looked, not a diagnosis. The model only works on "
        "axial chest CT slices and will still give an answer for any image, "
        "including non-CT images."
    )
else:
    st.info("Upload a CT image to see a prediction.")

with st.expander("How well does this model work? (read before trusting it)"):
    st.markdown(
        """
**Held-out test results (threshold 0.12)**

| Test set | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Kaggle SARS-CoV-2 CT (held-out split) | 94.6% | 96.2% | 93.1% |
| COVID-CT (held-out images, n=193) | 80.3% | 82.2% | 78.6% |

**Limitations**
- The Kaggle data has no patient IDs and only 120 patients, so near-identical
  slices from one patient can sit in both train and test. Its numbers are inflated.
- A model trained on Kaggle alone scored close to chance (AUC about 0.5) on COVID-CT.
  The 80% above comes after adding part of COVID-CT to training, so it shows
  adaptation to a second data source, not generalization to unseen scanners.
- The COVID-CT test set is small, so these numbers are uncertain by about ±6 points.
- Many COVID-CT images are cropped from papers and contain arrows and text, which
  the model may react to.
        """
    )