import streamlit as st
import numpy as np
from tqdm.auto import trange
import torch

from source import Model, Block


def cvtImg(img):
    img = img - img.min()
    img = (img / img.max())
    return img.numpy().astype(np.float32)


def load_DM():
    PATH = "./Diffusion-cuda"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write("Devise used:", device)
    model = Model().to(device)
    st.write("Diffusion Model's architecture created.")

    try:
        # Load the entire model object
        model = torch.load(PATH, map_location=device)
        st.write("Diffusion Model loaded.")
    except FileNotFoundError:
        st.error("Model file not found at specified path.")
    except Exception as e:
        st.error(f"Error occurred while loading model: {e}")

    return model, device


def predict_step_new(model, device):
    xs = []
    IMG_SIZE = 32
    timesteps = 16

    progress_text = "Denoising the image."
    my_bar = st.progress(0, text=progress_text)
    x = torch.randn(size=(1, 3, IMG_SIZE, IMG_SIZE), device=device)
    placeholder = st.empty()
    li = []
    with torch.no_grad():
        for i in trange(timesteps):
            my_bar.progress(i*100//15, text=progress_text)
            t = i
            x = model(x, torch.full([8, 1], t, dtype=torch.float, device=device))
            xs = x[0].permute(1, 2, 0)
            xs = torch.clip(xs, -1, 1)
            xs = cvtImg(xs.cpu())
            images = np.transpose(xs, (2, 0, 1))
            images = np.transpose(images, (1, 2, 0))
            placeholder.image(images, caption=f"Time step {t}", width=500)
            li.append(images)
    st.markdown("Brand new **car** generated!")
    return li


def main():
    st.set_page_config(page_title="Diffusion Model App Demo")
    st.header("Diffusion Model App 🖼️ 🚕", divider="rainbow")
    st.markdown("""Made by:
    Marijan$^\dag$, Thomas, Mauritz, Amanda, Baris for **COGITO** Diffusion Model""")

    if "model" not in st.session_state:
        st.session_state.model = []

    if "generation" not in st.session_state:
        st.session_state.generation = []

    if "device" not in st.session_state:
        st.session_state.device = None

    placeholder_1 = st.empty()
    isclick = placeholder_1.button('Import Diffusion Model')
    if isclick:
        st.session_state.model, st.session_state.device = load_DM()
        placeholder_1.empty()

    if st.button('Generate!'):
        li = predict_step_new(st.session_state.model, st.session_state.device)
        st.session_state.generation.append(li)
        for num, gen in enumerate(st.session_state.generation):
            st.write("Generation number", num+1)
            st.image(gen, width=40)


if __name__ == "__main__":
    main()
