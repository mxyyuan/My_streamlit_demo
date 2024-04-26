import torch
from PIL import Image
import numpy as np
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import streamlit as st
from model import CNNnet_cifar10

def main():
    with open("gradcam.md", 'r') as file:
        markdown_text = file.read()
    readme_text=st.markdown(markdown_text)
    mode = st.sidebar.selectbox("Choose the dataset",
        ["Cifar10", "ImageNet"])
    if mode == "Cifar10":
       cifar10()
    elif mode == "ImageNet":
       imagenet()


def cifar10():
    uploaded_file = st.file_uploader("please input a Cifar10 image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # 将图像转换为 NumPy 数组
        img_array = np.array(image)
        st.write(img_array.shape)
        # 使用 Streamlit 显示图片
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        model=torch.load("cifar10", map_location="cpu")
        transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(32),

                ]
            )
        image=transformer(img_array).unsqueeze(0)
        cam = GradCAM(model,input_shape=(3,32,32),target_layer="conv3")
        output= model(image)
        class_idx = torch.argmax(output.squeeze(0)).item()
        activation_map = cam(class_idx, output)
        alpha=st.sidebar.slider("Alpha", 0.0, 1.0, 0.5, 0.01)
        gradcam=overlay_mask(
                            to_pil_image(image.squeeze(0)),
                            to_pil_image(activation_map[0].squeeze(0), mode='F'),
                            alpha=alpha,
                        )
        st.image(gradcam,use_column_width=True)
def imagenet():
    uploaded_file = st.file_uploader("please input an ImageNet image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # 将图像转换为 NumPy 数组
        img_array = np.array(image)
        st.write(img_array.shape)
        # 使用 Streamlit 显示图片
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        model=torch.load("./imagenet",map_location="cpu")
        transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(227),
                ]
            )
        image=transformer(img_array).unsqueeze(0)
        st.write(image.shape)
        cam = GradCAM(model,input_shape=(3,227,227))
        output= model(image)
        class_idx = torch.argmax(output.squeeze(0)).item()
        activation_map = cam(class_idx, output)
        alpha=st.sidebar.slider("Alpha", 0.0, 1.0, 0.5, 0.01)
        gradcam=overlay_mask(
                            to_pil_image(image.squeeze(0)),
                            to_pil_image(activation_map[0].squeeze(0), mode='F'),
                            alpha=alpha,
                        )
        st.image(gradcam,use_column_width=True)
if __name__ == "__main__":
    main()
