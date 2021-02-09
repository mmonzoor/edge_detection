import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import signal
from PIL import Image
import argparse
import Derivatives
import Linker 


def rgb_to_gray(I_rgb):
    '''
    Helper function that coverts RGB image to grey
    '''
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return I_gray

def show_img(img_array):
    '''Helper function to turn image array for display'''
    im = Image.fromarray(img_array)
    imrgb = im.convert("RGB")
    return imrgb

def canny_edge(I, low_thresh, high_thresh):
    '''Function to apply the canny edge algorithm steps'''
    I_gray = rgb_to_gray(I)
    derivs = Derivatives.findDerivs(I_gray)
    Mag, Magx, Magy = derivs.find_derivatives()
    
    Ori = derivs.calculate_orientation()
    cm_hot = plt.cm.get_cmap("nipy_spectral")
    ori = cm_hot(Ori)
    im = np.uint8(ori * 255)
    im = Image.fromarray(im)
    imrgb = im.convert("RGB")

    link = Linker.linkedEdges(Mag, Ori)
    nms = link.non_max_suppression()

    e = link.edge_link(low_thresh, high_thresh)
    return Mag, nms, e

def set_st_values(Images, filename):
    '''
    Helper function to fetch images from a file and 
    run canny edge. 
    '''
    img_path = os.path.join(Images, filename)
    I = plt.imread(img_path)
    
    Mag, nms, edge_link = canny_edge(I, 20, 40)
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        st.image(I)
    with col2: 
        st.image(show_img(Mag))
    with col3: 
        st.image(show_img(nms))
    with col4: 
        st.image(show_img(edge_link))

st.title("Got an Edge?")
st.markdown('''Author: [Mumtahin Monzoor](https://www.linkedin.com/in/mmonzoor/) Data Scientist | Strategist | Life Long Learner]''')


st.header("Canny Edge Detection")

st.markdown(
    """A commonly used edge detection algorithm developed by [John F. Canny in 1986](https://10.1109/TPAMI.1986.4767851). 
When we look at a photo, we see color, black and white elements, and most importantly we see boundaries between elements. The 
boundaries between elements in a photo is called an *edge*.  

Edges can be created due to many reasons such as 3-D depth changes, lighting changes between objects, surface color
discontinuity, etc. Before we delve into the steps required for Canny Edge detection, it is important to understand the concepts of 
*Image Filtering* and *Image convolution*.  

**Image Filtering**: A process of replacing pixel values in an image by the weighted average of its neighbors. It is done by 
taking an image matrix *I* and applying the weights from a kernel matrix *f* to get a new weighted matrix. This process allows for certain 
features in an image to become more pronounced as the pixels become representative of its neighborhood.  
"""
)
st.latex(
    r"""
g[m, n] = \sum_{k,l} I(m+k, n+l))*f(k,l)
"""
)

st.markdown(
    """
 - g[m, n] is the output image
 - I(m+k,n+l) is the input image
 - f(k, l) is the kernel image
"""
)

# cover convolution
st.markdown(
    """
**Convolution**: This is a similar operation to filtering with slight differences.
"""
)
st.latex(
    r"""
g[m, n] = \sum_{k,l} I(m-k, n-l))*f(k,l)
"""
)

st.markdown(
    """Notice, we have a formula very similar to that of image filtering but 
we are subtracting k and l. What this means is if we want to look at a pixel at location
(m, n) -- instead of directly looking at its corresponding kernel value, we will look at the 
location on the opposite spot of the kernel (minus direction)."""
)

st.subheader(
    """
Objective: Find the Edges!
"""
)

st.markdown(
    """
We want to look at a pixel and ask the following question: based on the region around our target
pixel *i, j*, can we make a decision on whether we have an edge going through *i, j*? 
Posing this question for each pixel *i, j*, we turn this into binary problem such that *B(i, j)*
equals 1 if I(i, j) has an edge and
0 otherwise.  
"""
)

st.subheader(
    """
1. Filter out noise
"""
)

st.write(
    """Edges can be impacted by noise in the image. In order to make the 
noise less pronounced, we convolve the original image using the derivative of
the Gaussian filter G. To approximate the derivatives of the Gaussians in the x and y directions
we will use the [Sobel Operator](https://www.researchgate.net/publication/285159837_A_33_isotropic_gradient_operator_for_image_processing).
The operator consists of a pair of 3x3 convolutional kernel."""
)

st.latex(
    r"""
I_o \bigotimes (\frac{\delta}{\delta x}G) = I_1
"""
)

st.latex(r"""
S_x = (\frac{1}{8})\begin{bmatrix}
-1 & 0 & 1\\
-2 & 0 & 2\\
-1 & 0 & 1\\
\end{bmatrix}""")

st.latex(r"""
S_y = (\frac{1}{8})
\begin{bmatrix}
1 & 2 & 1\\
0 & 0 & 0\\
-1 & -2 & -1\\
\end{bmatrix}
""")
st.write("For the purposes of edge detection, we can ignore the 1/8.")
st.latex(r"""G_x = S_y \bigotimes G
""")
st.latex(r"""G_y = S_y\bigotimes G""")
st.write("Using this we can convolve the original image I with Gx and Gy.")
st.text(
    "I_o is the original image and I_1 is the final image after the convolution step."
)

# SHOW AN EXAMPLE
I = plt.imread("Images/tutorial.jpg")
I_gray = rgb_to_gray(I)
tut_deriv = Derivatives.findDerivs(I_gray)
Mag, Magx, Magy = tut_deriv.find_derivatives()
Ori = tut_deriv.calculate_orientation()

st.subheader("""2. Calculate magnitude of change""")
st.write(
    """Once we have calculated the changes in x and y direction, 
we can calculate the magnitude of the gradient using I_x and I_y"""
)
st.latex("""I_m = \sqrt(I_x^2 + I_y^2)""")

with st.beta_container():
    initialcol, magxcol, magycol, magcol = st.beta_columns(4)
    with initialcol:
        st.subheader(r"$$I_o$$")
        st.image(I)
        st.text("Original Image")
    with magxcol:
        st.subheader(r"$$I_x$$")
        st.image(show_img(Magx))
        st.text("X change (step 1)")
    with magycol:
        st.subheader(r"$$I_y$$")
        st.image(show_img(Magy))
        st.text("""Y change (step 1)""")
    with magcol:
        st.subheader(r"$$I_m$$")
        st.image(show_img(Mag))
        st.text("Magnitude (step 2)")
st.markdown(
    """[Ref Image](https://i.pinimg.com/originals/45/c6/74/45c67453f0916f7ee0723d2475ee08cb.jpg)"""
)

st.subheader("""3. Edge Orientation""")

st.write(
    """We want to know which way an edge is going so we can interpolate a direction
    for that edge. Looking at the gradient of the pixel intensity change in the x and y ddirection, 
    we can compute this orientation. Take the intensity gradient of an image I:"""
)

st.latex(r"""\Delta I = [(\frac{\delta I}{\delta x}), (\frac{\delta I}{\delta y})]""")

st.write(
    "Breaking this into partial derivatives to calculate change in the x and then the y direction:"
)
st.latex(r"""\Delta I = [(\frac{\delta I}{\delta x}), 0)]""")
st.latex(r"""\Delta I = [0, (\frac{\delta I}{\delta y})]""")

with st.beta_container():
    desc_orient, img_orient = st.beta_columns(2)
    with desc_orient:

        st.write(
            """The idea is that if we have a diagonal edge, than the derivative function y and 
        derivative function x, points perpendicular to that edge. This is the derivative function
        at a point that provides the direction in which the intensity is changing. If grident is 
        pointing to the way where intensity change is maximum, then perpendicular to it is the 
        orientation of that edge"""
        )
        st.latex(
            r"\theta = tan^{-1} ([(\frac{\delta I}{\delta x}), (\frac{\delta I}{\delta y})])"
        )
    with img_orient:
        # apply cmap
        cm_hot = plt.cm.get_cmap("nipy_spectral")
        ori = cm_hot(Ori)
        im = np.uint8(ori * 255)
        im = Image.fromarray(im)
        imrgb = im.convert("RGB")
        st.image(imrgb)

st.subheader("""4. Non-maximum Suppression""")
st.write(
    """Now that we know edge magnitude and edge orientation, we can use these components 
to localize the edges to a finer degree. We detect the local maximum in this step and remove unwanted pixels. 
We do this by asking for each pixel *i, j*, is this the local maximum in its neighborhood of pixels pointing 
to edge orientation? If so, we keep and otherwise we remove."""
)

tut_linker = Linker.linkedEdges(Mag, Ori)
NMS = tut_linker.non_max_suppression()
st.image(
    show_img(NMS)
)
st.subheader("""5. Hysteresis Thresholding""")
st.write(
    """At this point, we have identified a lot of edges. As you can see from image above, there are many 
outlines on the objects and we need to decide which edges are **actually** edges. For this step 
we will have two threshold values, *min* and *max* values. Edges that score below the *min*
values will be considered as noise and not edges. Edges that are above *max* values will be 
deemed as **strong** edges. All the edges scoring in the range between min and max will 
only be considered as an edge if they link to a strong edge. This link can be direct 
or through a chain of linkages."""
)

with st.beta_container():
    st.subheader(
        "keeping the low threshold consistent, observe changes in edge linking."
    )

    edge_col1, edge_col2, edge_col3, edge_col4 = st.beta_columns(4)

    with edge_col1:
        st.text("low:20, high:30")
        E = tut_linker.edge_link(20, 30)
        st.image(show_img(E))
    with edge_col2:
        st.text("low: 20, high: 40")
        E = tut_linker.edge_link(20, 40)
        st.image(show_img(E))
    with edge_col3:
        st.text("low: 20, high: 50")
        E = tut_linker.edge_link(20, 50)
        st.image(show_img(E))
    with edge_col4:
        st.text("low: 20, high: 100")
        E = tut_linker.edge_link(20, 100)
        st.image(show_img(E))

with st.beta_container():
    st.subheader(
        "keeping the high threshold consistent, observe changes in edge linking."
    )
    edge_col1, edge_col2, edge_col3, edge_col4 = st.beta_columns(4)

    with edge_col1:
        st.text("low:30, high:50")
        E = tut_linker.edge_link(30, 50)
        st.image(show_img(E))
    with edge_col2:
        st.text("low: 25, high: 50")
        E = tut_linker.edge_link(25, 50)
        st.image(show_img(E))
    with edge_col3:
        st.text("low: 20, high: 50")
        E = tut_linker.edge_link(20, 50)
        st.image(show_img(E))
    with edge_col4:
        st.text("low: 15, high: 50")
        E = tut_linker.edge_link(15, 50)
        st.image(show_img(E))

st.write(
    """Observing the output images, it seems like when *low = 20* and *high = 40*, we get the best edge detection. At this 
step, you can fine tune and apply a lot more combinations of high and low thresholds for a more robust comparison."""
)

st.subheader("Result of Canny Edge Detection")
col_original, col_result = st.beta_columns(2)
with col_original:
    st.image(I)
with col_result:
    E = tut_linker.edge_link(20, 40)
    st.image(show_img(E))
    
st.markdown("From the images above, we see that there is scope to change the low and high thresholds for the very last step to add more or less details.")

st.header("Expanding the concepts to other images")
col1, col2, col3, col4 = st.beta_columns(4)
with col1:
    st.write('Original')
with col2:
    st.write(r'Mag $$\Delta$$')
with col3:
    st.write("Non-max Suppression")
with col4:
    st.write("Hysteresis")

for filename in os.listdir("Images"):
    if filename != "tutorial.jpg":
        set_st_values("Images", filename)


st.subheader('Your Turn!')
st.write('''Upload an image below 2MBs to see what edges get detected. Feel free to
input your own custom thresholds.''')

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer:
    image = plt.imread(img_file_buffer)
    if image.size < 2000000*8: 
        low_thresh = st.number_input('Low thresh', value=20)
        high_thresh = st.number_input('High thresh', value=40)
        cols_user1, cols_user2 = st.beta_columns(2) 
        st.image(
            image,
            caption="submitted image",
            use_column_width=True,
        )
        st.write(low_thresh, high_thresh)
        st.image(
            show_img(canny_edge(image, low_thresh, high_thresh)[2]),
            caption="edge detection result",
            use_column_width=True,
        )
        st.write(f'''Current thresholds being used: {low_thresh}, {high_thresh}''')
    else:
        st.error("The file you tried to upload is too large!")


st.subheader("""Conclusion""")
st.markdown(
    """ Edge detection is an image processing technique that can be used in a wide range of industries such as agriculture, 
medicine, finance, etc. The ability to detect the presence of certain boundaries in an image can result in us knowing 
whether a certain object appears in said image or not. If you want to chat more about Canny edge detection or would 
like to provide feedback on the walkthrough above, please contact me on [github](https://github.com/mmonzoor)!
"""
)

st.subheader("""References""")
st.markdown(
    """
- SZELISKI, R. (2020). Computer Vision: Algorithms and applications. SPRINGER NATURE.
- Forsyth, D., Ponce, J., Mukherjee, S., &amp; Bhattacharjee, A. K. (2015), Computer vision: A modern approach.
Uttar Pradesh, India: Pearson India Education Services Pvt.
"""
)
