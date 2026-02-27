<a id="readme-top"></a>

<br />
<div align="center">

<h3 align="center">StegRevealer</h3>

  <p align="center">
    Application to analyze steganography algorithm in the given image.
    <br />
    <br />
    <a href="https://github.com/Binus-D6954/stegrevealer/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/Binus-D6954/stegrevealer/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][step-3-screenshot]](https://github.com/Binus-D6954/stegrevealer)

This application purpose is to detect the algorithm used for the stegaography behind image used. Algorithm that can be detected in this application are as follows:

* Least Significant Bit (LSB)
* Bit-plane Complexity Segmentation (BPCS)
* Pixel Value Differencing (PVD)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python-shield]][Python-url]
* [![PyTorch][PyTorch-shield]][PyTorch-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Here's the list of softwares that you will need to install to use the application.
* Python 3.13.11

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Binus-D6954/stegrevealer
   ```
2. Install Python dependencies
   ```sh
   pip install -r requirements.txt
   ```
3. Download compressed file of SRNet model from [this Google Drive link][SRNet-trained-url]
4. Extract the compressed file of SRNet model to current directory which consists of `main.py` file
  ```
  stegrevealer
  |-- main.py
  |-- srnet_best.pth (get from srnet_best.pth.zip)
   \- ...
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

To use the application, here are the steps that you need to do:
1. Execute main.py script
    ```sh
    python main.py
    ```
2. A window will be shown as below, please click "Select Image File" and choose an image file that you want to check.
    ![step-2][step-2-screenshot]
3. After choosing an image file, result will be shown as below.
    ![step-3][step-3-screenshot]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TRAIN MODEL -->
## Train Model
To train your own SRNet model, you can use `srnet_steganalysis-train.ipynb` file to train your own SRNet model. You may follow step-by-step provided in IPYNB file.

<!-- CONTACT -->
## Contact

Sandyka Bala - D7207 - sandyka.bala@binus.ac.id

Christopher Limawan - D6954 - christopher.limawan@binus.ac.id

Project Link: [https://github.com/Binus-D6954/stegrevealer](https://github.com/Binus-D6954/stegrevealer)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[step-2-screenshot]: images/step-2.png
[step-3-screenshot]: images/step-3.png
[Python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org
[PyTorch-shield]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org
[SRnet-trained-url]: https://drive.google.com/file/d/1_3_krafQOfuUb9FmKL5kCJoxfWJysadm/view?usp=sharing