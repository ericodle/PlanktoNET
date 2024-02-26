# PlanktoNET
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://github.com/ericodle/PlanktoNET/blob/main/img/D20230307T053258_IFCB108_02078.png" alt="Logo" width="600" height="300">
  </a>

<h3 align="center">PlanktoNET: Automated identification of planktonic biodiversity *Research Ongoing* </h3>

  <p align="center">
  This repository houses code generated during development of PlanktoNET, which is intended for a broader project exploring the biodiversity of plankton around Japan. This project is a joint effort between research teams at the Okinawa Institute of Science and Technology and Hokkaido University.
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About this Project

Automated data collection is enabled using an imaging flow cytobot (FCB) which
generates large datasets that can be used for monitoring marine biodiversity.
However, the bottleneck for analysis is often that species identification requires
specific expertise and is extremely time consuming.
Currently, the available software packages do not have their training dataset and
code available which restricts the flexibility for application by groups or locations.
Here we propose to develop an open-source software package designed for batch
identification of plankton using a classifier to process data collected from an FCB.
Samples will be collected from ecologically different locations, Okinawa and
Hokkaido; 3 sites in Okinawa (Sesoko island, Ikei Island, and Seragaki) and 2 sites in
Hokkaido (Otaru, Akkeshi). A training dataset will be generated from samples
collected which we will use for training and testing.

<p align="right">(<a href="#top">back to top</a>)</p>

## Prerequisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Enter this into your computer's command line interface (terminal, control panel, etc.) to check the version:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.

## Setup

Here is an easy way to use our GitHub repository.

### Step 1: Clone the repository


Open the command line interface and run:
  ```sh
  git clone git@github.com:ericodle/PlanktoNET.git
  ```

You have now downloaded the entire project, including all its sub-directories (folders) and files.
(We will avoid using Git commands.)

### Step 2: Navigate to the project directory
Find where your computer saved the project, then enter:

  ```sh
  cd /path/to/project/directory
  ```

If performed correctly, your command line interface should resemble

```
user@user:~/PlanktoNET$
```

### Step 3: Create a virtual environment: 
Use a **virtual environment** so library versions on your computer match the versions used during development and testing.


```sh
python3 -m venv -p python3.8 planktonet-env
```

A virtual environment named "planktonet-env" has been created. 
Enter the environment to do our work by using the following command:

```sh
source planktonet-env/bin/activate
```

When performed correctly, your command line interface prompt should look like 

```
(planktonet-env) user@user:~/PlanktoNET$
```

### Step 3: Install requirements.txt

Avoid "dependency hell" by installing specific software versions known to work well together.

  ```sh
pip install -r requirements.txt
  ```

### Step 4: Use the project

The core of this project is contained in the **src** directory. 

For example, if you want to pre-process your own data using the image_preprocessing.py script, enter:


```sh
python3 src/image_preprocessing.py
```


### Step 5: Deactivate the virtual environment

When finished working, it is best to deactivate the virtual environment and change directory (cd) out of the project directory. Enter the following command:

  ```sh
deactivate
cd ~
  ```

...or you can just close the command line interface window.


### Step 6: Access publicly available datasets

 We used the following public image datasets for model training and validation. Feel free to download them as needed.

> McLane Labs set: around 7,000 photos obtained using the Imaging FlowCytobot (https://mclanelabs.com/imaging-flowcytobot). 
> EcoTaxa set: projects 9621 (https://ecotaxa.obs-vlfr.fr/prj/9621) and 9942 (https://ecotaxa.obs-vlfr.fr/prj/9942) containing approx. xyz manually labeled photos.



## src scripts

  

### Fine-Tuning a Pre-trained Model

    Fine-tune a pre-trained model (e.g., VGG, ResNet) using one of the provided fine-tuning scripts in the repository.
    Specify the path to your dataset and the number of classes.
    Train the model for a specified number of epochs.

### Classifying Unsorted Images

    Place your unsorted images in a directory.
    Use the classification script to organize unsorted images based on the fine-tuned model's predictions.

<!-- LICENSE -->
## License

Distributed under the GNU Lesser General Public License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


Citing
------

Please cite the future paper, coming soon!



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
