# PlanktoNET
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="img/planktonet_logo.png" width="350" title="logo">
  </a>

<h3 align="center">PlanktoNET: Automated identification of planktonic biodiversity *Research Ongoing* </h3>

  <p align="center">
  This repository houses code generated during development of PlanktoNET, which is intended for a broader project exploring the biodiversity of plankton around Japan. This project is a joint effort between research teams at the Okinawa Institute of Science and Technology and Hokkaido University.
    <br />

<!-- ABOUT THE PROJECT -->
## About this Project

  <p align="center">
  The primary objective of this project is to develop an integrated "default" sorting mechanism. Upon completion of supervised training, users will have the capability to sort their images without the necessity of manual intervention. Subsequently, the model can be further refined and expanded to accommodate individual datasets and additional locations within the database. This has consistently been the overarching objective of the project.
    <br />
  </p>

  <p align="center">
  <img src="https://github.com/ericodle/PlanktoNET/blob/main/img/workflow_diagram.png" alt="Logo" width="700" height="900">
    <br />
  </p>
  
 
## Prerequisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Enter this into your computer's command line interface (terminal, control panel, etc.) to check the version:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.

## Setup

Currently, this is the easest way to run the project.

### Step 1: Download the repository

Download it to your computer. 

### Step 2: Unpack and change directory to the repository

Find where your computer saved the project. 
Unzip/unpack/decompress it, then enter:

  ```sh
  cd /path/to/project/directory
  ```

This is now the working directory.

### Step 3: Create a virtual environment: 
The default size limit on PyPI is 60MB.
Therefore, we will have to take the **virtual environment** route.

Create a virtual environment called *env* inside the working directory.

```sh
python3 -m venv env
```

Then, activate the virtual environment.


```sh
source env/bin/activate
```

### Step 4: Install requirements.txt

Avoid "dependency hell" by installing specific software versions known to work well together.

  ```sh
pip3 install -r requirements.txt
  ```

### Step 4: Run PlanktoNET

From inside the working directory, with virtual environment active and dependencies installed, run:

```sh
python3 -m planktonet.run_planktonet
```

#### Sort with New Model ####

This utility allows users to perform plankton image sorting using state-of-the-art neural network models. Users have the option to choose between Convolutional Neural Networks (CNNs) and Transformer Neural Networks. By selecting this option, users can specify input images, choose an appropriate model, and designate an output directory for the sorted images. This functionality is ideal for users who want to classify plankton images using newly trained models.

#### Sort with Existing Model ####

This utility enables users to utilize previously trained models for sorting plankton images. Users can select an existing model file and provide input images to initiate the sorting process. This feature is beneficial for users who have already trained models on specific datasets and wish to apply them to new plankton image sorting tasks.

#### Finetune Model ####

This utility allows users to refine the performance of pre-trained neural network models using their own dataset. Users can select a base model, specify training data, adjust hyperparameters such as learning rate and the number of images per class, and designate an output directory for the fine-tuned model. This feature is useful for users who want to adapt pre-trained models to better suit their specific plankton image sorting requirements.

#### Evaluate Model ####

This utility provides users with the capability to assess the performance of trained neural network models for plankton image sorting tasks. While this functionality is currently under development in the application, it will offer users valuable insights into the accuracy and effectiveness of their trained models, aiding in further refinement and optimization of sorting processes.

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
